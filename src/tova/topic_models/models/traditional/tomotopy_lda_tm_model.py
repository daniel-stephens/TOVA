import logging
import pathlib
import time
from typing import Optional

import numpy as np
import tomotopy as tp  # type: ignore
from sklearn.preprocessing import normalize  # type: ignore
from tqdm import tqdm

from tova.utils.cancel import CancellationToken, check_cancel
from tova.utils.progress import ProgressCallback  # type: ignore

from .base import TradTMmodel

class TomotopyLDATMmodel(TradTMmodel):
    """
    TomotopyLdaModel class for training and inferring topics using the Tomotopy library.
    """

    def __init__(
        self,
        model_name: str = None,
        corpus_id: str = None,
        id: str = None,
        model_path: str = None,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path("./static/config/config.yaml"),
        load_model: bool = False,
        **kwargs
    ) -> None:
        """
        Initialize the TomotopyLdaModel class.

        Parameters
        ----------
        model_path : str, optional
            Path to save the trained model.
        logger : logging.Logger, optional
            Logger object to log activity.
        config_path : pathlib.Path, optional
            Path to the configuration file.
        **kwargs :
            Override configuration parameters with explicit arguments.
        """

        model_name = model_name if model_name else f"{self.__class__.__name__}_{int(time.time())}"
        
        super().__init__(model_name, corpus_id, id, model_path, logger, config_path, load_model)

        # Load parameters from config
        tomotopy_config = self.config.get("tomotopy", {})
        self.num_iters = int(tomotopy_config.get("num_iters", 1000))
        self.alpha = float(tomotopy_config.get("alpha", 0.1))
        self.eta = float(tomotopy_config.get("eta", 0.01))
        self.iter_interval = int(tomotopy_config.get("iter_interval", 10))

        # Allow overriding parameters from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Log initialization
        self._logger.info(
            f"{self.__class__.__name__} initialized with num_topics={self.num_topics}, num_iters={self.num_iters}, alpha={self.alpha}, eta={self.eta}.")
        

    def train_core(self, prs: Optional[ProgressCallback] = None, cancel: Optional[CancellationToken] = None) -> float:
        """
        Train the topic model and save the data to the specified path.

        Returns
        -------
        float
            Time taken to train the model.
        """

        if not hasattr(self, "train_data"):
            raise RuntimeError("Training data not set. Call train_model(data) with normalized input first.")

        t_start = time.time()
        
        # 1. CREATE TOMOTOPY OBJECT (0-2%)
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.0, 0.02) if prs else None
        prss.report(0.0, "Creating TomotopyLDA object")
        self.model = tp.LDAModel(
            k=self.num_topics, tw=tp.TermWeight.ONE, alpha=self.alpha, eta=self.eta)
        prss.report(1.0, "TomotopyLDA object created")
        
        # 2. ADD DOCUMENTS TO TOMO OBJECT (2-30%)
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.02, 0.3) if prs else None
        prss.report(0.02, "Adding documents to TomotopyLDA object")
        [self.model.add_doc(doc) for doc in self.train_data]
        prss.report(0.3, "Documents added to TomotopyLDA object")

        # 3. TRAIN MODEL (30-90%)
        prss = prs.report_subrange(0.3, 0.9) if prs else None
        pbar = tqdm(total=self.num_iters, desc='Training Progress')
        for i in range(0, self.num_iters, self.iter_interval):
            check_cancel(cancel, self._logger)
            self.model.train(self.iter_interval)
            pbar.update(self.iter_interval)
            prss.report(i / self.num_iters, f"Training iteration {i}/{self.num_iters}, Log-likelihood: {self.model.ll_per_word}, Perplexity: {self.model.perplexity}")
        pbar.close()
        prss.report(1.0, "Training complete")

        # 4. CALCULATE TOPICS AND DISTRIBUTIONS (90-100%)
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.9, 1.0) if prs else None
        prss.report(0.9, "Calculating topics and distributions...")
        probs = [d.get_topic_dist() for d in self.model.docs]
        thetas = np.array(probs)
        self._logger.info(f"Thetas shape: {thetas.shape}")

        topic_dist = [self.model.get_topic_word_dist(
            k) for k in range(self.num_topics)]
        betas = np.array(topic_dist)
        self._logger.info(f"Betas shape: {betas.shape}")

        keys = self.print_topics(verbose=False)
        with self.model_path.joinpath('orig_tpc_descriptions.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join([' '.join(topic) for topic in keys]))
        self.maked_docs = [self.model.make_doc(doc) for doc in self.train_data]
        vocab = [word for word in self.model.used_vocabs]
        prss.report(1.0, "Topics and distributions calculated")
        
        return time.time() - t_start, thetas, betas, vocab
        

    def print_topics(self, verbose: bool = False) -> list:
        """
        Print the list of topics for the topic model.

        Parameters
        ----------
        verbose : bool, optional
            If True, print the topics to the console, by default False.

        Returns
        -------
        list
            List with the keywords for each topic.
        """

        keys = [[tup[0] for tup in self.model.get_topic_words(
            k, self.topn)] for k in range(self.model.k)]

        if verbose:
            for k, words in enumerate(keys):
                print(f"Topic {k}: {words}")

        return keys

    def infer_core(self, infer_data, df_infer, embeddings_infer) -> np.ndarray:
        """
        Perform inference on unseen documents.

        Parameters
        ----------
        docs : List[str]
            List of documents to perform inference on.

        Returns
        -------
        np.ndarray
            Array of inferred thetas.
        """
        _ = df_infer, embeddings_infer

        time_start = time.perf_counter()
        self._logger.info("Performing inference on unseen documents...")

        self._logger.info("Adding docs to TomotopyLDA...")
        doc_inst = [self.model.make_doc(text) for text in infer_data]

        self._logger.info("Inferring thetas...")
        topic_prob, _ = self.model.infer(doc_inst)
        thetas = np.array(topic_prob)
        self._logger.info(f"Inferred thetas shape {thetas.shape}")

        # sparsify thetas
        thetas[thetas < self.thetas_thr] = 0
        thetas = normalize(thetas, axis=1, norm='l1')
        
        t_end = time.perf_counter() - time_start
        
        return thetas, t_end

    def save_model(self):
        # save model object for later use
        save_path = self.model_path.joinpath('model.bin').as_posix()
        self._logger.info(f"Saving model to {save_path}")
        self.model.save(save_path)
        self._logger.info("Model saved successfully!")

    @classmethod
    def from_saved_model(cls, model_path: str):
        """
        Load a previously saved TomotopyLDATMmodel from disk.
        
        Parameters
        ----------
        model_path : str
            Path where the model is stored (directory containing model.bin).
        
        Returns
        -------
        TomotopyLDATMmodel
            An instance with the model loaded in `.model`.
        """
        obj = cls(model_path=model_path, load_model=True)
        load_path = pathlib.Path(model_path).joinpath('model.bin').as_posix()
        obj._logger.info(f"Loading Tomotopy model from {load_path}")
        obj.model = tp.LDAModel.load(load_path)
        obj._logger.info("Model loaded successfully!")
        return obj
