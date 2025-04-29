import logging
import pathlib
import time
from typing import List
import numpy as np
from tqdm import tqdm # type: ignore
import tomotopy as tp # type: ignore
from .base import TradTMmodel

class TomotopyLDATMmodel(TradTMmodel):
    """
    TomotopyLdaModel class for training and inferring topics using the Tomotopy library.
    """

    def __init__(
        self,
        model_path: str = None,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path("./static/config/config.yaml"),
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

        super().__init__(model_path, logger, config_path)

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
        
        # Add 

    def train_model(self, path_to_data, text_col: str = "tokenized_text") -> float:
        """
        Train the topic model and save the data to the specified path.

        Parameters
        ----------
        path_to_data : str
            Path to the training data.
        text_col : str, default='tokenized_text'
            Column name containing the text data.

        Returns
        -------
        float
            Time taken to train the model.
        """

        self._load_train_data(
            path_to_data, get_embeddings=False, text_data=text_col)
        t_start = time.perf_counter()

        self._logger.info("Creating TomotopyLDA object and adding docs...")
        self.model = tp.LDAModel(
            k=self.num_topics, tw=tp.TermWeight.ONE, alpha=self.alpha, eta=self.eta)
        [self.model.add_doc(doc) for doc in self.train_data]

        self._logger.info(f"Training TomotopyLDA model with {self.num_topics} topics...")
        pbar = tqdm(total=self.num_iters, desc='Training Progress')
        for i in range(0, self.num_iters, self.iter_interval):
            self.model.train(self.iter_interval)
            pbar.update(self.iter_interval)
            if i % 300 == 0 and i > 0:
                topics = self.print_topics(verbose=False)
                pbar.write(f'Iteration: {i}, Log-likelihood: {self.model.ll_per_word}, Perplexity: {self.model.perplexity}')
        pbar.close()

        self._logger.info("Calculating topics and distributions...")
        probs = [d.get_topic_dist() for d in self.model.docs]
        thetas = np.array(probs)
        self._logger.info(f"Thetas shape: {thetas.shape}")

        topic_dist = [self.model.get_topic_word_dist(
            k) for k in range(self.num_topics)]
        betas = np.array(topic_dist)
        self._logger.info(f"Betas shape: {betas.shape}")

        keys = self.print_topics(verbose=False)
        self.maked_docs = [self.model.make_doc(doc) for doc in self.train_data]
        vocab = [word for word in self.model.used_vocabs]

        t_end = time.perf_counter() - t_start

        self._save_model_results(thetas, betas, vocab, keys)
        self.save_to_json()

        return t_end

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

    def infer(self, path_to_data: str, text_col: str) -> np.ndarray:
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

        # docs, _ = super().infer(docs)

        # self._logger.info("Performing inference on unseen documents...")
        # docs_tokens = [doc.split() for doc in docs]

        # self._logger.info("Adding docs to TomotopyLDA...")
        # doc_inst = [self.model.make_doc(text) for text in docs_tokens]

        # self._logger.info("Inferring thetas...")
        # topic_prob, _ = self.model.infer(doc_inst)
        # thetas = np.array(topic_prob)
        # self._logger.info(f"Inferred thetas shape {thetas.shape}")
        
        # TODO: Implement the inference logic

        return
    

    def save_model(self, path: str):
        # @TODO: @lcalvobartolome
        print("TO DO: Implement save_model")

    def load_model(self, path: str):
        # @TODO: @lcalvobartolome
        print("TO DO: Implement load_model")