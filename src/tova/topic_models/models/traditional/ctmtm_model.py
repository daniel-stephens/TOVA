import logging
import pathlib
import time
from typing import Optional, List

import numpy as np
from sklearn.preprocessing import normalize  # type: ignore
from tqdm import tqdm

from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation  # type: ignore
from contextualized_topic_models.models.ctm import CombinedTM  # type: ignore

from tova.utils.cancel import CancellationToken, check_cancel
from tova.utils.progress import ProgressCallback  # type: ignore
from tova.topic_models.models.traditional.base import TradTMmodel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CTMTMmodel(TradTMmodel):
    """
    CTMTMmodel class for training and inferring topics using the Combined Topic Model (CTM).
    Mirrors TomotopyLDATMmodel structure & logs. Requires precomputed embeddings.
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
        model_name = model_name if model_name else f"{self.__class__.__name__}_{int(time.time())}"
        super().__init__(model_name, corpus_id, id, model_path, logger, config_path, load_model, do_embeddings=True)

        ctm_cfg = self.config.get("ctm", {})
        self.num_epochs = int(ctm_cfg.get("num_epochs", 100))
        self.sbert_model = str(ctm_cfg.get("sbert_model", "all-MiniLM-L6-v2"))
        self.sbert_context = int(ctm_cfg.get("sbert_context", 384))
        self.batch_size = int(ctm_cfg.get("batch_size", 32))
        self.contextual_size = int(ctm_cfg.get("contextual_size", 384))
        self.inference_type = str(ctm_cfg.get("inference_type", "combined"))
        self.n_components = int(ctm_cfg.get("n_components", 10))
        self.model_type = str(ctm_cfg.get("model_type", "prodLDA"))
        self.hidden_sizes = ctm_cfg.get("hidden_sizes", [100, 100])
        self.activation = str(ctm_cfg.get("activation", "softplus"))
        self.dropout = float(ctm_cfg.get("dropout", 0.2))
        self.learn_priors = bool(ctm_cfg.get("learn_priors", True))
        self.lr = float(ctm_cfg.get("lr", 0.002))
        self.momentum = float(ctm_cfg.get("momentum", 0.99))
        self.solver = str(ctm_cfg.get("solver", "adam"))
        self.reduce_on_plateau = bool(ctm_cfg.get("reduce_on_plateau", False))
        self.num_data_loader_workers = int(ctm_cfg.get("num_data_loader_workers", 4))
        self.label_size = int(ctm_cfg.get("label_size", 0))
        self.loss_weights = ctm_cfg.get("loss_weights", None)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._logger.info(
            f"{self.__class__.__name__} initialized with configuration:\n"
            f"  num_topics={self.num_topics}, num_epochs={self.num_epochs}\n"
            f"  sbert_model='{self.sbert_model}', sbert_context={self.sbert_context}\n"
            f"  contextual_size={self.contextual_size}, inference_type='{self.inference_type}'\n"
            f"  n_components={self.n_components}, model_type='{self.model_type}'\n"
            f"  hidden_sizes={self.hidden_sizes}, activation='{self.activation}'\n"
            f"  dropout={self.dropout}, learn_priors={self.learn_priors}\n"
            f"  lr={self.lr}, momentum={self.momentum}, solver='{self.solver}'\n"
            f"  batch_size={self.batch_size}, reduce_on_plateau={self.reduce_on_plateau}\n"
            f"  num_data_loader_workers={self.num_data_loader_workers}, label_size={self.label_size}\n"
            f"  loss_weights={self.loss_weights}"
        )

        self.qt: Optional[TopicModelDataPreparation] = None
        self.vocab: Optional[List[str]] = None


    def train_core(
        self,
        prs: Optional[ProgressCallback] = None,
        cancel: Optional[CancellationToken] = None
    ):
        """
        Train CTM and return (elapsed_time, thetas, betas, vocab).
        """

        self._logger.debug(f"train_data is set: {hasattr(self, 'train_data') and self.train_data is not None}")
        self._logger.debug(f"embeddings is set: {hasattr(self, 'embeddings') and self.embeddings is not None}")

        t_start = time.time()

        # 1) TopicModelDataPreparation (0–10%)
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.0, 0.1) if prs else None
        prss and prss.report(0.0, "Creating TopicModelDataPreparation and fitting dataset")
        self.qt = TopicModelDataPreparation(self.sbert_model)

        raw_texts = self.df["raw_text"].tolist()
        bow_texts = [" ".join(lemmas) for lemmas in self.train_data]

        tr_ds = self.qt.fit(
            text_for_contextual=raw_texts,
            text_for_bow=bow_texts,
            custom_embeddings=self.embeddings
        )
        self.vocab = list(tr_ds.idx2token.values())
        prss and prss.report(1.0, "Training dataset ready")

        # 2) Create CTM (10–15%)
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.1, 0.15) if prs else None
        prss and prss.report(0.0, "Creating CombinedTM model")
        self.model = CombinedTM(
            bow_size=len(self.vocab),
            contextual_size=self.contextual_size,
            n_components=self.num_topics,
            model_type=self.model_type,
            hidden_sizes=tuple(self.hidden_sizes),
            activation=self.activation,
            dropout=self.dropout,
            learn_priors=self.learn_priors,
            batch_size=self.batch_size,
            lr=self.lr,
            momentum=self.momentum,
            solver=self.solver,
            num_epochs=self.num_epochs,
            reduce_on_plateau=self.reduce_on_plateau,
            num_data_loader_workers=self.num_data_loader_workers,
            label_size=self.label_size,
            loss_weights=self.loss_weights
        )
        self.model.qt = self.qt
        prss and prss.report(1.0, "CombinedTM model created")

        # 3) Train (15–90%)
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.15, 0.9) if prs else None
        pbar = tqdm(total=1, desc="Training CTM")
        prss and prss.report(0.0, "Training CTM model...")
        self.model.fit(tr_ds)
        pbar.update(1)
        pbar.close()
        prss and prss.report(1.0, "Training complete")

        # 4) Distributions (90–100%)
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.9, 1.0) if prs else None
        prss and prss.report(0.9, "Calculating topics and distributions...")

        thetas = np.asarray(self.model.get_doc_topic_distribution(tr_ds))
        self._logger.info(f"Thetas shape: {thetas.shape}")

        betas = np.asarray(self.model.get_topic_word_distribution())
        self._logger.info(f"Betas shape: {betas.shape}")

        keys = self.print_topics(verbose=False)
        with self.model_path.joinpath('orig_tpc_descriptions.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join([' '.join(topic) for topic in keys]))

        prss and prss.report(1.0, "Topics and distributions calculated")

        return time.time() - t_start, thetas, betas, self.vocab

    def print_topics(self, verbose: bool = False) -> list:
        topics_dict = self.model.get_topics(self.topn)  # {k: [w1, w2, ...]}
        keys = [[w for w in topics_dict[k]] for k in sorted(topics_dict.keys())]
        if verbose:
            for k, words in enumerate(keys):
                print(f"Topic {k}: {words}")
        return keys

    # --------------------------- Core inference --------------------------

    def infer_core(self, infer_data, df_infer, embeddings_infer):
        """
        Inference on unseen docs. Requires precomputed embeddings in df_infer.
        Returns (thetas, elapsed_seconds).
        """
        t0 = time.perf_counter()
        self._logger.info("Performing inference on unseen documents...")

        if embeddings_infer is None:
            raise RuntimeError(
                "CTM requires precomputed embeddings. "
                "Call prepare_infer_data(..., compute_embeddings=True) before inference."
            )
        if self.qt is None:
            raise RuntimeError("TopicModelDataPreparation not initialized. Train or load a model first.")

        self._logger.info("Preparing holdout dataset...")
        raw_texts = df_infer["raw_text"].tolist()
        bow_texts = [" ".join(lemmas) for lemmas in infer_data]

        ho_ds = self.qt.transform(
            text_for_contextual=raw_texts,
            text_for_bow=bow_texts,
            custom_embeddings=embeddings_infer
        )

        self._logger.info("Inferring thetas...")
        thetas = np.asarray(self.model.get_doc_topic_distribution(ho_ds))
        self._logger.info(f"Inferred thetas shape: {thetas.shape}")

        thetas[thetas < self.thetas_thr] = 0
        thetas = normalize(thetas, axis=1, norm="l1")

        return thetas, (time.perf_counter() - t0)


    def save_model(self):
        """
        Save CTM artifacts to disk.
        - Model object (pickle): model.pkl
        - Vocabulary: vocab.txt (one token per line)
        - TopicModelDataPreparation (pickle): qt.pkl
        """
        import pickle

        model_pkl = self.model_path.joinpath('model.pkl')
        vocab_txt = self.model_path.joinpath('vocab.txt')
        qt_pkl = self.model_path.joinpath('qt.pkl')

        self._logger.info(f"Saving CTM model to {model_pkl.as_posix()}")
        with model_pkl.open('wb') as f:
            pickle.dump(self.model, f)

        if self.vocab is not None:
            self._logger.info(f"Saving vocabulary to {vocab_txt.as_posix()}")
            with vocab_txt.open('w', encoding='utf8') as f:
                f.write('\n'.join(self.vocab))

        if self.qt is not None:
            self._logger.info(f"Saving TopicModelDataPreparation to {qt_pkl.as_posix()}")
            with qt_pkl.open('wb') as f:
                pickle.dump(self.qt, f)

        self._logger.info("Model saved successfully!")

    @classmethod
    def from_saved_model(cls, model_path: str):
        """
        Load a previously saved CTMTMmodel from disk.

        Expects:
        - model.pkl (pickled CombinedTM)
        - vocab.txt (optional)
        - qt.pkl (optional but recommended for inference transforms)
        """
        import pickle

        obj = cls(model_path=model_path, load_model=True)

        model_pkl = pathlib.Path(model_path).joinpath('model.pkl')
        vocab_txt = pathlib.Path(model_path).joinpath('vocab.txt')
        qt_pkl = pathlib.Path(model_path).joinpath('qt.pkl')

        if not model_pkl.exists():
            raise FileNotFoundError(
                f"CTM model not found at {model_pkl.as_posix()}. "
                "Expected a pickled CombinedTM at 'model.pkl'."
            )

        obj._logger.info(f"Loading CTM model from {model_pkl.as_posix()}")
        with model_pkl.open('rb') as f: obj.model = pickle.load(f)
        obj._logger.info("Model loaded successfully!")

        if vocab_txt.exists():
            obj._logger.info(f"Loading vocabulary from {vocab_txt.as_posix()}")
            with vocab_txt.open('r', encoding='utf8') as f:
                obj.vocab = [w.strip() for w in f if w.strip()]
        else:
            obj.vocab = None
            obj._logger.warning("Vocabulary file 'vocab.txt' not found. Proceeding without vocab.")

        if qt_pkl.exists():
            obj._logger.info(f"Loading TopicModelDataPreparation from {qt_pkl.as_posix()}")
            with qt_pkl.open('rb') as f:
                obj.qt = pickle.load(f)
            # Restore reference for transforms
            if hasattr(obj.model, "qt") is False or obj.model.qt is None:
                obj.model.qt = obj.qt
        else:
            obj.qt = getattr(obj.model, "qt", None)
            if obj.qt is None:
                obj._logger.warning(
                    "TopicModelDataPreparation 'qt.pkl' not found and model has no attached `qt`. "
                    "Inference will fail until `qt` is provided (train again or supply qt.pkl)."
                )

        return obj
