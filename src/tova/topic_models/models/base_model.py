import json
import logging
import pathlib
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
import tomotopy as tp  # type: ignore
from contextualized_topic_models.models.ctm import CombinedTM  # type: ignore
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation  # type: ignore
from scipy import sparse
from sklearn.preprocessing import normalize

from tova.topic_models.tm_model import TMmodel
from tova.utils.common import init_logger, load_yaml_config_file


class BaseTMModel(ABC):
    """
    Abstract base class for topic models.
    """

    def __init__(
        self,
        model_name: str,
        corpus_id: str,
        id: str,
        model_path: str = None,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path(
            "./static/config/config.yaml"),
        load_model: bool = False
    ) -> None:
        """
        Initialize the BaseTMModel class.

        Parameters
        ----------
        model_path : str, optional
            Path to save the trained model.
        model_name : str, optional
            Name of the model.
        logger : logging.Logger, optional
            Logger object to log activity.
        config_path : pathlib.Path, optional
            Path to the configuration file.
        """

        self._logger = logger if logger else init_logger(config_path, __name__)

        if not load_model:
            self.model_path = pathlib.Path(model_path)
            if self.model_path.exists():
                self._logger.info(
                    f"Model path {self.model_path} already exists. Renaming it..."
                )

                old_model_path = self.model_path.with_name(
                    self.model_path.name + "_old")

                if old_model_path.exists():
                    shutil.rmtree(old_model_path)

                shutil.move(str(self.model_path), str(old_model_path))
                self._logger.info(f"Old model moved to {old_model_path}")

            # Create a fresh model path
            self.model_path.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.corpus_id = corpus_id
        self.id = id

        # load config
        self.config = load_yaml_config_file(
            config_path, "topic_modeling", logger)

        self.num_topics = int(self.config.get(
            "general", {}).get("num_topics", 50))
        self.topn = int(self.config.get("general", {}).get("topn", 15))
        self.thetas_thr = float(self.config.get(
            "general", {}).get("thetas_thr", 3e-3))
        self.not_include = self.config.get(
            "general", {}).get("not_include", [])

        # llm params
        self.llm_provider = self.config.get("general", {}).get("llm_provider")
        self.llm_model_type = self.config.get(
            "general", {}).get("llm_model_type")
        self.llm_server = self.config.get("general", {}).get("llm_server")

    def to_dict(self) -> dict:
        def safe_value(val):
            if isinstance(val, pathlib.Path):
                return str(val)
            elif isinstance(val, (np.ndarray, pd.DataFrame, pd.Series)):
                return f"<{type(val).__name__}>"
            return val

        return {
            # TODO: make all the "no_include" as start with "_" to simplify this
            k: safe_value(v)
            for k, v in self.__dict__.items()
            if not k.startswith('_') and k not in self.not_include
            and not k.startswith('_') and not isinstance(v, (np.ndarray, pd.DataFrame, pd.Series, list, dict, pathlib.Path, tp.Document, tp.LDAModel, TopicModelDataPreparation, CombinedTM, TMmodel))}

    def save_to_json(self, path: pathlib.Path = None):
        json_path = path or (self.model_path / 'metadata.json')

        model_info = {
            "corpus_id": self.corpus_id,
            "path": self.model_path.as_posix(),
            "type": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "location": "temporal",
            "created_at": pd.Timestamp.now().isoformat(),
            "tr_params": self.to_dict(),
        }

        with json_path.open('w') as f:
            json.dump(model_info, f, indent=2)

    def _createTMmodel(self, thetas, betas, vocab, tpc_labels=None, tpc_summaries=None, add_info=None):
        """Creates an object of class TMmodel hosting the topic model
        that has been trained and whose output is available at the
        provided folder

        Parameters
        ----------
        thetas: np.ndarray
            Document-topic distributions matrix
        betas: np.ndarray
            Topic-word distributions matrix
        vocab: List[str]
            Vocabulary list
        tpc_labels: List[str], optional
            List of topic labels
        tpc_summaries: List[str], optional
            List of topic summaries
        add_info: dict, optional
            Dictionary containing additional information per topic

        Returns
        -------
        tm: TMmodel
            The topic model as an object of class TMmodel

        """
        # Sparsification of thetas matrix
        # self._save_thr_fig(thetas, self.model_path.joinpath('thetasDist.pdf'))

        # Set to zeros all thetas below threshold, and renormalize
        thetas[thetas < self.thetas_thr] = 0
        thetas = normalize(thetas, axis=1, norm='l1')
        thetas = sparse.csr_matrix(thetas, copy=True)

        # Recalculate topic weights to avoid errors due to sparsification
        alphas = np.asarray(np.mean(thetas, axis=0)).ravel()

        tm = TMmodel(
            TMfolder=self.model_path.joinpath('TMmodel'),
            config_path=self._config_path,
            df_corpus_train=self.df,
            do_labeller=self.do_labeller,
            do_summarizer=self.do_summarizer,
            llm_model_type=self.llm_model_type,
            labeller_prompt=self.labeller_prompt,
            summarizer_prompt=self.summarizer_prompt,
        )
        tm.create(
            betas=betas, thetas=thetas, alphas=alphas, vocab=vocab, tpc_labels=tpc_labels, tpc_summaries=tpc_summaries, add_info=add_info)

        self.tm_model = tm

        return tm

    @abstractmethod
    def set_training_data(self, data: List[Dict]): pass

    @abstractmethod
    def prepare_infer_data(self, data: List[Dict]): pass

    @abstractmethod
    def save_model(self, path: str): pass

    @classmethod
    def from_saved_model(cls, model_path: str):
        
        model_config = pathlib.Path(model_path).joinpath('metadata.json')            
        with open(model_config, 'r') as f:
            model_config = json.load(f)
            
        obj = cls(model_path=model_path, load_model=True, **model_config["tr_params"])
        obj.model_path = pathlib.Path(model_path)
        
        return obj    

    @abstractmethod
    def train_model(self, data: List[Dict]): pass

    @abstractmethod
    def infer(self, data: List[Dict]): pass
