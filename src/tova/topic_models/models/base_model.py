import json
import logging
import pathlib
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd  # type: ignore
import tomotopy as tp  # type: ignore
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation  # type: ignore
from contextualized_topic_models.models.ctm import CombinedTM  # type: ignore
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

    def to_dict(self) -> dict:
        def safe_value(val):
            if isinstance(val, pathlib.Path):
                return str(val)
            elif isinstance(val, (np.ndarray, pd.DataFrame, pd.Series)):
                return f"<{type(val).__name__}>"
            return val

        return {
            k: safe_value(v)
            for k, v in self.__dict__.items()
            if not k.startswith('_') and k not in self.not_include
            and not k.startswith('_') and not isinstance(v, (np.ndarray, pd.DataFrame, pd.Series, list, dict, pathlib.Path, tp.Document, tp.LDAModel, TopicModelDataPreparation, CombinedTM))}

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

    @abstractmethod
    def set_training_data(self, data: List[Dict]): pass

    @abstractmethod
    def prepare_infer_data(self, data: List[Dict]): pass

    @abstractmethod
    def save_model(self, path: str): pass

    @classmethod
    @abstractmethod
    def from_saved_model(self, path: str): pass

    @abstractmethod
    def train_model(self, data: List[Dict]): pass

    @abstractmethod
    def infer(self, data: List[Dict]): pass
