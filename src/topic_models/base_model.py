from abc import ABC, abstractmethod
import json
import logging
import os
import pathlib
import shutil
import tomotopy as tp # type: ignore

import numpy as np
import pandas as pd  # type: ignore

from ..utils.common import init_logger, load_yaml_config_file
from ..utils.tm_utils import get_embeddings_from_str, read_dataframe


class BaseTMModel(ABC):
    """
    Abstract base class for topic models.
    """

    def __init__(
        self,
        model_path: str = None,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path("config/config.yaml")
    ) -> None:
        """
        Initialize the BaseTMModel class.

        Parameters
        ----------
        model_path : str, optional
            Path to save the trained model.
        logger : logging.Logger, optional
            Logger object to log activity.
        config_path : pathlib.Path, optional
            Path to the configuration file.
        """

        self._logger = logger if logger else init_logger(config_path, __name__)

        self.model_path = pathlib.Path(model_path)
        if self.model_path.exists():
            self._logger.info(
                f"Model path {self.model_path} already exists. Saving a copy..."
            )
            old_model_dir = self.model_path.parent / \
                (self.model_path.name + "_old")
            if not old_model_dir.is_dir():
                os.makedirs(old_model_dir)
                shutil.move(self.model_path, old_model_dir)

        self.model_path.mkdir(exist_ok=True, parents=True)

        # load config
        self.config = load_yaml_config_file(config_path, "topic_modeling", logger)
        
        self.num_topics = int(self.config.get("general", {}).get("num_topics", 50))
        self.topn = int(self.config.get("general", {}).get("topn", 15))
        self.not_include = self.config.get("general", {}).get("not_include", [])

    def _load_train_data(
        self,
        path_to_data: str,
        get_embeddings: bool = False,
        text_data: str = "tokenized_text",
        raw_text_data: str = "text"
    ) -> None:
        """
        Load the training data.

        Parameters
        ----------
        path_to_data : str
            Path to the training data.
        get_embeddings : bool, default=False
            Whether to load embeddings from the data.
        text_data : str, default='tokenized_text'
            Column name containing the text data.
        """

        path_to_data = pathlib.Path(path_to_data)
        self.text_col = text_data

        df = read_dataframe(path_to_data, self._logger)

        self.df = df
        self.train_data = [doc.split() for doc in df[text_data]]
        self._logger.info(f"Loaded processed data from {path_to_data}")

        if get_embeddings:
            if "embeddings" not in df.columns:
                self._logger.warning(
                    f"Embeddings required but not present in data. They will be calculated from the training data...")
                self.embeddings = None
                self.raw_text = [doc.split() for doc in df[raw_text_data]]
            else:
                self.embeddings = get_embeddings_from_str(df, self._logger)
                self.raw_text = None
                self._logger.info(f"Loaded embeddings from the DataFrame")
        else:
            self.embeddings = None

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
            and not k.startswith('_') and not isinstance(v, (np.ndarray, pd.DataFrame, pd.Series, list, dict, pathlib.Path, tp.Document, tp.LDAModel))}
            
    def save_to_json(self, path: pathlib.Path = None):
        json_path = path or (self.model_path / 'model_config.json')
        with json_path.open('w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @abstractmethod
    def save_model(self, path: str): pass

    @abstractmethod
    def load_model(self, path: str): pass

    @abstractmethod
    def train_model(self, path_to_data: str, text_col: str): pass

    @abstractmethod
    def infer(self, path_to_data: str, text_col: str): pass