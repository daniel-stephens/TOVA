from abc import ABC, abstractmethod
import json
import logging
import pathlib
import shutil
from typing import Dict, List

import tomotopy as tp # type: ignore

import numpy as np
import pandas as pd  # type: ignore

from ...utils.common import init_logger, load_yaml_config_file

class BaseTMModel(ABC):
    """
    Abstract base class for topic models.
    """

    def __init__(
        self,
        model_name: str,
        model_path: str = None,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path("./static/config/config.yaml"),
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

                old_model_path = self.model_path.with_name(self.model_path.name + "_old")

                if old_model_path.exists():
                    shutil.rmtree(old_model_path)

                shutil.move(str(self.model_path), str(old_model_path))
                self._logger.info(f"Old model moved to {old_model_path}")

            # Create a fresh model path
            self.model_path.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name

        # load config
        self.config = load_yaml_config_file(config_path, "topic_modeling", logger)
        
        self.num_topics = int(self.config.get("general", {}).get("num_topics", 50))
        self.topn = int(self.config.get("general", {}).get("topn", 15))
        self.thetas_thr = float(self.config.get("general", {}).get("thetas_thr", 3e-3))
        self.not_include = self.config.get("general", {}).get("not_include", [])
    
    def set_training_data(self, data: List[Dict]):
        """
        Set training data from a normalized list of dicts.
        Assumes each dict has: 'id', 'raw_text', and optionally 'embeddings'.
        """
        
        required_keys = {"id", "raw_text"}
        for row in data:
            if not required_keys.issubset(row):
                raise ValueError(f"Missing required keys in data row: {row}")
        
        self.train_data = [row["raw_text"].split() for row in data]
        # TODO: this should be changed once we to lemmatization
        
        # create dataframe from the data
        self.df = pd.DataFrame(data) #this should have "id", "raw_text" and additional "lemmas" column once preprocessed

        if all("embeddings" in row and row["embeddings"] is not None for row in data):
            self.embeddings = np.array([row["embeddings"] for row in data])
            self._logger.info(f"Embeddings loaded for {len(self.embeddings)} documents.")
        else:
            self.embeddings = None
            self._logger.info("No valid embeddings found. Proceeding without embeddings.")

        self._logger.info(f"Training data set with {len(self.train_data)} documents.")
        
    def prepare_infer_data(self, data: List[Dict]):
        
        required_keys = {"id", "raw_text"}
        for row in data:
            if not required_keys.issubset(row):
                raise ValueError(f"Missing required keys in data row: {row}")
        
        infer_data = [row["raw_text"].split() for row in data]
        df_infer = pd.DataFrame(data)
        
        if all("embeddings" in row and row["embeddings"] is not None for row in data):
            embeddings_infer = np.array([row["embeddings"] for row in data])
            self._logger.info(f"Embeddings loaded for {len(self.embeddings_infer)} documents.")
        else:
            embeddings_infer = None
            self._logger.info("No valid embeddings found. Proceeding without embeddings.")
        
        self._logger.info(f"Prepared inference data set with {len(infer_data)} documents.")
        
        return infer_data, df_infer, embeddings_infer
        
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
        json_path = path or (self.model_path / 'metadata.json')

        model_info = {
            "name": self.model_name,
            "path": self.model_path.as_posix(),
            "type": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "creation_date": str(pathlib.Path().resolve()),
            "tr_params": self.to_dict(),
        }
        
        with json_path.open('w') as f:
            json.dump(model_info, f, indent=2)

    @abstractmethod
    def save_model(self, path: str): pass

    @classmethod
    @abstractmethod
    def from_saved_model(self, path: str): pass

    @abstractmethod
    def train_model(self, data: List[Dict]): pass

    @abstractmethod
    def infer(self, data: List[Dict]): pass