import logging
import pathlib
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tova.topic_models.models.base_model import BaseTMModel
from tova.utils.cancel import CancellationToken, check_cancel
from tova.utils.progress import (ProgressCallback,  # type: ignore
                                 ProgressReporter)


class LLMTModel(BaseTMModel, ABC):
    def __init__(
        self,
        model_name: str,
        corpus_id: str,
        id: str,
        model_path: str = None,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path(
            "./static/config/config.yaml"),
        load_model: bool = False,
    ) -> None:
        """
        Initialize the LLMTModel class.
        """

        super().__init__(model_name, corpus_id, id,
                         model_path, logger, config_path, load_model)

        self.do_labeller = False  # LLM-based models do not use traditional labelling
        self.do_summarizer = False  # LLM-based models do not use traditional summarization
        self.labeller_prompt = ""
        self.summarizer_prompt = ""
        self._config_path = config_path

    def set_training_data(self, data):
        """
        Set training data from a normalized list of dicts.
         Each dict must contain at least 'id' and 'raw_text' keys.
         """
        required_keys = {"id", "raw_text"}
        for row in data:
            if not required_keys.issubset(row):
                raise ValueError(f"Missing required keys in data row: {row}")

        self.df = pd.DataFrame(data)
        
    
    def prepare_infer_data(self, data):
        """
        Prepare inference data from a normalized list of dicts.
         Each dict must contain at least 'id' and 'raw_text' keys.
         """
        required_keys = {"id", "raw_text"}
        for row in data:
            if not required_keys.issubset(row):
                raise ValueError(f"Missing required keys in data row: {row}")

        infer_df = pd.DataFrame(data)
        return infer_df

    
    def train_model(
        self,
        data: List[Dict],
        progress_callback: Optional[ProgressCallback] = None,
        cancel: Optional[CancellationToken] = None
    ):

        t_start = time.time()
        pr = ProgressReporter(callback=progress_callback, logger=self._logger)
        pr.report(0, "Starting training")

        # 1. PREPROCESSING (0-20%)
        check_cancel(cancel, self._logger)
        prs = pr.report_subrange(0.0, 0.2)
        prs.report(0.0, "Preprocessing")
        self.set_training_data(data)
        prs.report(1.0, "Preprocessing completed")

        # 2. TRAIN CORE (depends on the underlying topic modeling algorithm)
        # (20-70%)
        check_cancel(cancel, self._logger)
        prs = pr.report_subrange(0.2, 0.7)
        prs.report(0.0, "Training core")
        tr_core, thetas, betas, vocab = self.train_core(
            prs=prs,
            cancel=cancel
        )
        prs.report(
            1.0, "Training core completed in {:.2f} minutes".format(tr_core / 60))

        # 3. TM_MODEL CREATION (metrics calculation, etc.)
        # (70-90%)
        check_cancel(cancel, self._logger)
        prs = pr.report_subrange(0.7, 0.9)
        prs.report(0.0, "Creating TMmodel object")
        self._createTMmodel(thetas, betas, vocab)
        prs.report(1.0, "TMmodel object created")

        # 4. SAVE MODEL
        # (90-100%)
        check_cancel(cancel, self._logger)
        prs = pr.report_subrange(0.9, 1.0)
        prs.report(0.0, "Saving topic model")
        self.save_to_json()
        self.save_model()
        prs.report(1.0, "Topic model saved")

        return time.time() - t_start

    def infer(
        self,
        data: List[Dict],
        progress_callback: Optional[ProgressCallback] = None,
        cancel: Optional[CancellationToken] = None
    ):
        t_start = time.time()
        pr = ProgressReporter(callback=progress_callback, logger=self._logger)
        pr.report(0, "Starting inference")

        # 1. PREPROCESSING (0-50%)
        check_cancel(cancel, self._logger)
        prs = pr.report_subrange(0.0, 0.5)
        prs.report(0.0, "Preprocessing")
        
        # TODO: compute embeddings only if needed
        df_infer = self.prepare_infer_data(data)
        prs.report(1.0, "Preprocessing completed")

        # 2. INFERENCE (50-100%)
        check_cancel(cancel, self._logger)
        prs = pr.report_subrange(0.2, 0.7)
        prs.report(0.0, "Inference")
        thetas, _ = self.infer_core(df_infer)
        prs.report(1.0, "Inference completed")

        return thetas, time.time() - t_start
    
    
    @abstractmethod
    def train_core(
        self,
        prs: Optional[ProgressCallback] = None,
        cancel: Optional[CancellationToken] = None
    ) -> float: pass

    @abstractmethod
    def infer_core(self, infer_data, df_infer, embeddings_infer) -> Tuple[np.ndarray, float]:
        pass
