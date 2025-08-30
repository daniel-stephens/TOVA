import logging
import pathlib
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt # type: ignore
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from tova.utils.cancel import CancellationToken, check_cancel
from tova.utils.progress import ProgressCallback, ProgressReporter  # type: ignore

from ...tm_model import TMmodel
from ..base_model import BaseTMModel


class TradTMmodel(BaseTMModel, ABC):
    
    def __init__(
        self,
        model_path: str = None,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path("./static/config/config.yaml"),
        load_model: bool = False
    ) -> None:
        """
        Initialize the TradTMmodel class.
        """
        
        super().__init__(model_path, logger, config_path, load_model)
        
        # Load parameters from config specific to traditional models
        self.num_topics = int(self.config.get("traditional", {}).get("num_topics", 50))
        self.thetas_thr = float(self.config.get("traditional", {}).get("thetas_thr", 3e-3))
        self.topn = int(self.config.get("traditional", {}).get("topn", 15))
        self.do_labeller = bool(self.config.get("traditional", {}).get("do_labeller", False))
        self.do_summarizer = bool(self.config.get("traditional", {}).get("do_summarizer", False))
        self.llm_model_type = self.config.get("traditional", {}).get("llm_model_type", "qwen:32b")
        self.labeller_prompt = self.config.get("traditional", {}).get("labeller_model_path", "./src/prompter/prompts/labelling_dft.txt")
        self.summarizer_prompt = self.config.get("traditional", {}).get("summarizer_prompt", "./src/prompter/prompts/summarization_dft.txt")
        self._config_path = config_path
    
    def preprocess(self, data):
        print("Preprocessing with traditional methods")
        # TODO: change saving of raw_text / add lemmas
        
    def _save_thr_fig(
        self,
        thetas: np.ndarray,
        plot_file: pathlib.Path
    ) -> None:
        """
        Creates a figure to illustrate the effect of thresholding.

        Parameters
        ----------
        thetas : np.ndarray
            The doc-topics matrix for a topic model.
        plot_file : pathlib.Path
            The name of the file where the plot will be saved.
        """

        all_values = np.sort(thetas.flatten())
        step = int(np.round(len(all_values) / 1000))
        plt.semilogx(all_values[::step], (100 / len(all_values))
                     * np.arange(0, len(all_values))[::step])
        plt.savefig(plot_file)
        plt.close()
            
    def _createTMmodel(self, thetas, betas, vocab):
        """Creates an object of class TMmodel hosting the topic model
        that has been trained and whose output is available at the
        provided folder

        Parameters
        ----------
        modelFolder: Path
            the folder with the mallet output files

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
            betas=betas, thetas=thetas, alphas=alphas,vocab=vocab)
        
        return tm
                    
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
        # self.preprocess(data) # @TODO
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
        prs.report(1.0, "Training core completed in {:.2f} minutes".format(tr_core / 60))

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
    
    def infer(self, data: List[Dict]):
        
        infer_data, df_infer, embeddings_infer = self.prepare_infer_data(data)
        
        return self.infer_core(infer_data, df_infer, embeddings_infer)
        
    @abstractmethod
    def train_core(
        self, 
        prs: Optional[ProgressCallback] = None,
        cancel: Optional[CancellationToken] = None
    ) -> float: pass

    @abstractmethod
    def infer_core(self, infer_data, df_infer, embeddings_infer) -> Tuple[np.ndarray, float]:
        pass