from ..base_model import BaseTMModel
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from typing import List
import pathlib
from abc import ABC, abstractmethod

class TradTMmodel(BaseTMModel, ABC):
    def preprocess(self, data):
        print("Preprocessing with traditional methods")
        
    def _save_model_results(
        self,
        thetas: np.ndarray,
        betas: np.ndarray,
        vocab: List[str],
        keys: List[List[str]]
    ) -> None:
        """
        Save the model results.

        Parameters
        ----------
        thetas : np.ndarray
            The doc-topics matrix.
        betas : np.ndarray
            The topic-word distributions.
        vocab : List[str]
            The vocabulary of the model.
        keys : List[List[str]]
            The top words for each topic.
        """

        # self._save_thr_fig(thetas, self.model_path.joinpath('thetasDist.pdf'))
        thetas = sparse.csr_matrix(thetas, copy=True)

        alphas = np.asarray(np.mean(thetas, axis=0)).ravel()

        #bow = self.get_bow(vocab)
        #bow = sparse.csr_matrix(bow, copy=True)

        np.save(self.model_path.joinpath('alphas.npy'), alphas)
        np.save(self.model_path.joinpath('betas.npy'), betas)
        sparse.save_npz(self.model_path.joinpath('thetas.npz'), thetas)
        #sparse.save_npz(self.model_path.joinpath('bow.npz'), bow)
        #with self.model_path.joinpath('vocab.txt').open('w', encoding='utf8') as fout:
        #    fout.write('\n'.join(vocab))

        with self.model_path.joinpath('tpc_descriptions.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join([' '.join(topic) for topic in keys]))
            
            
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