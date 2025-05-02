"""This module is similar to the one available in the topicmodeler (https://github.com/IntelCompH2020/topicmodeler/blob/main/src/topicmodeling/manageModels.py). It provides a generic representation of all topic models used for curation purposes.

Authors: Jerónimo Arenas-García, J.A. Espinosa-Melchor, Lorena Calvo-Bartolomé
Modified: 02/05/2025 (Updated for TOVA)
"""


from collections import defaultdict
import itertools
import json
import logging
import shutil
import time
from typing import List, Optional
import warnings
from pathlib import Path
from gensim.corpora import Dictionary # type: ignore
from gensim.models.coherencemodel import CoherenceModel # type: ignore
import numpy as np
import pandas as pd
import rbo
import scipy.sparse as sparse
from ..prompter.prompter import Prompter

class TMmodel(object):
    # This class represents a Topic Model according to the LDA generative model
    # Essentially the TM is characterized by
    # _alphas: The weight of each topic
    # _betas: The weight of each word in the vocabulary
    # _thetas: The weight of each topic in each document
    #
    # and needs to be backed up with a folder in which all the associated
    # files will be saved
    #
    # The TM can be trained with Blei's LDA, Mallet, or any other toolbox
    # that produces a model according to this representation

    # The following variables will store original values of matrices alphas, betas, thetas
    # They will be used to reset the model to original values

    _TMfolder = None

    _betas_orig = None
    _thetas_orig = None
    _alphas_orig = None

    _betas = None
    _thetas = None
    _alphas = None
    _edits = None  # Store all editions made to the model
    _ntopics = None
    _betas_ds = None
    _coords = None
    _topic_entropy = None
    _topic_coherence = None
    _ndocs_active = None
    _tpc_descriptions = None
    _tpc_labels = None
    _vocab_w2id = None
    _vocab_id2w = None
    _vocab = None
    _size_vocab = None
    _s3 = None
    _most_representative_docs = None

    def __init__(
        self, 
        TMfolder: Path, 
        df_corpus_train: pd.DataFrame=None,
        config_path: Path=None,
        do_labeller: bool = True, 
        labeller_model_type: str = "qwen:32b",
        labeller_prompt: str = "src/prompter/prompts/labelling_dft.txt",
        logger: logging.Logger = None,
        ):

        """Class initializer

        We just need to make sure that we have a folder where the
        model will be stored. If the folder does not exist, it will
        create a folder for the model

        Parameters
        ----------
        TMfolder: Path
            Contains the name of an existing folder or a new folder
            where the model will be created
        logger:
            External logger to use. If None, a logger will be created for the object
        """
        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('TMmodel')

        # Convert strings to Paths if necessary
        self._TMfolder = Path(TMfolder)

        # If folder already exists no further action is needed
        # in other case, the folder is created
        if not self._TMfolder.is_dir():
            try:
                self._TMfolder.mkdir(parents=True)
            except:
                self._logger.error(
                    '-- -- Topic model object (TMmodel) could not be created')
                
        self._df_corpus_train = df_corpus_train
        self._config_path = config_path
        self._do_labeller = do_labeller
        self._labeller_model_type = labeller_model_type
        self._labeller_prompt = labeller_prompt

        self._logger.info(
            '-- -- -- Topic model object (TMmodel) successfully created')

    def create(self, betas=None, thetas=None, alphas=None, vocab=None):
        """Creates the topic model from the relevant matrices that characterize it. In addition to the initialization of the corresponding object's variables, all the associated variables and visualizations which are computationally costly are calculated so they are available for the other methods.

        Parameters
        ----------
        betas:
            Matrix of size n_topics x n_words (vocab of each topic)
        thetas:
            Matrix of size  n_docs x n_topics (document composition)
        alphas: 
            Vector of length n_topics containing the importance of each topic
        vocab: list
            List of words sorted according to betas matrix
        """

        # If folder already exists no further action is needed
        # in other case, the folder is created
        if not self._TMfolder.is_dir():
            self._logger.error(
                '-- -- Topic model object (TMmodel) folder not ready')
            return

        self._alphas_orig = alphas
        self._betas_orig = betas
        self._thetas_orig = thetas
        self._alphas = alphas
        self._betas = betas
        self._thetas = thetas
        self._vocab = vocab
        self._size_vocab = len(vocab)
        self._ntopics = thetas.shape[1]
        self._edits = []

        # Save original variables
        np.save(self._TMfolder.joinpath('alphas_orig.npy'), alphas)
        np.save(self._TMfolder.joinpath('betas_orig.npy'), betas)
        sparse.save_npz(self._TMfolder.joinpath('thetas_orig.npz'), thetas)
        with self._TMfolder.joinpath('vocab.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(vocab))

        # Initial sort of topics according to size. Calculate other variables
        self._sort_topics()
        self._logger.info("-- -- Sorted")
        self._calculate_beta_ds()
        self._logger.info("-- -- betas ds")
        self._calculate_topic_entropy()
        self._logger.info("-- -- entropy")
        self._ndocs_active = np.array((self._thetas != 0).sum(0).tolist()[0])
        self._logger.info("-- -- active")
        self._tpc_descriptions = [el[1] for el in self.get_tpc_word_descriptions()]
        self._logger.info("-- -- descriptions")
        self.calculate_topic_coherence()
        
        self._load_vocab_dicts()
        #self._calculate_s3()
        
        # get most representative documents
        self.get_most_representative_per_tpc(self._thetas, topn=3)     
        
        if self._do_labeller:
            try:
                self._tpc_labels = [el[1] for el in self.get_tpc_labels()]
            except Exception as e:
                self._logger.warning(
                    f"Error in labeller: {e}")
                self._tpc_labels = ["Topic " + str(i) for i in range(self._ntopics)]
        else:
            self._tpc_labels = ["Topic " + str(i) for i in range(self._ntopics)]

        # calculate the rank-biased overlap and topic diversity
        try:
            self.calculate_rbo()
            self.calculate_topic_diversity()
        except Exception as e:
            self._logger.warning(
                f"Error in rbo or topic diversity: {e}")

        # We are ready to save all variables in the model
        self._save_all()

        self._logger.info(
            '-- -- Topic model variables were computed and saved to file')
        return

    def _save_all(self):
        """Saves all variables in Topic Model
        * alphas, betas, thetas
        * edits
        * betas_ds, topic_entropy, ndocs_active
        * tpc_descriptions, tpc_labels
        This function should only be called after making sure all these
        variables exist and are not None
        """
        np.save(self._TMfolder.joinpath('alphas.npy'), self._alphas)
        np.save(self._TMfolder.joinpath('betas.npy'), self._betas)
        sparse.save_npz(self._TMfolder.joinpath('thetas.npz'), self._thetas)
        #sparse.save_npz(self._TMfolder.joinpath('s3.npz'), self._s3)

        with self._TMfolder.joinpath('edits.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._edits))
        np.save(self._TMfolder.joinpath('betas_ds.npy'), self._betas_ds)
        np.save(self._TMfolder.joinpath(
            'topic_entropy.npy'), self._topic_entropy)
        np.save(self._TMfolder.joinpath(
            'topic_coherence.npy'), self._topic_coherence)
        np.save(self._TMfolder.joinpath(
            'ndocs_active.npy'), self._ndocs_active)
        with self._TMfolder.joinpath('tpc_descriptions.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._tpc_descriptions))
            
        self.save_most_representative_docs()
        
        with self._TMfolder.joinpath('tpc_labels.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._tpc_labels))

        # Generate also pyLDAvisualization
        # pyLDAvis currently raises some Deprecation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import pyLDAvis # type: ignore

        # We will compute the visualization using ndocs random documents
        # In case the model has gone through topic deletion, we may have rows
        # in the thetas matrix that sum up to zero (active topics have been
        # removed for these problematic documents). We need to take this into
        # account
        try:
            ndocs = 10000
            validDocs = np.sum(self._thetas.toarray(), axis=1) > 0
            nValidDocs = np.sum(validDocs)
            if ndocs > nValidDocs:
                ndocs = nValidDocs
            perm = np.sort(np.random.permutation(nValidDocs)[:ndocs])
            # We consider all documents are equally important
            doc_len = ndocs * [1]
            vocabfreq = np.round(ndocs*(self._alphas.dot(self._betas))).astype(int)
            vis_data = pyLDAvis.prepare(
                self._betas,
                self._thetas[validDocs, ][perm, ].toarray(),
                doc_len,
                self._vocab,
                vocabfreq,
                lambda_step=0.05,
                sort_topics=False,
                n_jobs=-1)

            # Save html
            with self._TMfolder.joinpath("pyLDAvis.html").open("w") as f:
                pyLDAvis.save_html(vis_data, f)

            # Get coordinates of topics in the pyLDAvis visualization
            vis_data_dict = vis_data.to_dict()
            self._coords = list(
                zip(*[vis_data_dict['mdsDat']['x'], vis_data_dict['mdsDat']['y']]))

            with self._TMfolder.joinpath('tpc_coords.txt').open('w', encoding='utf8') as fout:
                for item in self._coords:
                    fout.write(str(item) + "\n")
        except Exception as e:
            print(f"Error in pyLDAvis: {e}")
        return

    def _save_cohr(self):
        np.save(self._TMfolder.joinpath(
            'topic_coherence.npy'), self._topic_coherence)
        
    def _sort_topics(self):
        """Sort topics according to topic size"""

        # Load information if necessary
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_edits()

        # Indexes for topics reordering
        idx = np.argsort(self._alphas)[::-1]
        self._edits.append('s ' + ' '.join([str(el) for el in idx]))

        # Sort data matrices
        self._alphas = self._alphas[idx]
        self._betas = self._betas[idx, :]
        self._thetas = self._thetas[:, idx]

        return

    def _load_alphas(self):
        if self._alphas is None:
            self._alphas = np.load(self._TMfolder.joinpath('alphas.npy'))
            self._ntopics = self._alphas.shape[0]

    def _load_betas(self):
        if self._betas is None:
            self._betas = np.load(self._TMfolder.joinpath('betas.npy'))
            self._ntopics = self._betas.shape[0]
            self._size_vocab = self._betas.shape[1]

    def _load_thetas(self):
        if self._thetas is None:
            self._thetas = sparse.load_npz(
                self._TMfolder.joinpath('thetas.npz'))
            self._ntopics = self._thetas.shape[1]
            # self._ndocs_active = np.array((self._thetas != 0).sum(0).tolist()[0])
    
    def _load_s3(self):
        if self._s3 is None:
            self._s3 = sparse.load_npz(
                self._TMfolder.joinpath('s3.npz'))
            
    def _load_ndocs_active(self):
        if self._ndocs_active is None:
            self._ndocs_active = np.load(
                self._TMfolder.joinpath('ndocs_active.npy'))
            self._ntopics = self._ndocs_active.shape[0]

    def _load_edits(self):
        if self._edits is None:
            with self._TMfolder.joinpath('edits.txt').open('r', encoding='utf8') as fin:
                self._edits = fin.readlines()

    def _calculate_beta_ds(self):
        """Calculates beta with down-scoring
        Emphasizes words appearing less frequently in topics
        """
        # Load information if necessary
        self._load_betas()

        self._betas_ds = np.copy(self._betas)
        if np.min(self._betas_ds) < 1e-12:
            self._betas_ds += 1e-12
        deno = np.reshape((sum(np.log(self._betas_ds)) / self._ntopics), (self._size_vocab, 1))
        deno = np.ones((self._ntopics, 1)).dot(deno.T)
        self._betas_ds = self._betas_ds * (np.log(self._betas_ds) - deno)

    def _load_betas_ds(self):
        if self._betas_ds is None:
            self._betas_ds = np.load(self._TMfolder.joinpath('betas_ds.npy'))
            self._ntopics = self._betas_ds.shape[0]
            self._size_vocab = self._betas_ds.shape[1]

    def _load_vocab(self):
        if self._vocab is None:
            with self._TMfolder.joinpath('vocab.txt').open('r', encoding='utf8') as fin:
                self._vocab = [el.strip() for el in fin.readlines()]

    def _load_vocab_dicts(self):
        """Creates two vocabulary dictionaries, one that utilizes the words as key, and a second one with the words' id as key. 
        """
        if self._vocab_w2id is None and self._vocab_w2id is None:
            self._vocab_w2id = {}
            self._vocab_id2w = {}
            with self._TMfolder.joinpath('vocab.txt').open('r', encoding='utf8') as fin:
                for i, line in enumerate(fin):
                    wd = line.strip()
                    self._vocab_w2id[wd] = i
                    self._vocab_id2w[str(i)] = wd

    def _calculate_topic_entropy(self):
        """Calculates the entropy of all topics in model
        """
        # Load information if necessary
        self._load_betas()

        if np.min(self._betas) < 1e-12:
            self._betas += 1e-12
        self._topic_entropy = - \
            np.sum(self._betas * np.log(self._betas), axis=1)
        self._topic_entropy = self._topic_entropy / np.log(self._size_vocab)

    def _load_topic_entropy(self):
        if self._topic_entropy is None:
            self._topic_entropy = np.load(
                self._TMfolder.joinpath('topic_entropy.npy'))

    def calculate_rbo(self,
                      weight: float = 1.0,
                      n_words: int = 15) -> float:
        """Calculates the rank_biased_overlap over the topics in a topic model.

        Parameters
        ----------
        weigth : float, optional
            Weight of each agreement at depth d: p**(d-1). When set to 1.0, there is no weight, the rbo returns to average overlap. The defau>
        n_words : int, optional
            Number of words to be used for calculating the rbo. The default is 15.

        Returns
        -------
        rbo : float
            Rank_biased_overlap
        """

        # Load topic information
        if self._tpc_descriptions is None:
            self._tpc_descriptions = \
                [el[1] for el in self.get_tpc_word_descriptions(n_words)]

        collect = []
        for list1, list2 in itertools.combinations(self._tpc_descriptions, 2):
            rbo_val = rbo.RankingSimilarity(
                list1.split(", "), list2.split(", ")).rbo(p=weight)
            collect.append(rbo_val)

        irbo = 1 - np.mean(collect)

        # Save rbo
        try:
            with self._TMfolder.joinpath('rbo.txt').open('w', encoding='utf8') as fout:
                fout.write(str(irbo))
        except:
            self._logger.warning(
                "Rank-biased overlap could not be saved to file")
        return rbo

    def calculate_topic_diversity(
        self,
        n_words: int = 15) -> float:
        """Calculates the percentage of unique words in the topn words of all topics. Diversity close to 0 indicates redundant topics; diversity close to 1 indicates more varied topics.

        Parameters
        ----------
        n_words : int, optional
            Number of words to be used for calculating the rbo. The default is 15.

        Returns
        -------
        td : float
            Topic diversity
        """

        # Load topic information
        if self._tpc_descriptions is None:
            self._tpc_descriptions = \
                [el[1] for el in self.get_tpc_word_descriptions(n_words)]

        unique_words = set()
        for topic in self._tpc_descriptions:
            unique_words = unique_words.union(set(topic.split(", ")))
        td = len(unique_words) / (n_words * len(self._tpc_descriptions))
        
        # save topic diversity
        try:
            with self._TMfolder.joinpath('topic_diversity.txt').open('w', encoding='utf8') as fout:
                fout.write(str(td))
        except:
            self._logger.warning(
                "Topic diversity could not be saved to file")
            return td 

        return td

    def calculate_topic_coherence(
        self,
        metrics: List[str] = ["c_npmi", "c_v"],
        n_words: int = 15,
        reference_text: Optional[List[List[str]]] = None,
        only_one: bool = True,
        aggregated: bool = False
    ) -> list:
        """Calculates the per-topic coherence of a topic model, given as TMmodel, or its average coherence when aggregated is True.

        If only_one is False and metrics is a list of different coherence metrics, the function returns a list of lists, where each sublist contains the coherence values for the respective metric.

        If reference_text is given, the coherence is calculated with respect to this text. Otherwise, the coherence is calculated with respect to the corpus used to train the topic model.

        Parameters
        ----------
        metrics : list of str, optional
            List of coherence metrics to be calculated. Possible values are 'c_v', 'c_uci', 'c_npmi', 'u_mass'. 
            The default is ["c_v", "c_npmi"].
        n_words : int, optional
            Number of words to be used for calculating the coherence. The default is 15.
        reference_text : List[List[str]]
            Text to use for calculating the coherence. If None, the corpus used to train the topic model is used. The default is None.
        only_one : bool, optional
            If True, only one coherence value is returned. If False, a list of coherence values is returned. The default is True.
        aggregated : bool, optional
            If True, the average coherence of the topic model is returned. If False, the coherence of each topic is returned. The default is False.
        """

        # Load topic information
        if self._tpc_descriptions is None:
            self._tpc_descriptions = \
                [el[1] for el in self.get_tpc_word_descriptions()]

        # Convert topic information into list of lists (Gensim's Coherence Model format)
        tpc_descriptions_ = \
            [tpc.split(', ') for tpc in self._tpc_descriptions]

        if reference_text is None:
            corpus = [el.split() for el in self._df_corpus_train["raw_text"].values.tolist()]
        else:
            # Texts should be given as a list of lists of strings
            corpus = reference_text
            
        # Get Gensim dictionary
        dictionary = None
        if self._TMfolder.parent.joinpath('dictionary.gensim').is_file():
            try:
                dictionary = Dictionary.load_from_text(
                    self._TMfolder.parent.joinpath('dictionary.gensim').as_posix())
            except:
                self._logger.warning(
                    "Gensim dictionary could not be load from vocabulary file.")
        else:
            if dictionary is None:
                dictionary = Dictionary(corpus)

        if n_words > len(tpc_descriptions_[0]):
            self._logger.error(
                '-- -- -- Coherence calculation failed: The number of words per topic must be equal to n_words.')
            return None
        else:
            if only_one:
                metric = metrics[0]
                self._logger.info(
                    f"Calculating just coherence {metric}.")
                if metric in ["c_npmi", "u_mass", "c_v", "c_uci"]:
                    cm = CoherenceModel(topics=tpc_descriptions_, texts=corpus,
                                        dictionary=dictionary, coherence=metric, topn=n_words)
                    self._topic_coherence = cm.get_coherence_per_topic()

                    if aggregated:
                        mean = cm.aggregate_measures(self._topic_coherence)
                        return mean
                    return self._topic_coherence
                else:
                    self._logger.error(
                        '-- -- -- Coherence metric provided is not available.')
                    return None
            else:
                cohrs_aux = []
                for metric in metrics:
                    self._logger.info(
                        f"Calculating coherence {metric}.")
                    if metric in ["c_npmi", "u_mass", "c_v", "c_uci"]:
                        cm = CoherenceModel(topics=tpc_descriptions_, texts=corpus,
                                            dictionary=dictionary, coherence=metric, topn=n_words)
                        aux = cm.get_coherence_per_topic()
                        cohrs_aux.extend(aux)
                        self._logger.info(cohrs_aux)
                    else:
                        self._logger.error(
                            '-- -- -- Coherence metric provided is not available.')
                        return None
                self._topic_coherence = cohrs_aux

        return self._topic_coherence
                
    def _load_topic_coherence(self):
        if self._topic_coherence is None:
            self._topic_coherence = np.load(
                self._TMfolder.joinpath('topic_coherence.npy'))

    def _largest_indices(self, ary, n):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        idx0, idx1 = np.unravel_index(indices, ary.shape)
        idx0 = idx0.tolist()
        idx1 = idx1.tolist()
        selected_idx = []
        for id0, id1 in zip(idx0, idx1):
            if id0 < id1:
                selected_idx.append((id0, id1, ary[id0, id1]))
        return selected_idx

    def get_model_info_for_hierarchical(self):
        """Returns the objects necessary for the creation of a level-2 topic model.
        """
        self._load_betas()
        self._load_thetas()
        self._load_vocab_dicts()

        return self._betas, self._thetas, self._vocab_w2id, self._vocab_id2w

    def get_model_info_for_vis(self):
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_vocab()
        self.load_tpc_coords()
        
        return self._alphas, self._betas, self._thetas, self._vocab, self._coords

    def get_tpc_word_descriptions(self, n_words=15, tfidf=True, tpc=None):
        """returns the chemical description of topics

        Parameters
        ----------
        n_words:
            Number of terms for each topic that will be included
        tfidf:
            If true, downscale the importance of words that appear
            in several topics, according to beta_ds (Blei and Lafferty, 2009)
        tpc:
            Topics for which the descriptions will be computed, e.g.: tpc = [0,3,4]
            If None, it will compute the descriptions for all topics  

        Returns
        -------
        tpc_descs: list of tuples
            Each element is a a term (topic_id, "word0, word1, ...")                      
        """

        # Load betas (including n_topics) and vocabulary
        if tfidf:
            self._load_betas_ds()
        else:
            self._load_betas()
        self._load_vocab()

        if not tpc:
            tpc = range(self._ntopics)

        tpc_descs = []
        for i in tpc:
            if tfidf:
                words = [self._vocab[idx2] for idx2 in np.argsort(self._betas_ds[i])[::-1][0:n_words]]
            else:
                words = [self._vocab[idx2] for idx2 in np.argsort(self._betas[i])[::-1][0:n_words]]
            tpc_descs.append((i, ', '.join(words)))

        return tpc_descs

    def load_tpc_descriptions(self):
        if self._tpc_descriptions is None:
            with self._TMfolder.joinpath('tpc_descriptions.txt').open('r', encoding='utf8') as fin:
                self._tpc_descriptions = [el.strip() for el in fin.readlines()]
                
    def _calculate_s3(self):
        """Given the path to a TMmodel, it calculates the similarities between documents and saves them in a sparse matrix.
        """
        
        t_start = time.perf_counter()
        
        corpusFile = self._TMfolder.parent  / 'corpus.txt'
        with corpusFile.open("r", encoding="utf-8") as f:
            lines = f.readlines()  
            f.seek(0)
            try:
                documents_texts = [line.rsplit(" 0 ")[1].strip().split() for line in lines]
            except:
                documents_texts = [line.rsplit("\t0\t")[1].strip().split() for line in lines]
        
        D = len(self._thetas.toarray())
        K = len(self._betas)
        S3 = np.zeros((D, K))

        for doc in range(D):
            for topic in range(K):
                wd_ids = [
                    self._vocab_w2id[word] 
                    for word in documents_texts[doc] 
                    if word in self._vocab_w2id
                ]
                S3[doc, topic] = np.sum(self._betas[topic, wd_ids])
                
        sparse_S3 = sparse.csr_matrix(S3)
        self._s3  = sparse_S3

        t_end = time.perf_counter()
        t_total = (t_end - t_start)/60
        self._logger.info(f"Total computation time: {t_total}")

    def get_most_representative_per_tpc(self, mat, topn=3):
        # Find the most representative document for each topic based on a matrix mat
        top_docs_per_topic = []
        
        aux = mat.toarray()
        
        for doc_distr in aux.T:
            sorted_docs_indices = np.argsort(doc_distr)[::-1]
            top = sorted_docs_indices[:topn].tolist()
            top_docs_per_topic.append(top)
        
        most_representative_docs = []
        for topic_docs in top_docs_per_topic:
            reps = [
                (self._df_corpus_train.iloc[doc].id, self._df_corpus_train.iloc[doc].raw_text)
                for doc in topic_docs
            ]
            most_representative_docs.append(reps)
        
        self._logger.info("Most representative documents for each topic:")
        for i, topic_docs in enumerate(most_representative_docs):
            ids = [doc[0] for doc in topic_docs]
            self._logger.info(f"Topic {i} -> Doc IDs: {ids}")

        self._most_representative_docs = most_representative_docs
        
    def save_most_representative_docs(self):
        """Saves the most representative documents for each topic in JSONL format."""
        if self._most_representative_docs is None:
            self._logger.warning("Most representative documents not calculated yet")
            return

        output_path = self._TMfolder.joinpath("most_representative_docs.jsonl")
        with output_path.open("w", encoding="utf-8") as fout:
            for tpc_id, docs in enumerate(self._most_representative_docs):
                for rank, (doc_id, raw_text) in enumerate(docs):
                    json_line = {
                        "topic_id": tpc_id,
                        "rank": rank + 1,
                        "doc_id": doc_id,
                        "text": raw_text
                    }
                    fout.write(json.dumps(json_line, ensure_ascii=False) + "\n")

        self._logger.info(f"Most representative documents saved to {output_path}")
            
    def load_most_representative_docs(self):
        """Loads the most representative documents for each topic from a JSONL file."""
        jsonl_path = self._TMfolder.joinpath("most_representative_docs.jsonl")

        if not jsonl_path.is_file():
            self._logger.warning(f"File not found: {jsonl_path}")
            return

        topic_docs = defaultdict(list)

        with jsonl_path.open("r", encoding="utf8") as fin:
            for line in fin:
                try:
                    record = json.loads(line)
                    topic_id = record["topic_id"]
                    doc_id = record["doc_id"]
                    text = record["text"]
                    topic_docs[topic_id].append((doc_id, text))
                except (json.JSONDecodeError, KeyError) as e:
                    self._logger.error(f"Skipping malformed line: {e}")

        # Convert defaultdict to list of lists sorted by topic ID
        max_topic_id = max(topic_docs.keys(), default=-1)
        self._most_representative_docs = [
            topic_docs[t] for t in range(max_topic_id + 1)
        ]

    def get_tpc_labels(self):
        """Returns the labels of the topics in the model
        
        Returns
        -------
        tpc_labels: list of tuples
            Each element is a a term (topic_id, "label for topic topic_id")                    
        """

        # Load tpc descriptions
        self.load_tpc_descriptions()
        
        # Get the most representative documents for each topic
        self.get_most_representative_per_tpc(self._thetas, topn=3) # TODO: Check whether s3 or thetas should be used
        
        prompter =  Prompter(
            config_path=self._config_path,
            model_type=self._labeller_model_type
        )
        with open(self._labeller_prompt, 'r') as file: TEMPLATE = file.read()
        
        labels = []
        for tpc_id, most_repr in enumerate(self._most_representative_docs):
            docs = "\n- " + "\n- ".join([doc_tuple[1] for doc_tuple in most_repr])
            template = TEMPLATE.format(keywords=self._tpc_descriptions[tpc_id], docs = docs)
            label, _ = prompter.prompt(question=template, system_prompt_template_path=None)
            labels.append(label)    
        labels_format = [(i, p) for i, p in enumerate(labels)]
            
        return labels_format

    def load_tpc_labels(self):
        if self._tpc_labels is None:
            with self._TMfolder.joinpath('tpc_labels.txt').open('r', encoding='utf8') as fin:
                self._tpc_labels = [el.strip() for el in fin.readlines()]
                
    def load_tpc_coords(self):
        if self._coords is None:
            with self._TMfolder.joinpath('tpc_coords.txt').open('r', encoding='utf8') as fin:
                # read the data from the file and convert it back to a list of tuples
                self._coords = \
                    [tuple(map(float, line.strip()[1:-1].split(', ')))
                        for line in fin]

    def get_alphas(self):
        self._load_alphas()
        return self._alphas

    def showTopics(self):
        self._load_alphas()
        self._load_ndocs_active()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        TpcsInfo = [{"Size": str(round(el[0], 4)), "Label": el[1].strip(), "Word Description": el[2].strip(), "Ndocs Active": str(el[3])} for el in zip(
            self._alphas, self._tpc_labels, self._tpc_descriptions, self._ndocs_active)]

        return TpcsInfo

    def showTopicsAdvanced(self):
        self._load_alphas()
        self._load_ndocs_active()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_topic_entropy()
        self._load_topic_coherence()
        TpcsInfo = [{"Size": str(round(el[0], 4)), "Label": el[1].strip(), "Word Description": el[2].strip(), "Ndocs Active": str(el[3]), "Topics entropy": str(round(
            el[4], 4)), "Topics coherence": str(round(el[5], 4))} for el in zip(self._alphas, self._tpc_labels, self._tpc_descriptions, self._ndocs_active, self._topic_entropy, self._topic_coherence)]

        return TpcsInfo

    def setTpcLabels(self, TpcLabels):
        self._tpc_labels = [el.strip() for el in TpcLabels]
        self._load_alphas()
        # Check that the number of labels is consistent with model
        if len(TpcLabels) == self._ntopics:
            with self._TMfolder.joinpath('tpc_labels.txt').open('w', encoding='utf8') as fout:
                fout.write('\n'.join(self._tpc_labels))
            return 1
        else:
            return 0

    def deleteTopics(self, tpcs):
        """This is a costly operation, almost everything
        needs to get modified"""
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_betas_ds()
        self._load_topic_entropy()
        self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_ndocs_active()
        self._load_edits()
        self._load_vocab()

        try:
            # Get a list of the topics that should be kept
            tpc_keep = [k for k in range(self._ntopics) if k not in tpcs]
            tpc_keep = [k for k in tpc_keep if k < self._ntopics]

            # Calculate new variables
            self._thetas = self._thetas[:, tpc_keep]
            from sklearn.preprocessing import normalize # type: ignore
            self._thetas = normalize(self._thetas, axis=1, norm='l1')
            self._alphas = np.asarray(np.mean(self._thetas, axis=0)).ravel()
            self._ntopics = self._thetas.shape[1]
            self._betas = self._betas[tpc_keep, :]
            self._betas_ds = self._betas_ds[tpc_keep, :]
            self._ndocs_active = self._ndocs_active[tpc_keep]
            self._topic_entropy = self._topic_entropy[tpc_keep]
            self._topic_coherence = self._topic_coherence[tpc_keep]
            self._tpc_labels = [self._tpc_labels[i] for i in tpc_keep]
            self._tpc_descriptions = [
                self._tpc_descriptions[i] for i in tpc_keep]
            self._edits.append('d ' + ' '.join([str(k) for k in tpcs]))

            # We are ready to save all variables in the model
            self._save_all()

            self._logger.info(
                '-- -- Topics deletion successful. All variables saved to file')
            return 1
        except:
            self._logger.info(
                '-- -- Topics deletion generated an error. Operation failed')
            return 0

    def getSimilarTopics(self, npairs, thr=1e-3):
        """Obtains pairs of similar topics
        npairs: number of pairs of words
        thr: threshold for vocabulary thresholding
        """

        self._load_thetas()
        self._load_betas()

        # Part 1 - Coocurring topics
        # Highly correlated topics co-occure together
        # Topic mean
        med = np.asarray(np.mean(self._thetas, axis=0)).ravel()
        # Topic square mean
        thetas2 = self._thetas.multiply(self._thetas)
        med2 = np.asarray(np.mean(thetas2, axis=0)).ravel()
        # Topic stds
        stds = np.sqrt(med2 - med ** 2)
        # Topic correlation
        num = self._thetas.T.dot(
            self._thetas).toarray() / self._thetas.shape[0]
        num = num - med[..., np.newaxis].dot(med[np.newaxis, ...])
        deno = stds[..., np.newaxis].dot(stds[np.newaxis, ...])
        corrcoef = num / deno
        selected_coocur = self._largest_indices(
            corrcoef, self._ntopics + 2 * npairs)
        selected_coocur = [(el[0], el[1], el[2].astype(float))  for el in selected_coocur]

        # Part 2 - Topics with similar word composition
        # Computes inter-topic distance based on word distributions
        # using scipy implementation of Jensen Shannon distance
        from scipy.spatial.distance import jensenshannon

        # For a more efficient computation with very large vocabularies
        # we implement a threshold for restricting the distance calculation
        # to columns where any element is greater than threshold thr
        betas_aux = self._betas[:, np.where(self._betas.max(axis=0) > thr)[0]]
        js_mat = np.zeros((self._ntopics, self._ntopics))
        for k in range(self._ntopics):
            for kk in range(self._ntopics):
                js_mat[k, kk] = jensenshannon(
                    betas_aux[k, :], betas_aux[kk, :])
        JSsim = 1 - js_mat
        selected_worddesc = self._largest_indices(
            JSsim, self._ntopics + 2 * npairs)
        selected_worddesc = [(el[0], el[1], el[2].astype(float)) for el in selected_worddesc]

        similarTopics = {
            'Coocurring': selected_coocur,
            'Worddesc': selected_worddesc
        }

        return similarTopics

    def fuseTopics(self, tpcs):
        """This is a costly operation, almost everything
        needs to get modified"""
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_edits()
        self._load_vocab()

        try:
            # List of topics that will be merged
            tpcs = sorted(tpcs)

            # Calculate new variables
            # For beta we keep a weighted average of topic vectors
            weights = self._alphas[tpcs]
            bet = weights[np.newaxis, ...].dot(
                self._betas[tpcs, :]) / (sum(weights))
            # keep new topic vector in upper position and delete the others
            self._betas[tpcs[0], :] = bet
            self._betas = np.delete(self._betas, tpcs[1:], 0)
            # For theta we need to keep the sum. Since adding implies changing
            # structure, we need to convert to full matrix first
            # No need to renormalize
            thetas_full = self._thetas.toarray()
            thet = np.sum(thetas_full[:, tpcs], axis=1)
            thetas_full[:, tpcs[0]] = thet
            thetas_full = np.delete(thetas_full, tpcs[1:], 1)
            self._thetas = sparse.csr_matrix(thetas_full, copy=True)
            # Compute new alphas and number of topics
            self._alphas = np.asarray(np.mean(self._thetas, axis=0)).ravel()
            self._ntopics = self._thetas.shape[1]
            # Compute all other variables
            self._calculate_beta_ds()
            self._calculate_topic_entropy()
            self._ndocs_active = np.array(
                (self._thetas != 0).sum(0).tolist()[0])

            # Keep label and description of most significant topic
            for tpc in tpcs[1:][::-1]:
                del self._tpc_descriptions[tpc]
            # Recalculate chemical description of most significant topic
            self._tpc_descriptions[tpcs[0]] = self.get_tpc_word_descriptions(tpc=[tpcs[0]])[
                0][1]
            for tpc in tpcs[1:][::-1]:
                del self._tpc_labels[tpc]

            self.calculate_topic_coherence()
            self._edits.append('f ' + ' '.join([str(el) for el in tpcs]))
            # We are ready to save all variables in the model
            self._save_all()

            self._logger.info(
                '-- -- Topics merging successful. All variables saved to file')
            return 1
        except:
            self._logger.info(
                '-- -- Topics merging generated an error. Operation failed')
            return 0

    def sortTopics(self):
        """This is a costly operation, almost everything
        needs to get modified"""
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_betas_ds()
        self._load_topic_entropy()
        self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_ndocs_active()
        self._load_edits()
        self._load_vocab()

        try:
            # Calculate order for the topics
            idx = np.argsort(self._alphas)[::-1]
            self._edits.append('s ' + ' '.join([str(el) for el in idx]))

            # Calculate new variables
            self._thetas = self._thetas[:, idx]
            self._alphas = self._alphas[idx]
            self._betas = self._betas[idx, :]
            self._betas_ds = self._betas_ds[idx, :]
            self._ndocs_active = self._ndocs_active[idx]
            self._topic_entropy = self._topic_entropy[idx]
            self._topic_coherence = self._topic_coherence[idx]
            self._tpc_labels = [self._tpc_labels[i] for i in idx]
            self._tpc_descriptions = [self._tpc_descriptions[i] for i in idx]
            self._edits.append('s ' + ' '.join([str(el) for el in idx]))

            # We are ready to save all variables in the model
            self._save_all()

            self._logger.info(
                '-- -- Topics reordering successful. All variables saved to file')
            return 1
        except:
            self._logger.info(
                '-- -- Topics reordering generated an error. Operation failed')
            return 0

    def resetTM(self):
        self._alphas_orig = np.load(self._TMfolder.joinpath('alphas_orig.npy'))
        self._betas_orig = np.load(self._TMfolder.joinpath('betas_orig.npy'))
        self._thetas_orig = sparse.load_npz(
            self._TMfolder.joinpath('thetas_orig.npz'))
        self._load_vocab()

        try:
            self.create(betas=self._betas_orig, thetas=self._thetas_orig,
                        alphas=self._alphas_orig, vocab=self._vocab)
            return 1
        except:
            return 0

    def recalculate_cohrs(self):

        self.load_tpc_descriptions()

        try:
            self.calculate_topic_coherence()

            self._save_cohr()

            self._logger.info(
                '-- -- Topics coherence recalculation successful. All variables saved to file')
            return 1
        except:
            self._logger.info(
                '-- -- Topics coherence recalculation  an error. Operation failed')
            return 0

    def to_dataframe(self):
        self._load_alphas()
        self._load_betas()
        self._load_betas_ds()
        self._load_thetas()
        self._load_betas_ds()
        self._load_topic_entropy()
        self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_ndocs_active()
        self._load_vocab()
        self._load_vocab_dicts()
        self.load_most_representative_docs()

        data = {
            "betas": [self._betas],
            "betas_ds": [self._betas_ds],
            "alphas": [self._alphas],
            "topic_entropy": [self._topic_entropy],
            "topic_coherence": [self._topic_coherence],
            "ndocs_active": [self._ndocs_active],
            "tpc_descriptions": [self._tpc_descriptions],
            "tpc_labels": [self._tpc_labels],
            "top_docs_per_topic": [self._most_representative_docs],
        }
        df = pd.DataFrame(data)
        return df, self._vocab_id2w, self._vocab