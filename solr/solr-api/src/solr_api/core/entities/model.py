"""
This module implements a class to manage and hold all the information associated with a TMmodel. It provides methods to retrieve information about the model such as topic distribution over documents, topic-word probabilities, and betas. 

Note:
-----
- This module assumes that the topic model has been as an associated TMmodel object (tova.topic_models.tm_model.TMmodel).
- This module assumes that the TMmodel folder has already been created.

Author: Lorena Calvo-Bartolomé
Date: 27/03/2023
Modified: 26/08/2025 (Updated for TOVA Project)
"""


import configparser
import json
import logging
import os
import pathlib
from typing import List

import numpy as np
import pandas as pd
from tova.topic_models.tm_model import TMmodel  # type: ignore
from solr_api.core.entities.utils import sum_up_to
# from utils import sum_up_to


class Model(object):
    """
    A class to manage and hold all the information associated with a TMmodel so it can be indexed in Solr.
    """

    def __init__(
        self,
        path_to_model: str,
        logger: logging.Logger | None = None,
        #config_file: str = "/config/config.cf"
    ) -> None:
        """Init method.

        Parameters
        ----------
        path_to_model: str
            Path to the TMmodel folder.
        logger : logging.Logger | None
            Logger to use; if None, a module-level logger is created.
        """

        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('Entity Model')

        path_to_model = pathlib.Path(path_to_model)
        if not path_to_model.is_dir():
            err = f"Model path is not a directory: {path_to_model}"
            self._logger.error(err)
            raise FileNotFoundError(err)
        self.path_to_model = path_to_model

        # Get model and corpus names
        self.name = path_to_model.stem.lower()
        self.corpus_name = None

        # Read configuration from config file
        cf = configparser.ConfigParser()
        cf.read(config_file)
        if self.name.startswith('prodlda') or self.name.startswith('ctm'):
            self.thetas_max_sum = int(
                cf.get('restapi', 'max_sum_neural_models'))
        else:
            self.thetas_max_sum = int(cf.get('restapi', 'thetas_max_sum'))
        self.betas_max_sum = int(cf.get('restapi', 'betas_max_sum'))
        # self.thetas_max_sum = 1000
        # self.betas_max_sum = 10000

        # Get model information from TMmodel
        self.tmmodel = TMmodel(self.path_to_model.joinpath("TMmodel"))
        self.alphas, self.betas, self.thetas, self.vocab, self.sims, self.coords, self.s3 = self.tmmodel.get_model_info_for_vis()

        return

    def get_model_info(self) -> List[dict]:
        """It retrieves the information about a topic model as a list of dictionaries.

        Returns:
        --------
        json_lst: list[dict]
            A list of dictionaries containing information about the topic model.
        """

        # Read training config ('trainconfig.json')
        tr_config = self.path_to_model.joinpath("trainconfig.json")
        with pathlib.Path(tr_config).open('r', encoding='utf8') as fin:
            tr_config = json.load(fin)

        # Get model information as dataframe, where each row is a topic
        df, vocab_id2w, vocab = self.tmmodel.to_dataframe()
        df = df.apply(pd.Series.explode)
        df.reset_index(drop=True)
        df["id"] = [f"t{i}" for i in range(len(df))]

        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]

        # Get betas scale to self.max_sum
        def get_betas_scale(vector: np.array,
                            max_sum: int):
            vector = sum_up_to(vector, max_sum)
            return vector

        df["betas_scale"] = df["betas"].apply(
            lambda x: get_betas_scale(x, self.betas_max_sum))

        # Get words in each topic
        def get_tp_words(vector: np.array,
                         vocab_id2w: dict) -> str:
            return ", ".join([vocab_id2w[str(idx)] for idx, val in enumerate(vector) if val != 0])

        df["vocab"] = df["betas"].apply(
            lambda x: get_tp_words(x, vocab_id2w))

        # Get betas string representation
        def get_tp_str_rpr(vector: np.array,
                           vocab_id2w: dict) -> str:
            rpr = " ".join([f"{vocab_id2w[str(idx)]}|{val}" for idx,
                           val in enumerate(vector) if val != 0]).rstrip()
            return rpr

        df["betas"] = df["betas_scale"].apply(
            lambda x: get_tp_str_rpr(x, vocab_id2w))

        def get_top_words_betas(row, vocab, n_words=15):
            # a row is a topic
            word_tfidf_dict = {vocab[idx2]: row["betas_ds"][idx2]
                               for idx2 in np.argsort(row["betas_ds"])[::-1][:n_words]}

            # Transform such as value0 = 100 % and following values are relative to value0
            value0 = word_tfidf_dict[list(word_tfidf_dict.keys())[0]]
            word_tfidf_dict = {word: round((val/value0)*100, 3)
                               for word, val in word_tfidf_dict.items()}
            rpr = " ".join(
                [f"{word}|{word_tfidf_dict[word]}" for word in word_tfidf_dict.keys()]).rstrip()

            return rpr

        df['top_words_betas'] = df.apply(
            lambda row: get_top_words_betas(row, vocab), axis=1)

        # Drop betas scale because it is not needed
        df = df.drop(columns=["betas_scale", "betas_ds"])

        # Get topic coordinates in cluster space
        df["coords"] = self.coords

        json_str = df.to_json(orient='records')
        json_lst = json.loads(json_str)

        return json_lst

    def get_model_info_update(self, action: str) -> List[dict]:
        """
        Retrieves the information from the model that goes to a corpus collection (document-topic proportions) and save it as an update in the format required by Solr.

        Parameters
        ----------
        action: str
            Action to be performed ('set', 'remove')

        Returns:
        --------
        json_lst: list[dict]
            A list of dictionaries with thr document-topic proportions update.
        """

        # Read training configuration
        tr_config = self.path_to_model.joinpath("trainconfig.json")
        with pathlib.Path(tr_config).open('r', encoding='utf8') as fin:
            tr_config = json.load(fin)

        # Get corpus path and name of the collection
        self.corpus = tr_config["TrDtSet"]
        self.corpus_name = pathlib.Path(tr_config["TrDtSet"]).name
        if self.corpus_name.endswith(".parquet") or self.corpus_name.endswith(".json"):
            self.corpus_name = self.corpus_name.split(".")[0].lower()

        # Keys for dodument-topic proportions and similarity that will be used within the corpus collection
        model_key = 'doctpc_' + self.name
        sim_model_key = 'sim_' + self.name
        s3_model_key = 's3_' + self.name
        
        # Get ids of documents kept in the tr corpus
        if tr_config["trainer"].lower() == "mallet":
            def process_line(line):
                id_ = line.rsplit(' 0 ')[0].strip()
                id_ = int(id_.strip('"'))
                return id_
            with open(self.path_to_model.joinpath("corpus.txt"), encoding="utf-8") as file:
                ids_corpus = [process_line(line) for line in file]
        elif tr_config["trainer"].lower() == "prodlda" or \
                tr_config["trainer"].lower() == "ctm":
            ddf = dd.read_parquet(
                self.path_to_model.joinpath("corpus.parquet"))
            with ProgressBar():
                ids_corpus = ddf["id"].compute(scheduler='processes')
        else:
            self._logger.error(
                '-- -- The trainer used to train the model is not supported.')

        # Actual topic model's information only needs to be retrieved if action is "set"
        if action == "set":
            # Get doc-topic representation
            def get_doc_str_rpr(vector, max_sum):
                """Calculates the string representation of a document's topic proportions in the format 't0|100 t1|200 ...', so that the sum of the topic proportions is at most max_sum.

                Parameters
                ----------
                vector: numpy.array
                    Array with the topic proportions of a document.
                max_sum: int
                    Maximum sum of the topic proportions.

                Returns 
                -------
                rpr: str
                    String representation of the document's topic proportions.
                """
                vector = sum_up_to(vector, max_sum)
                rpr = ""
                for idx, val in enumerate(vector):
                    if val != 0:
                        rpr += "t" + str(idx) + "|" + str(val) + " "
                rpr = rpr.rstrip()
                return rpr
            
            def get_s3_str_rpr(vector):
                """Calculates the s3 string representation.

                Parameters
                ----------
                vector: numpy.array
                    Array with the topic proportions of a document.
                max_sum: int
                    Maximum sum of the topic proportions.

                Returns 
                -------
                rpr: str
                    String representation of the document's topic proportions.
                """
                rpr = ""
                for idx, val in enumerate(np.asarray(vector).flatten()):
                    if val != 0:
                        rpr += "t" + str(idx) + "|" + str(val) + " "
                rpr = rpr.rstrip()
                return rpr

            self._logger.info("Attaining thetas rpr...")
            thetas_dense = self.thetas.todense()
            doc_tpc_rpr = [get_doc_str_rpr(thetas_dense[row, :], self.thetas_max_sum) for row in range(len(thetas_dense))]
            
            self._logger.info("Attaining S3 rpr...")
            s3_dense = self.s3.todense()
            self._logger.info(f"Some s3_dense: {s3_dense[:5]}")
            s3_rpr = [get_s3_str_rpr(s3_dense[row, :]) for row in range(len(s3_dense))]

            # Get similarities string representation
            self._logger.info("Attaining sims rpr...")

            with open(self.path_to_model.joinpath("TMmodel").joinpath('distances.txt'), 'r') as f:
                sim_rpr = [line.strip() for line in f]
            self._logger.info(
                "Thetas and sims attained. Creating dataframe...")

            # Save the information in a dataframe
            df = pd.DataFrame(list(zip(ids_corpus, doc_tpc_rpr, sim_rpr,s3_rpr)), columns=['id', model_key, sim_model_key,s3_model_key])
            self._logger.info(
                f"Dataframe created. Printing columns:{df.columns.tolist()}")

        elif action == "remove":
            doc_tpc_rpr = ["" for _ in range(len(ids_corpus))]
            sim_rpr = doc_tpc_rpr
            # Save the information in a dataframe
            df = pd.DataFrame(list(zip(ids_corpus, doc_tpc_rpr, sim_rpr, s3_rpr)), columns=['id', model_key, sim_model_key,s3_model_key])

        # Create json from dataframe
        json_str = df.to_json(orient='records')
        json_lst = json.loads(json_str)

        # Updating json in the format required by Solr
        new_list = []
        if action == 'set':
            for d in json_lst:
                tpc_dict = {'set': d[model_key]}
                d[model_key] = tpc_dict
                sim_dict = {'set': d[sim_model_key]}
                d[sim_model_key] = sim_dict
                s3_dict = {'set': d[s3_model_key]}
                d[s3_model_key] = s3_dict
                new_list.append(d)
        elif action == 'remove':
            for d in json_lst:
                tpc_dict = {'set': []}
                d[model_key] = tpc_dict
                sim_dict = {'set': []}
                d[sim_model_key] = sim_dict
                s3_dict = {'set': []}
                d[s3_model_key] = s3_dict
                new_list.append(d)

        return new_list, self.corpus_name

    def get_corpora_model_update(
        self,
        id: int,
        action: str
    ) -> List[dict]:
        """Generates an update for the CORPUS_COL collection.
        
        Parameters
        ----------
        id: int
            Identifier of the corpus collection in CORPUS_COL
        action: str
            Action to be performed ('add', 'remove')

        Returns:
        --------
        json_lst: list[dict]
            A list of dictionaries with the update.
        """

        json_lst = [{"id": id,
                    "fields": {action: ['doctpc_' + self.name,
                                        'sim_' + self.name,
                                        's3_' + self.name
                                        ]},
                     "models": {action: self.name}
                     }]

        return json_lst


# if __name__ == '__main__':
#    model = Model(pathlib.Path(
#       "/export/data_ml4ds/IntelComp/EWB/data/source/HFRI-30"))
    # json_lst = model.get_model_info_update(action='set')
    # pos = model.get_topic_pos()
    # print(json_lst[0])
#    df = model.get_model_info()
    # print(df[0].keys())
    # upt = model.get_corpora_model_update()
    # print(upt)