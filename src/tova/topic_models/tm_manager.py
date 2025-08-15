"""This module is similar to the one available in the topicmodeler (https://github.com/IntelCompH2020/topicmodeler/blob/main/src/topicmodeling/manageModels.py). It provides functions to manage topic models.

Authors: Jerónimo Arenas-García, J.A. Espinosa-Melchor, Lorena Calvo-Bartolomé
Modified: 02/05/2025 (Updated for TOVA)
"""

from pathlib import Path
import json
import shutil

class TMManager(object):
    """
    Main class to manage functionality for the management of topic models
    """

    def listTMmodels(self, path_TMmodels: Path):
        """
        Returns a dictionary with all topic models or all DC models

        Parameters
        ----------
        path_TMmodels : pathlib.Path
            Path to the folder hosting the topic models or the dc models

        Returns
        -------
        allTMmodels : Dictionary (path -> dictionary)
            One dictionary entry per model
            key is the topic model name
            value is a dictionary with metadata
        """
        allTMmodels = {}
        modelFolders = [el for el in path_TMmodels.iterdir()]

        for TMf in modelFolders:
            # For topic models
            if TMf.joinpath('trainconfig.json').is_file():
                # print(f"{TMf.as_posix()} is a topic model")
                modelConfig = TMf.joinpath('trainconfig.json')
                with modelConfig.open('r', encoding='utf8') as fin:
                    modelInfo = json.load(fin)
                    allTMmodels[modelInfo['name']] = {
                        "name": modelInfo['name'],
                        "description": modelInfo['description'],
                        "visibility": modelInfo['visibility'],
                        "creator": modelInfo['creator'],
                        "trainer": modelInfo['trainer'],
                        "TrDtSet": modelInfo['TrDtSet'],
                        "creation_date": modelInfo['creation_date'],
                        "hierarchy-level": modelInfo['hierarchy-level'],
                        "htm-version": modelInfo['htm-version']
                    }
                    submodelFolders = [
                        el for el in TMf.iterdir()
                        if not el.as_posix().endswith("modelFiles")
                        and not el.as_posix().endswith("corpus.parquet")
                        and not el.as_posix().endswith("_old")]
                    for sub_TMf in submodelFolders:
                        submodelConfig = sub_TMf.joinpath('trainconfig.json')
                        if submodelConfig.is_file():
                            with submodelConfig.open('r', encoding='utf8') as fin:
                                submodelInfo = json.load(fin)
                                corpus = "Subcorpus created from " + \
                                    str(modelInfo['name'])
                                allTMmodels[submodelInfo['name']] = {
                                    "name": submodelInfo['name'],
                                    "description": submodelInfo['description'],
                                    "visibility": submodelInfo['visibility'],
                                    "creator": modelInfo['creator'],
                                    "trainer": submodelInfo['trainer'],
                                    "TrDtSet": corpus,
                                    "creation_date": submodelInfo['creation_date'],
                                    "hierarchy-level": submodelInfo['hierarchy-level'],
                                    "htm-version": submodelInfo['htm-version']
                                }
            # For DC models
            elif TMf.joinpath('dc_config.json').is_file():
                # print(f"{TMf.as_posix()} is a domain classifier model")
                modelConfig = TMf.joinpath('dc_config.json')
                with modelConfig.open('r', encoding='utf8') as fin:
                    modelInfo = json.load(fin)
                allTMmodels[modelInfo['name']] = {
                    "name": modelInfo['name'],
                    "description": modelInfo['description'],
                    "visibility": modelInfo['visibility'],
                    "creator": modelInfo['creator'],
                    "type": modelInfo['type'],
                    "corpus": modelInfo['corpus'],
                    "tag": modelInfo['tag'],
                    "creation_date": modelInfo['creation_date']
                }
            # This condition only applies for Mac OS
            elif TMf.name == ".DS_Store":
                pass
            else:
                print(f"No valid JSON file provided for Topic models or DC models")
                return 0
        return allTMmodels

    def getTMmodel(self, path_TMmodel: Path):
        """
        Returns a dictionary with a topic model and it's sub-models

        Parameters
        ----------
        path_TMmodel : pathlib.Path
            Path to the folder hosting the topic model

        Returns
        -------
        result : Dictionary (path -> dictionary)
            One dictionary entry per model
            key is the topic model name
            value is a dictionary with metadata
        """
        result = {}

        modelConfig = path_TMmodel.joinpath('trainconfig.json')
        if modelConfig.is_file():
            with modelConfig.open('r', encoding='utf8') as fin:
                modelInfo = json.load(fin)
                result[modelInfo['name']] = {
                    "name": modelInfo['name'],
                    "description": modelInfo['description'],
                    "visibility": modelInfo['visibility'],
                    "creator": modelInfo['creator'],
                    "trainer": modelInfo['trainer'],
                    "TMparam": modelInfo['TMparam'],
                    "creation_date": modelInfo['creation_date'],
                    "hierarchy-level": modelInfo['hierarchy-level'],
                    "htm-version": modelInfo['htm-version']
                }
            submodelFolders = [el for el in path_TMmodel.iterdir() if not el.as_posix().endswith(
                "modelFiles") and not el.as_posix().endswith("corpus.parquet") and not el.as_posix().endswith("_old")]
            for sub_TMf in submodelFolders:
                submodelConfig = sub_TMf.joinpath('trainconfig.json')
                if submodelConfig.is_file():
                    with submodelConfig.open('r', encoding='utf8') as fin:
                        submodelInfo = json.load(fin)
                        corpus = "Subcorpus created from " + \
                            str(modelInfo['name'])
                        result[submodelInfo['name']] = {
                            "name": submodelInfo['name'],
                            "description": submodelInfo['description'],
                            "visibility": submodelInfo['visibility'],
                            "creator": modelInfo['creator'],
                            "trainer": submodelInfo['trainer'],
                            "TrDtSet": corpus,
                            "TMparam": submodelInfo['TMparam'],
                            "creation_date": submodelInfo['creation_date'],
                            "hierarchy-level": submodelInfo['hierarchy-level'],
                            "htm-version": submodelInfo['htm-version']
                        }
        return result

    def deleteTMmodel(self, path_TMmodel: Path):
        """
        Deletes a Topic Model or a DC model

        Parameters
        ----------
        path_TMmodel : pathlib.Path
            Path to the folder containing the Topic Model or the DC model

        Returns
        -------
        status : int
            - 0 if the model could not be deleted
            - 1 if the model was deleted successfully
        """

        if not path_TMmodel.is_dir():
            print(f"File '{path_TMmodel.as_posix()}' does not exist.")
            return 0
        else:
            try:
                shutil.rmtree(path_TMmodel)
                return 1
            except:
                return 0

    def renameTMmodel(self, name: Path, new_name: Path):
        """
        Renames a topic model or a DC model

        Parameters
        ----------
        name : pathlib.Path
            Path to the model to be renamed

        new_name : pathlib.Path
            Path to the new name for the model

        Returns
        -------
        status : int
            - 0 if the model could not be renamed
            - 1 if the model was renamed successfully

        """
        if not name.is_dir():
            print(f"Model '{name.as_posix()}' does not exist.")
            return 0
        if new_name.is_file():
            print(
                f"Model '{new_name.as_posix()}' already exists. Rename or delete it first.")
            return 0
        try:
            # Checking whether it is a TM or DC model
            if name.joinpath('trainconfig.json').is_file():
                config_file = name.joinpath('trainconfig.json')
            elif name.joinpath('dc_config.json').is_file():
                config_file = name.joinpath('dc_config.json')
            with config_file.open("r", encoding="utf8") as fin:
                TMmodel = json.load(fin)
            TMmodel["name"] = new_name.stem
            with config_file.open("w", encoding="utf-8") as fout:
                json.dump(TMmodel, fout, ensure_ascii=False,
                          indent=2, default=str)
            shutil.move(name, new_name)
            return 1
        except:
            return 0

    def copyTMmodel(self, name: Path, new_name: Path):
        """
        Makes a copy of an existing TM or DC model

        Parameters
        ----------
        name : pathlib.Path
            Path to the model to be copied

        new_name : pathlib.Path
            Path to the new name for the model

        Returns
        -------
        status : int
            - 0 if the model could not be copied
            - 1 if the model was copied successfully

        """
        if not name.is_dir():
            print(f"Model '{name.as_posix()}' does not exist.")
            return 0
        if new_name.is_file():
            print(
                f"Model '{new_name.as_posix()}' already exists. Rename or delete it first.")
            return 0
        try:
            shutil.copytree(name, new_name)

            # Checking whether it is a TM or DC model
            if new_name.joinpath('trainconfig.json').is_file():
                config_file = name.joinpath('trainconfig.json')
            elif new_name.joinpath('dc_config.json').is_file():
                config_file = name.joinpath('dc_config.json')
            with config_file.open("r", encoding="utf8") as fin:
                TMmodel = json.load(fin)
            TMmodel["name"] = new_name.stem
            with config_file.open("w", encoding="utf-8") as fout:
                json.dump(TMmodel, fout, ensure_ascii=False,
                          indent=2, default=str)
            return 1
        except:
            return 0