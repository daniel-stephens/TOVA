from collections import defaultdict
import json
import importlib
import logging
from pathlib import Path
import pandas as pd # type: ignore
from typing import Dict, List, Optional, Union

from src.topic_models.tm_model import TMmodel
from src.utils.common import init_logger, load_yaml_config_file

# -------------------- #
# AUXILIARY FUNCTIONS  #
# -------------------- #

def load_class_from_path(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

# -------------------- #
#    MODEL REGISTRY    #
# -------------------- #

with open("./static/config/modelRegistry.json", "r") as f:
    model_classes = json.load(f)

MODEL_REGISTRY = {
    key: load_class_from_path(path) for key, path in model_classes.items()
}

# -------------------- #
#     DISPATCHERS      #
# -------------------- #

def train_model_dispatch(
    model: str,
    data: List[Dict],
    output: str,
    config_path: Path = Path("./static/config/config.yaml"),
    do_preprocess: bool = False,
    tr_params: Optional[Union[Dict, defaultdict]] = None,
    logger: Optional[logging.Logger] = None
) -> float:
    model_cls = MODEL_REGISTRY.get(model)
    if model_cls is None:
        raise ValueError(f"Unknown model: {model}")
    
    _ = do_preprocess # @lcalvobartolome: TODO: implement preprocessing

    # Use empty dict if no training params provided
    tr_params = tr_params or {}

    tm_model = model_cls(
        model_path=output,
        config_path=config_path,
        load_model=False,
        logger=logger,
        **tr_params
    )
    
    return tm_model.train_model(data)

def infer_model_dispatch(
    model_path: str,
    data: List[Dict],
    config_path: Path = Path("./static/config/config.yaml"),
    logger: Optional[logging.Logger] = None
) -> float:
    """
    Infer the model using the provided data.
    
    Parameters
    ----------
    model_path : str
        Path to the model directory that is going to be used for inference.
    data : List[Dict]
        List of input records to make inference on.
    config_path : Path
        Path to the YAML config file.
    logger : Optional[logging.Logger]
        Logger instance for logging. If None, a new logger will be created.
    
    Returns
    -------
    thetas : List[Dict]
        List of dictionaries containing topic weights for each document in the format:
        {
            "doc_id": {
                "topic_id0": "topic_weight",
                "topic_id1": "topic_weight",
                ...
            }
        }
    duration : float
        Duration of the inference process in seconds.
    """
    
    logger = logger or init_logger(config_file=config_path)
    
    logger.info(f"Getting model info from {model_path}")
    with open(f"{model_path}/model_config.json", "r") as f:
        config = json.load(f)
    model_cls = load_class_from_path(config["model_type"])
    logger.info(f"Loading model {model_cls} from {model_path}")

    tm_model = model_cls.from_saved_model(model_path)
    logger.info(f"Loading model {tm_model} from {model_path}")
    
    thetas, duration = tm_model.infer(data)
    # get ids from the data
    ids = [record["id"] for record in data]
    
    thetas = [
        {
            ids[i]: {
                **{
                str(j): float(thetas[i][j])
                for j in range(len(thetas[i]))
                if thetas[i][j] > 0
            }
        }
            
        }
        for i in range(len(thetas))
    ]

    return thetas, duration

def get_model_info_dispatch(
    model_path: str,
    config_path: Path = Path("./static/config/config.yaml"),
    logger: Optional[logging.Logger] = None
) -> Dict:
    
    # load configuration
    config = load_yaml_config_file(config_path, "topic_modeling", logger)
    n_similar_tpcs = int(config.get("general", {}).get("n_similar_tpcs", 5))
    similar_tpc_thr = float(config.get("general", {}).get("similar_tpc_thr", 0.5))
    n_top_docs = int(config.get("general", {}).get("n_top_docs", 20))
    
    tmmodel = TMmodel(Path(model_path).joinpath("TMmodel"), logger=logger, config_path=config_path)
    
    topic_info, _, _, irbo, td, similar, _ = tmmodel.get_all_model_info(nsimilar=n_similar_tpcs, thr=similar_tpc_thr, n_most=n_top_docs)
    
    topic_info = topic_info.set_index("ID").to_dict(orient="index")
    for tpc in topic_info.keys():
        for key in ["Coocurring"]: #similar.keys():#
            most_similars = []
            for most_similar in similar[key][tpc]:
                most_similars.append({
                    "ID": str(most_similar[0]),
                    "Label": topic_info[most_similar[0]]["Label"],
                    "Similarity": most_similar[1]
                })
            topic_info[tpc][f"Similar Topics ({key})"] = most_similars

    topic_info = {str(k): v for k, v in topic_info.items()}
    topic_cohrs = [tpc["Coherence (NPMI)"] for tpc in topic_info.values()]
    topic_entrs = [tpc["Entropy"] for tpc in topic_info.values()]
    
    model_info = {
        "Model Path": model_path,
        "Topics Info": topic_info,
        "Model-Level Metrics": {
            "Average Coherence (NPMI)": sum(topic_cohrs) / len(topic_cohrs) if topic_cohrs else 0,
            "Average Entropy": sum(topic_entrs) / len(topic_entrs) if topic_entrs else 0,
            "Topic Diversity": td,
            "IRBO": irbo
        },
    }
    return model_info
    
def get_topic_info_dispatch(
    topic_id: int,
    model_path: str,
    config_path: Path = Path("./static/config/config.yaml"),
    logger: Optional[logging.Logger] = None
) -> Dict:

    # call get_model_info_dispatch to get all topics info and then filter by topic_id
    model_info = get_model_info_dispatch(
        model_path=model_path,
        config_path=config_path,
        logger=logger
    )
    topic_info = model_info["Topics Info"]
    topic_info = {k: v for k, v in topic_info.items() if int(k) == topic_id}
    if not topic_info:
        return None
    topic_info = topic_info[str(topic_id)]
    topic_info["ID"] = str(topic_id)    
    topic_info = {k: topic_info[k] for k in ["ID"] + [k for k in topic_info if k != "ID"]}
    return topic_info


def get_thetas_documents_by_id_dispatch(
    docs_ids: int | List[int],
    model_path: str,
    config_path: Path = Path("./static/config/config.yaml"),
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    It retrieves the topic weights for specific documents by their IDs from a trained topic model. Follows this structure:
    
    {
        'doc_id': {
            'topic_id0': topic_weight,
            'topic_id1': topic_weight,
            ...
        }, 
        'doc_id1': {
            'topic_id0': topic_weight,
            'topic_id1': topic_weight,
            ...
        },
        ...
    }
    """
    
    # load configuration
    config = load_yaml_config_file(config_path, "topic_modeling", logger)
    n_similar_tpcs = int(config.get("general", {}).get("n_similar_tpcs", 5))
    similar_tpc_thr = float(config.get("general", {}).get("similar_tpc_thr", 0.5))
    n_top_docs = int(config.get("general", {}).get("n_top_docs", 20))

    tmmodel = TMmodel(Path(model_path).joinpath("TMmodel"), logger=logger, config_path=config_path)

    _, _, _, _, _, _, thetas_rpr = tmmodel.get_all_model_info(nsimilar=n_similar_tpcs, thr=similar_tpc_thr, n_most=n_top_docs)
    
    filtered_thetas = {
        doc_id: {topic_id: proportion for topic_id, proportion in thetas_rpr[doc_id]}
        for doc_id in docs_ids if doc_id in thetas_rpr
    }
    
    return filtered_thetas