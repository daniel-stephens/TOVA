from collections import defaultdict
import json
import importlib
import logging
from pathlib import Path
import pandas as pd # type: ignore
from typing import Dict, List, Optional, Union

from src.topic_models.tm_model import TMmodel
from src.utils.common import init_logger

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
                f"t{j}": float(thetas[i][j])
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
    
    tmmodel = TMmodel(Path(model_path).joinpath("TMmodel"), logger=logger, config_path=config_path)
    df, _, _ = tmmodel.to_dataframe()
    df = df.apply(pd.Series.explode)
    # keep only alphas, tpc_labels and tpc_descriptions, top_docs_per_topic
    df = df[["alphas", "tpc_labels", "tpc_descriptions", "top_docs_per_topic"]]
    # scale alphas to percentage (f"{}:.2%}")
    df["alphas"] = df["alphas"].apply(lambda x: f"{x:.2%}")
    # convert top_docs_per_topic, which is a list of tuples, with the first element being the doc_id and the second being the score to a nested dict
    df["top_docs_per_topic"] = df["top_docs_per_topic"].apply(
        lambda x: {f"doc_{i[0]}": i[1] for i in x}
    )
    
    # assign topic id
    df = df.reset_index(drop=True)
    df["id"] = df.index

    # convert to dict of dicts
    df = df.set_index("id").to_dict(orient="index")
    df = {f"t{str(k)}": v for k, v in df.items()}

    return df

def get_thetas_dispatch(
    model_path: str,
    config_path: Path = Path("./static/config/config.yaml"),
    logger: Optional[logging.Logger] = None
) -> Dict:
    
    tmmodel = TMmodel(Path(model_path).joinpath("TMmodel"), logger=logger, config_path=config_path)
    tmmodel._load_thetas()
    return tmmodel._thetas.toarray().tolist()