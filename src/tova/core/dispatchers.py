import importlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tova.api.models.data_schemas import DataRecord
from tova.topic_models.tm_model import TMmodel
from tova.utils.cancel import CancellationToken
from tova.utils.common import init_logger, load_yaml_config_file, pydantic_to_dict
from tova.utils.progress import ProgressCallback

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
    model_name: str,
    corpus_id: str,
    id: str,
    config_path: Path = Path("./static/config/config.yaml"),
    do_preprocess: bool = False,
    tr_params: Optional[Union[Dict, defaultdict]] = None,
    logger: Optional[logging.Logger] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cancel: Optional[CancellationToken] = None,
) -> float:

    # instantiate model type based on the registered classes
    model_cls = MODEL_REGISTRY.get(model)
    if model_cls is None:
        raise ValueError(f"Unknown model: {model}")

    _ = do_preprocess  # TODO: implement preprocessing

    tr_params = tr_params or {}

    tm_model = model_cls(
        model_name=model_name,
        corpus_id=corpus_id,
        id=id,
        model_path=output,
        config_path=config_path,
        load_model=False,
        logger=logger,
        **tr_params
    )

    return tm_model.train_model(data, progress_callback=progress_callback, cancel=cancel)


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


def _to_dashboard_bundle(
    raw: Dict[str, Any],
    model_path: Path,
    model_metadata: Optional[Dict[str, Any]] = None,
    model_training_corpus: Optional[Dict[str, Any]] = None,
    doc_thetas: Optional[Dict[str, Dict[str, float]]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    topics: Dict[str, Dict[str, Any]] = raw.get("Topics Info", {}) or {}
    metrics: Dict[str, Any] = raw.get("Model-Level Metrics", {}) or {}
    
    logger.info(f"These is the model metadata being used to create the dashboard bundle: {model_metadata}")

    all_ids: List[str] = []
    by_id: Dict[str, Any] = {}
    diags_by_id: Dict[str, Any] = {}
    sims_by_id: Dict[str, List[Tuple[str, float]]] = {}
    coords_by_id: Dict[str, Any] = {}

    def _as_list(*vals):
        for v in vals:
            if v is None:
                continue
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            if isinstance(v, str):
                if "," in v:
                    return [w.strip() for w in v.split(",") if w.strip()]
                return [w.strip() for w in v.split() if w.strip()]
        return []

    def _first_nonempty(*vals):
        for v in vals:
            if v is None:
                continue
            if isinstance(v, str) and v.strip():
                return v
            if isinstance(v, (int, float)):
                return v
            if isinstance(v, (list, dict)) and v:
                return v
        return None

    def _num(x, default=None):
        try:
            if isinstance(x, str):
                s = x.strip()
                if s.endswith("%"):
                    return float(s[:-1])
                return float(x)
            return float(x)
        except Exception:
            return default

    # --- topics and diagnostics
    for rid, t in topics.items():
        tid = str(rid)
        all_ids.append(tid)

        label = _first_nonempty(t.get("Label"), f"Topic {tid}")
        keywords = _as_list(t.get("Top Keywords"), t.get(
            "Top Words"), t.get("Keywords"))

        raw_cnt = _first_nonempty(
            t.get("# Docs Active"), t.get("Document Count"),
            t.get("N docs"), t.get("N_docs"), t.get("Docs"), 0
        )
        try:
            doc_count = int(raw_cnt)
        except Exception:
            doc_count = 0

        by_id[tid] = {
            "id": int(tid) if tid.isdigit() else tid,
            "label": label,
            "keywords": keywords,
            "documentCount": doc_count,
            "summary": t.get("Summary", "") or ""
        }

        diags_by_id[tid] = {
            "size": _num(t.get("Size")),
            "entropy": _num(t.get("Entropy")),
            "coherence": _num(t.get("Coherence (NPMI)")),
            "docsActive": doc_count
        }

        sims_raw = t.get("Similar Topics (Coocurring)") or t.get(
            "Similar Topics") or t.get("Coocurring")
        if isinstance(sims_raw, list):
            pairs = []
            for entry in sims_raw:
                other = str(entry.get("ID"))
                sim = _num(entry.get("Similarity"))
                if other is not None and sim is not None:
                    pairs.append([other, sim])
            if pairs:
                sims_by_id[tid] = pairs

        if isinstance(t.get("Coordinates"), (list, tuple)) and len(t["Coordinates"]) >= 2:
            cx, cy = t["Coordinates"][:2]
            coords_by_id[tid] = {"x": _num(cx, 0.0), "y": _num(
                cy, 0.0), "size": doc_count or 1}

    # --- documents and inferences
    documents_by_id: Dict[str, Any] = {}
    all_doc_ids: List[str] = []
    inferences: Dict[str, Any] = doc_thetas or {}

    docs_list = model_training_corpus.get(
        "documents", []) if model_training_corpus else []

    for d in docs_list:
        did = str(d.get("id"))
        if not did:
            continue
        text_snippet = (d.get("text", "") or "").strip()[:150]
        theta_dist = inferences.get(did, {})
        if theta_dist:
            # find dominant topic
            top_tid = max(theta_dist.items(), key=lambda kv: kv[1])[0]
            score = theta_dist[top_tid]
        else:
            top_tid, score = None, None

        documents_by_id[did] = {
            "id": did,
            "text": text_snippet or "—",
            "themeId": int(top_tid) if str(top_tid).isdigit() else top_tid,
            "score": score,
        }
        all_doc_ids.append(did)

    # --- metric cards
    metric_cards = []
    for k, v in metrics.items():
        try:
            v = float(v)
        except Exception:
            pass
        metric_cards.append({"label": k, "value": v})

    return {
        "model": model_path.name,
        "color": {"seed": 0, "saturation": [70, 75, 80], "lightness": [55, 60, 65]},
        "themes": {"allIds": [int(i) if i.isdigit() else i for i in all_ids], "byId": by_id},
        "diagnostics": {"byThemeId": diags_by_id},
        "similarities": sims_by_id,
        "metrics": metric_cards,
        "documents": {"allIds": all_doc_ids, "byId": documents_by_id},
        "coordinates": {"byThemeId": coords_by_id},
        "inferences": inferences,
        "modelInfos": {
            model_path.name: {
                "model_name": model_metadata.get("tr_params", {}).get("model_name", ""),
                "model_type": model_metadata.get("model_type", ""),
                "num_topics": model_metadata.get("tr_params", {}).get("num_topics", 0),
                "trained_on": model_training_corpus.get("name", ""),
                "training_params": model_metadata.get("tr_params", {}),
                "training_metrics": metrics,
            }
        }
    }


def get_model_info_dispatch(
    model_path: str,
    config_path: Path = Path("./static/config/config.yaml"),
    model_metadata: Optional[Dict[str, Any]] = None,
    model_training_corpus: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    output: str = "raw",
) -> Dict[str, Any]:
    """
    When output='dashboard', returns the normalized dashboard bundle with topic,
    model, and per-document theta information.
    """
    
    model_training_corpus = pydantic_to_dict(model_training_corpus) or {}

    config = load_yaml_config_file(config_path, "topic_modeling", logger)
    n_similar_tpcs = int(config.get("general", {}).get("n_similar_tpcs", 5))
    similar_tpc_thr = float(config.get(
        "general", {}).get("similar_tpc_thr", 0.5))
    n_top_docs = int(config.get("general", {}).get("n_top_docs", 20))

    # load model topics/metrics
    tmmodel = TMmodel(Path(model_path).joinpath("TMmodel"),
                      logger=logger, config_path=config_path)
    topic_info, _, _, irbo, td, similar, _ = tmmodel.get_all_model_info(
        nsimilar=n_similar_tpcs, thr=similar_tpc_thr, n_most=n_top_docs
    )

    topic_info = topic_info.set_index("ID").to_dict(orient="index")

    #  similar topics
    for tpc in list(topic_info.keys()):
        most_similars = []
        for most_similar in similar.get("Coocurring", {}).get(tpc, []):
            most_similars.append({
                "ID": str(most_similar[0]),
                "Label": topic_info.get(most_similar[0], {}).get("Label", f"Theme {most_similar[0]}"),
                "Similarity": most_similar[1],
            })
        topic_info[tpc]["Similar Topics (Coocurring)"] = most_similars

    topic_info = {str(k): v for k, v in topic_info.items()}
    topic_cohrs = [tpc.get("Coherence (NPMI)") for tpc in topic_info.values(
    ) if tpc.get("Coherence (NPMI)") is not None]
    topic_entrs = [tpc.get("Entropy") for tpc in topic_info.values(
    ) if tpc.get("Entropy") is not None]

    raw_model_info = {
        "Model Path": model_path,
        "Topics Info": topic_info,
        "Model-Level Metrics": {
            "Average Coherence (NPMI)": sum(topic_cohrs) / len(topic_cohrs) if topic_cohrs else 0,
            "Average Entropy": sum(topic_entrs) / len(topic_entrs) if topic_entrs else 0,
            "Topic Diversity": td,
            "IRBO": irbo,
        },
    }

    if output == "dashboard":
        docs_list_raw = model_training_corpus.get("documents", []) or []
        docs_list = [
                d if isinstance(d, dict) else pydantic_to_dict(d)
                for d in docs_list_raw
            ]
        doc_ids = [d.get("id") for d in docs_list if d.get("id")]

        try:
            thetas_info = get_thetas_documents_by_id_dispatch(
                docs_ids=doc_ids,
                model_path=model_path,
                config_path=config_path,
                logger=logger
            )
        except Exception as e:
            thetas_info = {}
            if logger:
                logger.warning(f"Could not compute thetas for docs: {e}")

        return _to_dashboard_bundle(
            raw_model_info,
            Path(model_path),
            model_metadata=model_metadata or {},
            model_training_corpus=pydantic_to_dict(model_training_corpus) or {},
            doc_thetas=thetas_info,
            logger=logger,
        )

    return raw_model_info


def get_topic_info_dispatch(
    topic_id: int,
    model_path: str,
    config_path: Path = Path("./static/config/config.yaml"),
    logger: Optional[logging.Logger] = None
) -> Dict:

    #  call get_model_info_dispatch to get all topics info and then filter by topic_id
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
    topic_info = {k: topic_info[k] for k in [
        "ID"] + [k for k in topic_info if k != "ID"]}
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
    similar_tpc_thr = float(config.get(
        "general", {}).get("similar_tpc_thr", 0.5))
    n_top_docs = int(config.get("general", {}).get("n_top_docs", 20))

    tmmodel = TMmodel(Path(model_path).joinpath("TMmodel"),
                      logger=logger, config_path=config_path)

    _, _, _, _, _, _, thetas_rpr = tmmodel.get_all_model_info(
        nsimilar=n_similar_tpcs, thr=similar_tpc_thr, n_most=n_top_docs)

    filtered_thetas = {
        doc_id: {topic_id: proportion for topic_id,
                 proportion in thetas_rpr[doc_id]}
        for doc_id in docs_ids if doc_id in thetas_rpr
    }

    return filtered_thetas
