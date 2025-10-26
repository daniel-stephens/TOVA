import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def pydantic_to_dict(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if isinstance(model, dict):
        return model
    if isinstance(model, (list, tuple)):
        return {"items": [pydantic_to_dict(m) if not isinstance(m, dict) else m for m in model]}

    dump_method = getattr(model, "model_dump", None)
    if callable(dump_method):
        return dump_method(by_alias=True)
    dict_method = getattr(model, "dict", None)
    if callable(dict_method):
        return dict_method(by_alias=True)

    try:
        return dict(model)
    except Exception:
        try:
            import json
            return json.loads(json.dumps(model, default=lambda o: getattr(o, "__dict__", {})))
        except Exception:
            return {}

def to_dashboard_bundle(
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

    # topics and diagnostics
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

    # documents and "inferences"??
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

    # metric cards
    metric_cards = []
    for k, v in metrics.items():
        try:
            v = float(v)
        except Exception:
            pass
        metric_cards.append({"label": k, "value": v})

    model_name = model_metadata.get("tr_params", {}).get("model_name", "")
    
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
            model_name: {
                "model_name": model_name,
                "model_type": model_metadata.get("type", "").split(".")[-1],
                "num_topics": model_metadata.get("tr_params", {}).get("num_topics", 0),
                "trained_on": model_training_corpus.get("name", ""),
                "training_params": model_metadata.get("tr_params", {}),
                "training_metrics": metrics,
            }
        }
    }