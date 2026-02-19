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
    model_id: str,
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
        "model": model_id,
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


def dashboard_context_for_llm(
    context: Dict[str, Any],
    *,
    max_themes: Optional[int] = 30,
    max_doc_snippets: int = 0,
    include_diagnostics: bool = True,
    include_model_info: bool = True,
    max_keywords_per_theme: int = 15,
) -> str:
    """
    Convert dashboard context (as sent from the chat UI) into a single text block
    suitable for LLM system prompts or RAG-style context.

    Expected context keys (all optional):
      - themes: list of { id, label, keywords, documentCount?, summary? }
      - diagnostics: dict theme_id -> { size, entropy, coherence, docsActive }
      - similarities: dict theme_id -> list of [other_theme_id, similarity_score] (for comparison)
      - model_info: { model_name, model_type, num_topics, trained_on? }
      - document_count: int
      - document_samples: list of { id, text, themeId, score } (snippets)

    Returns a formatted string the LLM can use to answer questions about the model.
    """
    sections: List[str] = []

    # Model overview
    if include_model_info:
        model_info = context.get("model_info") or {}
        if model_info:
            name = model_info.get("model_name") or "Unknown"
            mtype = model_info.get("model_type") or ""
            n_topics = model_info.get("num_topics")
            trained_on = model_info.get("trained_on") or ""
            parts = [f"Topic model: {name}"]
            if mtype:
                parts.append(f"type={mtype}")
            if n_topics is not None:
                parts.append(f"num_topics={n_topics}")
            if trained_on:
                parts.append(f"trained_on={trained_on}")
            sections.append("## Model\n" + ", ".join(parts))

    doc_count = context.get("document_count")
    if doc_count is not None:
        if not sections:
            sections.append("## Corpus")
        sections[-1] += f"\nTotal documents: {doc_count}"

    # Themes (id, label, keywords, optional diagnostics)
    themes = context.get("themes") or []
    if themes and max_themes:
        themes = themes[:max_themes]
    if themes:
        lines = ["## Themes (topics)"]
        diags = context.get("diagnostics") or {}
        for t in themes:
            tid = t.get("id")
            label = t.get("label") or f"Theme {tid}"
            kw = t.get("keywords") or []
            if max_keywords_per_theme and len(kw) > max_keywords_per_theme:
                kw = kw[:max_keywords_per_theme]
            keywords_str = ", ".join(kw) if kw else "—"
            line = f"- {label} (id={tid}): {keywords_str}"
            doc_count_t = t.get("documentCount") or t.get("document_count")
            if doc_count_t is not None:
                line += f" — {doc_count_t} documents"
            if include_diagnostics and str(tid) in diags:
                d = diags[str(tid)]
                metrics = []
                if d.get("coherence") is not None:
                    metrics.append(f"coherence={d['coherence']}")
                if d.get("entropy") is not None:
                    metrics.append(f"entropy={d['entropy']}")
                if d.get("size") is not None:
                    metrics.append(f"size={d['size']}")
                if metrics:
                    line += f" [{', '.join(metrics)}]"
            lines.append(line)
            summary = t.get("summary") or ""
            if summary and isinstance(summary, str) and summary.strip():
                lines.append(f"  Summary: {summary.strip()}")
        sections.append("\n".join(lines))

    # Theme similarities (for comparison questions)
    similarities = context.get("similarities") or {}
    if similarities and themes:
        label_by_id = {str(t.get("id")): (t.get("label") or f"Theme {t.get('id')}") for t in themes}
        sim_lines = ["## Theme similarities (for comparison)"]
        sim_lines.append("Similarity scores: 1 = most similar, 0 = unrelated, negative = rarely co-occur.")
        for tid, sim_list in similarities.items():
            if not sim_list:
                continue
            theme_label = label_by_id.get(str(tid), f"Theme {tid}")
            parts = []
            for item in sim_list[:10]:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    other_id, score = str(item[0]), item[1]
                elif isinstance(item, dict):
                    other_id = str(item.get("ID") or item.get("id", ""))
                    score = item.get("Similarity") or item.get("similarity")
                else:
                    continue
                other_label = label_by_id.get(other_id, f"Theme {other_id}")
                try:
                    s = f"{other_label} ({float(score):.2f})"
                except (TypeError, ValueError):
                    s = f"{other_label} ({score})"
                parts.append(s)
            if parts:
                sim_lines.append(f"- {theme_label}: similar to {', '.join(parts)}")
        if len(sim_lines) > 2:
            sections.append("\n".join(sim_lines))

    # Analysis cues: pre-computed rankings to support insightful analysis
    if themes:
        diags = context.get("diagnostics") or {}
        total_docs = context.get("document_count")
        analysis_lines = ["## Analysis cues (for insightful analysis)"]
        analysis_lines.append("Use these rankings to answer analysis and insight questions. All values are from the context above.")
        with_count = [(t, t.get("documentCount") or t.get("document_count") or 0) for t in themes]
        with_count.sort(key=lambda x: -x[1])
        top_by_docs = [t[0] for t in with_count if t[1] > 0][:5]
        if top_by_docs:
            labels = [t.get("label") or f"Theme {t.get('id')}" for t in top_by_docs]
            analysis_lines.append(f"Dominant themes (most documents): {', '.join(labels)}.")
        with_coh = []
        for t in themes:
            tid = str(t.get("id"))
            c = (diags.get(tid) or {}).get("coherence")
            if c is not None:
                with_coh.append((t.get("label") or f"Theme {tid}", float(c)))
        with_coh.sort(key=lambda x: -x[1])
        if with_coh:
            top_coh = [x[0] for x in with_coh[:5]]
            analysis_lines.append(f"Themes with highest coherence (most interpretable): {', '.join(top_coh)}.")
        with_ent = []
        for t in themes:
            tid = str(t.get("id"))
            e = (diags.get(tid) or {}).get("entropy")
            if e is not None:
                with_ent.append((t.get("label") or f"Theme {tid}", float(e)))
        with_ent.sort(key=lambda x: x[1])
        if with_ent:
            focused = [x[0] for x in with_ent[:5]]
            analysis_lines.append(f"Most focused themes (lowest entropy): {', '.join(focused)}.")
        if total_docs is not None and total_docs > 0:
            analysis_lines.append(f"Overall: {len(themes)} themes over {total_docs} documents.")
        if len(analysis_lines) > 2:
            sections.append("\n".join(analysis_lines))

    # Optional document samples (for “find documents about X” style questions)
    if max_doc_snippets and context.get("document_samples"):
        samples = context["document_samples"][:max_doc_snippets]
        lines = ["## Sample documents (snippets)"]
        for s in samples:
            text = (s.get("text") or "—").strip()[:200]
            theme_id = s.get("themeId")
            score = s.get("score")
            parts = [f"id={s.get('id')}", text]
            if theme_id is not None:
                parts.append(f"primary_theme={theme_id}")
            if score is not None:
                parts.append(f"score={score:.3f}")
            lines.append(" | ".join(str(p) for p in parts))
        sections.append("\n".join(lines))

    if not sections:
        return ""
    return "\n\n".join(sections)


