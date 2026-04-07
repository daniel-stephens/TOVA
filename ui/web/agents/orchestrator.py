"""Agent orchestration entrypoints for UI workflows.

Right now the orchestration is mostly "RAG chat -> LLM call", but keeping it behind
an explicit boundary makes it much easier to add tools/agents later.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, TypedDict

import requests

from dashboard_utils import dashboard_context_for_llm
from web.services import runtime as R
from web.services.llm_client import chat_llm, chat_llm_stream, get_chat_llm_defaults, get_openai_key_from_env

logger = logging.getLogger(__name__)

_RAG_TIMEOUT = (5, 30)  # (connect, read) seconds


def _rag_retrieve(model_id: str, query: str, top_k: int = 3, top_n: int = 5) -> str | None:
    """Call the FastAPI RAG endpoint and format the result as a context block.

    Returns a formatted string on success, or *None* on any failure (so the
    caller can fall back to the dashboard context).
    """
    try:
        resp = requests.post(
            f"{R.API}/queries/rag-retrieve",
            json={
                "query": query,
                "model_id": model_id,
                "top_k_topics": top_k,
                "top_n_docs": top_n,
            },
            timeout=_RAG_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning("RAG retrieve returned %d: %s", resp.status_code, resp.text[:300])
            return None
        data = resp.json()
    except Exception:
        logger.exception("RAG retrieve call failed")
        return None

    topics: List[dict] = data.get("topics", [])
    if not topics:
        return None

    lines: list[str] = ["## Retrieved Topics (most relevant to your question)\n"]
    for t in topics:
        lines.append(f"### Topic: \"{t['label']}\" (relevance: {t['similarity']:.2f})")
        if t.get("keywords"):
            lines.append(f"Keywords: {t['keywords']}")
        if t.get("summary"):
            lines.append(f"Summary: {t['summary']}")
        docs = t.get("documents", [])
        if docs:
            lines.append("\nTop documents:")
            for d in docs:
                text_preview = d["text"][:500] if d["text"] else "(no text)"
                lines.append(f"- [{d['doc_id']}] {text_preview}")
        lines.append("")

    return "\n".join(lines)


class ChatOrchestrationResult(TypedDict, total=False):
    response_text: str | None
    err: str | None
    chat_model: str | None
    provider: str | None
    host: str | None


def run_chat_assistant(
    *,
    user_id: str | None,
    model_id: str,
    message: str,
    context: dict[str, Any] | None,
    llm_settings: dict[str, Any] | None,
    load_user_config: Callable[[str], dict[str, Any]] | None = None,
) -> ChatOrchestrationResult:
    """Run the UI assistant for a single user question.

    This is intentionally UI-independent of Django `request` objects. Views can handle:
    - auth
    - persistence of chat history
    - HTTP/JSON concerns
    """
    context = context or {}
    llm_settings = llm_settings or {}

    # Try query-driven RAG retrieval first; fall back to dashboard context
    rag_context_str = _rag_retrieve(model_id, message)

    llm_context_str = dashboard_context_for_llm(
        context,
        max_themes=30,
        max_doc_snippets=5,
        include_diagnostics=True,
        include_model_info=True,
    )
    if rag_context_str:
        logger.debug("RAG context length: %d chars", len(rag_context_str))
    elif llm_context_str:
        logger.debug("Dashboard context length: %d chars", len(llm_context_str))

    # RAG LLM: user overrides from chat UI take precedence over config
    provider, chat_model, host, _ollama_list, _gpt_list, _ = get_chat_llm_defaults()

    if llm_settings.get("provider") and llm_settings.get("model"):
        p = llm_settings.get("provider")
        if hasattr(p, "strip"):
            p = p.strip().lower()
        else:
            p = str(p).strip().lower()

        if p == "openai":
            p = "gpt"

        provider = p
        chat_model = llm_settings.get("model")
        if chat_model is not None and not isinstance(chat_model, str):
            chat_model = str(chat_model)

        if provider == "ollama" and llm_settings.get("host"):
            host = str(llm_settings.get("host")).strip().rstrip("/") or host

    if host is not None and not isinstance(host, str):
        host = str(host)

    # No language model configured -> let views return the existing UX message.
    if not chat_model:
        return {"response_text": None, "err": None, "chat_model": chat_model, "provider": provider, "host": host}

    # RAG-style: system = strict rules; user = retrieved context + question
    if rag_context_str:
        system_content = (
            "You are a retrieval-augmented assistant for topic model analysis. "
            "The user message contains topics and documents retrieved from a topic model "
            "that are most relevant to the user's question. Ground your answer in the "
            "retrieved documents — quote or cite them when possible. "
            "Do not invent facts. If the retrieved context does not contain enough "
            "information to answer, say so clearly."
        )
        context_block = rag_context_str
    else:
        system_content = (
            "You are a retrieval-augmented assistant for topic model analysis. You must answer ONLY "
            "using the retrieved context provided in the user message. Do not use external knowledge "
            "or invent themes, counts, or metrics. If the answer is not in the context, say so clearly. "
            "When you use numbers or theme names, they must come directly from the context.\n\n"
            "For comparison questions: use the Themes section and the Theme similarities section.\n\n"
            "For analysis and insight questions (e.g. 'give me an analysis', 'key insights', 'what stands out'): "
            "synthesize from the context. Use the 'Analysis cues' section (dominant themes, highest coherence, "
            "most focused themes, overall stats). Offer 2-4 clear insights: e.g. which themes dominate, which are "
            "most interpretable or focused, how themes relate or differ, and what the corpus seems to be about. "
            "Keep insights grounded in the data; cite theme names and metrics from the context."
        )
        context_block = llm_context_str if llm_context_str else "(No dashboard context loaded yet.)"

    user_content = (
        "[Retrieved context from the topic model]\n"
        "---\n"
        f"{context_block}\n"
        "---\n\n"
        f"Question: {message}"
    )

    # OpenAI key precedence:
    #   chat UI override > user saved (DB) > config.yaml llm.gpt.api_key > .env
    from_ui = (llm_settings.get("api_key") or "").strip()
    stored_key: str | None = None
    if not from_ui and user_id and load_user_config:
        try:
            user_config = load_user_config(user_id) or {}
            stored_llm = user_config.get("llm_config") or {}
            stored_key = (stored_llm.get("api_key") or "").strip() or None
        except Exception:
            stored_key = None

    from_config = (R._safe_dict(R.RAW_CONFIG.get("llm", {}))).get("gpt") or {}
    if isinstance(from_config, dict):
        config_key = (from_config.get("api_key") or "").strip()
    else:
        config_key = ""

    openai_key = from_ui or stored_key or config_key or get_openai_key_from_env()

    response_text, err = chat_llm(
        provider,
        chat_model,
        system_content,
        user_content,
        host=host,
        api_key=openai_key,
    )
    return {
        "response_text": response_text,
        "err": err,
        "chat_model": chat_model,
        "provider": provider,
        "host": host,
    }


def _resolve_chat_params(
    *,
    model_id: str,
    message: str,
    context: dict[str, Any] | None,
    llm_settings: dict[str, Any] | None,
    user_id: str | None,
    load_user_config: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Shared setup logic for both sync and streaming chat paths.

    Returns a dict with resolved parameters or ``None`` when no model is
    configured (caller should handle this).
    """
    context = context or {}
    llm_settings = llm_settings or {}

    rag_context_str = _rag_retrieve(model_id, message)

    llm_context_str = dashboard_context_for_llm(
        context, max_themes=30, max_doc_snippets=5,
        include_diagnostics=True, include_model_info=True,
    )

    provider, chat_model, host, _ol, _gl, _ = get_chat_llm_defaults()

    if llm_settings.get("provider") and llm_settings.get("model"):
        p = llm_settings["provider"]
        p = p.strip().lower() if hasattr(p, "strip") else str(p).strip().lower()
        if p == "openai":
            p = "gpt"
        provider = p
        chat_model = llm_settings.get("model")
        if chat_model is not None and not isinstance(chat_model, str):
            chat_model = str(chat_model)
        if provider == "ollama" and llm_settings.get("host"):
            host = str(llm_settings["host"]).strip().rstrip("/") or host

    if host is not None and not isinstance(host, str):
        host = str(host)

    if not chat_model:
        return None

    if rag_context_str:
        system_content = (
            "You are a retrieval-augmented assistant for topic model analysis. "
            "The user message contains topics and documents retrieved from a topic model "
            "that are most relevant to the user's question. Ground your answer in the "
            "retrieved documents — quote or cite them when possible. "
            "Do not invent facts. If the retrieved context does not contain enough "
            "information to answer, say so clearly."
        )
        context_block = rag_context_str
    else:
        system_content = (
            "You are a retrieval-augmented assistant for topic model analysis. You must answer ONLY "
            "using the retrieved context provided in the user message. Do not use external knowledge "
            "or invent themes, counts, or metrics. If the answer is not in the context, say so clearly. "
            "When you use numbers or theme names, they must come directly from the context.\n\n"
            "For comparison questions: use the Themes section and the Theme similarities section.\n\n"
            "For analysis and insight questions: synthesize from the context. Use the 'Analysis cues' section."
        )
        context_block = llm_context_str if llm_context_str else "(No dashboard context loaded yet.)"

    user_content = (
        "[Retrieved context from the topic model]\n"
        "---\n"
        f"{context_block}\n"
        "---\n\n"
        f"Question: {message}"
    )

    from_ui = (llm_settings.get("api_key") or "").strip()
    stored_key: str | None = None
    if not from_ui and user_id and load_user_config:
        try:
            stored_key = ((load_user_config(user_id) or {}).get("llm_config") or {}).get("api_key") or None
            if stored_key:
                stored_key = stored_key.strip() or None
        except Exception:
            stored_key = None

    from_config = R._safe_dict(R.RAW_CONFIG.get("llm", {})).get("gpt") or {}
    config_key = (from_config.get("api_key") or "").strip() if isinstance(from_config, dict) else ""
    openai_key = from_ui or stored_key or config_key or get_openai_key_from_env()

    return {
        "provider": provider,
        "chat_model": chat_model,
        "system_content": system_content,
        "user_content": user_content,
        "host": host,
        "openai_key": openai_key,
    }


def run_chat_assistant_stream(
    *,
    user_id: str | None,
    model_id: str,
    message: str,
    context: dict[str, Any] | None,
    llm_settings: dict[str, Any] | None,
    load_user_config: Callable[[str], dict[str, Any]] | None = None,
):
    """Yield text tokens as the LLM generates them (SSE-friendly)."""
    params = _resolve_chat_params(
        model_id=model_id, message=message, context=context,
        llm_settings=llm_settings, user_id=user_id,
        load_user_config=load_user_config,
    )
    if params is None:
        yield "[ERROR]No language model configured."
        return

    yield from chat_llm_stream(
        params["provider"],
        params["chat_model"],
        params["system_content"],
        params["user_content"],
        host=params["host"],
        api_key=params["openai_key"],
    )

