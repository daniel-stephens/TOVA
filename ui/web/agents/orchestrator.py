"""Agent orchestration entrypoints for UI workflows.

Right now the orchestration is mostly "RAG chat -> LLM call", but keeping it behind
an explicit boundary makes it much easier to add tools/agents later.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypedDict

from dashboard_utils import dashboard_context_for_llm
from web.services import runtime as R
from web.services.llm_client import chat_llm, get_chat_llm_defaults, get_openai_key_from_env

logger = logging.getLogger(__name__)


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

    llm_context_str = dashboard_context_for_llm(
        context,
        max_themes=30,
        max_doc_snippets=5,
        include_diagnostics=True,
        include_model_info=True,
    )
    if llm_context_str:
        logger.debug("LLM context length: %d chars", len(llm_context_str))

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
        "[Retrieved context from the topic model dashboard]\n"
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

