"""Chat endpoints for the UI.

These endpoints delegate the assistant logic to `ui/web/agents/orchestrator.py`
and provider-specific code to `ui/web/services/llm_client.py`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from django.contrib.auth.decorators import login_required
import json

from django.http import JsonResponse, HttpRequest, StreamingHttpResponse

from web import views as views_monolith
from web.agents.orchestrator import run_chat_assistant, run_chat_assistant_stream
from web.models import ChatMessage
from web.services import runtime as R
from web.services.llm_client import get_openai_key_from_env

logger = logging.getLogger(__name__)


@login_required
def chat_openai_key_status(request: HttpRequest):
    """Diagnostic: report where OPENAI_API_KEY was found (or not). No key value is returned."""
    # Match the existing behavior in `ui/web/views.py` (best-effort: compute `ui/` as the root).
    project_root = Path(__file__).resolve().parents[2]  # ui/
    env_project = project_root / ".env"
    env_cwd = Path.cwd() / ".env"
    tried = [
        {"path": str(env_project), "exists": env_project.exists()},
        {"path": str(env_cwd), "exists": env_cwd.exists()},
    ]

    ui_root = project_root
    tried.append({"path": str(ui_root / ".env"), "exists": (ui_root / ".env").exists()})
    tried.append({"path": str(ui_root.parent / ".env"), "exists": (ui_root.parent / ".env").exists()})

    key = get_openai_key_from_env()
    gpt_cfg = R._safe_dict(R.RAW_CONFIG.get("llm", {})).get("gpt") or {}
    config_has_key = bool((gpt_cfg.get("api_key") or "").strip()) if isinstance(gpt_cfg, dict) else False
    path_api_key = (gpt_cfg.get("path_api_key") or ".env") if isinstance(gpt_cfg, dict) else ".env"

    return JsonResponse(
        {
            "key_configured": bool(key) or config_has_key,
            "key_from_config": config_has_key,
            "env_has_key": bool((os.environ.get("OPENAI_API_KEY") or "").strip()),
            "path_api_key_from_config": path_api_key,
            "note": "Labeler (Prompter) uses load_dotenv(path_api_key) then os.getenv('OPENAI_API_KEY'); UI uses same.",
            "project_root_from_file": str(project_root),
            "cwd": str(Path.cwd()),
            "server_root_path": str(ui_root),
            "tried_env_paths": tried,
        },
        status=200,
    )


@login_required
def chat_llm_options(request: HttpRequest):
    """Return available RAG/chat LLM options for the chat UI. Does not expose secrets."""
    llm_cfg = R._safe_dict(R.RAW_CONFIG.get("llm"))
    ollama_cfg = R._safe_dict(llm_cfg.get("ollama"))
    gpt_cfg = R._safe_dict(llm_cfg.get("gpt"))
    tm_gen = R._safe_dict(R._safe_dict(R.RAW_CONFIG.get("topic_modeling")).get("general"))
    default_host = tm_gen.get("llm_server") or ollama_cfg.get("host", "http://host.docker.internal:11434")
    if default_host is not None and not isinstance(default_host, str):
        default_host = str(default_host)

    return JsonResponse(
        {
            "ollama": {
                "models": list(ollama_cfg.get("available_models") or []),
                "default_host": default_host,
            },
            "gpt": {
                "models": list(gpt_cfg.get("available_models") or []),
            },
        },
        status=200,
    )


@login_required
def get_chat_messages(request: HttpRequest):
    """Return saved chat messages for the current user and `model_id` (query param)."""
    model_id = (request.GET.get("model_id") or "").strip()
    if not model_id:
        return JsonResponse({"error": "model_id is required"}, status=400)

    user_id = views_monolith._request_user_id(request)
    if not user_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)

    try:
        messages = (
            ChatMessage.objects.filter(user_id=user_id, model_id=model_id).order_by("created_at")
        )
        return JsonResponse(
            {
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "created_at": m.created_at.isoformat() + "Z" if m.created_at else None,
                    }
                    for m in messages
                ]
            },
            status=200,
        )
    except Exception as e:
        logger.exception("Failed to load chat messages: %s", e)
        return JsonResponse({"error": "Failed to load messages"}, status=500)


@login_required
def chat(request: HttpRequest):
    """Chat interface endpoint for topic modeling assistance (agent-driven)."""
    try:
        payload = views_monolith._get_json(request) or {}
        message = payload.get("message", "").strip()
        model_id = payload.get("model_id", "")
        context: dict[str, Any] = payload.get("context", {}) or {}
        llm_settings: dict[str, Any] = R._safe_dict(payload.get("llm_settings"))

        if not message:
            return JsonResponse({"error": "Message is required"}, status=400)

        logger.info("Chat request: model_id=%s, message_length=%d", model_id, len(message))

        user_id = views_monolith._request_user_id(request)

        result = run_chat_assistant(
            user_id=user_id,
            model_id=model_id,
            message=message,
            context=context,
            llm_settings=llm_settings,
            load_user_config=views_monolith._load_user_config if user_id else None,
        )

        response_text = result.get("response_text")
        err = result.get("err")
        chat_model = result.get("chat_model")

        if response_text:
            # Preserve existing behavior: persist user+assistant turns for this model_id.
            if user_id and model_id:
                try:
                    ChatMessage(user_id=user_id, model_id=model_id.strip(), role="user", content=message).save()
                    ChatMessage(
                        user_id=user_id,
                        model_id=model_id.strip(),
                        role="assistant",
                        content=response_text,
                    ).save()
                except Exception as save_err:
                    logger.warning("Failed to save chat messages: %s", save_err)

            return JsonResponse({"response": response_text}, status=200)

        # No rule-based fallback: LLM only. If we get here, model is missing or the call failed.
        if not chat_model:
            return JsonResponse(
                {
                    "response": "No language model is configured for the assistant. Open the chat sidebar, expand \"Assistant LLM\", and choose a provider and model (and set the Ollama host if using Ollama). You can also set defaults in config under topic_modeling.general."
                },
                status=200,
            )

        return JsonResponse(
            {
                "response": f"The assistant could not get a response from the model ({err or 'unknown error'}). Check that the LLM service is running and reachable, then try again."
            },
            status=200,
        )
    except Exception as e:
        logger.exception("Chat error: %s", e)
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def chat_stream(request: HttpRequest):
    """SSE streaming chat endpoint -- tokens are sent as `data:` events."""
    try:
        payload = views_monolith._get_json(request) or {}
        message = payload.get("message", "").strip()
        model_id = payload.get("model_id", "")
        context: dict[str, Any] = payload.get("context", {}) or {}
        llm_settings: dict[str, Any] = R._safe_dict(payload.get("llm_settings"))

        if not message:
            return JsonResponse({"error": "Message is required"}, status=400)

        user_id = views_monolith._request_user_id(request)

        def event_stream():
            collected: list[str] = []
            for token in run_chat_assistant_stream(
                user_id=user_id,
                model_id=model_id,
                message=message,
                context=context,
                llm_settings=llm_settings,
                load_user_config=views_monolith._load_user_config if user_id else None,
            ):
                collected.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

            full_text = "".join(collected)
            if user_id and model_id and full_text and not full_text.startswith("[ERROR]"):
                try:
                    ChatMessage(user_id=user_id, model_id=model_id.strip(), role="user", content=message).save()
                    ChatMessage(user_id=user_id, model_id=model_id.strip(), role="assistant", content=full_text).save()
                except Exception as save_err:
                    logger.warning("Failed to save chat messages: %s", save_err)

            yield "data: [DONE]\n\n"

        resp = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        return resp
    except Exception as e:
        logger.exception("Chat stream error: %s", e)
        return JsonResponse({"error": str(e)}, status=500)

