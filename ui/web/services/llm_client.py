"""LLM client helpers for the UI.

This module isolates provider-specific request logic (Ollama/OpenAI) so views stay thin
and we can later plug in agent/tool orchestration without duplicating provider code.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover (python-dotenv should be installed for UI)

    def load_dotenv(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        return None

from web.services import runtime as R

logger = logging.getLogger(__name__)


def chat_llm(
    provider: str,
    model: str,
    system_content: str,
    user_content: str,
    *,
    host: str | None = None,
    api_key: str | None = None,
) -> tuple[str | None, str | None]:
    """Call configured LLM (Ollama or OpenAI).

    Returns:
        (response_text, None) on success,
        (None, error_message) on failure.
    """
    llm_cfg = R._safe_dict(R.RAW_CONFIG.get("llm"))

    provider = (provider or "").strip().lower()
    if provider == "ollama":
        ollama_cfg = R._safe_dict(llm_cfg.get("ollama"))
        base_url = host or ollama_cfg.get("host", "http://host.docker.internal:11434")
        base_url = str(base_url).strip().rstrip("/")
        try:
            r = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    "stream": False,
                },
                timeout=(5, 120),
            )
            r.raise_for_status()
            data = r.json()
            content = (data.get("message") or {}).get("content", "").strip()
            return (content, None) if content else (None, "Empty response from Ollama")
        except requests.RequestException as e:
            return None, str(e)
        except Exception as e:
            return None, str(e)

    if provider in {"gpt", "openai"}:
        key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not key:
            return None, (
                "OpenAI API key not set. Add your key in Assistant LLM settings "
                "(chat sidebar) or set OPENAI_API_KEY in the server .env file."
            )
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": 1024,
                },
                timeout=(5, 120),
            )
            r.raise_for_status()
            data = r.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            return (content, None) if content else (None, "Empty response from OpenAI")
        except requests.RequestException as e:
            return None, str(e)
        except Exception as e:
            return None, str(e)

    return None, f"Unsupported LLM provider: {provider}"


def _get_openai_key_from_env() -> str | None:
    """Read OPENAI_API_KEY from one .env file (project root) and env."""
    # This mirrors the current UI behavior in `ui/web/views.py` so we don't surprise users.
    # `ui/web/views.py` historically used `Path(__file__).parent.parent` (which resolves to `ui/`).
    # Since this file is deeper (`ui/web/services/`), we use `parents[2]` to land on `ui/`.
    project_root = Path(__file__).resolve().parents[2]  # ui/
    env_path = project_root / ".env"

    load_dotenv(dotenv_path=env_path, override=False)

    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    return key or None


# Startup: confirm OpenAI key visibility from .env/config (no key value logged)
_openai_key_at_startup = _get_openai_key_from_env()
logger.info(
    "OpenAI API key at startup: %s",
    "configured"
    if _openai_key_at_startup
    else "not set (set in .env or llm.gpt.api_key in config)",
)


def get_openai_key_from_env() -> str | None:
    """Public wrapper for other modules."""
    return _get_openai_key_from_env()


def get_chat_llm_defaults():
    """Return default provider, model, host from config.

    Used when the user has not set overrides.
    """
    tm_gen = R._safe_dict(R._safe_dict(R.RAW_CONFIG.get("topic_modeling")).get("general"))
    llm_cfg = R._safe_dict(R.RAW_CONFIG.get("llm"))

    provider = tm_gen.get("llm_provider") or "ollama"
    if hasattr(provider, "strip"):
        provider = provider.strip().lower()
    else:
        provider = str(provider).strip().lower()

    if provider == "openai":
        provider = "gpt"

    _ollama = R._safe_dict(llm_cfg.get("ollama")).get("available_models")
    _gpt = R._safe_dict(llm_cfg.get("gpt")).get("available_models")
    ollama_list = list(_ollama) if _ollama else []
    gpt_list = list(_gpt) if _gpt else []

    chat_model = (
        tm_gen.get("llm_model_type")
        or (ollama_list[0] if ollama_list else None)
        or (gpt_list[0] if gpt_list else None)
    )
    if chat_model is not None and not isinstance(chat_model, str):
        chat_model = str(chat_model)

    host = tm_gen.get("llm_server") or R._safe_dict(llm_cfg.get("ollama")).get("host")
    if host is not None and not isinstance(host, str):
        host = str(host)

    return (
        provider,
        chat_model,
        host,
        ollama_list,
        gpt_list,
        R._safe_dict(llm_cfg.get("ollama")).get("host"),
    )

