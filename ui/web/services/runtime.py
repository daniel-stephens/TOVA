"""Config, caches, and helpers shared across Django views."""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from time import time

from ruamel.yaml import YAML

logger = logging.getLogger(__name__)

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 1000
_yaml.indent(mapping=2, sequence=4, offset=2)


def represent_list(self, data):
    if len(data) <= 10:
        return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
    return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)


_yaml.Representer.add_representer(list, represent_list)

_UI_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_STATIC_ROOT = (
    _REPO_ROOT / "static"
    if (_REPO_ROOT / "static").is_dir()
    else (_UI_ROOT / "static")
)
_drafts = os.getenv("DRAFTS_SAVE")
DRAFTS_SAVE = Path(_drafts) if _drafts else _REPO_ROOT / "data" / "drafts"
API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

_CACHE_MODELS: dict = {}
_CACHE_CORPORA: dict = {}
_CACHE_MODEL_INFO: dict = {}
_CACHE_THETAS: dict = {}
_CACHE_THETAS_BULK: dict = {}
CACHE_TTL = 600

CONFIG_PATH = _STATIC_ROOT / "config" / "config.yaml"
USER_CONFIGS_DIR = _STATIC_ROOT / "config" / "users"
USER_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

try:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        RAW_CONFIG = _yaml.load(f) or {}
except FileNotFoundError:
    logger.warning("config.yaml not found at %s", CONFIG_PATH)
    RAW_CONFIG = {}
except Exception as e:
    logger.exception("Error loading config.yaml: %s", e)
    RAW_CONFIG = {}


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(v) for v in obj)
    return obj


DEFAULT_CONFIG = _json_safe(RAW_CONFIG) if RAW_CONFIG else {}


def _deep_merge(base: dict, overrides: dict | None) -> dict:
    result = deepcopy(base) if isinstance(base, dict) else {}
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _verify_corpus_ownership(corpus_id: str, user_id: str) -> bool:
    import requests

    if not user_id:
        return False
    try:
        response = requests.get(f"{API}/data/corpora/{corpus_id}", timeout=5)
        if response.status_code == 200:
            corpus = response.json()
            owner = corpus.get("owner_id") or corpus.get("metadata", {}).get("owner_id")
            return str(owner or "") == str(user_id or "")
        if response.status_code == 404:
            return False
    except Exception:
        pass
    return False


def _verify_model_ownership(model_id: str, user_id: str) -> bool:
    import requests

    if not user_id:
        return False
    uid = str(user_id)
    try:
        response = requests.get(f"{API}/data/models/{model_id}", timeout=5)
        if response.status_code == 200:
            model = response.json()
            meta = model.get("metadata") or {}
            owner = model.get("owner_id") or meta.get("owner_id")
            if str(owner or "") == uid:
                return True
            # Match trained-models list: model may lack owner_id while corpus belongs to the user
            corpus_id = model.get("corpus_id") or meta.get("corpus_id")
            if corpus_id:
                return _verify_corpus_ownership(str(corpus_id), uid)
    except Exception:
        pass
    return False


def _verify_dataset_ownership(dataset_id: str, user_id: str) -> bool:
    import requests

    if not user_id:
        return False
    try:
        response = requests.get(f"{API}/data/datasets/{dataset_id}", timeout=5)
        if response.status_code == 200:
            dataset = response.json()
            owner = dataset.get("owner_id") or (dataset.get("metadata") or {}).get("owner_id")
            return str(owner or "") == str(user_id or "")
    except Exception:
        pass
    return False


def _filter_list_by_owner(items: list, user_id: str) -> list:
    if not user_id or not items:
        return [] if not user_id else items
    out = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        owner = item.get("owner_id") or (item.get("metadata") or {}).get("owner_id")
        if owner == user_id:
            out.append(item)
    return out


def _safe_dict(val):
    return val if isinstance(val, dict) else {}


def _cache_get(cache: dict, key):
    entry = cache.get(key)
    if not entry:
        return None, "miss"
    if time() - entry["ts"] > CACHE_TTL:
        return None, "expired"
    return entry["data"], "hit"


def _cache_set(cache: dict, key, data):
    cache[key] = {"ts": time(), "data": data}


def _admin_emails_from_env():
    from web.admin_emails import admin_emails_from_env

    return admin_emails_from_env()


def build_llm_ui_config(config: dict | None = None):
    cfg = config or DEFAULT_CONFIG
    llm = cfg.get("llm", {}) or {}
    ui: dict[str, dict] = {}

    gpt_cfg = llm.get("gpt") or {}
    if gpt_cfg:
        models_raw = gpt_cfg.get("available_models") or []
        if isinstance(models_raw, (set, list, tuple)):
            models = sorted(models_raw)
        elif isinstance(models_raw, dict):
            models = sorted(models_raw.keys())
        else:
            models = []
        ui["gpt"] = {
            "label": "OpenAI (GPT)",
            "models": models,
            "show_api_key": True,
            "show_host": False,
            "default_host": "",
            "api_key_label": "OpenAI API key",
            "api_key_placeholder": "sk-...",
            "api_key_help": "Your key is only used to call OpenAI on your behalf.",
        }

    ollama_cfg = llm.get("ollama") or {}
    if ollama_cfg:
        models_raw = ollama_cfg.get("available_models") or []
        if isinstance(models_raw, (set, list, tuple)):
            models = sorted(models_raw)
        elif isinstance(models_raw, dict):
            models = sorted(models_raw.keys())
        else:
            models = []
        ui["ollama"] = {
            "label": "Ollama",
            "models": models,
            "show_api_key": False,
            "show_host": True,
            "default_host": ollama_cfg.get("host", ""),
            "host_help": "Base URL of your Ollama server.",
        }

    llama_cfg = llm.get("llama_cpp") or {}
    if llama_cfg:
        ui["llama_cpp"] = {
            "label": "llama.cpp",
            "models": [],
            "show_api_key": False,
            "show_host": True,
            "default_host": llama_cfg.get("host", ""),
            "host_help": "Base URL of your llama.cpp server.",
        }

    return ui
