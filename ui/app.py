import json
import logging
import os
from pathlib import Path
from time import time, sleep, monotonic
from datetime import timedelta
from functools import wraps
from uuid import uuid4
from copy import deepcopy

import requests
import yaml
from authlib.integrations.flask_client import OAuth
from dashboard_utils import pydantic_to_dict, to_dashboard_bundle
from flask import (
    Flask,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
    session,
    url_for,
    flash,
    redirect,
    g,
)
from sqlalchemy import text
from models import db, User, UserConfig

# ------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------
server = Flask(__name__)
server.logger.setLevel(logging.INFO)

# IMPORTANT: change this in real deployments
server.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me")
server.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

# Database config
server.config.from_object("config")
db.init_app(server)

RUN_DB_INIT_ON_START = os.getenv("RUN_DB_INIT_ON_START", "1") == "1"


def _run_db_init_once():
    """
    Initialize tables once, guarded by an advisory lock to avoid
    gunicorn worker races that can cause duplicate type errors.
    """
    if not RUN_DB_INIT_ON_START:
        return
    with server.app_context():
        with db.engine.begin() as conn:
            conn.execute(text("SELECT pg_advisory_lock(420420)"))
            try:
                db.create_all()
            finally:
                conn.execute(text("SELECT pg_advisory_unlock(420420)"))


_run_db_init_once()

DRAFTS_SAVE = Path(os.getenv("DRAFTS_SAVE", "/data/drafts"))  # TODO: remove if unused
API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

_CACHE_MODELS: dict = {}
_CACHE_CORPORA: dict = {}
_CACHE_MODEL_INFO: dict = {}      # key: model_id
_CACHE_THETAS: dict = {}          # key: (model_id, doc_id)
_CACHE_THETAS_BULK: dict = {}     # key: (model_id, tuple(sorted_doc_ids))
CACHE_TTL = 600

# ------------------------------------------------------------------------------
# LLM config from YAML
# ------------------------------------------------------------------------------
BASE_DIR = Path(server.root_path)
CONFIG_PATH = BASE_DIR / "static" / "config" / "config.yaml"
USER_CONFIGS_DIR = BASE_DIR / "static" / "config" / "users"
USER_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
JOB_STATUS_TIMEOUT_SECONDS = 10
JOB_STATUS_POLLING_INTERVAL_SECONDS = 1
try:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        RAW_CONFIG = yaml.safe_load(f) or {}
except FileNotFoundError:
    server.logger.warning("config.yaml not found at %s; RAW_CONFIG = {}", CONFIG_PATH)
    RAW_CONFIG = {}
except Exception as e:
    server.logger.exception("Error loading config.yaml: %s", e)
    RAW_CONFIG = {}


def _json_safe(obj):
    """
    Convert YAML/py objects into JSON-serializable structures (sets -> list, tuples -> list).
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(v) for v in obj)
    return obj


DEFAULT_CONFIG = _json_safe(RAW_CONFIG) if RAW_CONFIG else {}


def _deep_merge(base: dict, overrides: dict | None) -> dict:
    """
    Recursively merge two dictionaries without mutating the originals.
    """
    result = deepcopy(base) if isinstance(base, dict) else {}
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_user_config(user_id: str) -> dict | None:
    """Return raw stored user config (if any)."""
    if not user_id:
        return None
    entry = UserConfig.query.filter_by(user_id=user_id).first()
    return entry.config if entry else None


def _get_user_config_file_path(user_id: str) -> Path | None:
    """Get the file path for a user's config file."""
    if not user_id:
        return None
    # Sanitize user_id for filename (remove any path separators or dangerous chars)
    safe_user_id = user_id.replace("/", "_").replace("\\", "_").replace("..", "")
    return USER_CONFIGS_DIR / f"{safe_user_id}.yaml"


def _ensure_user_config_file(user_id: str) -> str | None:
    """
    Ensure a user's config file exists on disk with their effective config.
    Returns the relative path to the config file (e.g., "static/config/users/{user_id}.yaml")
    or None if user_id is missing.
    """
    if not user_id:
        return None
    
    config_file_path = _get_user_config_file_path(user_id)
    if config_file_path is None:
        return None
    
    # Get the effective config (merged default + user overrides)
    effective_config = get_effective_user_config(user_id)
    
    # Write to file
    try:
        with config_file_path.open("w", encoding="utf-8") as f:
            yaml.dump(effective_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        server.logger.debug("Saved user config to file: %s", config_file_path)
    except Exception as e:
        server.logger.exception("Failed to write user config file %s: %s", config_file_path, e)
        # Fallback to default config path
        return "static/config/config.yaml"
    
    # Return relative path from project root
    relative_path = config_file_path.relative_to(BASE_DIR)
    return str(relative_path).replace("\\", "/")  # Normalize path separators


def _save_user_config(user_id: str, config: dict):
    """Upsert user configuration in the database and static file."""
    if not user_id:
        return
    sanitized = _json_safe(config or {})
    entry = UserConfig.query.filter_by(user_id=user_id).first()
    if entry:
        entry.config = sanitized
    else:
        entry = UserConfig(user_id=user_id, config=sanitized)
        db.session.add(entry)
    db.session.commit()
    
    # Also save to static file
    _ensure_user_config_file(user_id)


def _reset_user_config(user_id: str):
    """Remove user-specific configuration (falls back to defaults) and update static file."""
    if not user_id:
        return
    entry = UserConfig.query.filter_by(user_id=user_id).first()
    if entry:
        db.session.delete(entry)
        db.session.commit()
    
    # Update the static config file (will now contain only defaults)
    _ensure_user_config_file(user_id)


def get_effective_user_config(user_id: str) -> dict:
    """
    Return the default config merged with the user's overrides (if any).
    Cached per-request on g to avoid duplicate queries.
    """
    if hasattr(g, "_effective_user_config"):
        return g._effective_user_config

    base = deepcopy(DEFAULT_CONFIG)
    if not user_id:
        g._effective_user_config = base
        return base

    overrides = _load_user_config(user_id) or {}
    merged = _deep_merge(base, overrides if isinstance(overrides, dict) else {})
    g._effective_user_config = merged
    return merged


def build_llm_ui_config(config: dict | None = None):
    """
    Build a UI-friendly LLM provider config based on a configuration dict.

    Returns a dict like:
    {
      "gpt": {
        "label": "OpenAI (GPT)",
        "models": [...],
        "show_api_key": true,
        "show_host": false,
        "default_host": "",
        "api_key_label": "...",
        "api_key_placeholder": "...",
        "api_key_help": "..."
      },
      "ollama": {
        "label": "Ollama",
        "models": [...],
        "show_api_key": false,
        "show_host": true,
        "default_host": "http://...",
        "host_help": "Base URL ..."
      },
      "llama_cpp": { ... }
    }
    """
    cfg = config or DEFAULT_CONFIG
    llm = cfg.get("llm", {}) or {}
    ui: dict[str, dict] = {}

    # GPT / OpenAI
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

    # Ollama
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

    # llama.cpp
    llama_cfg = llm.get("llama_cpp") or {}
    if llama_cfg:
        ui["llama_cpp"] = {
            "label": "llama.cpp",
            "models": [],  # typically one model configured server-side
            "show_api_key": False,
            "show_host": True,
            "default_host": llama_cfg.get("host", ""),
            "host_help": "Base URL of your llama.cpp server.",
        }

    return ui


# ------------------------------------------------------------------------------
# Okta / OAuth setup (optional)
# ------------------------------------------------------------------------------
server.config["OKTA_CLIENT_ID"] = os.getenv("OKTA_CLIENT_ID")
server.config["OKTA_CLIENT_SECRET"] = os.getenv("OKTA_CLIENT_SECRET")
server.config["OKTA_ISSUER"] = os.getenv("OKTA_ISSUER")  # e.g. https://dev-xxx.okta.com/oauth2/default

oauth = OAuth(server)

okta = None
if server.config["OKTA_CLIENT_ID"] and server.config["OKTA_ISSUER"]:
    try:
        okta = oauth.register(
            name="okta",
            client_id=server.config["OKTA_CLIENT_ID"],
            client_secret=server.config["OKTA_CLIENT_SECRET"],
            server_metadata_url=f'{server.config["OKTA_ISSUER"]}/.well-known/openid-configuration',
            client_kwargs={"scope": "openid profile email"},
        )
    except Exception as e:
        server.logger.exception("Failed to register Okta OAuth client: %s", e)
        okta = None
else:
    server.logger.info("Okta not configured (missing OKTA_CLIENT_ID or OKTA_ISSUER)")


# ------------------------------------------------------------------------------
# Simple cache utilities
# ------------------------------------------------------------------------------
def _cache_get(cache: dict, key):
    entry = cache.get(key)
    if not entry:
        return None, "miss"
    if time() - entry["ts"] > CACHE_TTL:
        return None, "expired"
    return entry["data"], "hit"


def _cache_set(cache: dict, key, data):
    cache[key] = {"ts": time(), "data": data}


# ------------------------------------------------------------------------------
# Auth helpers
# ------------------------------------------------------------------------------
def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        # check the same key you actually set in login/signup/Okta
        if "user_id" not in session:
            next_url = request.path  # or request.full_path
            return redirect(url_for("login", next=next_url))
        return view_func(*args, **kwargs)

    return wrapped_view


@server.before_request
def load_logged_in_user():
    """Attach the current user to g before every request, based on session."""
    user_id = session.get("user_id")
    if not user_id:
        g.user = None
        return

    user = User.query.filter_by(id=user_id).first()
    if user:
        g.user = {
            "id": user.id,
            "email": user.email,
            "name": user.name or user.email,
        }
        session["user_email"] = user.email
        session["user_name"] = user.name or user.email
    else:
        g.user = None
        session.clear()


@server.context_processor
def inject_user():
    """Make `user` available in all templates (e.g., base.html)."""
    return {"user": getattr(g, "user", None)}


# ------------------------------------------------------------------------------
# Auth routes (local username/password + optional Okta)
# ------------------------------------------------------------------------------
@server.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        password_confirm = request.form.get("password_confirm", "")

        from werkzeug.security import generate_password_hash

        # Basic validation
        if not name or not email or not password:
            flash("Please fill in all required fields.", "danger")
            return redirect(url_for("signup"))

        if password != password_confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("signup"))

        existing = User.query.filter_by(email=email).first()
        if existing:
            flash("An account with that email already exists. Please sign in.", "warning")
            return redirect(url_for("login"))

        # Create user
        user = User(
            id=str(uuid4()),
            name=name or email,
            email=email,
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        # Log them in
        session.clear()
        session["user_id"] = user.id
        session["user_email"] = user.email
        session["user_name"] = user.name or user.email
        session.permanent = True

        flash("Account created! Welcome to TOVA.", "success")
        return redirect(url_for("home"))

    return render_template("signup.html")


@server.route("/login", methods=["GET", "POST"])
def login():
    from werkzeug.security import check_password_hash

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if not user or not user.password_hash or not check_password_hash(
            user.password_hash, password
        ):
            flash("Invalid email or password.", "danger")
            return redirect(url_for("login"))

        # Credentials OK
        session.clear()
        session["user_id"] = user.id
        session["user_email"] = user.email
        session["user_name"] = user.name or user.email
        session.permanent = True

        flash("Signed in successfully.", "success")

        next_url = request.args.get("next")
        return redirect(next_url or url_for("home"))

    return render_template("login.html")


@server.route("/login/okta")
def login_okta():
    # Optionally remember where the user was trying to go
    next_url = request.args.get("next")
    if next_url:
        session["next_url"] = next_url

    if okta is None:
        flash("Okta sign-in is not configured.", "warning")
        return redirect(url_for("login"))

    redirect_uri = url_for("auth_okta_callback", _external=True)
    return okta.authorize_redirect(redirect_uri)


@server.route("/auth/okta/callback")
def auth_okta_callback():
    if okta is None:
        flash("Okta sign-in is not configured.", "warning")
        return redirect(url_for("login"))

    # Exchange authorization code for tokens
    try:
        token = okta.authorize_access_token()
        userinfo = okta.parse_id_token(token)
    except Exception as e:
        current_app.logger.exception("Okta auth callback failed: %s", e)
        flash("Failed to sign in with Okta.", "danger")
        return redirect(url_for("login"))

    email = (userinfo.get("email") or "").lower()
    okta_sub = userinfo.get("sub")
    name = userinfo.get("name") or email

    if not email:
        flash("Okta did not provide an email address.", "danger")
        return redirect(url_for("login"))

    # Find or create user in the database
    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(
            id=str(uuid4()),
            name=name or email,
            email=email,
            password_hash=None,  # They may not have a local password
            auth_source="okta",
        )
        db.session.add(user)
    else:
        # Link the Okta account to an existing user
        user.auth_source = "okta"

    db.session.commit()

    # Log them in
    session.clear()
    session["user_id"] = user.id
    session["user_email"] = user.email
    session["user_name"] = user.name or name or email
    session.permanent = True

    flash("Signed in with Okta.", "success")

    next_url = session.pop("next_url", None)
    return redirect(next_url or url_for("home"))


@server.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("login"))


# ------------------------------------------------------------------------------
# Basic pages & health
# ------------------------------------------------------------------------------
@server.route("/")
@login_required
def home():
    """Homepage after login."""
    return render_template("homepage.html")


@server.route("/check-backend")
def check_backend():
    try:
        response = requests.get(f"{API}/health", timeout=5)
        if response.status_code == 200:
            return jsonify(status="success", message="Backend is healthy"), 200
        else:
            return jsonify(status="error", message="Backend is not healthy"), 500
    except Exception as e:
        return jsonify(status="error", message=str(e)), 500


@server.route("/terms")
def terms():
    return render_template("terms.html")


@server.route("/privacy")
def privacy():
    return render_template("privacy.html")


# ------------------------------------------------------------------------------
# LLM config routes
# ------------------------------------------------------------------------------
@server.route("/llm/ui-config", methods=["GET"])
@login_required
def llm_ui_config():
    cfg = get_effective_user_config(session.get("user_id"))
    return jsonify(build_llm_ui_config(cfg))


@server.route("/get-llm-config", methods=["GET"])
@login_required
def get_llm_config():
    cfg = session.get("llm_config") or {}
    # DO NOT send api_key back to browser in a real app
    return jsonify(cfg)


@server.route("/save-llm-config", methods=["POST"])
@login_required
def save_llm_config():
    data = request.get_json(force=True) or {}

    provider = data.get("provider")
    model = data.get("model") or None
    host = data.get("host") or None
    api_key = data.get("api_key")  # Sensitive: don't put this in session cookie

    if not provider:
        return jsonify({"success": False, "message": "Provider is required."}), 400

    # Store only non-sensitive info in the session
    cfg = {
        "provider": provider,
        "model": model,
        "host": host,
    }
    session["llm_config"] = cfg

    # TODO: Persist api_key securely on server side (DB / KMS / secret store) if needed

    return jsonify({"success": True})


# ------------------------------------------------------------------------------
# User configuration (persisted per user)
# ------------------------------------------------------------------------------
@server.route("/api/user-config", methods=["GET"])
@login_required
def api_get_user_config():
    cfg = get_effective_user_config(session.get("user_id"))
    return jsonify(cfg), 200


@server.route("/api/user-config", methods=["POST"])
@login_required
def api_update_user_config():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"success": False, "message": "Config must be a JSON object."}), 400

    user_id = session.get("user_id")
    merged = _deep_merge(DEFAULT_CONFIG, _json_safe(payload))
    _save_user_config(user_id, merged)
    config_path = _ensure_user_config_file(user_id)
    return jsonify({
        "success": True, 
        "config": merged, 
        "message": "Configuration saved.",
        "config_path": config_path  # Return the file path where config was saved
    }), 200


@server.route("/api/user-config/reset", methods=["POST"])
@login_required
def api_reset_user_config():
    user_id = session.get("user_id")
    _reset_user_config(user_id)
    fresh = get_effective_user_config(user_id)
    config_path = _ensure_user_config_file(user_id)
    return jsonify({
        "success": True, 
        "config": fresh, 
        "message": "Configuration reset to defaults.",
        "config_path": config_path  # Return the file path where config was saved
    }), 200


# ------------------------------------------------------------------------------
# Data / Corpus-related routes
# ------------------------------------------------------------------------------
@server.route("/load-data-page/")
@login_required
def load_data_page():
    """Page for uploading/creating a dataset."""
    return render_template("loadData.html")


@server.route("/load-corpus-page/")
@login_required
def load_corpus_page():
    return render_template("manageCorpora.html")


@server.route("/data/create/corpus/", methods=["POST"])
@login_required
def create_corpus():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "No JSON payload received"}), 400

    # List of dictionaries with dataset IDs
    datasets = payload.get("datasets", [])
    datasets_lst = []

    for el in datasets:
        try:
            upstream = requests.get(
                f"{API}/data/datasets/{el['id']}",
                params={"owner_id": session.get("user_id")},
                timeout=(3.05, 30),
            )
            if not upstream.ok:
                return Response(
                    upstream.content,
                    status=upstream.status_code,
                    mimetype=upstream.headers.get("Content-Type", "application/json"),
                )
            datasets_lst.append(upstream.json())
        except requests.Timeout:
            return jsonify({"error": "Upstream timeout"}), 504
        except requests.RequestException as e:
            return jsonify({"error": f"Upstream connection error: {e}"}), 502

    owner_id = payload.get("owner_id") or session.get("user_id")

    corpus_payload = {
        "name": payload.get("corpus_name", ""),
        "description": f"Corpus from datasets {', '.join([d['id'] for d in datasets])}",
        "owner_id": owner_id,
        "datasets": datasets_lst,
    }

    current_app.logger.info("Payload sent to /data/corpora: %s", corpus_payload)

    try:
        upstream = requests.post(
            f"{API}/data/corpora",
            json=corpus_payload,
            timeout=(3.05, 30),
        )
        if not upstream.ok:
            return Response(
                upstream.content,
                status=upstream.status_code,
                mimetype=upstream.headers.get("Content-Type", "application/json"),
            )

        return Response(
            upstream.content,
            status=upstream.status_code,
            mimetype=upstream.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream connection error: {e}"}), 502


@server.route("/data/corpus/add_model", methods=["POST"])
@login_required
def add_model_to_corpus():
    payload = request.get_json(silent=True) or {}
    model_id = payload.get("model_id")
    corpus_id = payload.get("corpus_id")
    if not model_id:
        return jsonify({"error": "Missing model_id"}), 400
    if not corpus_id:
        return jsonify({"error": "Missing corpus_id"}), 400

    upstream_url = f"{API}/data/corpora/{corpus_id}/add_model"
    try:
        up = requests.post(
            upstream_url,
            params={"model_id": model_id},  # Send model_id as query parameter
            timeout=(3.05, 30),
        )
        up.raise_for_status()
        return Response(
            up.content,
            status=up.status_code,
            mimetype=up.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream connection error: {e}"}), 502


@server.route("/delete-corpus/", methods=["POST"])
@login_required
def delete_corpus():
    """
    Delete a corpus by name.
    Body (JSON): { "corpus_name": "..." }
    """
    payload = request.get_json(silent=True) or {}
    corpus_name = payload.get("corpus_name")
    owner_id = payload.get("owner_id") or session.get("user_id")

    if not corpus_name:
        return jsonify({"error": "Missing corpus_name"}), 400

    # Find the corpus by name to get its ID
    try:
        r = requests.get(f"{API}/data/corpora", params={"owner_id": owner_id}, timeout=10)
        r.raise_for_status()
        corpora = r.json()
    except requests.RequestException as e:
        server.logger.exception("Failed to fetch corpora: %s", e)
        return jsonify({"error": "Failed to fetch corpora"}), 502

    corpus_id = None
    corpus = None
    for c in corpora:
        metadata = c.get("metadata", {})
        if metadata.get("name") == corpus_name:
            corpus_id = c.get("id")
            corpus = c
            break

    if not corpus_id:
        return jsonify({"error": f"Corpus '{corpus_name}' not found"}), 404

    models = corpus.get("models") or []

    # Delete each associated model first
    model_results = []
    for model_id in models:
        server.logger.info("Deleting model '%s' associated with corpus '%s'", model_id, corpus_id)
        try:
            mresp = requests.delete(f"{API}/data/models/{model_id}", timeout=(3.05, 30))
            if mresp.status_code == 204:
                server.logger.info("Model '%s' deleted", model_id)
                model_results.append({"model_id": model_id, "status": "deleted"})
            elif mresp.status_code == 404:
                server.logger.warning("Model '%s' not found during deletion", model_id)
                model_results.append({"model_id": model_id, "status": "not_found"})
            else:
                server.logger.error(
                    "Failed to delete model '%s': status=%s body=%s",
                    model_id,
                    mresp.status_code,
                    mresp.text[:500],
                )
                model_results.append(
                    {
                        "model_id": model_id,
                        "status": "error",
                        "upstream_status": mresp.status_code,
                        "upstream_body": mresp.text[:500],
                    }
                )
        except requests.Timeout:
            server.logger.exception("Timeout deleting model '%s'", model_id)
            model_results.append({"model_id": model_id, "status": "timeout"})
        except requests.RequestException as e:
            server.logger.exception("Connection error deleting model '%s': %s", model_id, e)
            model_results.append(
                {"model_id": model_id, "status": "connection_error", "detail": str(e)}
            )

    # Now delete the corpus itself
    try:
        del_resp = requests.delete(f"{API}/data/corpora/{corpus_id}", timeout=(3.05, 30))
        if del_resp.status_code == 204:
            return jsonify(
                {
                    "message": f"Corpus '{corpus_id}' deleted successfully",
                    "models": model_results,
                }
            ), 200
        elif del_resp.status_code == 404:
            return jsonify(
                {
                    "error": f"Corpus '{corpus_id}' not found during deletion",
                    "models": model_results,
                }
            ), 404
        else:
            return Response(
                json.dumps(
                    {
                        "error": "Upstream error deleting corpus",
                        "upstream_status": del_resp.status_code,
                        "upstream_body": del_resp.text[:1000],
                        "models": model_results,
                    }
                ),
                status=del_resp.status_code,
                mimetype="application/json",
            )
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout deleting corpus", "models": model_results}), 504
    except requests.RequestException as e:
        return jsonify(
            {"error": f"Upstream connection error deleting corpus: {e}", "models": model_results}
        ), 502


@server.route("/data/create/dataset/", methods=["POST"])
@login_required
def create_dataset():
    payload = request.get_json(silent=True) or {}
    metadata = payload.get("metadata", {})
    data = payload.get("data", {})
    documents = data.get("documents", [])
    owner_id = payload.get("owner_id") or session.get("user_id")

    dataset_payload = {
        "name": metadata.get("name", ""),
        "description": metadata.get("description", ""),
        "owner_id": owner_id,
        "documents": documents,
        "metadata": metadata,
    }

    try:
        upstream = requests.post(
            f"{API}/data/datasets",
            json=dataset_payload,
            timeout=(3.05, 30),
        )
        if not upstream.ok:
            return Response(
                upstream.content,
                status=upstream.status_code,
                mimetype=upstream.headers.get("Content-Type", "application/json"),
            )
        return Response(
            upstream.content,
            status=upstream.status_code,
            mimetype=upstream.headers.get("Content-Type", "application/json"),
        )

    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream connection error: {e}"}), 502


# ------------------------------------------------------------------------------
# Training routes
# ------------------------------------------------------------------------------
@server.get("/training/")
@login_required
def training_page_get():
    return render_template("training.html")


@server.route("/get-training-session", methods=["GET"])
@login_required
def get_training_session():
    corpus_id = session.get("corpus_id")
    tmReq = session.get("tmReq")

    if not corpus_id or not tmReq:
        return jsonify({"error": "Training session data not found"}), 404

    return jsonify({"corpus_id": corpus_id, "tmReq": tmReq}), 200


@server.route("/train/corpus/<corpus_id>/tfidf/", methods=["GET"])
@login_required
def proxy_corpus_tfidf(corpus_id):
    try:
        up = requests.get(f"{API}/train/corpus/{corpus_id}/tfidf/", timeout=(3.05, 30))
        return Response(
            up.content,
            status=up.status_code,
            mimetype=up.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream connection error: {e}"}), 502


@server.route("/corpus/<corpus_id>/tfidf/", methods=["GET"])
@login_required
def get_tfidf_data(corpus_id):
    try:
        upstream = requests.get(
            f"{API}/train/corpus/{corpus_id}/tfidf/",
            params=request.args,
            timeout=(3.05, 60),
        )
        upstream.raise_for_status()
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream connection error: {e}"}), 502

    try:
        return jsonify(upstream.json()), upstream.status_code
    except ValueError:
        return Response(
            upstream.content,
            status=upstream.status_code,
            mimetype=upstream.headers.get("Content-Type", "application/json"),
        )


@server.route("/train/corpus/<corpus_id>/tfidf/", methods=["POST"])
@login_required
def train_tfidf_corpus(corpus_id):
    payload = request.get_json(silent=True) or {}
    try:
        n_clusters = int(payload.get("n_clusters") or 15)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid n_clusters"}), 400

    # fetch corpus from backend
    try:
        up = requests.get(
            f"{API}/data/corpora/{corpus_id}",
            params={"owner_id": session.get("user_id")},
            timeout=(3.05, 60),
        )
        up.raise_for_status()
        corpus = up.json()
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout fetching corpus"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream error fetching corpus: {e}"}), 502

    documents = [
        {"id": str(d.get("id")), "raw_text": str(d.get("text", ""))}
        for d in (corpus.get("documents") or [])
    ]
    if not documents:
        return jsonify({"error": "No documents found in corpus"}), 400

    upstream_payload = {"n_clusters": n_clusters, "documents": documents}
    try:
        upstream = requests.post(
            f"{API}/train/corpus/{corpus_id}/tfidf/",
            json=upstream_payload,
            timeout=(3.05, 120),
        )
        upstream.raise_for_status()
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout calling TF-IDF"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream connection error: {e}"}), 502

    try:
        return jsonify(upstream.json()), upstream.status_code
    except ValueError:
        return Response(
            upstream.content,
            status=upstream.status_code,
            mimetype=upstream.headers.get("Content-Type", "application/json"),
        )


@server.post("/training/start")
@login_required
def training_start():
    payload = request.get_json(silent=True) or {}
    corpus_id = payload.get("corpus_id")
    model = payload.get("model")
    model_name = payload.get("model_name")
    training_params = payload.get("training_params") or {}
    user_id = session.get("user_id")

    if not corpus_id or not model or not model_name:
        return jsonify({"error": "Missing corpus_id/model/model_name"}), 400

    # Merge training_params into user config for traditional models
    # This ensures the config file has the correct values when the base class reads from it
    if training_params and model in ["tomotopyLDA", "CTM"]:  # Traditional models
        current_config = get_effective_user_config(user_id)
        # Map training_params to config structure
        if "do_labeller" in training_params:
            current_config.setdefault("topic_modeling", {}).setdefault("traditional", {})["do_labeller"] = training_params["do_labeller"]
        if "do_summarizer" in training_params:
            current_config.setdefault("topic_modeling", {}).setdefault("traditional", {})["do_summarizer"] = training_params["do_summarizer"]
        if "llm_model_type" in training_params:
            current_config.setdefault("topic_modeling", {}).setdefault("traditional", {})["llm_model_type"] = training_params["llm_model_type"]
        if "labeller_prompt" in training_params:
            current_config.setdefault("topic_modeling", {}).setdefault("traditional", {})["labeller_model_path"] = training_params["labeller_prompt"]
        elif "labeller_model_path" in training_params:
            current_config.setdefault("topic_modeling", {}).setdefault("traditional", {})["labeller_model_path"] = training_params["labeller_model_path"]
        if "summarizer_prompt" in training_params:
            current_config.setdefault("topic_modeling", {}).setdefault("traditional", {})["summarizer_prompt"] = training_params["summarizer_prompt"]
        if "num_topics" in training_params:
            current_config.setdefault("topic_modeling", {}).setdefault("traditional", {})["num_topics"] = training_params["num_topics"]
        if "thetas_thr" in training_params:
            current_config.setdefault("topic_modeling", {}).setdefault("traditional", {})["thetas_thr"] = training_params["thetas_thr"]
        if "topn" in training_params:
            current_config.setdefault("topic_modeling", {}).setdefault("traditional", {})["topn"] = training_params["topn"]
        
        # Save the updated config
        _save_user_config(user_id, current_config)

    # Ensure user config file exists and get its path
    config_path = _ensure_user_config_file(user_id) or "static/config/config.yaml"

    # get corpus
    try:
        up = requests.get(
            f"{API}/data/corpora/{corpus_id}",
            params={"owner_id": user_id},
            timeout=(3.05, 60),
        )
        up.raise_for_status()
        corpus = up.json()
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout fetching corpus"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream error fetching corpus: {e}"}), 502

    docs = [
        {"id": str(d.get("id")), "text": str(d.get("text", ""))}
        for d in (corpus.get("documents") or [])
    ]
    if not docs:
        return jsonify({"error": "No documents found in corpus"}), 400

    tm_req = {
        "model": model,
        "corpus_id": corpus_id,
        "data": docs,
        "id_col": "id",
        "text_col": "text",
        "training_params": training_params,
        "config_path": config_path,  # Use user-specific config path
        "model_name": model_name,
        "owner_id": user_id,
    }

    try:
        tr = requests.post(f"{API}/train/json", json=tm_req, timeout=(3.05, 120))
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout starting training"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream error starting training: {e}"}), 502

    if tr.status_code >= 400:
        return Response(
            tr.content,
            status=tr.status_code,
            mimetype=tr.headers.get("Content-Type", "application/json"),
        )

    job_id = None
    model_id = None
    try:
        body = tr.json()
        job_id = (body or {}).get("job_id")
        model_id = (body or {}).get("model_id")
    except Exception:
        body = {}

    loc = tr.headers.get("Location")
    return jsonify(
        {
            "job_id": job_id,
            "status_url": loc or (f"/status/jobs/{job_id}" if job_id else None),
            "corpus_id": corpus_id,
            "model_id": model_id,
            "config_path": config_path,  # Return the config path used
        }
    ), 200


# ------------------------------------------------------------------------------
# Status routes
# ------------------------------------------------------------------------------
@server.route("/status/jobs/<job_id>", methods=["GET"])
@server.route("/status/", methods=["GET"])
def get_status(job_id=None):
    job_id = job_id or request.args.get("job_id") or request.headers.get("X-Job-Id")
    if not job_id:
        return jsonify({"error": "Missing job_id"}), 400

    headers = {}
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth

    upstream_url = f"{API}/status/jobs/{job_id}"
    try:
        up = requests.get(upstream_url, headers=headers, timeout=(3.05, 30))
        return Response(
            up.content,
            status=up.status_code,
            mimetype=up.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream connection error: {e}"}), 502


# ------------------------------------------------------------------------------
# Model-related routes
# ------------------------------------------------------------------------------
@server.route("/model")
@login_required
def loadModel():
    return render_template("loadModel.html")


@server.route("/getUniqueCorpusNames", methods=["GET"])
@login_required
def get_unique_corpus_names():
    """
    Return a deduped (case-insensitive), A→Z list of corpus names.
    Prefers the most recent 'created_at' when duplicates exist.
    """
    try:
        r = requests.get(
            f"{API}/data/corpora",
            params={"owner_id": session.get("user_id")},
            timeout=10,
        )
        r.raise_for_status()
        items = r.json()
    except requests.RequestException:
        server.logger.exception("Failed to fetch drafts from upstream")
        return jsonify({"error": "Upstream drafts service failed"}), 502

    pick = {}  # lower(name) -> {"name": original, "created_at": ts}
    for d in items or []:
        meta = (d or {}).get("metadata") or {}
        name = (meta.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        created = meta.get("created_at") or ""
        if key not in pick or created > pick[key]["created_at"]:
            pick[key] = {"name": name, "created_at": created}

    names = sorted((v["name"] for v in pick.values()), key=str.casefold)
    return jsonify(names), 200


@server.route("/getAllCorpora", methods=["GET"])
@login_required
def getAllCorpora():
    """
    Pulls all corpora (draft or permanent storage) from the upstream API.
    """
    try:
        r = requests.get(
            f"{API}/data/corpora",
            params={"owner_id": session.get("user_id")},
            timeout=10,
        )
        r.raise_for_status()
        items = r.json()
    except requests.RequestException:
        server.logger.exception("Failed to fetch drafts from upstream")
        return jsonify({"error": "Upstream drafts service failed"}), 502

    def _norm_corpus(c):
        location = c.get("location")
        is_draft = location != "database"
        return {
            "id": c.get("id"),
            "name": c.get("metadata", {}).get("name", ""),
            "is_draft": is_draft,
            "created_at": c.get("metadata", {}).get("created_at", ""),
        }

    corpora = [_norm_corpus(c) for c in items]
    corpora.sort(key=lambda x: (x["name"].lower(), not x["is_draft"]))
    return jsonify(corpora), 200


@server.route("/getCorpus/<corpus_id>")
@login_required
def get_corpus(corpus_id):
    try:
        response = requests.get(
            f"{API}/data/corpora/{corpus_id}",
            params={"owner_id": session.get("user_id")},
            timeout=10,
        )
        response.raise_for_status()
        return Response(
            response.content,
            status=response.status_code,
            mimetype=response.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.RequestException as e:
        current_app.logger.error(f"Failed to fetch corpus {corpus_id}: {e}")
        return jsonify({"error": f"Failed to fetch corpus {corpus_id}: {e}"}), 502


@server.route("/model-registry")
@login_required
def get_model_registry():
    with open("static/config/modelRegistry.json") as f:
        return jsonify(json.load(f))


@server.route("/trained-models")
@login_required
def trained_models():
    return render_template("trained_models.html")


def fetch_trained_models():
    try:
        owner_id = session.get("user_id")
        corpora_response = requests.get(
            f"{API}/data/corpora",
            params={"owner_id": owner_id},
            timeout=10,
        )
        corpora_response.raise_for_status()
        corpora = corpora_response.json()
        current_app.logger.info("Corpora response: %s", corpora)

        for corpus in corpora:
            corpus_id = corpus.get("id")
            server.logger.info("Processing corpus ID: %s", corpus_id)
            if not corpus_id:
                current_app.logger.warning("Corpus without ID: %s", corpus)
                continue

            try:
                models_response = requests.get(
                    f"{API}/data/corpora/{corpus_id}/models",
                    params={"owner_id": owner_id},
                    timeout=10,
                )
                models_response.raise_for_status()
                server.logger.info(
                    "Models response for corpus ID %s: %s", corpus_id, models_response.json()
                )
                corpus["models"] = models_response.json()
            except requests.Timeout:
                current_app.logger.error(
                    "Timeout fetching models for corpus ID %s", corpus_id
                )
                corpus["models"] = {"error": "Timeout fetching models"}
            except requests.RequestException as e:
                current_app.logger.error(
                    "Error fetching models for corpus ID %s: %s", corpus_id, str(e)
                )
                corpus["models"] = {
                    "error": "Failed to fetch models",
                    "detail": str(e),
                }

        return corpora

    except requests.Timeout:
        current_app.logger.error("Timeout fetching corpora")
        raise
    except requests.RequestException as e:
        current_app.logger.error("Error fetching corpora: %s", str(e))
        raise
    except ValueError:
        current_app.logger.error("Invalid JSON response from corpora endpoint")
        raise


@server.route("/get-trained-models", methods=["GET"])
@login_required
def get_trained_models():
    try:
        corpora = fetch_trained_models()
        return jsonify(corpora), 200
    except requests.Timeout:
        return jsonify({"error": "Upstream request timed out"}), 504
    except requests.RequestException as e:
        return jsonify({"error": "Upstream request failed", "detail": str(e)}), 502
    except ValueError:
        return jsonify({"error": "Invalid JSON from upstream"}), 502


@server.route("/get-models-names", methods=["GET"])
@login_required
def get_model_names():
    try:
        corpora = fetch_trained_models()

        names = []
        for corpus in corpora:
            models = corpus.get("models", [])
            for model in models:
                meta = model.get("metadata") or {}
                tr = meta.get("tr_params") or {}
                name = tr.get("model_name")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())

        seen = set()
        unique = []
        for n in names:
            key = n.lower()
            if key not in seen:
                seen.add(key)
                unique.append(n)

        return jsonify({"models": unique})

    except requests.Timeout:
        return jsonify({"error": "Upstream request timed out"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream request failed: {e}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@server.route("/delete-model/", methods=["POST"])
@login_required
def delete_model():
    """
    Delete a model by ID.
    """
    payload = request.get_json(silent=True) or {}
    model_id = payload.get("model_id")

    if not model_id:
        return jsonify({"error": "Missing 'model_id'"}), 400

    # get model entity
    try:
        t0 = time()
        up = requests.get(
            f"{API}/data/models/{model_id}",
            params={"owner_id": session.get("user_id")},
            timeout=(3.05, 60),
        )
        up.raise_for_status()
        model_metadata = up.json()
        server.logger.info(
            "get-dashboard model meta ok status=%s dt=%.3fs",
            up.status_code,
            time() - t0,
        )
    except requests.Timeout:
        server.logger.exception("model delete timeout fetching model")
        return jsonify({"error": "Upstream timeout fetching model"}), 504
    except requests.RequestException as e:
        server.logger.exception("model delete error fetching model: %s", e)
        return jsonify({"error": f"Upstream error fetching model: {e}"}), 502

    corpus_id = model_metadata.get("corpus_id")
    if not corpus_id:
        server.logger.error("model missing corpus_id")
        return jsonify({"error": "Model missing corpus_id"}), 400
    server.logger.info("Deleting model '%s' from corpus '%s'", model_id, corpus_id)

    # delete model from corpus
    try:
        up = requests.post(
            f"{API}/data/corpora/{corpus_id}/delete_model",
            params={"model_id": model_id},
            timeout=(3.05, 30),
        )
        up.raise_for_status()
    except requests.Timeout:
        server.logger.exception("model delete timeout removing from corpus")
        return jsonify(
            {"error": "Upstream timeout deleting model from corpus"}
        ), 504
    except requests.RequestException as e:
        server.logger.exception("model delete upstream corpus error: %s", e)
        return jsonify(
            {"error": f"Upstream error deleting from corpus: {e}"}
        ), 502
    server.logger.info("Model '%s' removed from corpus '%s'", model_id, corpus_id)

    # delete model
    try:
        up = requests.delete(
            f"{API}/data/models/{model_id}",
            params={"owner_id": session.get("user_id")},
            timeout=(3.05, 30),
        )
        server.logger.info("Model delete upstream response status=%s", up.status_code)
        if up.status_code == 204:
            return jsonify({"message": f"Model '{model_id}' deleted successfully"}), 200
        elif up.status_code == 404:
            return jsonify({"error": f"Model '{model_id}' not found"}), 404
        elif up.status_code >= 400:
            server.logger.error("model delete upstream error status=%s", up.status_code)
            return Response(
                up.content,
                status=up.status_code,
                mimetype=up.headers.get("Content-Type", "application/json"),
            )
        else:
            return jsonify(
                {
                    "message": f"Model '{model_id}' delete request completed",
                    "upstream_status": up.status_code,
                }
            ), 200
    except requests.Timeout:
        server.logger.exception("model delete timeout deleting model")
        return jsonify(
            {"error": "Upstream timeout deleting model"}
        ), 504
    except requests.RequestException as e:
        server.logger.exception("model delete upstream model error: %s", e)
        return jsonify(
            {"error": f"Upstream connection error deleting model: {e}"}
        ), 502


# ------------------------------------------------------------------------------
# Dashboard-related routes
# ------------------------------------------------------------------------------
@server.route("/dashboard", methods=["POST"])
@login_required
def dashboard():
    model_id = request.form.get("model_id", "")
    return render_template("dashboard.html", model_id=model_id)


@server.route("/get-dashboard-data", methods=["POST"])
@login_required
def proxy_dashboard_data():
    t0 = time()
    payload = request.get_json(silent=True) or {}
    # Prefer per-user config blob over static path for advanced dashboard config
    payload["config"] = get_effective_user_config(session.get("user_id"))
    payload.pop("config_path", None)

    model_id = payload.get("model_id", "")

    server.logger.info("get-dashboard start model_id=%s", model_id)

    if not model_id:
        server.logger.warning("get-dashboard missing model_id")
        return jsonify({"error": "model_id is required"}), 400

    # model metadata
    try:
        up0 = time()
        up = requests.get(
            f"{API}/data/models/{model_id}",
            params={"owner_id": session.get("user_id")},
            timeout=(3.05, 60),
        )
        up.raise_for_status()
        model_metadata = up.json()
        server.logger.info(
            "get-dashboard model meta ok status=%s dt=%.3fs",
            up.status_code,
            time() - up0,
        )
    except requests.Timeout:
        server.logger.exception("get-dashboard timeout fetching model")
        return jsonify({"error": "Upstream timeout fetching model"}), 504
    except requests.RequestException as e:
        server.logger.exception("get-dashboard error fetching model: %s", e)
        return jsonify({"error": f"Upstream error fetching model: {e}"}), 502

    corpus_id = model_metadata.get("corpus_id", "")
    if not corpus_id:
        server.logger.error("get-dashboard model missing corpus_id")
        return jsonify({"error": "Model missing corpus_id"}), 400

    # corpus data
    try:
        up0 = time()
        up = requests.get(
            f"{API}/data/corpora/{corpus_id}",
            params={"owner_id": session.get("user_id")},
            timeout=(3.05, 60),
        )
        up.raise_for_status()
        corpus_training_data = up.json()
        server.logger.info(
            "get-dashboard corpus ok status=%s dt=%.3fs docs=%s",
            up.status_code,
            time() - up0,
            len(corpus_training_data.get("documents", []) or []),
        )
    except requests.Timeout:
        server.logger.exception("get-dashboard timeout fetching corpus")
        return jsonify({"error": "Upstream timeout fetching corpus"}), 504
    except requests.RequestException as e:
        server.logger.exception("get-dashboard error fetching corpus: %s", e)
        return jsonify({"error": f"Upstream error fetching corpus: {e}"}), 502

    # model info (with cache)
    model_info_key = model_id
    cached_info, info_state = _cache_get(_CACHE_MODEL_INFO, model_info_key)
    if cached_info:
        raw_model_info = cached_info
        server.logger.debug("get-dashboard model-info cache %s", info_state)
    else:
        try:
            up0 = time()
            r = requests.post(f"{API}/queries/model-info", json=payload, timeout=60)
            r.raise_for_status()
            raw_model_info = r.json()
            _cache_set(_CACHE_MODEL_INFO, model_info_key, raw_model_info)
            server.logger.info(
                "get-dashboard model-info fetched status=%s dt=%.3fs (cached)",
                r.status_code,
                time() - up0,
            )
        except requests.HTTPError:
            try:
                return jsonify(r.json()), r.status_code
            except Exception:
                return jsonify({"detail": r.text}), r.status_code
        except Exception as e:
            server.logger.exception("get-dashboard model-info proxy error: %s", e)
            return jsonify({"detail": f"Proxy error: {e}"}), 502

    docs_list_raw = corpus_training_data.get("documents", []) or []
    docs_list = [
        d if isinstance(d, dict) else pydantic_to_dict(d) for d in docs_list_raw
    ]
    doc_ids = [str(d.get("id")) for d in docs_list if d.get("id") is not None]

    sorted_key = tuple(sorted(doc_ids))
    bulk_key = (model_id, sorted_key)
    cached_bulk, bulk_state = _cache_get(_CACHE_THETAS_BULK, bulk_key)
    if cached_bulk:
        doc_thetas = cached_bulk
        server.logger.debug(
            "get-dashboard thetas bulk cache %s | docs=%d", bulk_state, len(doc_ids)
        )
    else:
        payload_thetas = {"docs_ids": ",".join(doc_ids), "model_id": model_id}
        try:
            up0 = time()
            r = requests.post(
                f"{API}/queries/thetas-by-docs-ids", json=payload_thetas, timeout=60
            )
            r.raise_for_status()
            doc_thetas = r.json()
            _cache_set(_CACHE_THETAS_BULK, bulk_key, doc_thetas)
            server.logger.info(
                "get-dashboard thetas fetched status=%s dt=%.3fs (cached) docs=%d",
                r.status_code,
                time() - up0,
                len(doc_ids),
            )
        except requests.HTTPError:
            try:
                return jsonify(r.json()), r.status_code
            except Exception:
                return jsonify({"detail": r.text}), r.status_code
        except Exception as e:
            server.logger.exception("get-dashboard thetas proxy error: %s", e)
            return jsonify({"detail": f"Proxy error: {e}"}), 502

    try:
        model_training_corpus = pydantic_to_dict(corpus_training_data) or {}
        bundle = to_dashboard_bundle(
            raw_model_info,
            model_id,
            model_metadata=model_metadata.get("metadata", {}),
            model_training_corpus=model_training_corpus,
            doc_thetas=doc_thetas,
            logger=server.logger,
        )
        server.logger.info("get-dashboard done dt=%.3fs", time() - t0)
        return jsonify(bundle), 200
    except Exception as e:
        server.logger.exception("get-dashboard bundling error: %s", e)
        return jsonify({"error": f"Failed to build dashboard data: {e}"}), 500


@server.route("/text-info", methods=["POST"])
@login_required
def text_info():
    """
    Get text and model information for a given document according to a topic model.
    Used when a document row is clicked in dashboard.html.
    """
    t0 = time()
    payload = request.get_json(silent=True) or {}
    payload["config"] = get_effective_user_config(session.get("user_id"))
    payload.pop("config_path", None)
    model_id = payload.get("model_id", "")
    doc_id = str(payload.get("document_id", "")).strip()

    server.logger.info("TEXT-INFO start model_id=%s doc_id=%s", model_id, doc_id)

    if not model_id or not doc_id:
        server.logger.warning("TEXT-INFO missing required fields")
        return jsonify({"detail": "model_id and document_id are required."}), 400

    # model meta info (cached)
    model_entry = _CACHE_MODELS.get(model_id)
    if not model_entry or (time() - model_entry["ts"] > CACHE_TTL):
        try:
            up0 = time()
            up = requests.get(
                f"{API}/data/models/{model_id}",
                params={"owner_id": session.get("user_id")},
                timeout=(3.05, 30),
            )
            up.raise_for_status()
            model = up.json()
            _CACHE_MODELS[model_id] = {"ts": time(), "data": model}
            server.logger.info(
                "TEXT-INFO model meta fetched status=%s dt=%.3fs (cached)",
                up.status_code,
                time() - up0,
            )
        except requests.RequestException as e:
            server.logger.exception("TEXT-INFO fetch model error: %s", e)
            return jsonify({"detail": f"Failed to fetch model info: {e}"}), 502
    else:
        model = model_entry["data"]
        server.logger.debug("TEXT-INFO model meta cache hit")

    corpus_id = model.get("corpus_id", "")
    if not corpus_id:
        server.logger.error("TEXT-INFO model missing corpus_id")
        return jsonify({"detail": "Model missing corpus_id."}), 400

    # corpus info (cached)
    corpus_entry = _CACHE_CORPORA.get(corpus_id)
    if not corpus_entry or (time() - corpus_entry["ts"] > CACHE_TTL):
        try:
            up0 = time()
            up = requests.get(
                f"{API}/data/corpora/{corpus_id}",
                params={"owner_id": session.get("user_id")},
                timeout=(3.05, 60),
            )
            up.raise_for_status()
            corpus = up.json()
            _CACHE_CORPORA[corpus_id] = {"ts": time(), "data": corpus}
            server.logger.info(
                "TEXT-INFO corpus fetched status=%s dt=%.3fs (cached)",
                up.status_code,
                time() - up0,
            )
        except requests.RequestException as e:
            server.logger.exception("TEXT-INFO fetch corpus error: %s", e)
            return jsonify({"detail": f"Failed to fetch corpus info: {e}"}), 502
    else:
        corpus = corpus_entry["data"]
        server.logger.debug("TEXT-INFO corpus cache hit")

    docs = corpus.get("documents", [])
    doc_text = next((d.get("text", "") for d in docs if str(d.get("id")) == doc_id), "")
    if not doc_text:
        doc_text = f"(Text not found for document {doc_id})"

    # thetas for this document (cached)
    theta_key = (model_id, doc_id)
    cached_theta, theta_state = _cache_get(_CACHE_THETAS, theta_key)
    if cached_theta:
        thetas_by_doc = {doc_id: cached_theta}
        server.logger.debug("TEXT-INFO thetas cache %s for doc_id=%s", theta_state, doc_id)
    else:
        try:
            up0 = time()
            r = requests.post(
                f"{API}/queries/thetas-by-docs-ids",
                json={
                    "model_id": model_id,
                    "config_path": "static/config/config.yaml",
                    "docs_ids": doc_id,
                },
                timeout=60,
            )
            r.raise_for_status()
            thetas_by_doc = r.json()
            _cache_set(_CACHE_THETAS, theta_key, thetas_by_doc.get(doc_id) or {})
            server.logger.info(
                "TEXT-INFO thetas fetched status=%s dt=%.3fs (cached)",
                r.status_code,
                time() - up0,
            )
        except requests.RequestException as e:
            server.logger.exception("TEXT-INFO fetch thetas error: %s", e)
            return jsonify({"detail": f"Failed to fetch thetas: {e}"}), 502

    doc_thetas = thetas_by_doc.get(doc_id) or {}

    # model-info (cached)
    model_info_key = model_id
    cached_info, info_state = _cache_get(_CACHE_MODEL_INFO, model_info_key)
    if cached_info:
        raw_model_info = cached_info
        server.logger.debug("TEXT-INFO model-info cache %s", info_state)
    else:
        try:
            up0 = time()
            r = requests.post(f"{API}/queries/model-info", json=payload, timeout=60)
            r.raise_for_status()
            raw_model_info = r.json()
            _cache_set(_CACHE_MODEL_INFO, model_info_key, raw_model_info)
            server.logger.info(
                "TEXT-INFO model-info fetched status=%s dt=%.3fs (cached)",
                r.status_code,
                time() - up0,
            )
        except requests.HTTPError:
            try:
                return jsonify(r.json()), r.status_code
            except Exception:
                return jsonify({"detail": r.text}), r.status_code
        except Exception as e:
            server.logger.exception("TEXT-INFO model-info proxy error: %s", e)
            return jsonify({"detail": f"Proxy error: {e}"}), 502

    if not doc_thetas:
        server.logger.info("TEXT-INFO no thetas for doc_id=%s dt=%.3fs", doc_id, time() - t0)
        return jsonify(
            {
                "theme": "Unknown",
                "top_themes": [],
                "rationale": "",
                "text": doc_text,
            }
        )

    topics_info = raw_model_info.get("Topics Info", {}) or {}
    top_themes = sorted(
        (
            {
                "theme_id": int(k),
                "label": f"Topic {k}",
                "score": float(v),
                "keywords": (topics_info.get(str(k), {}) or {}).get("Keywords", ""),
            }
            for k, v in doc_thetas.items()
            if v is not None
        ),
        key=lambda x: x["score"],
        reverse=True,
    )
    top_theme = (
        top_themes[0]
        if top_themes
        else {"theme_id": None, "label": "Unknown", "score": 0, "keywords": ""}
    )

    server.logger.info(
        "TEXT-INFO done doc_id=%s top_theme=%s dt=%.3fs",
        doc_id,
        top_theme.get("theme_id"),
        time() - t0,
    )

    return jsonify(
        {
            "theme": top_theme["label"],
            "top_themes": top_themes,
            "rationale": "",
            "text": doc_text,
        }
    )

@server.post("/infer-text")
def infer_text():
    """
    Handles the initial inference request from the frontend.
    1. Sends the request body to the external API's inference endpoint.
    2. Receives a job_id.
    3. Polls the API's status endpoint using the job_id until results are available or a timeout occurs.
    4. Returns the final inference results to the frontend.
    """
    try:
        # 1. Get the JSON payload from the frontend request
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400

        # 2. Send the exact payload to the external API's inference endpoint
        inference_url = f"{API}/infer/json"
        
        # Note: The 'text' in your example is a single string. 
        # Ensure your external API expects this format.
        
        response = requests.post(
            inference_url,
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        # response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)

        # Assuming the response body contains a job_id (e.g., {"job_id": "..."})
        job_info = response.json()
        job_id = job_info.get("job_id")
        # print(f"Inference job submitted. Job ID: {job_id}")



        if not job_id:
            return jsonify({"error": "Inference API did not return a job_id"}), 500

        print(f"Inference job submitted. Job ID: {job_id}")
        
        
        # 3. Poll the job status (short delay) before returning
        status_url = f"{API}/status/jobs/{job_id}"
        poll_interval = float(os.getenv("INFER_POLL_INTERVAL_SEC", "1.0"))
        max_wait = float(os.getenv("INFER_POLL_TIMEOUT_SEC", "30.0"))

        deadline = monotonic() + max_wait
        last_status = None
        last_job_status = None

        while True:
            status_response = requests.get(status_url)
            status_response.raise_for_status()
            job_status = status_response.json()
            status = str(job_status.get("status", "")).lower()
            last_status = status
            last_job_status = job_status

            if status in ["completed", "succeeded", "success"]:
                return jsonify(job_status.get("results", job_status))
            if status in ["failed", "error"]:
                print(f"Job {job_id} failed.")
                return jsonify({"error": "Inference job failed", "details": job_status}), 500

            if monotonic() >= deadline:
                return jsonify({"status": last_status or "running", "details": last_job_status, "timeout": True}), 202

            sleep(poll_interval)

    except requests.exceptions.HTTPError as e:
        # Handle exceptions from the external API calls
        return jsonify({"error": f"External API HTTP Error: {e.response.text}", "status_code": e.response.status_code}), e.response.status_code
    except requests.exceptions.RequestException as e:
        # Handle network-level errors (e.g., connection refused)
        return jsonify({"error": f"Error communicating with external API: {e}"}), 503
    except Exception as e:
        # Catch any other unexpected errors
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


@server.post("/save-settings")
def save_settings():
    payload = request.get_json(silent=True) or {}
    server.logger.info("Dummy /save-settings received: %s", payload)
    return jsonify({"ok": True, "echo": payload}), 200


@server.route("/api/models/<model_id>/topics/<int:topic_id>/rename", methods=["POST"])
@login_required
def rename_topic(model_id: str, topic_id: int):
    """
    Rename a topic in a model. Proxies the request to the backend API.
    """
    payload = request.get_json(silent=True) or {}
    new_label = payload.get("new_label", "").strip()
    
    if not new_label:
        return jsonify({"error": "new_label is required"}), 400
    
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    
    # Forward request to backend API
    try:
        upstream_url = f"{API}/data/models/{model_id}/topics/{topic_id}/rename"
        upstream_response = requests.post(
            upstream_url,
            json={"new_label": new_label},
            params={"owner_id": session.get("user_id")},
            timeout=(3.05, 30),
        )
        
        if upstream_response.status_code == 200:
            return jsonify(upstream_response.json()), 200
        else:
            return Response(
                upstream_response.content,
                status=upstream_response.status_code,
                mimetype=upstream_response.headers.get("Content-Type", "application/json"),
            )
    except requests.Timeout:
        server.logger.exception("rename topic timeout")
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.RequestException as e:
        server.logger.exception("rename topic error: %s", e)
        return jsonify({"error": f"Upstream connection error: {e}"}), 502


@server.route("/api/models/<model_id>/topics/renames", methods=["GET"])
@login_required
def get_topic_renames(model_id: str):
    """
    Get all topic renames for a model from the backend API.
    """
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    
    try:
        upstream_url = f"{API}/data/models/{model_id}/topics/renames"
        upstream_response = requests.get(
            upstream_url,
            params={"owner_id": session.get("user_id")},
            timeout=(3.05, 30),
        )
        
        if upstream_response.status_code == 200:
            return jsonify(upstream_response.json()), 200
        elif upstream_response.status_code == 404:
            # No renames found, return empty object
            return jsonify({"topic_labels": {}}), 200
        else:
            return Response(
                upstream_response.content,
                status=upstream_response.status_code,
                mimetype=upstream_response.headers.get("Content-Type", "application/json"),
            )
    except requests.Timeout:
        server.logger.exception("get topic renames timeout")
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.RequestException as e:
        server.logger.exception("get topic renames error: %s", e)
        return jsonify({"error": f"Upstream connection error: {e}"}), 502

import yaml
import os

# Function to read the YAML config file
def load_config(filepath=CONFIG_PATH):
    """Reads and parses the YAML configuration file."""
    if not os.path.exists(filepath):
        # Handle the case where the file doesn't exist
        print(f"Error: Config file not found at {filepath}")
        return {}
    try:
        with open(filepath, 'r') as f:
            # Use safe_load to avoid potential security issues
            return yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(f"Error reading YAML file: {exc}")
        return {}
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")
        return {}


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # For local development; in production use gunicorn/uwsgi, etc.
    server.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
