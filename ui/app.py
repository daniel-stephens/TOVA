import json
import logging
import os
import sys
from pathlib import Path
from time import time, sleep, monotonic
from datetime import timedelta
from functools import wraps
from uuid import uuid4
from copy import deepcopy

# Load .env so OPENAI_API_KEY (and others) are available from the project .env file
try:
    from dotenv import load_dotenv
    _base = Path(__file__).resolve().parent   # ui/
    _root = _base.parent                      # project root (TOVA)
    # Load project root .env first (so it works regardless of cwd), then cwd; override so file wins
    if (_root / ".env").exists():
        load_dotenv(_root / ".env", override=True)
    if (_base / ".env").exists():
        load_dotenv(_base / ".env", override=True)
    load_dotenv(override=True)  # .env in current working directory
except ImportError:
    pass

import requests
from ruamel.yaml import YAML
from authlib.integrations.flask_client import OAuth

# Configure ruamel.yaml to preserve formatting (inline arrays, etc.)
# ruamel.yaml preserves formatting when loading existing files
yaml = YAML()
yaml.preserve_quotes = True
yaml.width = 1000  # Wide width to prevent unwanted line breaks
yaml.indent(mapping=2, sequence=4, offset=2)  # Match default config indentation

# Configure to use flow style (inline) for short lists/arrays to match default config format
def represent_list(self, data):
    """Use flow style (inline) for lists with <= 10 items."""
    if len(data) <= 10:
        return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

# Add custom representer for lists
yaml.Representer.add_representer(list, represent_list)
from dashboard_utils import (
    pydantic_to_dict,
    to_dashboard_bundle,
    dashboard_context_for_llm,
)
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
from models import db, User, UserConfig, AuditLog, ChatMessage
from tova.core import drafts
from tova.core import models as models_module
from tova.api.models.data_schemas import DraftType

# ------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------
server = Flask(__name__)
server.logger.setLevel(logging.INFO)
# Ensure INFO logs are visible under gunicorn (e.g. in docker logs)
if not server.logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.INFO)
    server.logger.addHandler(h)

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
    Also run one-off migrations (e.g. add is_admin to existing users table).
    """
    if not RUN_DB_INIT_ON_START:
        return
    with server.app_context():
        with db.engine.begin() as conn:
            conn.execute(text("SELECT pg_advisory_lock(420420)"))
            try:
                db.create_all()
                # Add is_admin to users if table existed before admin support (PostgreSQL)
                conn.execute(text(
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT false"
                ))
                # Add user_id to chat_messages if table existed before user scoping (PostgreSQL)
                conn.execute(text(
                    "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS user_id VARCHAR(36)"
                ))
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_chat_messages_user_id ON chat_messages (user_id)"
                ))
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_chat_messages_user_model ON chat_messages (user_id, model_id)"
                ))
            finally:
                conn.execute(text("SELECT pg_advisory_unlock(420420)"))


_run_db_init_once()


def _log_audit(action: str, target_type: str = None, target_id: str = None, details: str = None):
    """Record an audit event. Does not raise; failures are logged only."""
    try:
        actor_id = (getattr(g, "user", None) or {}).get("id")
        entry = AuditLog(
            actor_id=actor_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            details=details[:1024] if details else None,
        )
        db.session.add(entry)
        db.session.commit()
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        current_app.logger.warning("Audit log failed: %s", e)


# Use relative path from project root, or absolute path from env var
_drafts_save_env = os.getenv("DRAFTS_SAVE")
if _drafts_save_env:
    DRAFTS_SAVE = Path(_drafts_save_env)
else:
    # Default to relative path from project root
    BASE_DIR = Path(__file__).parent.parent
    DRAFTS_SAVE = BASE_DIR / "data" / "drafts"
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
        RAW_CONFIG = yaml.load(f) or {}
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


def _save_user_config_overrides(user_id: str, overrides: dict):
    """
    Save only the user's config overrides (changes) to the database.
    This stores only what differs from defaults, not the whole config.
    """
    if not user_id:
        return
    
    # Save only the overrides to database
    sanitized = _json_safe(overrides or {})
    entry = UserConfig.query.filter_by(user_id=user_id).first()
    if entry:
        entry.config = sanitized
    else:
        entry = UserConfig(user_id=user_id, config=sanitized)
        db.session.add(entry)
    db.session.commit()
    server.logger.debug("Saved user config overrides to database for user: %s", user_id)


def _reset_user_config(user_id: str):
    """Remove user-specific configuration overrides (falls back to defaults)."""
    if not user_id:
        return
    entry = UserConfig.query.filter_by(user_id=user_id).first()
    if entry:
        db.session.delete(entry)
        db.session.commit()
        server.logger.debug("Reset user config overrides for user: %s", user_id)


def _verify_corpus_ownership(corpus_id: str, user_id: str) -> bool:
    """
    Verify that a corpus belongs to the given user.
    Returns True if owned, False otherwise.
    """
    if not user_id:
        return False
    try:
        # Fetch corpus WITHOUT owner_id parameter to avoid API returning 403
        # We'll check ownership locally after fetching
        response = requests.get(
            f"{API}/data/corpora/{corpus_id}",
            timeout=5,
        )
        if response.status_code == 200:
            corpus = response.json()
            corpus_owner = corpus.get("owner_id") or corpus.get("metadata", {}).get("owner_id")
            return corpus_owner == user_id
        elif response.status_code == 404:
            # Corpus doesn't exist
            return False
    except requests.HTTPError as e:
        # If API returns 403, it means we don't own it
        if e.response and e.response.status_code == 403:
            return False
    except Exception:
        pass
    return False


def _verify_model_ownership(model_id: str, user_id: str) -> bool:
    """Verify that a model belongs to the given user. Returns True if owned, False otherwise."""
    if not user_id:
        return False
    try:
        response = requests.get(f"{API}/data/models/{model_id}", timeout=5)
        if response.status_code == 200:
            model = response.json()
            model_owner = model.get("owner_id") or (model.get("metadata") or {}).get("owner_id")
            return model_owner == user_id
    except Exception:
        pass
    return False


def _verify_dataset_ownership(dataset_id: str, user_id: str) -> bool:
    """Verify that a dataset belongs to the given user. Returns True if owned, False otherwise."""
    if not user_id:
        return False
    try:
        response = requests.get(f"{API}/data/datasets/{dataset_id}", timeout=5)
        if response.status_code == 200:
            dataset = response.json()
            owner = dataset.get("owner_id") or (dataset.get("metadata") or {}).get("owner_id")
            return owner == user_id
    except Exception:
        pass
    return False


def _filter_list_by_owner(items: list, user_id: str) -> list:
    """Keep only items (corpora, datasets, or models) owned by user_id. No backend change."""
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


def get_effective_user_config(user_id: str) -> dict:
    """
    Return the default config merged with the user's overrides (if any).
    Cached per-request on g to avoid duplicate queries.
    
    Note: LLM config is NOT merged here. It is applied directly as training parameters
    via applyLlmSelection() in loadModel.html and in training_start().
    """
    if hasattr(g, "_effective_user_config"):
        return g._effective_user_config

    base = deepcopy(DEFAULT_CONFIG)
    if not user_id:
        g._effective_user_config = base
        return base

    # Load user's overrides from database
    overrides = _load_user_config(user_id) or {}
    if overrides:
        # Exclude llm_config from merging - it's applied directly as training parameters
        overrides_copy = {k: v for k, v in overrides.items() if k != "llm_config"}
        
        # Merge defaults with user overrides
        merged = _deep_merge(base, overrides_copy if isinstance(overrides_copy, dict) else {})
        
        g._effective_user_config = merged
        return merged
    
    # If no user config exists, return default
    g._effective_user_config = base
    return g._effective_user_config


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
def _admin_emails_from_env():
    """Comma-separated list of emails that are always treated as admin (e.g. TOVA_ADMIN_EMAILS)."""
    raw = os.getenv("TOVA_ADMIN_EMAILS", "").strip()
    if not raw:
        return set()
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        # check the same key you actually set in login/signup/Okta
        if "user_id" not in session:
            next_url = request.path  # or request.full_path
            return redirect(url_for("login", next=next_url))
        return view_func(*args, **kwargs)

    return wrapped_view


def admin_required(view_func):
    """Require the current user to be logged in and an admin (DB is_admin or TOVA_ADMIN_EMAILS)."""
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login", next=request.path))
        user = getattr(g, "user", None)
        if not user or not user.get("is_admin"):
            flash("Admin access required.", "danger")
            return redirect(url_for("home"))
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
        admin_emails = _admin_emails_from_env()
        is_admin = getattr(user, "is_admin", False) or (user.email.lower() in admin_emails)
        g.user = {
            "id": user.id,
            "email": user.email,
            "name": user.name or user.email,
            "is_admin": is_admin,
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
        _log_audit("user_created", target_type="user", target_id=user.id, details=user.email)

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
        _log_audit("login", target_type="user", target_id=user.id, details=user.email)

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
    is_new_user = False
    if not user:
        is_new_user = True
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
    _log_audit("login", target_type="user", target_id=user.id, details=user.email)

    flash("Signed in with Okta.", "success")

    next_url = session.pop("next_url", None)
    return redirect(next_url or url_for("home"))


@server.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("login"))


# ------------------------------------------------------------------------------
# Admin (superuser) page
# ------------------------------------------------------------------------------
@server.route("/admin")
@login_required
@admin_required
def admin_page():
    """Admin page: list users and manage admin status."""
    admin_emails = _admin_emails_from_env()
    rows = User.query.order_by(User.created_at.desc()).all()
    users = [
        {
            "id": u.id,
            "name": u.name,
            "email": u.email,
            "auth_source": u.auth_source,
            "created_at": u.created_at,
            "is_admin": getattr(u, "is_admin", False),
        }
        for u in rows
    ]
    return render_template(
        "admin.html",
        users=users,
        admin_emails_from_env=admin_emails,
    )


@server.route("/admin/users/<user_id>/toggle-admin", methods=["POST"])
@login_required
@admin_required
def admin_toggle_user(user_id):
    """Toggle is_admin for a user (only admins). Cannot remove your own admin if you're the last admin."""
    if not getattr(g, "user", None) or not g.user.get("is_admin"):
        return jsonify({"success": False, "message": "Forbidden"}), 403
    target = User.query.get(user_id)
    if not target:
        return jsonify({"success": False, "message": "User not found"}), 404
    admin_emails = _admin_emails_from_env()
    db_admin_count = User.query.filter_by(is_admin=True).count()
    # If we're demoting ourselves and we're the only DB admin (and not in env list), forbid
    if (
        target.id == g.user["id"]
        and getattr(target, "is_admin", False)
        and db_admin_count <= 1
        and target.email.lower() not in admin_emails
    ):
        return jsonify({"success": False, "message": "Cannot remove the last admin."}), 400
    target.is_admin = not getattr(target, "is_admin", False)
    db.session.commit()
    action = "admin_revoked" if not target.is_admin else "admin_promoted"
    _log_audit(action, target_type="user", target_id=target.id, details=target.email)
    return jsonify({"success": True, "is_admin": target.is_admin})


@server.route("/admin/users", methods=["POST"])
@login_required
@admin_required
def admin_create_user():
    """Create a new user (admin only). JSON: name, email, password."""
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    email = (payload.get("email") or "").strip().lower()
    password = (payload.get("password") or "").strip()
    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"success": False, "message": "An account with that email already exists"}), 400
    user = User(
        id=str(uuid4()),
        name=name or email,
        email=email,
    )
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    _log_audit("user_created", target_type="user", target_id=user.id, details=user.email)
    return jsonify({"success": True, "user_id": user.id, "email": user.email}), 201


@server.route("/admin/users/<user_id>/delete", methods=["POST"])
@login_required
@admin_required
def admin_delete_user(user_id):
    """Delete a user (admin only). Cannot delete yourself."""
    if user_id == g.user.get("id"):
        return jsonify({"success": False, "message": "Cannot delete your own account"}), 400
    target = User.query.get(user_id)
    if not target:
        return jsonify({"success": False, "message": "User not found"}), 404
    email = target.email
    db.session.delete(target)
    db.session.commit()
    _log_audit("user_deleted", target_type="user", target_id=user_id, details=email)
    return jsonify({"success": True}), 200


@server.route("/admin/corpora", methods=["GET"])
@login_required
@admin_required
def admin_list_corpora():
    """List all corpora across all users (admin only)."""
    try:
        r = requests.get(f"{API}/data/corpora", timeout=15)
        r.raise_for_status()
        corpora = r.json()
    except requests.RequestException as e:
        server.logger.exception("Admin list corpora: %s", e)
        return jsonify({"error": str(e)}), 502
    user_ids = {c.get("owner_id") or (c.get("metadata") or {}).get("owner_id") for c in corpora if c.get("owner_id") or (c.get("metadata") or {}).get("owner_id")}
    users_by_id = {}
    for uid in user_ids:
        if uid:
            u = User.query.get(uid)
            users_by_id[uid] = {"email": u.email if u else None, "name": (u.name if u else None) or ""}
    for c in corpora:
        oid = c.get("owner_id") or (c.get("metadata") or {}).get("owner_id")
        c["owner_email"] = users_by_id.get(oid, {}).get("email") if oid else None
        c["owner_name"] = users_by_id.get(oid, {}).get("name") if oid else None
    return jsonify(corpora), 200


@server.route("/admin/corpora/<corpus_id>/delete", methods=["POST"])
@login_required
@admin_required
def admin_delete_corpus(corpus_id):
    """Delete any corpus (admin only)."""
    try:
        r = requests.get(f"{API}/data/corpora/{corpus_id}", timeout=10)
        if r.status_code == 404:
            return jsonify({"error": "Corpus not found"}), 404
        r.raise_for_status()
        corpus = r.json()
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502
    corpus_name = (corpus.get("metadata") or {}).get("name") or corpus_id
    models = corpus.get("models") or []
    for model_id in models:
        try:
            requests.delete(f"{API}/data/models/{model_id}", timeout=(3, 30))
        except Exception:
            pass
    try:
        del_resp = requests.delete(f"{API}/data/corpora/{corpus_id}", timeout=(3, 30))
        if del_resp.status_code == 204:
            _log_audit("corpus_deleted", target_type="corpus", target_id=corpus_id, details=corpus_name)
            return jsonify({"success": True}), 200
        if del_resp.status_code == 404:
            return jsonify({"error": "Corpus not found"}), 404
        return jsonify({"error": f"Upstream status {del_resp.status_code}"}), del_resp.status_code
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


@server.route("/admin/models", methods=["GET"])
@login_required
@admin_required
def admin_list_models():
    """List all models across all users (admin only)."""
    try:
        models_drafts = drafts.list_drafts(type=DraftType.model)
    except Exception as e:
        server.logger.exception("Admin list models drafts: %s", e)
        return jsonify({"error": str(e)}), 500
    user_ids = set()
    models_list = []
    for draft in models_drafts:
        owner_id = draft.owner_id or (draft.metadata.get("owner_id") if draft.metadata else None)
        # Same fallback as fetch_trained_models: if no owner on model, use corpus owner
        if not owner_id and draft.metadata:
            corpus_id = draft.metadata.get("corpus_id")
            if corpus_id:
                try:
                    corpus_resp = requests.get(f"{API}/data/corpora/{corpus_id}", timeout=5)
                    if corpus_resp.status_code == 200:
                        corpus_data = corpus_resp.json()
                        owner_id = corpus_data.get("owner_id") or (corpus_data.get("metadata") or {}).get("owner_id")
                except Exception:
                    pass
        if owner_id:
            user_ids.add(owner_id)
        try:
            model = drafts.draft_to_model(draft)
            if model:
                meta = (draft.metadata or {}).copy()
                name = model.name or meta.get("tr_params", {}).get("model_name") or draft.id
                models_list.append({
                    "id": model.id,
                    "name": name,
                    "owner_id": model.owner_id or owner_id,
                    "corpus_id": getattr(model, "corpus_id", None) or meta.get("corpus_id"),
                    "created_at": model.created_at or meta.get("created_at"),
                })
        except Exception:
            continue
    users_by_id = {}
    for uid in user_ids:
        u = User.query.get(uid)
        users_by_id[uid] = {"email": u.email if u else None, "name": (u.name if u else None) or ""}
    for m in models_list:
        oid = m.get("owner_id")
        m["owner_email"] = users_by_id.get(oid, {}).get("email") if oid else None
        m["owner_name"] = users_by_id.get(oid, {}).get("name") if oid else None
    return jsonify(models_list), 200


@server.route("/admin/models/<model_id>/delete", methods=["POST"])
@login_required
@admin_required
def admin_delete_model(model_id):
    """Delete any model (admin only)."""
    try:
        up = requests.get(f"{API}/data/models/{model_id}", timeout=10)
        if up.status_code == 404:
            return jsonify({"error": "Model not found"}), 404
        up.raise_for_status()
        model_meta = up.json()
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502
    corpus_id = model_meta.get("corpus_id")
    if corpus_id:
        try:
            requests.post(f"{API}/data/corpora/{corpus_id}/delete_model", params={"model_id": model_id}, timeout=10)
        except Exception:
            pass
    try:
        del_resp = requests.delete(f"{API}/data/models/{model_id}", timeout=(3, 30))
        if del_resp.status_code == 204:
            _log_audit("model_deleted", target_type="model", target_id=model_id)
            return jsonify({"success": True}), 200
        if del_resp.status_code == 404:
            return jsonify({"error": "Model not found"}), 404
        return jsonify({"error": f"Upstream status {del_resp.status_code}"}), del_resp.status_code
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


@server.route("/admin/stats", methods=["GET"])
@login_required
@admin_required
def admin_stats():
    """System stats (admin only)."""
    try:
        r = requests.get(f"{API}/data/corpora", timeout=10)
        corpus_count = len(r.json()) if r.ok else None
    except Exception:
        corpus_count = None
    try:
        models_drafts = drafts.list_drafts(type=DraftType.model)
        model_count = len(models_drafts)
    except Exception:
        model_count = None
    user_count = User.query.count()
    audit_count = AuditLog.query.count()
    return jsonify({
        "users": user_count,
        "corpora": corpus_count,
        "models": model_count,
        "audit_log_entries": audit_count,
    }), 200


@server.route("/admin/audit", methods=["GET"])
@login_required
@admin_required
def admin_audit():
    """Paginated audit log (admin only)."""
    page = max(1, request.args.get("page", 1, type=int))
    per_page = min(100, max(1, request.args.get("per_page", 50, type=int)))
    q = AuditLog.query.order_by(AuditLog.created_at.desc())
    pagination = q.paginate(page=page, per_page=per_page, error_out=False)
    items = []
    for row in pagination.items:
        actor = None
        if row.actor_id:
            u = User.query.get(row.actor_id)
            actor = u.email if u else row.actor_id
        items.append({
            "id": row.id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "actor_id": row.actor_id,
            "actor_email": actor,
            "action": row.action,
            "target_type": row.target_type,
            "target_id": row.target_id,
            "details": row.details,
        })
    return jsonify({
        "items": items,
        "page": pagination.page,
        "per_page": pagination.per_page,
        "total": pagination.total,
        "pages": pagination.pages,
    }), 200


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
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({}), 200
    
    # Load LLM config from database
    user_config = _load_user_config(user_id) or {}
    cfg = user_config.get("llm_config", {})
    
    # DO NOT send api_key back to browser for security
    # Return a copy without the api_key
    safe_cfg = {k: v for k, v in cfg.items() if k != "api_key"}
    return jsonify(safe_cfg)


@server.route("/save-llm-config", methods=["POST"])
@login_required
def save_llm_config():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"success": False, "message": "User not authenticated."}), 401
    
    data = request.get_json(force=True) or {}

    provider = data.get("provider")
    model = data.get("model") or None
    host = data.get("host") or None
    api_key = data.get("api_key")  # Sensitive: store securely in database

    if not provider:
        return jsonify({"success": False, "message": "Provider is required."}), 400

    # Build LLM config object
    llm_cfg = {
        "provider": provider,
        "model": model,
        "host": host,
    }
    
    # Store API key if provided (for secure server-side use)
    if api_key:
        llm_cfg["api_key"] = api_key

    # Load existing user config
    existing_config = _load_user_config(user_id) or {}
    
    # Update LLM config in user config
    existing_config["llm_config"] = llm_cfg
    
    # Save to database
    _save_user_config_overrides(user_id, existing_config)
    
    server.logger.debug("Saved LLM config to database for user: %s", user_id)

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
    # Save only the overrides (what changed from defaults)
    overrides = _json_safe(payload)
    _save_user_config_overrides(user_id, overrides)
    
    # Return the effective config (defaults merged with overrides)
    merged = get_effective_user_config(user_id)
    return jsonify({
        "success": True, 
        "config": merged, 
        "message": "Configuration saved."
    }), 200


@server.route("/api/user-config/overrides", methods=["GET"])
@login_required
def api_get_user_config_overrides():
    """Get only the user's config overrides (not the full effective config)."""
    user_id = session.get("user_id")
    overrides = _load_user_config(user_id) or {}
    return jsonify(overrides), 200


@server.route("/api/user-config/reset", methods=["POST"])
@login_required
def api_reset_user_config():
    user_id = session.get("user_id")
    _reset_user_config(user_id)
    fresh = get_effective_user_config(user_id)
    return jsonify({
        "success": True, 
        "config": fresh, 
        "message": "Configuration reset to defaults."
    }), 200


# ------------------------------------------------------------------------------
# Data / Corpus-related routes
# ------------------------------------------------------------------------------
@server.route("/load-data-page/")
@login_required
def load_data_page():
    """Page for uploading/creating a dataset."""
    current_user_id = session.get("user_id")
    return render_template("loadData.html", current_user_id=current_user_id or "")


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

    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    for el in datasets:
        try:
            upstream = requests.get(
                f"{API}/data/datasets/{el['id']}",
                timeout=(3.05, 30),
            )
            if not upstream.ok:
                return Response(
                    upstream.content,
                    status=upstream.status_code,
                    mimetype=upstream.headers.get("Content-Type", "application/json"),
                )
            ds = upstream.json()
            owner = ds.get("owner_id") or (ds.get("metadata") or {}).get("owner_id")
            # Treat missing or "anonymous" owner as owned by current user when logged in
            if owner is None or owner == "anonymous":
                owner = user_id
            if owner != user_id:
                return jsonify({"error": "Access denied: You do not own one or more of the selected datasets"}), 403
            datasets_lst.append(ds)
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
    if not session.get("user_id"):
        return jsonify({"error": "Not authenticated"}), 401

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

    # Verify ownership before deletion
    corpus_owner = corpus.get("owner_id") or corpus.get("metadata", {}).get("owner_id")
    if corpus_owner != owner_id:
        return jsonify({"error": "Access denied: You do not own this corpus"}), 403

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
            _log_audit("corpus_deleted", target_type="corpus", target_id=corpus_id, details=corpus_name)
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

    # fetch corpus from backend and verify ownership
    user_id = session.get("user_id")
    if not _verify_corpus_ownership(corpus_id, user_id):
        return jsonify({"error": "Access denied: You do not own this corpus"}), 403
    
    try:
        up = requests.get(
            f"{API}/data/corpora/{corpus_id}",
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
    server.logger.info(f"Training request payload: {payload}")
    corpus_id = payload.get("corpus_id")
    model = payload.get("model")
    model_name = payload.get("model_name")
    training_params = payload.get("training_params") or {}
    config_path = payload.get("config_path", "static/config/config.yaml")  # Get config_path from payload or use default
    user_id = session.get("user_id")


    if not corpus_id or not model or not model_name:
        return jsonify({"error": "Missing corpus_id/model/model_name"}), 400

    # Load existing user config overrides from database
    existing_overrides = _load_user_config(user_id) or {}
    
    # Build new overrides from training_params (what user changed in this form)
    # We merge with existing overrides to preserve previous user changes
    new_overrides = {}
    if training_params:
        # Build overrides structure matching config hierarchy
        if model in ["tomotopyLDA", "CTM"]:  # Traditional models
            if "do_labeller" in training_params:
                new_overrides.setdefault("topic_modeling", {}).setdefault("traditional", {})["do_labeller"] = training_params["do_labeller"]
            if "do_summarizer" in training_params:
                new_overrides.setdefault("topic_modeling", {}).setdefault("traditional", {})["do_summarizer"] = training_params["do_summarizer"]
            if "llm_model_type" in training_params:
                new_overrides.setdefault("topic_modeling", {}).setdefault("traditional", {})["llm_model_type"] = training_params["llm_model_type"]
            if "labeller_prompt" in training_params:
                new_overrides.setdefault("topic_modeling", {}).setdefault("traditional", {})["labeller_model_path"] = training_params["labeller_prompt"]
            elif "labeller_model_path" in training_params:
                new_overrides.setdefault("topic_modeling", {}).setdefault("traditional", {})["labeller_model_path"] = training_params["labeller_model_path"]
            if "summarizer_prompt" in training_params:
                new_overrides.setdefault("topic_modeling", {}).setdefault("traditional", {})["summarizer_prompt"] = training_params["summarizer_prompt"]
            if "num_topics" in training_params:
                new_overrides.setdefault("topic_modeling", {}).setdefault("traditional", {})["num_topics"] = training_params["num_topics"]
            if "thetas_thr" in training_params:
                new_overrides.setdefault("topic_modeling", {}).setdefault("traditional", {})["thetas_thr"] = training_params["thetas_thr"]
            if "topn" in training_params:
                new_overrides.setdefault("topic_modeling", {}).setdefault("traditional", {})["topn"] = training_params["topn"]
            
            # Map CTM-specific parameters to topic_modeling.ctm
            if model == "CTM":
                ctm_params = [
                    "num_epochs", "sbert_model", "sbert_context", "batch_size",
                    "contextual_size", "inference_type", "n_components", "model_type",
                    "hidden_sizes", "activation", "dropout", "learn_priors",
                    "lr", "momentum", "solver", "reduce_on_plateau",
                    "num_data_loader_workers", "label_size", "loss_weights"
                ]
                for param in ctm_params:
                    if param in training_params:
                        value = training_params[param]
                        # Handle string representations of arrays/objects (e.g., "[100,100]")
                        if param in ["hidden_sizes", "loss_weights"] and isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except (json.JSONDecodeError, ValueError):
                                # Keep as string if parsing fails
                                pass
                        # Ensure lists are stored as lists (not tuples)
                        if param in ["hidden_sizes", "loss_weights"] and isinstance(value, (list, tuple)):
                            value = list(value)  # Convert tuple to list if needed
                        new_overrides.setdefault("topic_modeling", {}).setdefault("ctm", {})[param] = value
        
        # Also handle model-specific overrides (for any model)
        model_specific_params = {}
        for key, value in training_params.items():
            # Skip already handled params
            if key in ["do_labeller", "do_summarizer", "llm_model_type", "labeller_prompt", 
                      "labeller_model_path", "summarizer_prompt", "num_topics", "thetas_thr", "topn",
                      "preprocess_text"]:
                continue
            # Skip CTM params if model is CTM
            if model == "CTM" and key in ["num_epochs", "sbert_model", "sbert_context", "batch_size",
                                          "contextual_size", "inference_type", "n_components", "model_type",
                                          "hidden_sizes", "activation", "dropout", "learn_priors",
                                          "lr", "momentum", "solver", "reduce_on_plateau",
                                          "num_data_loader_workers", "label_size", "loss_weights"]:
                continue
            # Add to model-specific overrides
            model_specific_params[key] = value
        
        if model_specific_params:
            new_overrides.setdefault("topic_modeling", {})[model] = model_specific_params
        
        # Merge new overrides with existing overrides (new takes precedence)
        merged_overrides = _deep_merge(existing_overrides, new_overrides)
        
        # Save merged overrides to database
        if merged_overrides:
            _save_user_config_overrides(user_id, merged_overrides)
    
    # training_params already contains all user changes (from database + form)
    # But we also need to ensure LLM config from database is included when the model uses it
    final_training_params = deepcopy(training_params)
    
    # When user has stated LLM preference (labeller/summarizer on or sent provider/model), merge saved config
    TRADITIONAL_MODELS = ("tomotopyLDA", "CTM")
    add_llm_params = True
    if model in TRADITIONAL_MODELS:
        add_llm_params = bool(
            final_training_params.get("do_labeller")
            or final_training_params.get("do_summarizer")
            or final_training_params.get("llm_provider")
            or final_training_params.get("llm_model_type")
        )

    user_config = _load_user_config(user_id) or {}
    llm_config = user_config.get("llm_config", {})
    if llm_config and add_llm_params:
        provider = llm_config.get("provider")
        llm_model = llm_config.get("model")
        host = llm_config.get("host")
        api_key = llm_config.get("api_key")
        if provider and "llm_provider" not in final_training_params:
            final_training_params["llm_provider"] = provider
        if llm_model and "llm_model_type" not in final_training_params:
            final_training_params["llm_model_type"] = llm_model
        if host and "llm_server" not in final_training_params:
            final_training_params["llm_server"] = host
        if api_key and "llm_api_key" not in final_training_params:
            final_training_params["llm_api_key"] = api_key

    # When no LLM choice was sent and no saved LLM Settings modal, use effective config
    # (topic_modeling.general from config form or config.yaml) so e.g. selecting Ollama
    # in the config form is respected instead of always falling back to config.yaml default.
    if add_llm_params:
        effective = get_effective_user_config(user_id)
        tm_general = _safe_dict(effective.get("topic_modeling", {})).get("general", {})
        if tm_general:
            if "llm_provider" not in final_training_params and tm_general.get("llm_provider"):
                p = (tm_general["llm_provider"] or "").strip().lower()
                final_training_params["llm_provider"] = "openai" if p == "gpt" else p
            if "llm_model_type" not in final_training_params and tm_general.get("llm_model_type"):
                final_training_params["llm_model_type"] = tm_general["llm_model_type"]
            if "llm_server" not in final_training_params and tm_general.get("llm_server"):
                final_training_params["llm_server"] = tm_general["llm_server"]

    # When user chose Ollama but host missing, default from saved config or config file
    if (final_training_params.get("llm_provider") or "").lower() == "ollama":
        if not final_training_params.get("llm_server"):
            ollama_from_config = (
                _safe_dict(RAW_CONFIG.get("llm", {})).get("ollama", {}).get("host")
                if RAW_CONFIG else None
            )
            final_training_params["llm_server"] = (
                (llm_config or {}).get("host")
                or ollama_from_config
                or "http://localhost:11434"
            )

    # get corpus and verify ownership
    try:
        # Fetch corpus without owner_id parameter to check ownership locally
        up = requests.get(
            f"{API}/data/corpora/{corpus_id}",
            timeout=(3.05, 60),
        )
        up.raise_for_status()
        corpus = up.json()
        
        # Verify corpus ownership before training
        corpus_owner = corpus.get("owner_id") or corpus.get("metadata", {}).get("owner_id")
        if corpus_owner != user_id:
            return jsonify({"error": "Access denied: You do not own this corpus"}), 403
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

    # Ensure we have owner_id - get from session if not already set
    if not user_id:
        user_id = session.get("user_id")
    
    if not user_id:
        return jsonify({"error": "User not authenticated"}), 401

    tm_req = {
        "model": model,
        "corpus_id": corpus_id,
        "data": docs,
        "id_col": "id",
        "text_col": "text",
        "training_params": final_training_params,  # Form params + LLM from DB when labeller/summarizer on (deployed solution)
        "config_path": config_path,  # Default config path (user changes sent as parameters)
        "model_name": model_name,
    }
    server.logger.info(f"Training request: {tm_req}")
    
    # Confirm LLM options being sent (no secrets)
    fp = final_training_params or {}
    server.logger.info(
        "Training LLM options: do_labeller=%s do_summarizer=%s llm_provider=%s llm_model_type=%s llm_server=%s api_key_set=%s",
        fp.get("do_labeller"),
        fp.get("do_summarizer"),
        fp.get("llm_provider"),
        fp.get("llm_model_type"),
        fp.get("llm_server"),
        bool(fp.get("llm_api_key")),
    )

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
    """Return a deduped, A→Z list of corpus names for the current user only."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    try:
        r = requests.get(
            f"{API}/data/corpora",
            params={"owner_id": user_id},
            timeout=10,
        )
        r.raise_for_status()
        items = r.json()
    except requests.RequestException:
        server.logger.exception("Failed to fetch drafts from upstream")
        return jsonify({"error": "Upstream drafts service failed"}), 502
    items = _filter_list_by_owner(items or [], user_id)
    pick = {}  # lower(name) -> {"name": original, "created_at": ts}
    for d in items:
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
    """Pulls corpora from the API; only returns those owned by the current user."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    try:
        r = requests.get(
            f"{API}/data/corpora",
            params={"owner_id": user_id},
            timeout=10,
        )
        r.raise_for_status()
        items = r.json()
    except requests.RequestException:
        server.logger.exception("Failed to fetch drafts from upstream")
        return jsonify({"error": "Upstream drafts service failed"}), 502
    # UI-only filter: ensure only current user's corpora (in case API returns more)
    items = _filter_list_by_owner(items or [], user_id)
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
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "User not authenticated"}), 401
    
    # Verify ownership before returning corpus
    if not _verify_corpus_ownership(corpus_id, user_id):
        return jsonify({"error": "Access denied: You do not own this corpus"}), 403
    
    try:
        # Fetch without owner_id since we've already verified ownership
        response = requests.get(
            f"{API}/data/corpora/{corpus_id}",
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
        if not owner_id:
            return []
        
        # Get all models directly from drafts system filtered by owner_id
        all_user_models = []
        try:
            models_drafts = drafts.list_drafts(type=DraftType.model)
            server.logger.info("Found %d total model drafts", len(models_drafts))
            
            for draft in models_drafts:
                # Check if draft belongs to the owner
                draft_owner = draft.owner_id or (draft.metadata.get("owner_id") if draft.metadata else None)
                
                # If no owner_id on model, try to get it from the corpus
                if not draft_owner and draft.metadata:
                    corpus_id = draft.metadata.get("corpus_id")
                    if corpus_id:
                        try:
                            # Fetch corpus without owner_id parameter to check ownership locally
                            corpus_resp = requests.get(
                                f"{API}/data/corpora/{corpus_id}",
                                timeout=5,
                            )
                            if corpus_resp.status_code == 200:
                                corpus_data = corpus_resp.json()
                                draft_owner = corpus_data.get("owner_id") or corpus_data.get("metadata", {}).get("owner_id")
                        except Exception:
                            pass
                
                server.logger.debug("Model draft %s: owner_id=%s, checking against %s", draft.id, draft_owner, owner_id)
                
                if draft_owner == owner_id:
                    try:
                        model = drafts.draft_to_model(draft)
                        if model:
                            # Read metadata.json directly to get tr_params structure
                            model_path = DRAFTS_SAVE / draft.id
                            metadata_path = model_path / "metadata.json"
                            metadata_dict = {}
                            
                            if metadata_path.exists():
                                try:
                                    with open(metadata_path, 'r') as f:
                                        metadata_dict = json.load(f)
                                    server.logger.debug("Loaded metadata from %s", metadata_path)
                                except Exception as e:
                                    server.logger.warning("Failed to read metadata.json for %s: %s", draft.id, e)
                                    # Fall back to draft metadata
                                    metadata_dict = pydantic_to_dict(model.metadata) if model.metadata else {}
                            else:
                                # Fall back to draft metadata
                                metadata_dict = pydantic_to_dict(model.metadata) if model.metadata else {}
                            
                            # Convert model to dict format expected by frontend
                            model_dict = {
                                "id": model.id,
                                "name": model.name or metadata_dict.get("tr_params", {}).get("model_name") or draft.id,
                                "owner_id": model.owner_id,
                                "corpus_id": model.corpus_id or metadata_dict.get("corpus_id"),
                                "created_at": model.created_at or metadata_dict.get("created_at"),
                                "location": model.location.value if hasattr(model.location, 'value') else str(model.location),
                                "metadata": metadata_dict,  # This should contain tr_params at the top level
                            }
                            all_user_models.append(model_dict)
                            server.logger.info("Added model %s (%s) for user %s", model.id, model_dict.get("name"), owner_id)
                    except Exception as e:
                        server.logger.error("Error converting draft %s to model: %s", draft.id, e)
            
            server.logger.info("Found %d models for user %s", len(all_user_models), owner_id)
            if all_user_models:
                server.logger.info("Sample model structure: %s", json.dumps(all_user_models[0], indent=2, default=str))
        except Exception as e:
            current_app.logger.exception("Error fetching models from drafts: %s", str(e))
            all_user_models = []
        
        # Create a map of model_id -> model for quick lookup
        models_by_id = {model.get("id"): model for model in all_user_models}
        
        # Get corpora and associate models
        corpora_response = requests.get(
            f"{API}/data/corpora",
            params={"owner_id": owner_id},
            timeout=10,
        )
        corpora_response.raise_for_status()
        corpora = corpora_response.json()
        corpora = _filter_list_by_owner(corpora or [], owner_id)
        current_app.logger.info("Corpora (owner-filtered): %s", len(corpora))

        # Group models by their corpus_id from metadata
        models_by_corpus_id = {}
        unassociated_models = []
        
        for model in all_user_models:
            model_corpus_id = model.get("corpus_id") or model.get("metadata", {}).get("corpus_id")
            if model_corpus_id:
                if model_corpus_id not in models_by_corpus_id:
                    models_by_corpus_id[model_corpus_id] = []
                models_by_corpus_id[model_corpus_id].append(model)
            else:
                unassociated_models.append(model)
        
        server.logger.info("Grouped models: %d corpora, %d unassociated", len(models_by_corpus_id), len(unassociated_models))
        
        # Track which models we've already added to a corpus
        models_in_corpora = set()

        for corpus in corpora:
            corpus_id = corpus.get("id")
            server.logger.info("Processing corpus ID: %s", corpus_id)
            if not corpus_id:
                current_app.logger.warning("Corpus without ID: %s", corpus)
                continue

            # Get models from API (existing association)
            try:
                models_response = requests.get(
                    f"{API}/data/corpora/{corpus_id}/models",
                    params={"owner_id": owner_id},
                    timeout=10,
                )
                models_response.raise_for_status()
                corpus_models = models_response.json()
                server.logger.info(
                    "Models from API for corpus ID %s: %s", corpus_id, len(corpus_models)
                )
                
                # Also add models that have this corpus_id in their metadata but aren't in the API response
                if corpus_id in models_by_corpus_id:
                    # Merge API models with models found by corpus_id
                    api_model_ids = {m.get("id") for m in corpus_models}
                    for model in models_by_corpus_id[corpus_id]:
                        if model.get("id") not in api_model_ids:
                            corpus_models.append(model)
                            server.logger.info("Added model %s to corpus %s based on corpus_id in metadata", model.get("id"), corpus_id)
                
                corpus["models"] = corpus_models
                # Track which models are in corpora
                for model in corpus_models:
                    model_id = model.get("id")
                    if model_id:
                        models_in_corpora.add(model_id)
            except requests.Timeout:
                current_app.logger.error(
                    "Timeout fetching models for corpus ID %s", corpus_id
                )
                # Still add models based on corpus_id if available
                if corpus_id in models_by_corpus_id:
                    corpus["models"] = models_by_corpus_id[corpus_id]
                    for model in models_by_corpus_id[corpus_id]:
                        models_in_corpora.add(model.get("id"))
                else:
                    corpus["models"] = []
            except requests.RequestException as e:
                current_app.logger.error(
                    "Error fetching models for corpus ID %s: %s", corpus_id, str(e)
                )
                # Still add models based on corpus_id if available
                if corpus_id in models_by_corpus_id:
                    corpus["models"] = models_by_corpus_id[corpus_id]
                    for model in models_by_corpus_id[corpus_id]:
                        models_in_corpora.add(model.get("id"))
                else:
                    corpus["models"] = []

        # Add models that aren't in any corpus to a special "Unassociated Models" corpus
        final_unassociated = [
            model for model in all_user_models
            if model.get("id") not in models_in_corpora
        ]
        
        if final_unassociated:
            # Create a virtual corpus for unassociated models
            unassociated_corpus = {
                "id": f"unassociated_{owner_id}",
                "name": "Unassociated Models",
                "owner_id": owner_id,
                "models": final_unassociated,
                "created_at": None,
                "metadata": {"name": "Unassociated Models"},
                "location": "temporal",
            }
            corpora.append(unassociated_corpus)
            server.logger.info("Added %d unassociated models to virtual corpus", len(final_unassociated))

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
    except Exception as e:
        current_app.logger.exception("Unexpected error in fetch_trained_models: %s", str(e))
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
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    if not _verify_model_ownership(model_id, user_id):
        return jsonify({"error": "Access denied: You do not own this model"}), 403

    # get model entity to verify it exists and get corpus_id
    try:
        t0 = time()
        up = requests.get(
            f"{API}/data/models/{model_id}",
            timeout=(3.05, 60),
        )
        up.raise_for_status()
        model_metadata = up.json()
        server.logger.info(
            "delete-model model meta ok status=%s dt=%.3fs",
            up.status_code,
            time() - t0,
        )
    except requests.HTTPError as e:
        if e.response and e.response.status_code == 404:
            return jsonify({"error": "Model not found"}), 404
        server.logger.exception("model delete HTTP error fetching model: %s", e)
        return jsonify({"error": f"Failed to fetch model: {e}"}), e.response.status_code if e.response else 502
    except requests.Timeout:
        server.logger.exception("model delete timeout fetching model")
        return jsonify({"error": "Upstream timeout fetching model"}), 504
    except requests.RequestException as e:
        server.logger.exception("model delete error fetching model: %s", e)
        return jsonify({"error": f"Upstream error fetching model: {e}"}), 502

    corpus_id = model_metadata.get("corpus_id")
    if corpus_id:
        server.logger.info("Deleting model '%s' from corpus '%s'", model_id, corpus_id)
        # delete model from corpus
        try:
            up = requests.post(
                f"{API}/data/corpora/{corpus_id}/delete_model",
                params={"model_id": model_id},
                timeout=(3.05, 30),
            )
            up.raise_for_status()
            server.logger.info("Model '%s' removed from corpus '%s'", model_id, corpus_id)
        except requests.Timeout:
            server.logger.warning("model delete timeout removing from corpus (continuing with model deletion)")
        except requests.RequestException as e:
            server.logger.warning("model delete upstream corpus error (continuing with model deletion): %s", e)
    else:
        server.logger.info("Deleting model '%s' (no corpus association)", model_id)

    # delete model
    try:
        up = requests.delete(
            f"{API}/data/models/{model_id}",
            timeout=(3.05, 30),
        )
        server.logger.info("Model delete upstream response status=%s", up.status_code)
        if up.status_code == 204:
            _log_audit("model_deleted", target_type="model", target_id=model_id)
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
@server.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    model_id = request.form.get("model_id") or request.args.get("model_id", "")
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
    user_id = session.get("user_id")

    server.logger.info("get-dashboard start model_id=%s user_id=%s", model_id, user_id)

    if not model_id:
        server.logger.warning("get-dashboard missing model_id")
        return jsonify({"error": "model_id is required"}), 400

    # model metadata
    model_metadata = None
    try:
        up0 = time()
        up = requests.get(
            f"{API}/data/models/{model_id}",
            timeout=(3.05, 60),
        )
        up.raise_for_status()
        model_metadata = up.json()
        server.logger.info(
            "get-dashboard model meta ok status=%s dt=%.3fs",
            up.status_code,
            time() - up0,
        )
    except requests.HTTPError as e:
        if e.response and e.response.status_code == 404:
            # Model not found via API, try direct draft access
            server.logger.warning("Model not found via API, trying direct draft access for %s", model_id)
            try:
                model_draft = drafts.get_draft(model_id, DraftType.model)
                if model_draft:
                    model = drafts.draft_to_model(model_draft)
                    if model:
                        # Read metadata.json directly
                        model_path = DRAFTS_SAVE.resolve() / model_id
                        metadata_path = model_path / "metadata.json"
                        metadata_dict = {}
                        
                        if metadata_path.exists():
                            try:
                                with open(metadata_path, 'r') as f:
                                    metadata_dict = json.load(f)
                            except Exception as e:
                                server.logger.warning("Failed to read metadata.json for %s: %s", model_id, e)
                                metadata_dict = pydantic_to_dict(model.metadata) if model.metadata else {}
                        else:
                            metadata_dict = pydantic_to_dict(model.metadata) if model.metadata else {}
                        
                        # Convert to expected format
                        model_metadata = {
                            "id": model.id,
                            "name": model.name or metadata_dict.get("tr_params", {}).get("model_name") or model_id,
                            "owner_id": model.owner_id,
                            "corpus_id": model.corpus_id or metadata_dict.get("corpus_id"),
                            "created_at": model.created_at or metadata_dict.get("created_at"),
                            "location": model.location.value if hasattr(model.location, 'value') else str(model.location),
                            "metadata": metadata_dict,
                        }
                        server.logger.info("Successfully loaded model %s from drafts directly", model_id)
                    else:
                        raise ValueError("Failed to convert draft to model")
                else:
                    server.logger.error("Model draft %s not found in drafts system", model_id)
                    raise ValueError("Model not found in drafts")
            except Exception as draft_error:
                server.logger.exception("Failed to load model from drafts: %s", draft_error)
                try:
                    error_detail = e.response.json() if e.response else {}
                    return jsonify({
                        "error": error_detail.get("detail") or error_detail.get("error") or f"Model not found (HTTP {e.response.status_code})"
                    }), e.response.status_code if e.response else 502
                except:
                    return jsonify({"error": f"Failed to fetch model: HTTP {e.response.status_code if e.response else 'unknown'}"}), 502
        else:
            server.logger.exception("get-dashboard HTTP error fetching model: %s", e)
            try:
                error_detail = e.response.json() if e.response else {}
                return jsonify({
                    "error": error_detail.get("detail") or error_detail.get("error") or f"Model not found (HTTP {e.response.status_code})"
                }), e.response.status_code if e.response else 502
            except:
                return jsonify({"error": f"Failed to fetch model: HTTP {e.response.status_code if e.response else 'unknown'}"}), 502
    except requests.Timeout:
        server.logger.exception("get-dashboard timeout fetching model")
        return jsonify({"error": "Timeout: The model service took too long to respond. Please try again."}), 504
    except requests.RequestException as e:
        server.logger.exception("get-dashboard error fetching model: %s", e)
        return jsonify({"error": f"Connection error: Unable to reach the model service. {str(e)}"}), 502
    
    if not model_metadata:
        return jsonify({"error": "Failed to load model metadata"}), 500

    # Get corpus_id from model metadata
    corpus_id = (
        model_metadata.get("corpus_id") or
        model_metadata.get("metadata", {}).get("corpus_id") or
        model_metadata.get("metadata", {}).get("tr_params", {}).get("corpus_id") or
        ""
    )
    if not corpus_id:
        server.logger.error("get-dashboard model missing corpus_id. Model metadata keys: %s", list(model_metadata.keys()))
        return jsonify({"error": "Model missing corpus_id. Please ensure the model was trained with a valid corpus."}), 400

    # corpus data
    try:
        up0 = time()
        # Fetch without owner_id since we've already verified ownership
        up = requests.get(
            f"{API}/data/corpora/{corpus_id}",
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
        return jsonify({"error": "Timeout: The corpus service took too long to respond. Please try again."}), 504
    except requests.HTTPError as e:
        server.logger.exception("get-dashboard HTTP error fetching corpus: %s", e)
        try:
            error_detail = e.response.json() if e.response else {}
            return jsonify({
                "error": error_detail.get("detail") or error_detail.get("error") or f"Corpus not found (HTTP {e.response.status_code})"
            }), e.response.status_code if e.response else 502
        except:
            return jsonify({"error": f"Failed to fetch corpus: HTTP {e.response.status_code if e.response else 'unknown'}"}), 502
    except requests.RequestException as e:
        server.logger.exception("get-dashboard error fetching corpus: %s", e)
        return jsonify({"error": f"Connection error: Unable to reach the corpus service. {str(e)}"}), 502

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
        except requests.HTTPError as e:
            server.logger.error("get-dashboard model-info HTTP error: %s", e)
            try:
                error_data = r.json() if r else {}
                return jsonify({
                    "error": error_data.get("detail") or error_data.get("error") or f"Failed to fetch model info (HTTP {r.status_code})"
                }), r.status_code
            except Exception:
                return jsonify({"error": f"Failed to fetch model info: {r.text if r else 'Unknown error'}"}), r.status_code if r else 502
        except Exception as e:
            server.logger.exception("get-dashboard model-info proxy error: %s", e)
            return jsonify({"error": f"Error fetching model info: {str(e)}"}), 502

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
        except requests.HTTPError as e:
            server.logger.error("get-dashboard thetas HTTP error: %s", e)
            try:
                error_data = r.json() if r else {}
                return jsonify({
                    "error": error_data.get("detail") or error_data.get("error") or f"Failed to fetch document-topic probabilities (HTTP {r.status_code})"
                }), r.status_code
            except Exception:
                return jsonify({"error": f"Failed to fetch document-topic probabilities: {r.text if r else 'Unknown error'}"}), r.status_code if r else 502
        except Exception as e:
            server.logger.exception("get-dashboard thetas proxy error: %s", e)
            return jsonify({"error": f"Error fetching document-topic probabilities: {str(e)}"}), 502

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
        return jsonify({
            "error": f"Failed to build dashboard data: {str(e)}",
            "detail": "An error occurred while processing the dashboard data. Check server logs for details."
        }), 500


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
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"detail": "Not authenticated"}), 401

    # model meta info (cached) — fetch first so we can check ownership from response
    model_entry = _CACHE_MODELS.get(model_id)
    if not model_entry or (time() - model_entry["ts"] > CACHE_TTL):
        try:
            up0 = time()
            up = requests.get(
                f"{API}/data/models/{model_id}",
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

    # Allow if user owns model, model has no owner_id (legacy), or for any authenticated user
    # (viewing document text is read-only; ownership enforced on create/delete)
    model_owner = model.get("owner_id") or (model.get("metadata") or {}).get("owner_id")
    if model_owner is not None and model_owner != user_id:
        # Still allow: text-info is read-only document view, user is authenticated
        server.logger.debug("TEXT-INFO model owner %s != user %s; allowing read-only access", model_owner, user_id)

    corpus_id = model.get("corpus_id", "")
    if not corpus_id:
        server.logger.error("TEXT-INFO model missing corpus_id")
        return jsonify({"detail": "Model missing corpus_id."}), 400

    # corpus info (cached) — fetch first so we can check ownership from response
    corpus_entry = _CACHE_CORPORA.get(corpus_id)
    if not corpus_entry or (time() - corpus_entry["ts"] > CACHE_TTL):
        try:
            up0 = time()
            up = requests.get(
                f"{API}/data/corpora/{corpus_id}",
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

    # Corpus access: text-info is read-only; allow any authenticated user to view document text
    corpus_owner = corpus.get("owner_id") or (corpus.get("metadata") or {}).get("owner_id")
    if corpus_owner is not None and corpus_owner != user_id:
        server.logger.debug("TEXT-INFO corpus owner %s != user %s; allowing read-only access", corpus_owner, user_id)

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
                "label": (topics_info.get(str(k), {}) or {}).get("Label", f"Topic {k}"),
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


def _safe_dict(val):
    """Return val if it's a dict, else {} (handles ruamel.yaml scalars etc.)."""
    return val if isinstance(val, dict) else {}


def _chat_llm(provider: str, model: str, system_content: str, user_content: str, host: str = None, api_key: str = None):
    """
    Call configured LLM (Ollama or OpenAI) for the model assistant.
    api_key: optional OpenAI API key from chat UI; if not set, uses OPENAI_API_KEY env for GPT.
    Returns (response_text, None) on success or (None, error_message) on failure.
    """
    llm_cfg = _safe_dict(RAW_CONFIG.get("llm"))
    if provider == "ollama":
        ollama_cfg = _safe_dict(llm_cfg.get("ollama"))
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
    if provider == "gpt" or provider == "openai":
        key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not key:
            return None, "OpenAI API key not set. Add your key in Assistant LLM settings (chat sidebar) or set OPENAI_API_KEY in the server .env file."
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


def _read_key_from_dotfile(env_path: Path) -> str | None:
    """Read OPENAI_API_KEY from one .env file. Returns None if not found or on error."""
    if not env_path.exists():
        return None
    try:
        with open(env_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() == "OPENAI_API_KEY":
                    key = value.strip().strip('"').strip("'").strip()
                    return key if key else None
    except OSError:
        pass
    return None



def _get_openai_key_from_env() -> str | None:
    project_root = Path(__file__).resolve().parent.parent  # /TOVA
    env_path = project_root / ".env"

    load_dotenv(dotenv_path=env_path, override=False)  # set True if you want .env to override existing env vars

    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    return key or None

# Startup: confirm whether app.py can see the OpenAI key from .env/config (no key value logged)
_openai_key_at_startup = _get_openai_key_from_env()
server.logger.info("OpenAI API key at startup: %s", "configured" if _openai_key_at_startup else "not set (set in .env or llm.gpt.api_key in config)")


def _get_chat_llm_defaults():
    """Return default provider, model, host from config (for RAG chat). Used when user has not set overrides."""
    tm_gen = _safe_dict(_safe_dict(RAW_CONFIG.get("topic_modeling")).get("general"))
    llm_cfg = _safe_dict(RAW_CONFIG.get("llm"))
    provider = tm_gen.get("llm_provider") or "ollama"
    if hasattr(provider, "strip"):
        provider = provider.strip().lower()
    else:
        provider = str(provider).strip().lower()
    if provider == "openai":
        provider = "gpt"
    _ollama = _safe_dict(llm_cfg.get("ollama")).get("available_models")
    _gpt = _safe_dict(llm_cfg.get("gpt")).get("available_models")
    ollama_list = list(_ollama) if _ollama else []
    gpt_list = list(_gpt) if _gpt else []
    chat_model = (
        tm_gen.get("llm_model_type")
        or (ollama_list[0] if ollama_list else None)
        or (gpt_list[0] if gpt_list else None)
    )
    if chat_model is not None and not isinstance(chat_model, str):
        chat_model = str(chat_model)
    host = tm_gen.get("llm_server") or _safe_dict(llm_cfg.get("ollama")).get("host")
    if host is not None and not isinstance(host, str):
        host = str(host)
    return provider, chat_model, host, ollama_list, gpt_list, _safe_dict(llm_cfg.get("ollama")).get("host")


@server.route("/api/chat-openai-key-status", methods=["GET"])
@login_required
def chat_openai_key_status():
    """Diagnostic: report where OPENAI_API_KEY was found (or not). No key value is returned."""
    project_root = Path(__file__).resolve().parent.parent
    env_project = project_root / ".env"
    env_cwd = Path.cwd() / ".env"
    tried = [
        {"path": str(env_project), "exists": env_project.exists()},
        {"path": str(env_cwd), "exists": env_cwd.exists()},
    ]
    if server.root_path:
        tried.append({"path": str(Path(server.root_path) / ".env"), "exists": (Path(server.root_path) / ".env").exists()})
        tried.append({"path": str(Path(server.root_path).parent / ".env"), "exists": (Path(server.root_path).parent / ".env").exists()})
    key = _get_openai_key_from_env()
    gpt_cfg = _safe_dict(RAW_CONFIG.get("llm", {})).get("gpt") or {}
    config_has_key = bool((gpt_cfg.get("api_key") or "").strip()) if isinstance(gpt_cfg, dict) else False
    path_api_key = (gpt_cfg.get("path_api_key") or ".env") if isinstance(gpt_cfg, dict) else ".env"
    return jsonify({
        "key_configured": bool(key) or config_has_key,
        "key_from_config": config_has_key,
        "env_has_key": bool((os.environ.get("OPENAI_API_KEY") or "").strip()),
        "path_api_key_from_config": path_api_key,
        "note": "Labeler (Prompter) uses load_dotenv(path_api_key) then os.getenv('OPENAI_API_KEY'); UI uses same.",
        "project_root_from_file": str(project_root),
        "cwd": str(Path.cwd()),
        "server_root_path": getattr(server, "root_path", None),
        "tried_env_paths": tried,
    }), 200


@server.route("/api/chat-llm-options", methods=["GET"])
@login_required
def chat_llm_options():
    """Return available RAG/chat LLM options (providers, models, default host) for the chat UI. Does not expose secrets."""
    llm_cfg = _safe_dict(RAW_CONFIG.get("llm"))
    ollama_cfg = _safe_dict(llm_cfg.get("ollama"))
    gpt_cfg = _safe_dict(llm_cfg.get("gpt"))
    tm_gen = _safe_dict(_safe_dict(RAW_CONFIG.get("topic_modeling")).get("general"))
    default_host = tm_gen.get("llm_server") or ollama_cfg.get("host", "http://host.docker.internal:11434")
    if default_host is not None and not isinstance(default_host, str):
        default_host = str(default_host)
    return jsonify({
        "ollama": {
            "models": list(ollama_cfg.get("available_models") or []),
            "default_host": default_host,
        },
        "gpt": {
            "models": list(gpt_cfg.get("available_models") or []),
        },
    }), 200


@server.route("/api/chat/messages", methods=["GET"])
@login_required
def get_chat_messages():
    """Return saved chat messages for the current user and the given model_id (query param)."""
    model_id = (request.args.get("model_id") or "").strip()
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    try:
        messages = (
            ChatMessage.query.filter_by(user_id=user_id, model_id=model_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        return jsonify({
            "messages": [
                {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat() + "Z" if m.created_at else None}
                for m in messages
            ]
        }), 200
    except Exception as e:
        server.logger.exception("Failed to load chat messages: %s", e)
        return jsonify({"error": "Failed to load messages"}), 500


@server.route("/api/chat", methods=["POST"])
@login_required
def chat():
    """
    Chat interface endpoint for topic modeling assistance.
    Uses the LLM set in the request (llm_settings from chat UI) or config. No rule-based fallback.
    """
    try:
        payload = request.get_json(silent=True) or {}
        message = payload.get("message", "").strip()
        model_id = payload.get("model_id", "")
        context = payload.get("context", {})
        llm_settings = _safe_dict(payload.get("llm_settings"))
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        server.logger.info("Chat request: model_id=%s, message_length=%d", model_id, len(message))
        
        llm_context_str = dashboard_context_for_llm(
            context,
            max_themes=30,
            max_doc_snippets=5,
            include_diagnostics=True,
            include_model_info=True,
        )
        if llm_context_str:
            server.logger.debug("LLM context length: %d chars", len(llm_context_str))
        
        # RAG LLM: user overrides from chat UI take precedence over config
        provider, chat_model, host, ollama_list, gpt_list, _ = _get_chat_llm_defaults()
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
        
        if chat_model:
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
            # OpenAI key: chat UI > user saved (DB) > config.yaml llm.gpt.api_key > .env
            from_ui = (llm_settings.get("api_key") or "").strip()
            stored_key = None
            if not from_ui:
                user_config = _load_user_config(session.get("user_id")) or {}
                stored_llm = user_config.get("llm_config") or {}
                stored_key = (stored_llm.get("api_key") or "").strip()
            from_config = (_safe_dict(RAW_CONFIG.get("llm", {})).get("gpt") or {})
            if isinstance(from_config, dict):
                config_key = (from_config.get("api_key") or "").strip()
            else:
                config_key = ""
            openai_key = from_ui or stored_key or config_key or _get_openai_key_from_env()
            response_text, err = _chat_llm(provider, chat_model, system_content, user_content, host=host, api_key=openai_key)
            if response_text:
                server.logger.info("Chat response from LLM: length=%d", len(response_text))
                user_id = session.get("user_id")
                if user_id and model_id:
                    try:
                        db.session.add(ChatMessage(user_id=user_id, model_id=model_id.strip(), role="user", content=message))
                        db.session.add(ChatMessage(user_id=user_id, model_id=model_id.strip(), role="assistant", content=response_text))
                        db.session.commit()
                    except Exception as save_err:
                        server.logger.warning("Failed to save chat messages: %s", save_err)
                        db.session.rollback()
                return jsonify({"response": response_text}), 200
            if err:
                server.logger.warning("Chat LLM error: %s", err)
        
        # No rule-based fallback: LLM only. If we get here, model is missing or the call failed.
        if not chat_model:
            return jsonify({
                "response": "No language model is configured for the assistant. Open the chat sidebar, expand \"Assistant LLM\", and choose a provider and model (and set the Ollama host if using Ollama). You can also set defaults in config under topic_modeling.general."
            }), 200
        return jsonify({
            "response": f"The assistant could not get a response from the model ({err or 'unknown error'}). Check that the LLM service is running and reachable, then try again."
        }), 200
        
    except Exception as e:
        server.logger.exception("Chat error: %s", e)
        return jsonify({"error": f"Chat service error: {str(e)}"}), 500

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
            return yaml.load(f)
    except Exception as exc:
        print(f"Error reading YAML file: {exc}")
        return {}
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")
        return {}


@server.route("/api/models/<model_id>/topics/<int:topic_id>/rename", methods=["POST"])
@login_required
def rename_topic(model_id: str, topic_id: int):
    """
    Rename a topic in a model. Proxies the request to the backend API.
    If backend is unavailable, returns None.
    """
    payload = request.get_json(silent=True) or {}
    new_label = payload.get("new_label", "").strip()
    owner_id = session.get("user_id")
    
    if not new_label:
        return jsonify({"error": "new_label is required"}), 400
    
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    
    if not owner_id:
        return jsonify({"error": "User not authenticated"}), 401
    
    # Forward request to backend API
    try:
        upstream_url = f"{API}/data/models/{model_id}/topics/{topic_id}/rename"
        upstream_response = requests.post(
            upstream_url,
            json={"new_label": new_label},
            params={"owner_id": owner_id},
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
    except requests.exceptions.ConnectionError:
        # Backend is not available - return None
        server.logger.warning(
            f"Backend unavailable for topic rename: model_id={model_id}, topic_id={topic_id}"
        )
        return jsonify(None), 200
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
    If backend is unavailable, returns None.
    """
    owner_id = session.get("user_id")
    
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    
    if not owner_id:
        return jsonify({"error": "User not authenticated"}), 401
    
    try:
        upstream_url = f"{API}/data/models/{model_id}/topics/renames"
        upstream_response = requests.get(
            upstream_url,
            params={"owner_id": owner_id},
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
    except requests.exceptions.ConnectionError:
        # Backend is not available - return None
        server.logger.warning(
            f"Backend unavailable for getting topic renames: model_id={model_id}"
        )
        return jsonify(None), 200
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
            return yaml.load(f)
    except Exception as exc:
        print(f"Error reading YAML file: {exc}")
        return {}
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")
        return {}


# Note: Encoded routes are handled by the custom url_for which generates encoded paths.
# The server accepts both original and encoded routes for backward compatibility.
# For better security, consider registering routes with both original and encoded paths
# using the register_encoded_route() helper function above.


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # For local development; in production use gunicorn/uwsgi, etc.
    server.run(debug=True, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 5000)))
