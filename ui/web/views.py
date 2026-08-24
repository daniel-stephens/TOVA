"""TOVA UI views (Django)."""
# flake8: noqa
from __future__ import annotations

import json
import logging
import re
import os
import sys
from copy import deepcopy
from pathlib import Path
from time import monotonic, sleep, time
from urllib.parse import urlencode
import requests

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(*args, **kwargs):
        return None


from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.core.exceptions import RequestDataTooBig
from django.core.paginator import Paginator
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from ruamel.yaml import YAML

from dashboard_utils import (
    DASHBOARD_COLOR_CFG,
    build_theme_color_map,
    dashboard_context_for_llm,
    pydantic_to_dict,
    to_dashboard_bundle,
)
from tova.api.models.data_schemas import DraftType
from tova.core import drafts
from tova.core import models as models_module

from web import oauth_okta
from web.admin_emails import admin_emails_from_env, promote_env_listed_user_to_staff
from web.authz import get_tova_user_session_dict
from web.decorators import block_study_participants, require_study_feature, require_tova_app_admin
from web.models import (
    AuditLog,
    ChatMessage,
    InteractionEvent,
    ParticipantSession,
    StudyResponse,
    TopicRename,
    User,
    UserConfig,
)
from web.study_features import FEATURE_ACTIVE_LEARNING, get_study_model_id, is_study_participant, post_login_redirect_url, study_participant_may_access_model, user_has_feature
from web.services import runtime as R
from web import study_flow

logger = logging.getLogger(__name__)


def _get_json(request):
    if not request.body:
        return {}
    try:
        return json.loads(request.body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def _request_user_id(request) -> str | None:
    user = getattr(request, "user", None)
    if user is not None and user.is_authenticated:
        return str(user.pk)
    return None


def _safe_redirect_target(*candidates) -> str | None:
    """Allow only same-site relative paths (avoid open redirects)."""
    for c in candidates:
        if not c:
            continue
        s = str(c).strip()
        if s.startswith("/") and not s.startswith("//"):
            return s
    return None


def _session_permanent(request, days: int = 7):
    request.session.set_expiry(60 * 60 * 24 * days)


def _log_audit(request, action: str, target_type: str = None, target_id: str = None, details: str = None):
    import logging

    log = logging.getLogger(__name__)
    try:
        actor = get_tova_user_session_dict(request) or {}
        entry = AuditLog(
            actor_id=actor.get("id"),
            action=action,
            target_type=target_type,
            target_id=target_id,
            details=details[:1024] if details else None,
        )
        entry.save()
    except Exception as e:
        log.warning("Audit log failed: %s", e)


def get_effective_user_config(request, user_id: str) -> dict:
    cache = getattr(request, "_effective_user_config_cache", None)
    if cache is None:
        cache = {}
        setattr(request, "_effective_user_config_cache", cache)
    if user_id in cache:
        return cache[user_id]
    base = deepcopy(R.DEFAULT_CONFIG)
    if not user_id:
        cache[user_id] = base
        return base
    entry = UserConfig.objects.filter(user_id=user_id).first()
    overrides = entry.config if entry else None
    if overrides:
        oc = {k: v for k, v in overrides.items() if k != "llm_config"}
        merged = R._deep_merge(base, oc if isinstance(oc, dict) else {})
        cache[user_id] = merged
        return merged
    cache[user_id] = base
    return base


def _load_user_config(user_id: str):
    if not user_id:
        return None
    entry = UserConfig.objects.filter(user_id=user_id).first()
    return entry.config if entry else None


def _save_user_config_overrides(user_id: str, overrides: dict):
    if not user_id:
        return
    sanitized = R._json_safe(overrides or {})
    entry = UserConfig.objects.filter(user_id=user_id).first()
    from django.utils import timezone
    if entry:
        entry.config = sanitized
        entry.updated_at = timezone.now()
        entry.save()
    else:
        UserConfig.objects.create(user_id=user_id, config=sanitized)


def _reset_user_config(user_id: str):
    if not user_id:
        return
    UserConfig.objects.filter(user_id=user_id).delete()


# ------------------------------------------------------------------------------
# Auth routes (local username/password + optional Okta)
# ------------------------------------------------------------------------------
def _default_post_auth_redirect(user, *next_candidates):
    """Resolve redirect after login/signup, honoring safe ``next`` when provided."""
    dest = _safe_redirect_target(*next_candidates)
    if dest:
        return dest
    return post_login_redirect_url(user)


def _study_model_access_denied(request, model_id: str, *, json_response: bool = False):
    if study_participant_may_access_model(request.user, model_id):
        return None
    if json_response:
        return JsonResponse({"error": "Access denied: model is not assigned to your study group."}, status=403)
    messages.error(request, "You can only access your assigned study model.")
    return redirect(post_login_redirect_url(request.user))


def signup(request):
    if not getattr(settings, "TOVA_ALLOW_SIGNUP", True):
        messages.error(request, "Account registration is closed.")
        return redirect(reverse("web:login"))

    if request.user.is_authenticated:
        return redirect(_default_post_auth_redirect(request.user))
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password", "")
        password_confirm = request.POST.get("password_confirm", "")

        # Basic validation
        if not name or not email or not password:
            messages.error(request, "Please fill in all required fields.")
            return redirect(reverse("web:signup"))

        if password != password_confirm:
            messages.error(request, "Passwords do not match.")
            return redirect(reverse("web:signup"))

        existing = User.objects.filter(email=email).first()
        if existing:
            messages.warning(request, "An account with that email already exists. Please sign in.")
            return redirect(reverse("web:login"))

        # Create user
        user = User(name=name or email, email=email)
        user.set_password(password)
        user.save()

        _log_audit(request, "user_created", target_type="user", target_id=user.id, details=user.email)

        auth_login(request, user)
        _session_permanent(request)
        promote_env_listed_user_to_staff(user)

        messages.success(request, "Account created! Welcome to TOVA.")
        return redirect(_default_post_auth_redirect(user))

    return render(request, "signup.html")


def login(request):
    return _login(request, template="login.html", url_name="web:login")


def study_login(request):
    """Sign-in for study participants.

    Deliberately the same page for every condition -- a separate door per arm
    would signal to participants which one they are in before they start.
    It differs from the main login only in dropping signup and SSO, neither of
    which applies to an account the research team created.
    """
    return _login(request, template="study/login.html", url_name="web:study_login")


def _login(request, *, template: str, url_name: str):
    if request.user.is_authenticated:
        dest = _safe_redirect_target(request.GET.get("next"))
        return redirect(dest or post_login_redirect_url(request.user))
    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password", "")

        user = authenticate(request, email=email, password=password)
        if user is None:
            messages.error(request, "Invalid email or password.")
            next_param = _safe_redirect_target(
                request.POST.get("next"),
                request.GET.get("next"),
            )
            if next_param:
                return redirect(f"{reverse(url_name)}?{urlencode({'next': next_param})}")
            return redirect(reverse(url_name))

        auth_login(request, user)
        _session_permanent(request)
        promote_env_listed_user_to_staff(user)
        _log_audit(request, "login", target_type="user", target_id=str(user.pk), details=user.email)

        if not is_study_participant(user):
            # Participants have no message block on their screens; queueing here
            # would just leave it unread until sign-out.
            messages.success(request, "Signed in successfully.")

        dest = _default_post_auth_redirect(
            user,
            request.POST.get("next"),
            request.GET.get("next"),
        )
        return redirect(dest)

    return render(request, template)


def login_okta(request):
    # Optionally remember where the user was trying to go
    next_url = request.GET.get("next")
    if next_url:
        request.session["next_url"] = next_url

    okta_client = oauth_okta.get_client()
    if okta_client is None:
        messages.warning(request, "Okta sign-in is not configured.")
        return redirect(reverse("web:login"))

    redirect_uri = request.build_absolute_uri(reverse("web:auth_okta_callback"))
    return okta_client.authorize_redirect(request, redirect_uri)


def auth_okta_callback(request):
    okta_client = oauth_okta.get_client()
    if okta_client is None:
        messages.warning(request, "Okta sign-in is not configured.")
        return redirect(reverse("web:login"))

    # Exchange authorization code for tokens
    try:
        token = okta_client.authorize_access_token(request)
        userinfo = token.get("userinfo") or {}
        if not userinfo:
            try:
                userinfo = okta_client.parse_id_token(request, token) or {}
            except Exception:
                userinfo = {}
    except Exception as e:
        logger.exception("Okta auth callback failed: %s", e)
        messages.error(request, "Failed to sign in with Okta.")
        return redirect(reverse("web:login"))

    email = (userinfo.get("email") or "").lower()
    okta_sub = userinfo.get("sub")
    name = userinfo.get("name") or email

    if not email:
        messages.error(request, "Okta did not provide an email address.")
        return redirect(reverse("web:login"))

    # Find or create user in the database
    user = User.objects.filter(email=email).first()
    is_new_user = False
    if not user:
        is_new_user = True
        user = User(name=name or email, email=email, auth_source="okta")
        user.set_unusable_password()
        user.save()
    else:
        user.auth_source = "okta"
        user.save(update_fields=["auth_source"])

    auth_login(request, user)
    _session_permanent(request)
    promote_env_listed_user_to_staff(user)
    _log_audit(request, "login", target_type="user", target_id=str(user.pk), details=user.email)

    messages.success(request, "Signed in with Okta.")

    next_url = request.session.pop("next_url", None)
    return redirect(next_url or post_login_redirect_url(user))


def logout(request):
    # Send participants back to the study door, not the main app's login, so a
    # shared machine is ready for the next person in the session.
    was_participant = is_study_participant(request.user)
    auth_logout(request)

    # Discard anything queued but never displayed. The study screens do not
    # render the message block, so without this a stale "Signed in
    # successfully." resurfaces at sign-out. Session flush is not enough:
    # Django's FallbackStorage writes messages to a cookie first.
    messages.get_messages(request).used = True

    messages.info(request, "You have been signed out.")
    return redirect(reverse("web:study_login" if was_participant else "web:login"))


# ------------------------------------------------------------------------------
# Admin (superuser) page
# ------------------------------------------------------------------------------
@login_required
@require_tova_app_admin(json_response=False)
def admin_page(request):
    """Admin entry: redirect to the TOVA console inside Django admin."""
    # The legacy Flask-like "staff tools" console is removed from the UI.
    # Instead, surface the corpora + dashboards directly inside Django admin.
    return redirect(reverse("tova_admin_corpora"))


@login_required
@require_tova_app_admin(
    forbidden_payload={"success": False, "message": "Forbidden"},
)
def admin_toggle_user(request, user_id):
    """Toggle is_admin for a user (only admins). Cannot remove your own admin if you're the last admin."""
    target = User.objects.filter(pk=user_id).first()
    if not target:
        return JsonResponse({"success": False, "message": "User not found"}, status=404)
    admin_emails = admin_emails_from_env()
    db_admin_count = User.objects.filter(is_admin=True).count()
    # If we're demoting ourselves and we're the only DB admin (and not in env list), forbid
    if (
        str(target.pk) == str(request.tova_user["id"])
        and getattr(target, "is_admin", False)
        and db_admin_count <= 1
        and target.email.lower() not in admin_emails
    ):
        return JsonResponse({"success": False, "message": "Cannot remove the last admin."}, status=400)
    target.is_admin = not getattr(target, "is_admin", False)
    
    action = "admin_revoked" if not target.is_admin else "admin_promoted"
    _log_audit(
        request,
        action,
        target_type="user",
        target_id=str(target.pk),
        details=target.email,
    )
    return JsonResponse({"success": True, "is_admin": target.is_admin})


@login_required
@require_tova_app_admin(
    forbidden_payload={"success": False, "message": "Forbidden"},
)
def admin_create_user(request):
    """Create a new user (admin only). JSON: name, email, password."""
    payload = _get_json(request) or {}
    name = (payload.get("name") or "").strip()
    email = (payload.get("email") or "").strip().lower()
    password = (payload.get("password") or "").strip()
    if not email or not password:
        return JsonResponse({"success": False, "message": "Email and password are required"}, status=400)
    if User.objects.filter(email=email).first():
        return JsonResponse({"success": False, "message": "An account with that email already exists"}, status=400)
    user = User(name=name or email, email=email)
    user.set_password(password)
    user.save()

    _log_audit(
        request,
        "user_created",
        target_type="user",
        target_id=str(user.pk),
        details=user.email,
    )
    return JsonResponse(
        {"success": True, "user_id": str(user.pk), "email": user.email},
        status=201,
    )


@login_required
@require_tova_app_admin(
    forbidden_payload={"success": False, "message": "Forbidden"},
)
def admin_delete_user(request, user_id):
    """Delete a user (admin only). Cannot delete yourself."""
    if user_id == request.tova_user.get("id"):
        return JsonResponse({"success": False, "message": "Cannot delete your own account"}, status=400)
    target = User.objects.filter(pk=user_id).first()
    if not target:
        return JsonResponse({"success": False, "message": "User not found"}, status=404)
    email = target.email
    target.delete()
    
    _log_audit(request, "user_deleted", target_type="user", target_id=user_id, details=email)
    return JsonResponse({"success": True}, status=200)


@login_required
@require_tova_app_admin()
def admin_list_corpora(request):
    """List all corpora across all users (admin only)."""
    try:
        r = requests.get(f"{R.API}/data/corpora", timeout=15)
        r.raise_for_status()
        corpora = r.json()
    except requests.RequestException as e:
        logger.exception("Admin list corpora: %s", e)
        return JsonResponse({"error": str(e)}, status=502)
    user_ids = {c.get("owner_id") or (c.get("metadata") or {}).get("owner_id") for c in corpora if c.get("owner_id") or (c.get("metadata") or {}).get("owner_id")}
    users_by_id = {}
    for uid in user_ids:
        if uid:
            u = User.objects.filter(pk=uid).first()
            users_by_id[uid] = {"email": u.email if u else None, "name": (u.name if u else None) or ""}
    for c in corpora:
        oid = c.get("owner_id") or (c.get("metadata") or {}).get("owner_id")
        c["owner_email"] = users_by_id.get(oid, {}).get("email") if oid else None
        c["owner_name"] = users_by_id.get(oid, {}).get("name") if oid else None
    return JsonResponse(corpora, safe=False, status=200)


@login_required
@require_tova_app_admin()
def admin_delete_corpus(request, corpus_id):
    """Delete any corpus (admin only)."""
    try:
        r = requests.get(f"{R.API}/data/corpora/{corpus_id}", timeout=10)
        if r.status_code == 404:
            return JsonResponse({"error": "Corpus not found"}, status=404)
        r.raise_for_status()
        corpus = r.json()
    except requests.RequestException as e:
        return JsonResponse({"error": str(e)}, status=502)
    corpus_name = (corpus.get("metadata") or {}).get("name") or corpus_id
    models = corpus.get("models") or []
    for model_id in models:
        try:
            requests.delete(f"{R.API}/data/models/{model_id}", timeout=(3, 30))
        except Exception:
            pass
    try:
        del_resp = requests.delete(f"{R.API}/data/corpora/{corpus_id}", timeout=(3, 30))
        if del_resp.status_code == 204:
            _log_audit(request, "corpus_deleted", target_type="corpus", target_id=corpus_id, details=corpus_name)
            return JsonResponse({"success": True}, status=200)
        if del_resp.status_code == 404:
            return JsonResponse({"error": "Corpus not found"}, status=404)
        return JsonResponse(
            {"error": f"Upstream status {del_resp.status_code}"},
            status=del_resp.status_code,
        )
    except requests.RequestException as e:
        return JsonResponse({"error": str(e)}, status=502)


@login_required
@require_tova_app_admin()
def admin_list_models(request):
    """List all models across all users (admin only)."""
    try:
        models_drafts = drafts.list_drafts(type=DraftType.model)
    except Exception as e:
        logger.exception("Admin list models drafts: %s", e)
        return JsonResponse({"error": str(e)}, status=500)
    user_ids = set()
    models_list = []
    for draft in models_drafts:
        owner_id = draft.owner_id or (draft.metadata.get("owner_id") if draft.metadata else None)
        # Same fallback as fetch_trained_models: if no owner on model, use corpus owner
        if not owner_id and draft.metadata:
            corpus_id = draft.metadata.get("corpus_id")
            if corpus_id:
                try:
                    corpus_resp = requests.get(f"{R.API}/data/corpora/{corpus_id}", timeout=5)
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
        u = User.objects.filter(pk=uid).first()
        users_by_id[uid] = {"email": u.email if u else None, "name": (u.name if u else None) or ""}
    for m in models_list:
        oid = m.get("owner_id")
        m["owner_email"] = users_by_id.get(oid, {}).get("email") if oid else None
        m["owner_name"] = users_by_id.get(oid, {}).get("name") if oid else None
    return JsonResponse(models_list, safe=False, status=200)


@login_required
@require_tova_app_admin()
def admin_delete_model(request, model_id):
    """Delete any model (admin only)."""
    try:
        up = requests.get(f"{R.API}/data/models/{model_id}", timeout=10)
        if up.status_code == 404:
            return JsonResponse({"error": "Model not found"}, status=404)
        up.raise_for_status()
        model_meta = up.json()
    except requests.RequestException as e:
        return JsonResponse({"error": str(e)}, status=502)
    corpus_id = (
        model_meta.get("corpus_id")
        or (model_meta.get("metadata") or {}).get("corpus_id")
        or ((model_meta.get("metadata") or {}).get("tr_params") or {}).get("corpus_id")
    )
    if corpus_id:
        try:
            requests.post(f"{R.API}/data/corpora/{corpus_id}/delete_model", params={"model_id": model_id}, timeout=10)
        except Exception:
            pass
    try:
        del_resp = requests.delete(f"{R.API}/data/models/{model_id}", timeout=(3, 30))
        if del_resp.status_code == 204:
            _log_audit(request, "model_deleted", target_type="model", target_id=model_id)
            return JsonResponse({"success": True}, status=200)
        if del_resp.status_code == 404:
            return JsonResponse({"error": "Model not found"}, status=404)
        return JsonResponse(
            {"error": f"Upstream status {del_resp.status_code}"},
            status=del_resp.status_code,
        )
    except requests.RequestException as e:
        return JsonResponse({"error": str(e)}, status=502)


@login_required
@require_tova_app_admin()
def admin_stats(request):
    """System stats (admin only)."""
    try:
        r = requests.get(f"{R.API}/data/corpora", timeout=10)
        corpus_count = len(r.json()) if r.ok else None
    except Exception:
        corpus_count = None
    try:
        models_drafts = drafts.list_drafts(type=DraftType.model)
        model_count = len(models_drafts)
    except Exception:
        model_count = None
    user_count = User.objects.count()
    audit_count = AuditLog.objects.count()
    return JsonResponse({
        "users": user_count,
        "corpora": corpus_count,
        "models": model_count,
        "audit_log_entries": audit_count,
    }, status=200)


@login_required
@require_tova_app_admin()
def admin_audit(request):
    """Paginated audit log (admin only)."""
    try:
        page = max(1, int(request.GET.get("page", 1)))
    except (TypeError, ValueError):
        page = 1
    try:
        per_page = min(100, max(1, int(request.GET.get("per_page", 50))))
    except (TypeError, ValueError):
        per_page = 50
    qs = AuditLog.objects.order_by("-created_at")
    _p = Paginator(qs, per_page)
    pagination = _p.get_page(page)
    pagination.items = list(pagination.object_list)
    items = []
    for row in pagination.object_list:
        actor = None
        if row.actor_id:
            u = User.objects.filter(pk=row.actor_id).first()
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
    return JsonResponse({
        "items": items,
        "page": pagination.number,
        "per_page": pagination.paginator.per_page,
        "total": pagination.paginator.count,
        "pages": pagination.paginator.num_pages,
    }, status=200)
# ------------------------------------------------------------------------------
# Basic pages & health
# ------------------------------------------------------------------------------
def home(request):
    """Public landing page; study participants are sent to their preset dashboard."""
    if request.user.is_authenticated and is_study_participant(request.user):
        return redirect(post_login_redirect_url(request.user))
    return render(request, "homepage.html")


def check_backend(request):
    try:
        response = requests.get(f"{R.API}/health", timeout=5)
        if response.status_code == 200:
            return JsonResponse({"status": "success", "message": "Backend is healthy"})
        else:
            return JsonResponse({"status": "error", "message": "Backend is not healthy"}, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


def django_health(request):
    return HttpResponse("ok", content_type="text/plain")


def terms(request):
    return render(request, "terms.html")


def privacy(request):
    return render(request, "privacy.html")


# ------------------------------------------------------------------------------
# LLM config routes
# ------------------------------------------------------------------------------
@login_required
def llm_ui_config(request):
    cfg = get_effective_user_config(request, _request_user_id(request))
    return JsonResponse(R.build_llm_ui_config(cfg))


@login_required
def llm_models(request):
    backend = request.GET.get("backend", "").strip()
    if not backend:
        return JsonResponse({"error": "backend is required"}, status=400)

    params = {"backend": backend}
    host = request.GET.get("host", "").strip()
    api_key = request.GET.get("api_key", "").strip()
    if host:
        params["host"] = host
    if api_key:
        params["api_key"] = api_key

    try:
        r = requests.get(f"{R.API}/llm/models", params=params, timeout=15)
        return JsonResponse(r.json(), status=r.status_code)
    except Exception as e:
        logger.warning("llm_models proxy failed: %s", e)
        return JsonResponse({"error": str(e)}, status=502)


@login_required
def get_llm_config(request):
    user_id = _request_user_id(request)
    if not user_id:
        return JsonResponse({}, status=200)
    
    # Load LLM config from database
    user_config = _load_user_config(user_id) or {}
    cfg = user_config.get("llm_config", {})
    
    # DO NOT send api_key back to browser for security
    # Return a copy without the api_key
    safe_cfg = {k: v for k, v in cfg.items() if k != "api_key"}
    return JsonResponse(safe_cfg)


@login_required
def save_llm_config(request):
    user_id = _request_user_id(request)
    if not user_id:
        return JsonResponse({"success": False, "message": "User not authenticated."}, status=401)

    data = _get_json(request) or {}

    provider = data.get("provider")
    model = data.get("model") or None
    host = data.get("host") or None
    api_key = data.get("api_key")  # Sensitive: store securely in database

    if not provider:
        return JsonResponse({"success": False, "message": "Provider is required."}, status=400)

    # Validate model against available models for the provider
    if model:
        try:
            from tova.prompter.prompter import Prompter
            _provider_norm = provider.strip().lower()
            if _provider_norm in ("openai", "gpt"):
                _key = api_key or _get_openai_key_from_env() or os.environ.get("OPENAI_API_KEY", "")
                available = Prompter.fetch_available_models(backend="openai", api_key=_key or None)
            elif _provider_norm == "ollama":
                available = Prompter.fetch_available_models(backend="ollama", ollama_host=host or None)
            elif _provider_norm in ("vllm", "llama_cpp"):
                available = Prompter.fetch_available_models(backend=_provider_norm, base_url=host or None)
            else:
                available = None
            if available is not None and model not in available:
                return JsonResponse(
                    {"success": False, "message": f"Model '{model}' is not available for provider '{provider}'. Available models: {available}"},
                    status=400,
                )
        except Exception as e:
            logger.warning("Could not validate model '%s' for provider '%s': %s", model, provider, e)

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

    logger.debug("Saved LLM config to database for user: %s", user_id)

    return JsonResponse({"success": True})


# ------------------------------------------------------------------------------
# User configuration (persisted per user)
# ------------------------------------------------------------------------------
@login_required
def api_get_user_config(request):
    cfg = get_effective_user_config(request, _request_user_id(request))
    return JsonResponse(cfg, status=200)


@login_required
def api_update_user_config(request):
    payload = _get_json(request)
    if not isinstance(payload, dict):
        return JsonResponse({"success": False, "message": "Config must be a JSON object."}, status=400)

    user_id = _request_user_id(request)
    # Save only the overrides (what changed from defaults)
    overrides = R._json_safe(payload)
    _save_user_config_overrides(user_id, overrides)
    
    # Return the effective config (defaults merged with overrides)
    merged = get_effective_user_config(request, user_id)
    return JsonResponse({
        "success": True, 
        "config": merged, 
        "message": "Configuration saved."
    }, status=200)


@login_required
def api_user_config(request):
    if request.method == "GET":
        return api_get_user_config(request)
    if request.method == "POST":
        return api_update_user_config(request)
    return JsonResponse({"error": "Method not allowed"}, status=405)


@login_required
def api_get_user_config_overrides(request):
    """Get only the user's config overrides (not the full effective config)."""
    user_id = _request_user_id(request)
    overrides = _load_user_config(user_id) or {}
    return JsonResponse(overrides, status=200)


@login_required
def api_reset_user_config(request):
    user_id = _request_user_id(request)
    _reset_user_config(user_id)
    fresh = get_effective_user_config(request, user_id)
    return JsonResponse({
        "success": True, 
        "config": fresh, 
        "message": "Configuration reset to defaults."
    }, status=200)
# ------------------------------------------------------------------------------
# Data / Corpus-related routes
# ------------------------------------------------------------------------------
@login_required
@block_study_participants()
def load_data_page(request):
    """Page for uploading/creating a dataset."""
    cid = _request_user_id(request) or ""
    return render(
        request,
        "loadData.html",
        {"current_user_id_js": json.dumps(cid)},
    )


@login_required
@block_study_participants()
def load_corpus_page(request):
    return render(request, "manageCorpora.html")


@login_required
@block_study_participants(json_response=True)
def create_corpus(request):
    payload = _get_json(request)
    if not payload:
        return JsonResponse({"error": "No JSON payload received"}, status=400)

    # List of dictionaries with dataset IDs
    datasets = payload.get("datasets", [])
    datasets_lst = []

    user_id = _request_user_id(request)
    if not user_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    for el in datasets:
        try:
            upstream = requests.get(
                f"{R.API}/data/datasets/{el['id']}",
                timeout=(3.05, 30),
            )
            if not upstream.ok:
                return HttpResponse(
                    upstream.content,
                    status=upstream.status_code,
                    content_type=upstream.headers.get("Content-Type", "application/json"),
                )
            ds = upstream.json()
            owner = ds.get("owner_id") or (ds.get("metadata") or {}).get("owner_id")
            # Treat missing or "anonymous" owner as owned by current user when logged in
            if owner is None or owner == "anonymous":
                owner = user_id
            if owner != user_id:
                return JsonResponse({"error": "Access denied: You do not own one or more of the selected datasets"}, status=403)
            datasets_lst.append(ds)
        except requests.Timeout:
            return JsonResponse({"error": "Upstream timeout"}, status=504)
        except requests.RequestException as e:
            return JsonResponse({"error": f"Upstream connection error: {e}"}, status=502)

    owner_id = payload.get("owner_id") or _request_user_id(request)
    corpus_name = (payload.get("corpus_name") or "").strip()
    if not corpus_name:
        return JsonResponse({"error": "Corpus name is required"}, status=400)

    corpus_payload = {
        "name": corpus_name,
        "description": f"Corpus from datasets {', '.join([d['id'] for d in datasets])}",
        "owner_id": owner_id,
        "datasets": datasets_lst,
    }

    logger.info("Payload sent to /data/corpora: %s", corpus_payload)

    try:
        upstream = requests.post(
            f"{R.API}/data/corpora",
            json=corpus_payload,
            timeout=(3.05, 30),
        )
        if not upstream.ok:
            return HttpResponse(
                upstream.content,
                status=upstream.status_code,
                content_type=upstream.headers.get("Content-Type", "application/json"),
            )

        return HttpResponse(
            upstream.content,
            status=upstream.status_code,
            content_type=upstream.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream connection error: {e}"}, status=502)


@login_required
@block_study_participants(json_response=True)
def add_model_to_corpus(request):
    payload = _get_json(request) or {}
    model_id = payload.get("model_id")
    corpus_id = payload.get("corpus_id")
    if not model_id:
        return JsonResponse({"error": "Missing model_id"}, status=400)
    if not corpus_id:
        return JsonResponse({"error": "Missing corpus_id"}, status=400)
    if not _request_user_id(request):
        return JsonResponse({"error": "Not authenticated"}, status=401)

    upstream_url = f"{R.API}/data/corpora/{corpus_id}/add_model"
    try:
        up = requests.post(
            upstream_url,
            params={"model_id": model_id},  # Send model_id as query parameter
            timeout=(3.05, 30),
        )
        up.raise_for_status()
        return HttpResponse(
            up.content,
            status=up.status_code,
            content_type=up.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream connection error: {e}"}, status=502)


@login_required
@block_study_participants(json_response=True)
def delete_corpus(request):
    """
    Delete a corpus by name.
    Body (JSON): { "corpus_name": "..." }
    """
    payload = _get_json(request) or {}
    corpus_name = payload.get("corpus_name")
    owner_id = payload.get("owner_id") or _request_user_id(request)

    if not corpus_name:
        return JsonResponse({"error": "Missing corpus_name"}, status=400)

    # Find the corpus by name to get its ID
    try:
        r = requests.get(f"{R.API}/data/corpora", params={"owner_id": owner_id}, timeout=10)
        r.raise_for_status()
        corpora = r.json()
    except requests.RequestException as e:
        logger.exception("Failed to fetch corpora: %s", e)
        return JsonResponse({"error": "Failed to fetch corpora"}, status=502)

    corpus_id = None
    corpus = None
    for c in corpora:
        metadata = c.get("metadata", {})
        if metadata.get("name") == corpus_name:
            corpus_id = c.get("id")
            corpus = c
            break

    if not corpus_id:
        return JsonResponse({"error": f"Corpus '{corpus_name}' not found"}, status=404)

    # Verify ownership before deletion
    corpus_owner = corpus.get("owner_id") or corpus.get("metadata", {}).get("owner_id")
    if corpus_owner != owner_id:
        return JsonResponse({"error": "Access denied: You do not own this corpus"}, status=403)

    models = corpus.get("models") or []

    # Delete each associated model first
    model_results = []
    for model_id in models:
        logger.info("Deleting model '%s' associated with corpus '%s'", model_id, corpus_id)
        try:
            mresp = requests.delete(f"{R.API}/data/models/{model_id}", timeout=(3.05, 30))
            if mresp.status_code == 204:
                logger.info("Model '%s' deleted", model_id)
                model_results.append({"model_id": model_id, "status": "deleted"})
            elif mresp.status_code == 404:
                logger.warning("Model '%s' not found during deletion", model_id)
                model_results.append({"model_id": model_id, "status": "not_found"})
            else:
                logger.error(
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
            logger.exception("Timeout deleting model '%s'", model_id)
            model_results.append({"model_id": model_id, "status": "timeout"})
        except requests.RequestException as e:
            logger.exception("Connection error deleting model '%s': %s", model_id, e)
            model_results.append(
                {"model_id": model_id, "status": "connection_error", "detail": str(e)}
            )

    # Now delete the corpus itself
    try:
        del_resp = requests.delete(f"{R.API}/data/corpora/{corpus_id}", timeout=(3.05, 30))
        if del_resp.status_code == 204:
            _log_audit(request, "corpus_deleted", target_type="corpus", target_id=corpus_id, details=corpus_name)
            return JsonResponse(
                {
                    "message": f"Corpus '{corpus_id}' deleted successfully",
                    "models": model_results,
                },
                status=200,
            )
        elif del_resp.status_code == 404:
            return JsonResponse(
                {
                    "error": f"Corpus '{corpus_id}' not found during deletion",
                    "models": model_results,
                },
                status=404,
            )
        else:
            return HttpResponse(
                json.dumps(
                    {
                        "error": "Upstream error deleting corpus",
                        "upstream_status": del_resp.status_code,
                        "upstream_body": del_resp.text[:1000],
                        "models": model_results,
                    }
                ),
                status=del_resp.status_code,
                content_type="application/json",
            )
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout deleting corpus", "models": model_results}, status=504)
    except requests.RequestException as e:
        return JsonResponse(
            {"error": f"Upstream connection error deleting corpus: {e}", "models": model_results},
            status=502,
        )


@login_required
@block_study_participants(json_response=True)
def create_dataset(request):
    try:
        payload = _get_json(request) or {}
    except RequestDataTooBig:
        max_mb = os.getenv("DJANGO_DATA_UPLOAD_MAX_MEMORY_SIZE", str(25 * 1024 * 1024))
        try:
            max_mb = f"{int(max_mb) / (1024 * 1024):.0f}"
        except (TypeError, ValueError):
            max_mb = "25"
        return JsonResponse(
            {
                "error": (
                    "Dataset payload too large. Reduce file size or increase "
                    f"DJANGO_DATA_UPLOAD_MAX_MEMORY_SIZE (currently ~{max_mb}MB)."
                )
            },
            status=413,
        )
    metadata = payload.get("metadata", {})
    data = payload.get("data", {})
    documents = data.get("documents", [])
    owner_id = payload.get("owner_id") or _request_user_id(request)

    dataset_payload = {
        "name": metadata.get("name", ""),
        "description": metadata.get("description", ""),
        "owner_id": owner_id,
        "documents": documents,
        "metadata": metadata,
    }

    try:
        upstream = requests.post(
            f"{R.API}/data/datasets",
            json=dataset_payload,
            timeout=(3.05, 30),
        )
        if not upstream.ok:
            return HttpResponse(
                upstream.content,
                status=upstream.status_code,
                content_type=upstream.headers.get("Content-Type", "application/json"),
            )
        return HttpResponse(
            upstream.content,
            status=upstream.status_code,
            content_type=upstream.headers.get("Content-Type", "application/json"),
        )

    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream connection error: {e}"}, status=502)


# ------------------------------------------------------------------------------
# Training routes
# ------------------------------------------------------------------------------
@login_required
def training_page_get(request):
    return render(request, "training.html")


@login_required
def get_training_session(request):
    corpus_id = request.session.get("corpus_id")
    tmReq = request.session.get("tmReq")

    if not corpus_id or not tmReq:
        return JsonResponse({"error": "Training session data not found"}, status=404)

    return JsonResponse({"corpus_id": corpus_id, "tmReq": tmReq}, status=200)


@login_required
def proxy_corpus_tfidf(request, corpus_id):
    try:
        up = requests.get(f"{R.API}/train/corpus/{corpus_id}/tfidf/", timeout=(3.05, 30))
        return HttpResponse(
            up.content,
            status=up.status_code,
            content_type=up.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream connection error: {e}"}, status=502)


@login_required
def get_tfidf_data(request, corpus_id):
    try:
        upstream = requests.get(
            f"{R.API}/train/corpus/{corpus_id}/tfidf/",
            params=request.GET,
            timeout=(3.05, 60),
        )
        upstream.raise_for_status()
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream connection error: {e}"}, status=502)

    try:
        return JsonResponse(upstream.json(), status=upstream.status_code)
    except ValueError:
        return HttpResponse(
            upstream.content,
            status=upstream.status_code,
            content_type=upstream.headers.get("Content-Type", "application/json"),
        )


@login_required
@block_study_participants(json_response=True)
def train_corpus_tfidf_route(request, corpus_id):
    if request.method == "GET":
        return proxy_corpus_tfidf(request, corpus_id)
    if request.method == "POST":
        return train_tfidf_corpus(request, corpus_id)
    return JsonResponse({"error": "Method not allowed"}, status=405)


@login_required
def train_tfidf_corpus(request, corpus_id):
    payload = _get_json(request) or {}
    try:
        n_clusters = int(payload.get("n_clusters") or 15)
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid n_clusters"}, status=400)

    # fetch corpus from backend and verify ownership
    user_id = _request_user_id(request)
    if not R._verify_corpus_ownership(corpus_id, user_id):
        return JsonResponse({"error": "Access denied: You do not own this corpus"}, status=403)
    
    try:
        up = requests.get(
            f"{R.API}/data/corpora/{corpus_id}",
            timeout=(3.05, 60),
        )
        up.raise_for_status()
        corpus = up.json()
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout fetching corpus"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream error fetching corpus: {e}"}, status=502)

    documents = [
        {"id": str(d.get("id")), "text": str(d.get("text", ""))}
        for d in (corpus.get("documents") or [])
    ]
    if not documents:
        return JsonResponse({"error": "No documents found in corpus"}, status=400)

    upstream_payload = {"n_clusters": n_clusters, "documents": documents}
    try:
        upstream = requests.post(
            f"{R.API}/train/corpus/{corpus_id}/tfidf/",
            json=upstream_payload,
            timeout=(3.05, 120),
        )
        upstream.raise_for_status()
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout calling TF-IDF"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream connection error: {e}"}, status=502)

    try:
        return JsonResponse(upstream.json(), status=upstream.status_code)
    except ValueError:
        return HttpResponse(
            upstream.content,
            status=upstream.status_code,
            content_type=upstream.headers.get("Content-Type", "application/json"),
        )


@login_required
@block_study_participants(json_response=True)
def training_start(request):

    payload = _get_json(request) or {}
    logger.info(f"Training request payload: {payload}")
    corpus_id = payload.get("corpus_id")
    model = payload.get("model")
    model_name = payload.get("model_name")
    training_params = payload.get("training_params") or {}
    config_path = payload.get("config_path", "static/config/config.yaml")  # Get config_path from payload or use default
    user_id = _request_user_id(request)


    if not corpus_id or not model or not model_name:
        return JsonResponse({"error": "Missing corpus_id/model/model_name"}, status=400)

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
        merged_overrides = R._deep_merge(existing_overrides, new_overrides)
        
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
            _src = (llm_config.get("provider") or "").lower()
            _dst = (final_training_params.get("llm_provider") or "").lower()
            if not _dst or _src == _dst:
                final_training_params["llm_server"] = host
        if api_key and "llm_api_key" not in final_training_params:
            final_training_params["llm_api_key"] = api_key

    # When no LLM choice was sent and no saved LLM Settings modal, use effective config
    # (topic_modeling.general from config form or config.yaml) so e.g. selecting Ollama
    # in the config form is respected instead of always falling back to config.yaml default.
    if add_llm_params:
        effective = get_effective_user_config(request, user_id)
        tm_general = R._safe_dict(effective.get("topic_modeling", {})).get("general", {})
        if tm_general:
            if "llm_provider" not in final_training_params and tm_general.get("llm_provider"):
                p = (tm_general["llm_provider"] or "").strip().lower()
                final_training_params["llm_provider"] = "openai" if p == "gpt" else p
            if "llm_model_type" not in final_training_params and tm_general.get("llm_model_type"):
                final_training_params["llm_model_type"] = tm_general["llm_model_type"]
            if "llm_server" not in final_training_params and tm_general.get("llm_server"):
                _src = (tm_general.get("llm_provider") or "").lower()
                _dst = (final_training_params.get("llm_provider") or "").lower()
                if not _dst or _src == _dst:
                    final_training_params["llm_server"] = tm_general["llm_server"]

    # When user chose Ollama but host missing, default from saved config or config file
    if (final_training_params.get("llm_provider") or "").lower() == "ollama":
        if not final_training_params.get("llm_server"):
            ollama_from_config = (
                R._safe_dict(R.RAW_CONFIG.get("llm", {})).get("ollama", {}).get("host")
                if R.RAW_CONFIG else None
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
            f"{R.API}/data/corpora/{corpus_id}",
            timeout=(3.05, 60),
        )
        up.raise_for_status()
        corpus = up.json()
        
        # Verify corpus ownership before training
        corpus_owner = corpus.get("owner_id") or corpus.get("metadata", {}).get("owner_id")
        if str(corpus_owner or "") != str(user_id or ""):
            return JsonResponse({"error": "Access denied: You do not own this corpus"}, status=403)
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout fetching corpus"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream error fetching corpus: {e}"}, status=502)

    docs = [
        {"id": str(d.get("id")), "text": str(d.get("text", ""))}
        for d in (corpus.get("documents") or [])
    ]
    if not docs:
        return JsonResponse({"error": "No documents found in corpus"}, status=400)

    # Ensure we have owner_id - get from session if not already set
    if not user_id:
        user_id = _request_user_id(request)
    
    if not user_id:
        return JsonResponse({"error": "User not authenticated"}, status=401)

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
    logger.info(f"Training request: {tm_req}")
    
    # Confirm LLM options being sent (no secrets)
    fp = final_training_params or {}
    logger.info(
        "Training LLM options: do_labeller=%s do_summarizer=%s llm_provider=%s llm_model_type=%s llm_server=%s api_key_set=%s",
        fp.get("do_labeller"),
        fp.get("do_summarizer"),
        fp.get("llm_provider"),
        fp.get("llm_model_type"),
        fp.get("llm_server"),
        bool(fp.get("llm_api_key")),
    )

    try:
        tr = requests.post(f"{R.API}/train/json", json=tm_req, timeout=(3.05, 120))
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout starting training"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream error starting training: {e}"}, status=502)

    if tr.status_code >= 400:
        return HttpResponse(
            tr.content,
            status=tr.status_code,
            content_type=tr.headers.get("Content-Type", "application/json"),
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
    return JsonResponse(
        {
            "job_id": job_id,
            "status_url": loc or (f"/status/jobs/{job_id}" if job_id else None),
            "corpus_id": corpus_id,
            "model_id": model_id,
            "config_path": config_path,  # Return the config path used
        },
        status=200,
    )


# ------------------------------------------------------------------------------
# Status routes
# ------------------------------------------------------------------------------
@login_required
def get_status(request, job_id=None):
    job_id = job_id or request.GET.get("job_id") or request.headers.get("X-Job-Id")
    if not job_id:
        return JsonResponse({"error": "Missing job_id"}, status=400)

    headers = {}
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth

    upstream_url = f"{R.API}/status/jobs/{job_id}"
    try:
        up = requests.get(upstream_url, headers=headers, timeout=(3.05, 30))
        return HttpResponse(
            up.content,
            status=up.status_code,
            content_type=up.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream connection error: {e}"}, status=502)


# ------------------------------------------------------------------------------
# Model-related routes
# ------------------------------------------------------------------------------
@login_required
@block_study_participants()
def loadModel(request):
    return render(request, "loadModel.html")


@login_required
def get_unique_corpus_names(request):
    """Return a deduped, A→Z list of corpus names for the current user only."""
    user_id = _request_user_id(request)
    if not user_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    try:
        r = requests.get(
            f"{R.API}/data/corpora",
            params={"owner_id": user_id},
            timeout=10,
        )
        r.raise_for_status()
        items = r.json()
    except requests.RequestException:
        logger.exception("Failed to fetch drafts from upstream")
        return JsonResponse({"error": "Upstream drafts service failed"}, status=502)
    items = R._filter_list_by_owner(items or [], user_id)
    pick = {}  # lower(name) -> {"name": original, "created_at": ts}
    for d in items:
        meta = (d or {}).get("metadata") or {}
        name = (meta.get("name") or d.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        created = meta.get("created_at") or ""
        if key not in pick or created > pick[key]["created_at"]:
            pick[key] = {"name": name, "created_at": created}

    names = sorted((v["name"] for v in pick.values()), key=str.casefold)
    return JsonResponse(names, safe=False, status=200)


@login_required
def getAllCorpora(request):
    """Pulls corpora from the API; only returns those owned by the current user."""
    user_id = _request_user_id(request)
    if not user_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    try:
        r = requests.get(
            f"{R.API}/data/corpora",
            params={"owner_id": user_id},
            timeout=10,
        )
        r.raise_for_status()
        items = r.json()
    except requests.RequestException:
        logger.exception("Failed to fetch drafts from upstream")
        return JsonResponse({"error": "Upstream drafts service failed"}, status=502)
    except ValueError:
        logger.exception("Invalid JSON from corpora upstream")
        return JsonResponse({"error": "Invalid JSON from upstream"}, status=502)

    if not isinstance(items, list):
        logger.warning("corpora upstream returned %s, expected list", type(items).__name__)
        items = []

    # UI-only filter: ensure only current user's corpora (in case R.API returns more).
    items = R._filter_list_by_owner(items, user_id)

    def _norm_corpus(c):
        meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
        location = c.get("location")
        is_draft = location != "database"
        name = meta.get("name") or c.get("name") or ""
        created = meta.get("created_at") or c.get("created_at") or ""
        return {
            "id": c.get("id"),
            "name": str(name) if name is not None else "",
            "is_draft": is_draft,
            "created_at": str(created) if created is not None else "",
        }

    corpora = [_norm_corpus(c) for c in items if isinstance(c, dict)]
    corpora.sort(key=lambda x: ((x.get("name") or "").lower(), not x["is_draft"]))
    return JsonResponse(corpora, safe=False, status=200)


def _corpus_owner_display_name(owner_id) -> str:
    if owner_id is None or owner_id == "":
        return "—"
    oid = str(owner_id)
    try:
        user = User.objects.filter(pk=oid).first()
    except (ValueError, TypeError):
        user = None
    if user:
        name = (user.name or "").strip()
        if name:
            return name
        if user.email:
            return user.email
    return oid


def _model_json_display_name(m: dict) -> str:
    if not isinstance(m, dict):
        return "—"
    n = m.get("name")
    if n is not None and str(n).strip():
        return str(n).strip()
    meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
    tr = meta.get("tr_params") if isinstance(meta.get("tr_params"), dict) else {}
    for key in ("model_name", "name"):
        v = tr.get(key) if tr else None
        if v is None:
            v = meta.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    mid = m.get("id")
    return str(mid) if mid is not None else "—"


def _linked_models_display(model_ids: list) -> list[dict]:
    out: list[dict] = []
    for raw_id in model_ids:
        if raw_id is None or raw_id == "":
            continue
        mid = str(raw_id)
        name = None
        try:
            r = requests.get(f"{R.API}/data/models/{mid}", timeout=5)
            if r.ok and isinstance(r.json(), dict):
                name = _model_json_display_name(r.json())
        except Exception:
            logger.debug("Could not fetch model %s for corpus detail UI", mid, exc_info=True)
        out.append({"id": mid, "name": name})
    return out


def _enrich_corpus_detail_for_ui(corpus: dict) -> dict:
    """Add owner_display and linked_models for the corpus details modal."""
    enriched = dict(corpus)
    oid = enriched.get("owner_id")
    if oid is None and isinstance(enriched.get("metadata"), dict):
        oid = enriched["metadata"].get("owner_id")
    enriched["owner_display"] = _corpus_owner_display_name(oid)
    mids = enriched.get("models")
    if isinstance(mids, list) and mids:
        enriched["linked_models"] = _linked_models_display(mids)
    else:
        enriched["linked_models"] = []
    return enriched


@login_required
def get_corpus(request, corpus_id):
    user_id = _request_user_id(request)
    if not user_id:
        return JsonResponse({"error": "User not authenticated"}, status=401)

    # Verify ownership before returning corpus.
    if not R._verify_corpus_ownership(corpus_id, user_id):
        return JsonResponse({"error": "Access denied: You do not own this corpus"}, status=403)
    
    try:
        # Fetch without owner_id since we've already verified ownership
        response = requests.get(
            f"{R.API}/data/corpora/{corpus_id}",
            timeout=10,
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            return HttpResponse(
                response.content,
                status=response.status_code,
                content_type=response.headers.get("Content-Type", "application/json"),
            )
        if not isinstance(payload, dict):
            return HttpResponse(
                response.content,
                status=response.status_code,
                content_type=response.headers.get("Content-Type", "application/json"),
            )
        payload = _enrich_corpus_detail_for_ui(payload)
        return JsonResponse(payload, status=response.status_code)
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout"}, status=504)
    except requests.RequestException as e:
        logger.error(f"Failed to fetch corpus {corpus_id}: {e}")
        return JsonResponse({"error": f"Failed to fetch corpus {corpus_id}: {e}"}, status=502)


@login_required
@block_study_participants(json_response=True)
def get_model_registry(request):
    with open("static/config/modelRegistry.json") as f:
        return JsonResponse(json.load(f))


@login_required
def trained_models(request):
    return render(request, "trained_models.html")


def _model_dict_from_id(model_id: str) -> dict | None:
    """Load a model dict for the trained-models UI, without owner filtering."""
    if not model_id:
        return None
    try:
        response = requests.get(f"{R.API}/data/models/{model_id}", timeout=10)
        if response.status_code == 200:
            model = response.json()
            meta = model.get("metadata") or {}
            tr = meta.get("tr_params") or {}
            return {
                "id": model.get("id") or model_id,
                "name": tr.get("model_name") or model.get("name") or model_id,
                "owner_id": model.get("owner_id") or meta.get("owner_id"),
                "corpus_id": model.get("corpus_id") or meta.get("corpus_id"),
                "created_at": model.get("created_at") or meta.get("created_at"),
                "location": model.get("location") or "temporal",
                "metadata": meta,
            }
    except Exception:
        logger.exception("Failed to fetch study model %s from API", model_id)

    try:
        draft = drafts.get_draft(model_id, DraftType.model)
        if not draft:
            return None
        model = drafts.draft_to_model(draft)
        if not model:
            return None
        model_path = R.DRAFTS_SAVE / model_id
        metadata_path = model_path / "metadata.json"
        metadata_dict = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)
        else:
            metadata_dict = pydantic_to_dict(model.metadata) if model.metadata else {}
        return {
            "id": model.id,
            "name": model.name or metadata_dict.get("tr_params", {}).get("model_name") or model_id,
            "owner_id": model.owner_id,
            "corpus_id": model.corpus_id or metadata_dict.get("corpus_id"),
            "created_at": model.created_at or metadata_dict.get("created_at"),
            "location": model.location.value if hasattr(model.location, "value") else str(model.location),
            "metadata": metadata_dict,
        }
    except Exception:
        logger.exception("Failed to load study model %s from drafts", model_id)
    return None


def _study_model_already_listed(corpora: list, model_id: str) -> bool:
    for corpus in corpora or []:
        for model in corpus.get("models") or []:
            if str(model.get("id")) == str(model_id):
                return True
    return False


def _inject_study_preset_model(corpora: list, request) -> list:
    """Ensure a study participant sees their group's preset model in trained-models."""
    if not is_study_participant(request.user):
        return corpora
    model_id = get_study_model_id(request.user)
    if not model_id or _study_model_already_listed(corpora, model_id):
        return corpora

    model_dict = _model_dict_from_id(model_id)
    if not model_dict:
        logger.warning("Study preset model %s could not be loaded for user %s", model_id, _request_user_id(request))
        return corpora

    study_corpus = {
        "id": f"study_{model_id}",
        "name": "Study model",
        "owner_id": _request_user_id(request),
        "models": [model_dict],
        "created_at": model_dict.get("created_at"),
        "metadata": {"name": "Study model"},
        "location": model_dict.get("location") or "temporal",
    }
    return [study_corpus] + list(corpora or [])


def _fetch_trained_models_corpora(request):
    try:
        owner_id = _request_user_id(request)
        if not owner_id:
            return []
        
        # Get all models directly from drafts system filtered by owner_id
        all_user_models = []
        try:
            models_drafts = drafts.list_drafts(type=DraftType.model)
            logger.info("Found %d total model drafts", len(models_drafts))
            
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
                                f"{R.API}/data/corpora/{corpus_id}",
                                timeout=5,
                            )
                            if corpus_resp.status_code == 200:
                                corpus_data = corpus_resp.json()
                                draft_owner = corpus_data.get("owner_id") or corpus_data.get("metadata", {}).get("owner_id")
                        except Exception:
                            pass
                
                logger.debug("Model draft %s: owner_id=%s, checking against %s", draft.id, draft_owner, owner_id)
                
                if draft_owner == owner_id:
                    try:
                        model = drafts.draft_to_model(draft)
                        if model:
                            # Read metadata.json directly to get tr_params structure
                            model_path = R.DRAFTS_SAVE / draft.id
                            metadata_path = model_path / "metadata.json"
                            metadata_dict = {}
                            
                            if metadata_path.exists():
                                try:
                                    with open(metadata_path, 'r') as f:
                                        metadata_dict = json.load(f)
                                    logger.debug("Loaded metadata from %s", metadata_path)
                                except Exception as e:
                                    logger.warning("Failed to read metadata.json for %s: %s", draft.id, e)
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
                            logger.info("Added model %s (%s) for user %s", model.id, model_dict.get("name"), owner_id)
                    except Exception as e:
                        logger.error("Error converting draft %s to model: %s", draft.id, e)
            
            logger.info("Found %d models for user %s", len(all_user_models), owner_id)
            if all_user_models:
                logger.info("Sample model structure: %s", json.dumps(all_user_models[0], indent=2, default=str))
        except Exception as e:
            logger.exception("Error fetching models from drafts: %s", str(e))
            all_user_models = []
        
        # Create a map of model_id -> model for quick lookup
        models_by_id = {model.get("id"): model for model in all_user_models}
        
        # Get corpora and associate models
        corpora_response = requests.get(
            f"{R.API}/data/corpora",
            params={"owner_id": owner_id},
            timeout=10,
        )
        corpora_response.raise_for_status()
        corpora = corpora_response.json()
        corpora = R._filter_list_by_owner(corpora or [], owner_id)
        logger.info("Corpora returned (owner-filtered): %s", len(corpora))

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
        
        logger.info("Grouped models: %d corpora, %d unassociated", len(models_by_corpus_id), len(unassociated_models))
        
        # Track which models we've already added to a corpus
        models_in_corpora = set()

        for corpus in corpora:
            corpus_id = corpus.get("id")
            logger.info("Processing corpus ID: %s", corpus_id)
            if not corpus_id:
                logger.warning("Corpus without ID: %s", corpus)
                continue

            # Get models from R.API (existing association)
            try:
                models_response = requests.get(
                    f"{R.API}/data/corpora/{corpus_id}/models",
                    params={"owner_id": owner_id},
                    timeout=10,
                )
                models_response.raise_for_status()
                corpus_models = models_response.json()
                logger.info(
                    "Models from R.API for corpus ID %s: %s", corpus_id, len(corpus_models)
                )
                
                # Also add models that have this corpus_id in their metadata but aren't in the R.API response
                if corpus_id in models_by_corpus_id:
                    # Merge R.API models with models found by corpus_id
                    api_model_ids = {m.get("id") for m in corpus_models}
                    for model in models_by_corpus_id[corpus_id]:
                        if model.get("id") not in api_model_ids:
                            corpus_models.append(model)
                            logger.info("Added model %s to corpus %s based on corpus_id in metadata", model.get("id"), corpus_id)
                
                corpus["models"] = corpus_models
                # Track which models are in corpora
                for model in corpus_models:
                    model_id = model.get("id")
                    if model_id:
                        models_in_corpora.add(model_id)
            except requests.Timeout:
                logger.error(
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
                logger.error(
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
            logger.info("Added %d unassociated models to virtual corpus", len(final_unassociated))

        return _inject_study_preset_model(corpora, request)

    except requests.Timeout:
        logger.error("Timeout fetching corpora")
        raise
    except requests.RequestException as e:
        logger.error("Error fetching corpora: %s", str(e))
        raise
    except ValueError:
        logger.error("Invalid JSON response from corpora endpoint")
        raise
    except Exception as e:
        logger.exception("Unexpected error in _fetch_trained_models_corpora: %s", str(e))
        raise


@login_required
def get_trained_models(request):
    try:
        corpora = _fetch_trained_models_corpora(request)
        return JsonResponse(corpora, safe=False, status=200)
    except requests.Timeout:
        return JsonResponse({"error": "Upstream request timed out"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": "Upstream request failed", "detail": str(e)}, status=502)
    except ValueError:
        return JsonResponse({"error": "Invalid JSON from upstream"}, status=502)


@login_required
def get_model_names(request):
    try:
        corpora = _fetch_trained_models_corpora(request)

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

        return JsonResponse({"models": unique})

    except requests.Timeout:
        return JsonResponse({"error": "Upstream request timed out"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream request failed: {e}"}, status=502)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def delete_model(request):
    """
    Delete a model by ID.
    """
    payload = _get_json(request) or {}
    model_id = payload.get("model_id")

    if not model_id:
        return JsonResponse({"error": "Missing 'model_id'"}, status=400)
    user_id = _request_user_id(request)
    if not user_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    if not R._verify_model_ownership(model_id, user_id):
        return JsonResponse({"error": "Access denied: You do not own this model"}, status=403)

    # get model entity to verify it exists and get corpus_id
    try:
        t0 = time()
        up = requests.get(
            f"{R.API}/data/models/{model_id}",
            timeout=(3.05, 60),
        )
        up.raise_for_status()
        model_metadata = up.json()
        logger.info(
            "delete-model model meta ok status=%s dt=%.3fs",
            up.status_code,
            time() - t0,
        )
    except requests.HTTPError as e:
        if e.response and e.response.status_code == 404:
            return JsonResponse({"error": "Model not found"}, status=404)
        logger.exception("model delete HTTP error fetching model: %s", e)
        return JsonResponse(
            {"error": f"Failed to fetch model: {e}"},
            status=e.response.status_code if e.response else 502,
        )
    except requests.Timeout:
        logger.exception("model delete timeout fetching model")
        return JsonResponse({"error": "Upstream timeout fetching model"}, status=504)
    except requests.RequestException as e:
        logger.exception("model delete error fetching model: %s", e)
        return JsonResponse({"error": f"Upstream error fetching model: {e}"}, status=502)

    corpus_id = (
        model_metadata.get("corpus_id")
        or (model_metadata.get("metadata") or {}).get("corpus_id")
        or ((model_metadata.get("metadata") or {}).get("tr_params") or {}).get("corpus_id")
    )
    if corpus_id:
        logger.info("Deleting model '%s' from corpus '%s'", model_id, corpus_id)
        # delete model from corpus
        try:
            up = requests.post(
                f"{R.API}/data/corpora/{corpus_id}/delete_model",
                params={"model_id": model_id},
                timeout=(3.05, 30),
            )
            up.raise_for_status()
            logger.info("Model '%s' removed from corpus '%s'", model_id, corpus_id)
        except requests.Timeout:
            logger.warning("model delete timeout removing from corpus (continuing with model deletion)")
        except requests.RequestException as e:
            logger.warning("model delete upstream corpus error (continuing with model deletion): %s", e)
    else:
        logger.info("Deleting model '%s' (no corpus association)", model_id)

    # delete model
    try:
        up = requests.delete(
            f"{R.API}/data/models/{model_id}",
            timeout=(3.05, 30),
        )
        logger.info("Model delete upstream response status=%s", up.status_code)
        if up.status_code == 204:
            _log_audit(request, "model_deleted", target_type="model", target_id=model_id)
            return JsonResponse({"message": f"Model '{model_id}' deleted successfully"}, status=200)
        elif up.status_code == 404:
            return JsonResponse({"error": f"Model '{model_id}' not found"}, status=404)
        elif up.status_code >= 400:
            logger.error("model delete upstream error status=%s", up.status_code)
            return HttpResponse(
                up.content,
                status=up.status_code,
                content_type=up.headers.get("Content-Type", "application/json"),
            )
        else:
            return JsonResponse(
                {
                    "message": f"Model '{model_id}' delete request completed",
                    "upstream_status": up.status_code,
                },
                status=200,
            )
    except requests.Timeout:
        logger.exception("model delete timeout deleting model")
        return JsonResponse(
            {"error": "Upstream timeout deleting model"},
            status=504,
        )
    except requests.RequestException as e:
        logger.exception("model delete upstream model error: %s", e)
        return JsonResponse(
            {"error": f"Upstream connection error deleting model: {e}"},
            status=502,
        )


# ------------------------------------------------------------------------------
# Dashboard-related routes
# ------------------------------------------------------------------------------
def _model_nav_context(model_id: str) -> dict:
    """model_id, model_name, corpus_id for dashboard and active-learning page shells."""
    model_id = (model_id or "").strip()
    model_name = ""
    corpus_id = ""
    if model_id:
        try:
            up = requests.get(
                f"{R.API}/data/models/{model_id}",
                timeout=(3.05, 20),
            )
            if up.ok:
                body = up.json() or {}
                model_name = (body.get("name") or "").strip()
                meta = body.get("metadata") or {}
                if not model_name:
                    tr = meta.get("tr_params") or {}
                    model_name = (tr.get("model_name") or "").strip()
                corpus_id = (body.get("corpus_id") or "").strip()
                if not corpus_id:
                    tr = meta.get("tr_params") or {}
                    corpus_id = (
                        (meta.get("corpus_id") or "").strip()
                        or (tr.get("corpus_id") or "").strip()
                    )
        except requests.RequestException:
            logger.warning(
                "model nav context: could not fetch metadata for %s",
                model_id,
                exc_info=True,
            )
    return {
        "model_id": model_id,
        "model_name": model_name,
        "corpus_id": corpus_id,
    }


@login_required
@require_study_feature(FEATURE_ACTIVE_LEARNING)
def active_learning_two(request):
    model_id = (request.GET.get("model_id") or "").strip()
    denied = _study_model_access_denied(request, model_id)
    if denied:
        return denied
    return render(
        request,
        "active_learning_two.html",
        _model_nav_context(model_id),
    )


@login_required
@require_study_feature(FEATURE_ACTIVE_LEARNING, json_response=True)
def active_learning_two_recommend(request):
    model_id = (request.GET.get("model_id") or "").strip()
    user_id = _request_user_id(request)
    if not model_id:
        return JsonResponse({"error": "model_id is required"}, status=400)
    if not user_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    if not R._verify_model_ownership(model_id, user_id):
        return JsonResponse({"error": "Access denied: You do not own this model"}, status=403)
    try:
        up = requests.get(
            f"{R.API}/active-learning/{model_id}/recommend",
            timeout=(3.05, 30),
        )
        return HttpResponse(
            up.content,
            status=up.status_code,
            content_type=up.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream connection error: {e}"}, status=502)


@login_required
@require_study_feature(FEATURE_ACTIVE_LEARNING, json_response=True)
def active_learning_two_status(request):
    model_id = (request.GET.get("model_id") or "").strip()
    user_id = _request_user_id(request)
    if not model_id:
        return JsonResponse({"error": "model_id is required"}, status=400)
    if not user_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    if not R._verify_model_ownership(model_id, user_id):
        return JsonResponse({"error": "Access denied: You do not own this model"}, status=403)
    try:
        up = requests.get(
            f"{R.API}/active-learning/{model_id}/status",
            timeout=(3.05, 20),
        )
        return HttpResponse(
            up.content,
            status=up.status_code,
            content_type=up.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream connection error: {e}"}, status=502)


@login_required
@require_study_feature(FEATURE_ACTIVE_LEARNING, json_response=True)
def active_learning_two_label(request):
    payload = _get_json(request) or {}
    model_id = str(payload.get("model_id") or "").strip()
    doc_id = str(payload.get("doc_id") or "").strip()
    label = str(payload.get("label") or "").strip()
    user_id = _request_user_id(request)

    if not model_id:
        return JsonResponse({"error": "model_id is required"}, status=400)
    if not doc_id:
        return JsonResponse({"error": "doc_id is required"}, status=400)
    if not label:
        return JsonResponse({"error": "label is required"}, status=400)
    if not user_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    if not R._verify_model_ownership(model_id, user_id):
        return JsonResponse({"error": "Access denied: You do not own this model"}, status=403)

    try:
        up = requests.post(
            f"{R.API}/active-learning/{model_id}/label",
            json={"doc_id": doc_id, "label": label},
            timeout=(3.05, 30),
        )
        return HttpResponse(
            up.content,
            status=up.status_code,
            content_type=up.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream connection error: {e}"}, status=502)


@login_required
@require_study_feature(FEATURE_ACTIVE_LEARNING, json_response=True)
def active_learning_two_unlabel(request):
    """Remove a human label from a document (undo). Proxies the upstream DELETE."""
    payload = _get_json(request) or {}
    model_id = str(payload.get("model_id") or "").strip()
    doc_id = str(payload.get("doc_id") or "").strip()
    user_id = _request_user_id(request)

    if not model_id:
        return JsonResponse({"error": "model_id is required"}, status=400)
    if not doc_id:
        return JsonResponse({"error": "doc_id is required"}, status=400)
    if not user_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    if not R._verify_model_ownership(model_id, user_id):
        return JsonResponse({"error": "Access denied: You do not own this model"}, status=403)

    try:
        up = requests.delete(
            f"{R.API}/active-learning/{model_id}/label/{doc_id}",
            timeout=(3.05, 30),
        )
        return HttpResponse(
            up.content,
            status=up.status_code,
            content_type=up.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream connection error: {e}"}, status=502)


def _al2_dominant_topic_id(theta_dist: dict) -> Optional[str]:
    """Dominant topic id for one document from theta distribution."""
    if not theta_dist:
        return None
    try:
        top_tid, _ = max(theta_dist.items(), key=lambda kv: float(kv[1] or 0))
    except (TypeError, ValueError):
        return None
    return str(top_tid)


def _al2_dominant_topic_label(theta_dist: dict, topics_info: dict) -> str:
    """Topic-model label for one document from theta distribution and Topics Info."""
    tid_str = _al2_dominant_topic_id(theta_dist)
    if not tid_str:
        return "—"
    meta = topics_info.get(tid_str)
    if meta is None:
        for k, v in topics_info.items():
            if str(k) == tid_str and isinstance(v, dict):
                meta = v
                break
    if isinstance(meta, dict):
        lab = meta.get("Label")
        if lab is not None and str(lab).strip():
            return str(lab).strip()
    return f"Topic {tid_str}"


def _al2_fetch_model_metadata(model_id: str):
    """Load model metadata from API or drafts (same strategy as get-dashboard)."""
    model_metadata = None
    try:
        up0 = time()
        up = requests.get(
            f"{R.API}/data/models/{model_id}",
            timeout=(3.05, 60),
        )
        up.raise_for_status()
        model_metadata = up.json()
        logger.info(
            "al2-bootstrap model meta ok status=%s dt=%.3fs",
            up.status_code,
            time() - up0,
        )
    except requests.HTTPError as e:
        if e.response and e.response.status_code == 404:
            logger.warning(
                "Model not found via R.API, trying direct draft access for %s", model_id
            )
            try:
                model_draft = drafts.get_draft(model_id, DraftType.model)
                if model_draft:
                    model = drafts.draft_to_model(model_draft)
                    if model:
                        model_path = R.DRAFTS_SAVE.resolve() / model_id
                        metadata_path = model_path / "metadata.json"
                        metadata_dict = {}
                        if metadata_path.exists():
                            try:
                                with open(metadata_path, "r") as f:
                                    metadata_dict = json.load(f)
                            except Exception as ex:
                                logger.warning(
                                    "Failed to read metadata.json for %s: %s", model_id, ex
                                )
                                metadata_dict = (
                                    pydantic_to_dict(model.metadata) if model.metadata else {}
                                )
                        else:
                            metadata_dict = (
                                pydantic_to_dict(model.metadata) if model.metadata else {}
                            )
                        model_metadata = {
                            "id": model.id,
                            "name": model.name
                            or metadata_dict.get("tr_params", {}).get("model_name")
                            or model_id,
                            "owner_id": model.owner_id,
                            "corpus_id": model.corpus_id or metadata_dict.get("corpus_id"),
                            "created_at": model.created_at or metadata_dict.get("created_at"),
                            "location": model.location.value
                            if hasattr(model.location, "value")
                            else str(model.location),
                            "metadata": metadata_dict,
                        }
                        logger.info("Successfully loaded model %s from drafts directly", model_id)
                    else:
                        raise ValueError("Failed to convert draft to model")
                else:
                    logger.error("Model draft %s not found in drafts system", model_id)
                    raise ValueError("Model not found in drafts")
            except Exception:
                logger.exception("Failed to load model from drafts")
                try:
                    error_detail = e.response.json() if e.response else {}
                    return None, JsonResponse(
                        {
                            "error": error_detail.get("detail")
                            or error_detail.get("error")
                            or f"Model not found (HTTP {e.response.status_code})",
                        },
                        status=e.response.status_code if e.response else 502,
                    )
                except Exception:
                    return None, JsonResponse(
                        {
                            "error": f"Failed to fetch model: HTTP {e.response.status_code if e.response else 'unknown'}"
                        },
                        status=502,
                    )
        else:
            logger.exception("al2-bootstrap HTTP error fetching model: %s", e)
            try:
                error_detail = e.response.json() if e.response else {}
                return None, JsonResponse(
                    {
                        "error": error_detail.get("detail")
                        or error_detail.get("error")
                        or f"Model not found (HTTP {e.response.status_code})",
                    },
                    status=e.response.status_code if e.response else 502,
                )
            except Exception:
                return None, JsonResponse(
                    {
                        "error": f"Failed to fetch model: HTTP {e.response.status_code if e.response else 'unknown'}"
                    },
                    status=502,
                )
    except requests.Timeout:
        logger.exception("al2-bootstrap timeout fetching model")
        return None, JsonResponse(
            {"error": "Timeout: The model service took too long to respond. Please try again."},
            status=504,
        )
    except requests.RequestException as e:
        logger.exception("al2-bootstrap error fetching model: %s", e)
        return None, JsonResponse(
            {"error": f"Connection error: Unable to reach the model service. {str(e)}"},
            status=502,
        )
    if not model_metadata:
        return None, JsonResponse({"error": "Failed to load model metadata"}, status=500)
    return model_metadata, None


@login_required
@require_study_feature(FEATURE_ACTIVE_LEARNING, json_response=True)
def active_learning_two_labels(request):
    model_id = (request.GET.get("model_id") or "").strip()
    user_id = _request_user_id(request)
    if not model_id:
        return JsonResponse({"error": "model_id is required"}, status=400)
    if not user_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    if not R._verify_model_ownership(model_id, user_id):
        return JsonResponse({"error": "Access denied: You do not own this model"}, status=403)
    try:
        up = requests.get(
            f"{R.API}/active-learning/{model_id}/labels",
            timeout=(3.05, 20),
        )
        return HttpResponse(
            up.content,
            status=up.status_code,
            content_type=up.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return JsonResponse({"error": "Upstream timeout"}, status=504)
    except requests.RequestException as e:
        return JsonResponse({"error": f"Upstream connection error: {e}"}, status=502)


@login_required
@require_study_feature(FEATURE_ACTIVE_LEARNING, json_response=True)
def active_learning_two_bootstrap(request):
    """Full corpus documents, dominant topic label per doc, and AL human labels."""
    t0 = time()
    payload = _get_json(request) or {}
    model_id = str(payload.get("model_id") or "").strip()
    user_id = _request_user_id(request)
    if not model_id:
        return JsonResponse({"error": "model_id is required"}, status=400)
    if not user_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    if not R._verify_model_ownership(model_id, user_id):
        return JsonResponse({"error": "Access denied: You do not own this model"}, status=403)

    q_payload = {
        "model_id": model_id,
        "config": get_effective_user_config(request, user_id),
    }
    q_payload.pop("config_path", None)

    model_metadata, err = _al2_fetch_model_metadata(model_id)
    if err is not None:
        return err

    corpus_id = (
        model_metadata.get("corpus_id")
        or model_metadata.get("metadata", {}).get("corpus_id")
        or model_metadata.get("metadata", {}).get("tr_params", {}).get("corpus_id")
        or ""
    )
    if not corpus_id:
        return JsonResponse(
            {
                "error": "Model missing corpus_id. Please ensure the model was trained with a valid corpus."
            },
            status=400,
        )

    try:
        up0 = time()
        up = requests.get(
            f"{R.API}/data/corpora/{corpus_id}",
            timeout=(3.05, 60),
        )
        up.raise_for_status()
        corpus_training_data = up.json()
        logger.info(
            "al2-bootstrap corpus ok status=%s dt=%.3fs docs=%s",
            up.status_code,
            time() - up0,
            len(corpus_training_data.get("documents", []) or []),
        )
    except requests.Timeout:
        return JsonResponse(
            {"error": "Timeout: The corpus service took too long to respond. Please try again."},
            status=504,
        )
    except requests.HTTPError as e:
        try:
            error_detail = e.response.json() if e.response else {}
            return JsonResponse(
                {
                    "error": error_detail.get("detail")
                    or error_detail.get("error")
                    or f"Corpus not found (HTTP {e.response.status_code})",
                },
                status=e.response.status_code if e.response else 502,
            )
        except Exception:
            return JsonResponse(
                {
                    "error": f"Failed to fetch corpus: HTTP {e.response.status_code if e.response else 'unknown'}"
                },
                status=502,
            )
    except requests.RequestException as e:
        return JsonResponse(
            {"error": f"Connection error: Unable to reach the corpus service. {str(e)}"},
            status=502,
        )

    model_info_key = model_id
    cached_info, info_state = R._cache_get(R._CACHE_MODEL_INFO, model_info_key)
    if cached_info:
        raw_model_info = cached_info
        logger.debug("al2-bootstrap model-info cache %s", info_state)
    else:
        try:
            up0 = time()
            r = requests.post(
                f"{R.API}/queries/model-info", json=q_payload, timeout=60
            )
            r.raise_for_status()
            raw_model_info = r.json()
            R._cache_set(R._CACHE_MODEL_INFO, model_info_key, raw_model_info)
            logger.info(
                "al2-bootstrap model-info fetched status=%s dt=%.3fs (cached)",
                r.status_code,
                time() - up0,
            )
        except requests.HTTPError:
            try:
                error_data = r.json() if r else {}
                return JsonResponse(
                    {
                        "error": error_data.get("detail")
                        or error_data.get("error")
                        or f"Failed to fetch model info (HTTP {r.status_code})",
                    },
                    status=r.status_code,
                )
            except Exception:
                return JsonResponse(
                    {
                        "error": f"Failed to fetch model info: {r.text if r else 'Unknown error'}",
                    },
                    status=r.status_code if r else 502,
                )
        except Exception as e:
            logger.exception("al2-bootstrap model-info proxy error: %s", e)
            return JsonResponse({"error": f"Error fetching model info: {str(e)}"}, status=502)

    topics_info = raw_model_info.get("Topics Info", {}) or {}

    docs_list_raw = corpus_training_data.get("documents", []) or []
    docs_list = [
        d if isinstance(d, dict) else pydantic_to_dict(d) for d in docs_list_raw
    ]

    raw_text_by_id: dict[str, str] = {}
    corpus_data_path = R.DRAFTS_SAVE / corpus_id / "data.json"
    if corpus_data_path.exists():
        try:
            raw_payload = json.loads(corpus_data_path.read_text(encoding="utf-8"))
            raw_items = raw_payload if isinstance(raw_payload, list) else raw_payload.get("documents", raw_payload)
            if isinstance(raw_items, list):
                for item in raw_items:
                    if not isinstance(item, dict):
                        continue
                    doc_id = str(item.get("id", ""))
                    if not doc_id:
                        continue
                    raw_val = item.get("raw_text")
                    if raw_val is not None and str(raw_val).strip():
                        raw_text_by_id[doc_id] = str(raw_val)
        except Exception as e:
            logger.warning("al2-bootstrap could not read raw_text from data.json: %s", e)

    doc_ids = [str(d.get("id")) for d in docs_list if d.get("id") is not None]

    sorted_key = tuple(sorted(doc_ids))
    bulk_key = (model_id, sorted_key)
    cached_bulk, bulk_state = R._cache_get(R._CACHE_THETAS_BULK, bulk_key)
    if cached_bulk:
        doc_thetas = cached_bulk
        logger.debug("al2-bootstrap thetas bulk cache %s | docs=%d", bulk_state, len(doc_ids))
    else:
        payload_thetas = {"docs_ids": ",".join(doc_ids), "model_id": model_id}
        try:
            up0 = time()
            r = requests.post(
                f"{R.API}/queries/thetas-by-docs-ids",
                json=payload_thetas,
                timeout=60,
            )
            r.raise_for_status()
            doc_thetas = r.json()
            R._cache_set(R._CACHE_THETAS_BULK, bulk_key, doc_thetas)
            logger.info(
                "al2-bootstrap thetas fetched status=%s dt=%.3fs docs=%d",
                r.status_code,
                time() - up0,
                len(doc_ids),
            )
        except requests.HTTPError:
            try:
                error_data = r.json() if r else {}
                return JsonResponse(
                    {
                        "error": error_data.get("detail")
                        or error_data.get("error")
                        or f"Failed to fetch document-topic probabilities (HTTP {r.status_code})",
                    },
                    status=r.status_code,
                )
            except Exception:
                return JsonResponse(
                    {
                        "error": f"Failed to fetch document-topic probabilities: {r.text if r else 'Unknown error'}",
                    },
                    status=r.status_code if r else 502,
                )
        except Exception as e:
            logger.exception("al2-bootstrap thetas proxy error: %s", e)
            return JsonResponse(
                {"error": f"Error fetching document-topic probabilities: {str(e)}"},
                status=502,
            )

    topic_colors = build_theme_color_map(topics_info, DASHBOARD_COLOR_CFG)
    topic_labels = {}
    for tid, meta in topics_info.items():
        tid_str = str(tid)
        if isinstance(meta, dict):
            lab = meta.get("Label")
            if lab is not None and str(lab).strip():
                topic_labels[tid_str] = str(lab).strip()
                continue
        topic_labels[tid_str] = f"Topic {tid_str}"

    documents = []
    model_label_by_id = {}
    topic_id_by_id = {}
    topic_confidence_by_id = {}
    for d in docs_list:
        did = str(d.get("id", ""))
        if not did:
            continue
        text = d.get("text", "")
        text_str = str(text) if text is not None else ""
        raw_text = d.get("raw_text") or raw_text_by_id.get(did) or text_str
        documents.append({
            "id": did,
            "text": text_str,
            "raw_text": str(raw_text) if raw_text is not None else text_str,
        })
        theta_row = doc_thetas.get(did)
        if theta_row is None:
            theta_row = doc_thetas.get(str(did)) or {}
        top_tid = _al2_dominant_topic_id(theta_row)
        if top_tid:
            topic_id_by_id[did] = top_tid
            # Confidence = weight of the dominant topic (theta) in [0, 1].
            try:
                conf = float(theta_row.get(top_tid, theta_row.get(str(top_tid), 0)) or 0)
            except (TypeError, ValueError):
                conf = 0.0
            topic_confidence_by_id[did] = max(0.0, min(1.0, conf))
        model_label_by_id[did] = _al2_dominant_topic_label(theta_row, topics_info)

    human_labels = {}
    try:
        lr = requests.get(
            f"{R.API}/active-learning/{model_id}/labels",
            timeout=(3.05, 20),
        )
        lr.raise_for_status()
        body = lr.json() if lr.content else {}
        if isinstance(body, dict):
            human_labels = body.get("labels") or {}
    except requests.RequestException as e:
        logger.warning("al2-bootstrap could not fetch human labels: %s", e)

    logger.info("al2-bootstrap done dt=%.3fs docs=%d", time() - t0, len(documents))
    return JsonResponse(
        {
            "documents": documents,
            "model_label_by_id": model_label_by_id,
            "topic_id_by_id": topic_id_by_id,
            "topic_confidence_by_id": topic_confidence_by_id,
            "topic_colors": topic_colors,
            "topic_labels": topic_labels,
            "human_labels": human_labels,
        },
        status=200,
    )


@login_required
def dashboard(request):
    model_id = request.POST.get("model_id") or request.GET.get("model_id", "")
    denied = _study_model_access_denied(request, model_id)
    if denied:
        return denied
    try:
        return render(
            request,
            "dashboard.html",
            _model_nav_context(model_id),
        )
    except Exception:
        logger.exception(
            "dashboard: render failed (check template syntax); model_id=%r",
            model_id,
        )
        raise


@login_required
def proxy_dashboard_data(request):
    t0 = time()
    payload = _get_json(request) or {}
    # Prefer per-user config blob over static path for advanced dashboard config
    payload["config"] = get_effective_user_config(request, _request_user_id(request))
    payload.pop("config_path", None)

    model_id = payload.get("model_id", "")
    user_id = _request_user_id(request)

    logger.info("get-dashboard start model_id=%s user_id=%s", model_id, user_id)

    if not model_id:
        logger.warning("get-dashboard missing model_id")
        return JsonResponse({"error": "model_id is required"}, status=400)

    denied = _study_model_access_denied(request, model_id, json_response=True)
    if denied:
        return denied

    # model metadata
    model_metadata = None
    try:
        up0 = time()
        up = requests.get(
            f"{R.API}/data/models/{model_id}",
            timeout=(3.05, 60),
        )
        up.raise_for_status()
        model_metadata = up.json()
        logger.info(
            "get-dashboard model meta ok status=%s dt=%.3fs",
            up.status_code,
            time() - up0,
        )
    except requests.HTTPError as e:
        if e.response and e.response.status_code == 404:
            # Model not found via R.API, try direct draft access
            logger.warning("Model not found via R.API, trying direct draft access for %s", model_id)
            try:
                model_draft = drafts.get_draft(model_id, DraftType.model)
                if model_draft:
                    model = drafts.draft_to_model(model_draft)
                    if model:
                        # Read metadata.json directly
                        model_path = R.DRAFTS_SAVE.resolve() / model_id
                        metadata_path = model_path / "metadata.json"
                        metadata_dict = {}
                        
                        if metadata_path.exists():
                            try:
                                with open(metadata_path, 'r') as f:
                                    metadata_dict = json.load(f)
                            except Exception as e:
                                logger.warning("Failed to read metadata.json for %s: %s", model_id, e)
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
                        logger.info("Successfully loaded model %s from drafts directly", model_id)
                    else:
                        raise ValueError("Failed to convert draft to model")
                else:
                    logger.error("Model draft %s not found in drafts system", model_id)
                    raise ValueError("Model not found in drafts")
            except Exception as draft_error:
                logger.exception("Failed to load model from drafts: %s", draft_error)
                try:
                    error_detail = e.response.json() if e.response else {}
                    return JsonResponse(
                        {
                            "error": error_detail.get("detail")
                            or error_detail.get("error")
                            or f"Model not found (HTTP {e.response.status_code})",
                        },
                        status=e.response.status_code if e.response else 502,
                    )
                except Exception:
                    return JsonResponse(
                        {
                            "error": f"Failed to fetch model: HTTP {e.response.status_code if e.response else 'unknown'}"
                        },
                        status=502,
                    )
        else:
            logger.exception("get-dashboard HTTP error fetching model: %s", e)
            try:
                error_detail = e.response.json() if e.response else {}
                return JsonResponse(
                    {
                        "error": error_detail.get("detail")
                        or error_detail.get("error")
                        or f"Model not found (HTTP {e.response.status_code})",
                    },
                    status=e.response.status_code if e.response else 502,
                )
            except Exception:
                return JsonResponse(
                    {
                        "error": f"Failed to fetch model: HTTP {e.response.status_code if e.response else 'unknown'}"
                    },
                    status=502,
                )
    except requests.Timeout:
        logger.exception("get-dashboard timeout fetching model")
        return JsonResponse({"error": "Timeout: The model service took too long to respond. Please try again."}, status=504)
    except requests.RequestException as e:
        logger.exception("get-dashboard error fetching model: %s", e)
        return JsonResponse({"error": f"Connection error: Unable to reach the model service. {str(e)}"}, status=502)
    
    if not model_metadata:
        return JsonResponse({"error": "Failed to load model metadata"}, status=500)

    # Get corpus_id from model metadata
    corpus_id = (
        model_metadata.get("corpus_id") or
        model_metadata.get("metadata", {}).get("corpus_id") or
        model_metadata.get("metadata", {}).get("tr_params", {}).get("corpus_id") or
        ""
    )
    if not corpus_id:
        logger.error("get-dashboard model missing corpus_id. Model metadata keys: %s", list(model_metadata.keys()))
        return JsonResponse({"error": "Model missing corpus_id. Please ensure the model was trained with a valid corpus."}, status=400)

    # corpus data
    try:
        up0 = time()
        # Fetch without owner_id since we've already verified ownership
        up = requests.get(
            f"{R.API}/data/corpora/{corpus_id}",
            timeout=(3.05, 60),
        )
        up.raise_for_status()
        corpus_training_data = up.json()
        logger.info(
            "get-dashboard corpus ok status=%s dt=%.3fs docs=%s",
            up.status_code,
            time() - up0,
            len(corpus_training_data.get("documents", []) or []),
        )
    except requests.Timeout:
        logger.exception("get-dashboard timeout fetching corpus")
        return JsonResponse({"error": "Timeout: The corpus service took too long to respond. Please try again."}, status=504)
    except requests.HTTPError as e:
        logger.exception("get-dashboard HTTP error fetching corpus: %s", e)
        try:
            error_detail = e.response.json() if e.response else {}
            return JsonResponse(
                {
                    "error": error_detail.get("detail")
                    or error_detail.get("error")
                    or f"Corpus not found (HTTP {e.response.status_code})",
                },
                status=e.response.status_code if e.response else 502,
            )
        except Exception:
            return JsonResponse(
                {
                    "error": f"Failed to fetch corpus: HTTP {e.response.status_code if e.response else 'unknown'}"
                },
                status=502,
            )
    except requests.RequestException as e:
        logger.exception("get-dashboard error fetching corpus: %s", e)
        return JsonResponse({"error": f"Connection error: Unable to reach the corpus service. {str(e)}"}, status=502)

    # model info (with cache)
    model_info_key = model_id
    cached_info, info_state = R._cache_get(R._CACHE_MODEL_INFO, model_info_key)
    if cached_info:
        raw_model_info = cached_info
        logger.debug("get-dashboard model-info cache %s", info_state)
    else:
        try:
            up0 = time()
            r = requests.post(f"{R.API}/queries/model-info", json=payload, timeout=60)
            r.raise_for_status()
            raw_model_info = r.json()
            R._cache_set(R._CACHE_MODEL_INFO, model_info_key, raw_model_info)
            logger.info(
                "get-dashboard model-info fetched status=%s dt=%.3fs (cached)",
                r.status_code,
                time() - up0,
            )
        except requests.HTTPError as e:
            logger.error("get-dashboard model-info HTTP error: %s", e)
            try:
                error_data = r.json() if r else {}
                return JsonResponse(
                    {
                        "error": error_data.get("detail")
                        or error_data.get("error")
                        or f"Failed to fetch model info (HTTP {r.status_code})",
                    },
                    status=r.status_code,
                )
            except Exception:
                return JsonResponse(
                    {
                        "error": f"Failed to fetch model info: {r.text if r else 'Unknown error'}",
                    },
                    status=r.status_code if r else 502,
                )
        except Exception as e:
            logger.exception("get-dashboard model-info proxy error: %s", e)
            return JsonResponse({"error": f"Error fetching model info: {str(e)}"}, status=502)

    docs_list_raw = corpus_training_data.get("documents", []) or []
    docs_list = [
        d if isinstance(d, dict) else pydantic_to_dict(d) for d in docs_list_raw
    ]
    doc_ids = [str(d.get("id")) for d in docs_list if d.get("id") is not None]

    sorted_key = tuple(sorted(doc_ids))
    bulk_key = (model_id, sorted_key)
    cached_bulk, bulk_state = R._cache_get(R._CACHE_THETAS_BULK, bulk_key)
    if cached_bulk:
        doc_thetas = cached_bulk
        logger.debug(
            "get-dashboard thetas bulk cache %s | docs=%d", bulk_state, len(doc_ids)
        )
    else:
        payload_thetas = {"docs_ids": ",".join(doc_ids), "model_id": model_id}
        try:
            up0 = time()
            r = requests.post(
                f"{R.API}/queries/thetas-by-docs-ids", json=payload_thetas, timeout=60
            )
            r.raise_for_status()
            doc_thetas = r.json()
            R._cache_set(R._CACHE_THETAS_BULK, bulk_key, doc_thetas)
            logger.info(
                "get-dashboard thetas fetched status=%s dt=%.3fs (cached) docs=%d",
                r.status_code,
                time() - up0,
                len(doc_ids),
            )
        except requests.HTTPError as e:
            logger.error("get-dashboard thetas HTTP error: %s", e)
            try:
                error_data = r.json() if r else {}
                return JsonResponse(
                    {
                        "error": error_data.get("detail")
                        or error_data.get("error")
                        or f"Failed to fetch document-topic probabilities (HTTP {r.status_code})",
                    },
                    status=r.status_code,
                )
            except Exception:
                return JsonResponse(
                    {
                        "error": f"Failed to fetch document-topic probabilities: {r.text if r else 'Unknown error'}",
                    },
                    status=r.status_code if r else 502,
                )
        except Exception as e:
            logger.exception("get-dashboard thetas proxy error: %s", e)
            return JsonResponse({"error": f"Error fetching document-topic probabilities: {str(e)}"}, status=502)

    try:
        model_training_corpus = pydantic_to_dict(corpus_training_data) or {}
        bundle = to_dashboard_bundle(
            raw_model_info,
            model_id,
            model_metadata=model_metadata.get("metadata", {}),
            model_training_corpus=model_training_corpus,
            doc_thetas=doc_thetas,
            logger=logger,
        )
        logger.info("get-dashboard done dt=%.3fs", time() - t0)
        return JsonResponse(bundle, status=200)
    except Exception as e:
        logger.exception("get-dashboard bundling error: %s", e)
        return JsonResponse({
            "error": f"Failed to build dashboard data: {str(e)}",
            "detail": "An error occurred while processing the dashboard data. Check server logs for details."
        }, status=500)


@login_required
def text_info(request):
    """
    Get text and model information for a given document according to a topic model.
    Used when a document row is clicked in dashboard.html.
    """
    t0 = time()
    payload = _get_json(request) or {}
    payload["config"] = get_effective_user_config(request, _request_user_id(request))
    payload.pop("config_path", None)
    model_id = payload.get("model_id", "")
    doc_id = str(payload.get("document_id", "")).strip()

    logger.info("TEXT-INFO start model_id=%s doc_id=%s", model_id, doc_id)

    if not model_id or not doc_id:
        logger.warning("TEXT-INFO missing required fields")
        return JsonResponse({"detail": "model_id and document_id are required."}, status=400)
    user_id = _request_user_id(request)
    if not user_id:
        return JsonResponse({"detail": "Not authenticated"}, status=401)

    # model meta info (cached) — fetch first so we can check ownership from response
    model_entry = R._CACHE_MODELS.get(model_id)
    if not model_entry or (time() - model_entry["ts"] > R.CACHE_TTL):
        try:
            up0 = time()
            up = requests.get(
                f"{R.API}/data/models/{model_id}",
                timeout=(3.05, 30),
            )
            up.raise_for_status()
            model = up.json()
            R._CACHE_MODELS[model_id] = {"ts": time(), "data": model}
            logger.info(
                "TEXT-INFO model meta fetched status=%s dt=%.3fs (cached)",
                up.status_code,
                time() - up0,
            )
        except requests.RequestException as e:
            logger.exception("TEXT-INFO fetch model error: %s", e)
            return JsonResponse({"detail": f"Failed to fetch model info: {e}"}, status=502)
    else:
        model = model_entry["data"]
        logger.debug("TEXT-INFO model meta cache hit")

    # Allow if user owns model, model has no owner_id (legacy), or for any authenticated user
    # (viewing document text is read-only; ownership enforced on create/delete)
    model_owner = model.get("owner_id") or (model.get("metadata") or {}).get("owner_id")
    if model_owner is not None and model_owner != user_id:
        # Still allow: text-info is read-only document view, user is authenticated
        logger.debug("TEXT-INFO model owner %s != user %s; allowing read-only access", model_owner, user_id)

    corpus_id = model.get("corpus_id", "")
    if not corpus_id:
        logger.error("TEXT-INFO model missing corpus_id")
        return JsonResponse({"detail": "Model missing corpus_id."}, status=400)

    # corpus info (cached) — fetch first so we can check ownership from response
    corpus_entry = R._CACHE_CORPORA.get(corpus_id)
    if not corpus_entry or (time() - corpus_entry["ts"] > R.CACHE_TTL):
        try:
            up0 = time()
            up = requests.get(
                f"{R.API}/data/corpora/{corpus_id}",
                timeout=(3.05, 60),
            )
            up.raise_for_status()
            corpus = up.json()
            R._CACHE_CORPORA[corpus_id] = {"ts": time(), "data": corpus}
            logger.info(
                "TEXT-INFO corpus fetched status=%s dt=%.3fs (cached)",
                up.status_code,
                time() - up0,
            )
        except requests.RequestException as e:
            logger.exception("TEXT-INFO fetch corpus error: %s", e)
            return JsonResponse({"detail": f"Failed to fetch corpus info: {e}"}, status=502)
    else:
        corpus = corpus_entry["data"]
        logger.debug("TEXT-INFO corpus cache hit")

    # Corpus access: text-info is read-only; allow any authenticated user to view document text
    corpus_owner = corpus.get("owner_id") or (corpus.get("metadata") or {}).get("owner_id")
    if corpus_owner is not None and corpus_owner != user_id:
        logger.debug("TEXT-INFO corpus owner %s != user %s; allowing read-only access", corpus_owner, user_id)

    docs = corpus.get("documents", [])
    # Prefer the original document; ``text`` is the preprocessed form.
    doc_text = next(
        (
            (d.get("raw_text") or d.get("text") or "")
            for d in docs
            if str(d.get("id")) == doc_id
        ),
        "",
    )
    if not doc_text:
        doc_text = f"(Text not found for document {doc_id})"

    # thetas for this document (cached)
    theta_key = (model_id, doc_id)
    cached_theta, theta_state = R._cache_get(R._CACHE_THETAS, theta_key)
    if cached_theta:
        thetas_by_doc = {doc_id: cached_theta}
        logger.debug("TEXT-INFO thetas cache %s for doc_id=%s", theta_state, doc_id)
    else:
        try:
            up0 = time()
            r = requests.post(
                f"{R.API}/queries/thetas-by-docs-ids",
                json={
                    "model_id": model_id,
                    "config_path": "static/config/config.yaml",
                    "docs_ids": doc_id,
                },
                timeout=60,
            )
            r.raise_for_status()
            thetas_by_doc = r.json()
            R._cache_set(R._CACHE_THETAS, theta_key, thetas_by_doc.get(doc_id) or {})
            logger.info(
                "TEXT-INFO thetas fetched status=%s dt=%.3fs (cached)",
                r.status_code,
                time() - up0,
            )
        except requests.RequestException as e:
            logger.exception("TEXT-INFO fetch thetas error: %s", e)
            return JsonResponse({"detail": f"Failed to fetch thetas: {e}"}, status=502)

    doc_thetas = thetas_by_doc.get(doc_id) or {}

    # model-info (cached)
    model_info_key = model_id
    cached_info, info_state = R._cache_get(R._CACHE_MODEL_INFO, model_info_key)
    if cached_info:
        raw_model_info = cached_info
        logger.debug("TEXT-INFO model-info cache %s", info_state)
    else:
        try:
            up0 = time()
            r = requests.post(f"{R.API}/queries/model-info", json=payload, timeout=60)
            r.raise_for_status()
            raw_model_info = r.json()
            R._cache_set(R._CACHE_MODEL_INFO, model_info_key, raw_model_info)
            logger.info(
                "TEXT-INFO model-info fetched status=%s dt=%.3fs (cached)",
                r.status_code,
                time() - up0,
            )
        except requests.HTTPError:
            try:
                return JsonResponse(r.json(), status=r.status_code)
            except Exception:
                return JsonResponse({"detail": r.text}, status=r.status_code)
        except Exception as e:
            logger.exception("TEXT-INFO model-info proxy error: %s", e)
            return JsonResponse({"detail": f"Proxy error: {e}"}, status=502)

    if not doc_thetas:
        logger.info("TEXT-INFO no thetas for doc_id=%s dt=%.3fs", doc_id, time() - t0)
        return JsonResponse(
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

    logger.info(
        "TEXT-INFO done doc_id=%s top_theme=%s dt=%.3fs",
        doc_id,
        top_theme.get("theme_id"),
        time() - t0,
    )

    return JsonResponse(
        {
            "theme": top_theme["label"],
            "top_themes": top_themes,
            "rationale": "",
            "text": doc_text,
        }
    )


@login_required
def infer_text(request):
    """
    Handles the initial inference request from the frontend.
    1. Sends the request body to the external API's inference endpoint.
    2. Receives a job_id.
    3. Polls the API's status endpoint using the job_id until results are available or a timeout occurs.
    4. Returns the final inference results to the frontend.
    """
    try:
        # 1. Get the JSON payload from the frontend request
        request_data = _get_json(request)
        if not request_data:
            return JsonResponse({"error": "Invalid or missing JSON payload"}, status=400)

        # 2. Send the exact payload to the external API's inference endpoint
        inference_url = f"{R.API}/infer/json"
        
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
            return JsonResponse({"error": "Inference API did not return a job_id"}, status=500)

        print(f"Inference job submitted. Job ID: {job_id}")
        
        
        # 3. Poll the job status (short delay) before returning
        status_url = f"{R.API}/status/jobs/{job_id}"
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
                return JsonResponse(job_status.get("results", job_status))
            if status in ["failed", "error"]:
                print(f"Job {job_id} failed.")
                return JsonResponse({"error": "Inference job failed", "details": job_status}, status=500)

            if monotonic() >= deadline:
                return JsonResponse({"status": last_status or "running", "details": last_job_status, "timeout": True}, status=202)

            sleep(poll_interval)

    except requests.exceptions.HTTPError as e:
        # Handle exceptions from the external API calls
        return JsonResponse(
            {"error": f"External API HTTP Error: {e.response.text}", "status_code": e.response.status_code},
            status=e.response.status_code,
        )
    except requests.exceptions.RequestException as e:
        # Handle network-level errors (e.g., connection refused)
        return JsonResponse({"error": f"Error communicating with external API: {e}"}, status=503)
    except Exception as e:
        # Catch any other unexpected errors
        return JsonResponse({"error": f"An unexpected error occurred: {e}"}, status=500)


@login_required
def save_settings(request):
    payload = _get_json(request) or {}
    logger.info("Dummy /save-settings received: %s", payload)
    return JsonResponse({"ok": True, "echo": payload}, status=200)


def _safe_dict_local(val):
    """Return val if it's a dict, else {} (handles ruamel.yaml scalars etc.)."""
    return val if isinstance(val, dict) else {}


def _chat_llm(provider: str, model: str, system_content: str, user_content: str, host: str = None, api_key: str = None):
    """
    Call configured LLM (Ollama or OpenAI) for the model assistant.
    api_key: optional OpenAI API key from chat UI; if not set, uses OPENAI_API_KEY env for GPT.
    Returns (response_text, None) on success or (None, error_message) on failure.
    """
    llm_cfg = R._safe_dict(R.RAW_CONFIG.get("llm"))
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

# Startup: confirm OpenAI key visibility from .env/config (no key value logged)
_openai_key_at_startup = _get_openai_key_from_env()
logger.info("OpenAI API key at startup: %s", "configured" if _openai_key_at_startup else "not set (set in .env or llm.gpt.api_key in config)")


def _get_chat_llm_defaults():
    """Return default provider, model, host from config (for RAG chat). Used when user has not set overrides."""
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
    return provider, chat_model, host, ollama_list, gpt_list, R._safe_dict(llm_cfg.get("ollama")).get("host")


@login_required
def chat_openai_key_status(request):
    """Diagnostic: report where OPENAI_API_KEY was found (or not). No key value is returned."""
    project_root = Path(__file__).resolve().parent.parent
    env_project = project_root / ".env"
    env_cwd = Path.cwd() / ".env"
    tried = [
        {"path": str(env_project), "exists": env_project.exists()},
        {"path": str(env_cwd), "exists": env_cwd.exists()},
    ]
    ui_root = Path(__file__).resolve().parents[1]
    tried.append({"path": str(ui_root / ".env"), "exists": (ui_root / ".env").exists()})
    tried.append({"path": str(ui_root.parent / ".env"), "exists": (ui_root.parent / ".env").exists()})
    key = _get_openai_key_from_env()
    gpt_cfg = R._safe_dict(R.RAW_CONFIG.get("llm", {})).get("gpt") or {}
    config_has_key = bool((gpt_cfg.get("api_key") or "").strip()) if isinstance(gpt_cfg, dict) else False
    path_api_key = (gpt_cfg.get("path_api_key") or ".env") if isinstance(gpt_cfg, dict) else ".env"
    return JsonResponse({
        "key_configured": bool(key) or config_has_key,
        "key_from_config": config_has_key,
        "env_has_key": bool((os.environ.get("OPENAI_API_KEY") or "").strip()),
        "path_api_key_from_config": path_api_key,
        "note": "Labeler (Prompter) uses load_dotenv(path_api_key) then os.getenv('OPENAI_API_KEY'); UI uses same.",
        "project_root_from_file": str(project_root),
        "cwd": str(Path.cwd()),
        "server_root_path": str(ui_root),
        "tried_env_paths": tried,
    }, status=200)


@login_required
def chat_llm_options(request):
    """Return available RAG/chat LLM options (providers, models, default host) for the chat UI. Does not expose secrets."""
    llm_cfg = R._safe_dict(R.RAW_CONFIG.get("llm"))
    ollama_cfg = R._safe_dict(llm_cfg.get("ollama"))
    gpt_cfg = R._safe_dict(llm_cfg.get("gpt"))
    tm_gen = R._safe_dict(R._safe_dict(R.RAW_CONFIG.get("topic_modeling")).get("general"))
    default_host = tm_gen.get("llm_server") or ollama_cfg.get("host", "http://host.docker.internal:11434")
    if default_host is not None and not isinstance(default_host, str):
        default_host = str(default_host)
    return JsonResponse({
        "ollama": {
            "models": list(ollama_cfg.get("available_models") or []),
            "default_host": default_host,
        },
        "gpt": {
            "models": list(gpt_cfg.get("available_models") or []),
        },
    }, status=200)


@login_required
def get_chat_messages(request):
    """Return saved chat messages for the current user and the given model_id (query param)."""
    model_id = (request.GET.get("model_id") or "").strip()
    if not model_id:
        return JsonResponse({"error": "model_id is required"}, status=400)
    user_id = _request_user_id(request)
    if not user_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    try:
        messages = ChatMessage.objects.filter(user_id=user_id, model_id=model_id).order_by(
            "created_at"
        )
        return JsonResponse({
            "messages": [
                {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat() + "Z" if m.created_at else None}
                for m in messages
            ]
        }, status=200)
    except Exception as e:
        logger.exception("Failed to load chat messages: %s", e)
        return JsonResponse({"error": "Failed to load messages"}, status=500)


@login_required
def chat(request):
    """
    Chat interface endpoint for topic modeling assistance.
    Uses the LLM set in the request (llm_settings from chat UI) or config. No rule-based fallback.
    """
    try:
        payload = _get_json(request) or {}
        message = payload.get("message", "").strip()
        model_id = payload.get("model_id", "")
        context = payload.get("context", {})
        llm_settings = R._safe_dict(payload.get("llm_settings"))
        
        if not message:
            return JsonResponse({"error": "Message is required"}, status=400)
        
        logger.info("Chat request: model_id=%s, message_length=%d", model_id, len(message))
        
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
                user_config = _load_user_config(_request_user_id(request)) or {}
                stored_llm = user_config.get("llm_config") or {}
                stored_key = (stored_llm.get("api_key") or "").strip()
            from_config = (R._safe_dict(R.RAW_CONFIG.get("llm", {})).get("gpt") or {})
            if isinstance(from_config, dict):
                config_key = (from_config.get("api_key") or "").strip()
            else:
                config_key = ""
            openai_key = from_ui or stored_key or config_key or _get_openai_key_from_env()
            response_text, err = _chat_llm(provider, chat_model, system_content, user_content, host=host, api_key=openai_key)
            if response_text:
                logger.info("Chat response from LLM: length=%d", len(response_text))
                user_id = _request_user_id(request)
                if user_id and model_id:
                    try:
                        ChatMessage(
                            user_id=user_id,
                            model_id=model_id.strip(),
                            role="user",
                            content=message,
                        ).save()
                        ChatMessage(
                            user_id=user_id,
                            model_id=model_id.strip(),
                            role="assistant",
                            content=response_text,
                        ).save()
                    except Exception as save_err:
                        logger.warning("Failed to save chat messages: %s", save_err)
                        
                return JsonResponse({"response": response_text}, status=200)
            if err:
                logger.warning("Chat LLM error: %s", err)
        
        # No rule-based fallback: LLM only. If we get here, model is missing or the call failed.
        if not chat_model:
            return JsonResponse({
                "response": "No language model is configured for the assistant. Open the chat sidebar, expand \"Assistant LLM\", and choose a provider and model (and set the Ollama host if using Ollama). You can also set defaults in config under topic_modeling.general."
            }, status=200)
        return JsonResponse({
            "response": f"The assistant could not get a response from the model ({err or 'unknown error'}). Check that the LLM service is running and reachable, then try again."
        }, status=200)
    except Exception as e:
        logger.exception("Chat error: %s", e)
        return JsonResponse({"error": str(e)}, status=500)


@login_required
@login_required
def rename_topic(request, model_id: str, topic_id: int):
    """Store a per-user label for one topic of one model.

    Renames are private to the user. They used to be proxied upstream, but that
    endpoint does not exist on the API and the failure was swallowed, so no
    rename ever persisted.
    """
    payload = _get_json(request) or {}
    new_label = str(payload.get("new_label") or "").strip()
    owner_id = _request_user_id(request)

    if not new_label:
        return JsonResponse({"error": "new_label is required"}, status=400)
    if not model_id:
        return JsonResponse({"error": "model_id is required"}, status=400)
    if not owner_id:
        return JsonResponse({"error": "User not authenticated"}, status=401)
    if not R._verify_model_ownership(model_id, owner_id):
        return JsonResponse({"error": "Access denied: You do not own this model"}, status=403)

    TopicRename.objects.update_or_create(
        user=request.user,
        model_id=model_id,
        topic_id=str(topic_id),
        defaults={"label": new_label[:255]},
    )
    return JsonResponse(
        {"model_id": model_id, "topic_id": str(topic_id), "new_label": new_label},
        status=200,
    )


@login_required
def get_topic_renames(request, model_id: str):
    """Return this user's topic renames for *model_id* as ``{topic_id: label}``."""
    owner_id = _request_user_id(request)

    if not model_id:
        return JsonResponse({"error": "model_id is required"}, status=400)
    if not owner_id:
        return JsonResponse({"error": "User not authenticated"}, status=401)

    labels = dict(
        TopicRename.objects.filter(user=request.user, model_id=model_id).values_list(
            "topic_id", "label"
        )
    )
    return JsonResponse({"topic_labels": labels}, status=200)


# Note: Encoded routes are handled by the custom url_for which generates encoded paths.
# The server accepts both original and encoded routes for backward compatibility.
# For better security, consider registering routes with both original and encoded paths
# using the register_encoded_route() helper function above.




# ------------------------------------------------------------------------------
# Study flow (participant screens: instructions -> task -> post-study -> done)
# ------------------------------------------------------------------------------
_EMAIL_HEADER_RE = re.compile(r"^\s*(subject|date|sent)\s*:\s*(.+)$", re.IGNORECASE)


def _study_email_meta(doc: dict) -> tuple[str, str]:
    """Best-effort ``(subject, date)`` for one corpus document.

    Corpus records carry neither field today, so fall back to parsing RFC-822
    style headers out of ``raw_text`` and finally to the first line of text.
    """
    subject = str(doc.get("subject") or "").strip()
    date = str(doc.get("date") or "").strip()
    if subject and date:
        return subject, date

    body = str(doc.get("raw_text") or doc.get("text") or "")
    for line in body.splitlines()[:12]:
        m = _EMAIL_HEADER_RE.match(line)
        if not m:
            continue
        field, value = m.group(1).lower(), m.group(2).strip()
        if field == "subject" and not subject:
            subject = value
        elif field in ("date", "sent") and not date:
            date = value

    if not subject:
        first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        subject = (first[:97] + "...") if len(first) > 100 else first
    return subject, date


def _study_session(request):
    return study_flow.get_or_create_session(request.user)


@login_required
def study_instructions(request):
    session = _study_session(request)
    corpus_size = None
    try:
        corpus_size = _study_corpus_size(session.model_id)
    except Exception:
        logger.warning("study: could not determine corpus size", exc_info=True)
    return render(
        request,
        "instructions.html",
        {
            "task_minutes": study_flow.task_minutes(),
            "corpus_size": corpus_size,
            "session": session,
        },
    )


@login_required
def study_begin(request):
    """Start the exploration timer. Idempotent: re-POSTing never extends it."""
    if request.method != "POST":
        return redirect(reverse("web:study_instructions"))
    session = _study_session(request)
    study_flow.begin_task(session, request.user)
    _log_audit(request, "study_begin", "study", str(request.user.pk), session.model_id)
    return redirect(reverse("web:study_task"))


@login_required
def study_task(request):
    """Canonical URL for the task stage; forwards to the participant's dashboard."""
    session = _study_session(request)
    model_id = session.model_id or get_study_model_id(request.user) or ""
    url = reverse("web:dashboard")
    if model_id:
        url += "?" + urlencode({"model_id": model_id})
    return redirect(url)


@login_required
def study_timer(request):
    session = _study_session(request)
    if session.stage == ParticipantSession.STAGE_TASK and session.expired:
        study_flow.finish_task(session, early=False)
    expired = session.stage != ParticipantSession.STAGE_TASK
    return JsonResponse(
        {
            "remaining": 0 if expired else session.seconds_remaining,
            "expired": expired,
            "stage": session.stage,
            "redirect": study_flow.stage_redirect_url(session),
        }
    )


@login_required
def study_finish(request):
    """Finish the exploration task early and move to the questions."""
    if request.method != "POST":
        return redirect(reverse("web:study_task"))
    session = _study_session(request)
    was_running = session.stage == ParticipantSession.STAGE_TASK
    study_flow.finish_task(session, early=was_running and not session.expired)
    _log_audit(request, "study_finish_task", "study", str(request.user.pk), session.stage)
    target = study_flow.stage_redirect_url(session)
    if _wants_json_request(request):
        return JsonResponse({"redirect": target, "stage": session.stage})
    return redirect(target)


def _wants_json_request(request) -> bool:
    return (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or "application/json" in (request.headers.get("Accept") or "")
    )


@login_required
def study_done(request):
    session = _study_session(request)
    response = StudyResponse.objects.filter(pk=request.user.pk).first()
    return render(request, "study/done.html", {"session": session, "response": response})


def _study_corpus_size(model_id: str) -> int | None:
    """Document count for the corpus behind *model_id*, or None if unavailable."""
    if not model_id:
        return None
    cached, _state = R._cache_get(R._CACHE_MODEL_INFO, f"corpus_size:{model_id}")
    if cached:
        return cached.get("n")

    model_metadata, err = _al2_fetch_model_metadata(model_id)
    if err is not None:
        return None
    corpus_id = _study_corpus_id_from_metadata(model_metadata)
    if not corpus_id:
        return None
    data_path = R.DRAFTS_SAVE / corpus_id / "data.json"
    if not data_path.exists():
        return None
    try:
        raw = json.loads(data_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    items = raw if isinstance(raw, list) else raw.get("documents", [])
    n = len(items) if isinstance(items, list) else None
    if n is not None:
        R._cache_set(R._CACHE_MODEL_INFO, f"corpus_size:{model_id}", {"n": n})
    return n


def _study_corpus_id_from_metadata(model_metadata: dict) -> str:
    md = model_metadata.get("metadata", {}) or {}
    return (
        model_metadata.get("corpus_id")
        or md.get("corpus_id")
        or (md.get("tr_params", {}) or {}).get("corpus_id")
        or ""
    )


def _study_base_documents(model_id: str):
    """Corpus documents plus the model's dominant-topic label for each.

    Identical for everyone in a study group, so it is cached per model. The
    per-participant overlay (renames, active-learning labels) is applied on top
    by :func:`study_docs`.

    Returns ``(docs, topic_label_by_doc, topic_id_by_doc, error_response)``.
    """
    cache_key = f"study_docs:{model_id}"
    cached, _state = R._cache_get(R._CACHE_MODEL_INFO, cache_key)
    if cached:
        return cached["docs"], cached["topic_label"], cached["topic_id"], None

    model_metadata, err = _al2_fetch_model_metadata(model_id)
    if err is not None:
        return None, None, None, err

    corpus_id = _study_corpus_id_from_metadata(model_metadata)
    if not corpus_id:
        return None, None, None, JsonResponse(
            {"error": "Model is missing corpus_id."}, status=400
        )

    try:
        up = requests.get(f"{R.API}/data/corpora/{corpus_id}", timeout=(3.05, 60))
        up.raise_for_status()
        corpus = up.json()
    except requests.RequestException as e:
        logger.exception("study-docs corpus fetch failed: %s", e)
        return None, None, None, JsonResponse(
            {"error": "Unable to reach the corpus service."}, status=502
        )

    docs_raw = corpus.get("documents", []) or []
    docs_list = [d if isinstance(d, dict) else pydantic_to_dict(d) for d in docs_raw]

    # raw_text is not returned by the corpus endpoint; read it off disk so
    # subject/date parsing has real email text to work with.
    raw_text_by_id: dict[str, str] = {}
    data_path = R.DRAFTS_SAVE / corpus_id / "data.json"
    if data_path.exists():
        try:
            raw_payload = json.loads(data_path.read_text(encoding="utf-8"))
            raw_items = (
                raw_payload if isinstance(raw_payload, list)
                else raw_payload.get("documents", [])
            )
            for item in raw_items or []:
                if isinstance(item, dict) and item.get("id"):
                    val = item.get("raw_text")
                    if val:
                        raw_text_by_id[str(item["id"])] = str(val)
        except Exception as e:
            logger.warning("study-docs could not read raw_text: %s", e)

    doc_ids = [str(d.get("id")) for d in docs_list if d.get("id") is not None]

    # Dominant topic per document.
    bulk_key = (model_id, tuple(sorted(doc_ids)))
    cached_bulk, _bulk_state = R._cache_get(R._CACHE_THETAS_BULK, bulk_key)
    if cached_bulk:
        doc_thetas = cached_bulk
    else:
        try:
            r = requests.post(
                f"{R.API}/queries/thetas-by-docs-ids",
                json={"docs_ids": ",".join(doc_ids), "model_id": model_id},
                timeout=60,
            )
            r.raise_for_status()
            doc_thetas = r.json()
            R._cache_set(R._CACHE_THETAS_BULK, bulk_key, doc_thetas)
        except Exception as e:
            logger.warning("study-docs thetas unavailable, falling back: %s", e)
            doc_thetas = {}

    info_cached, _s = R._cache_get(R._CACHE_MODEL_INFO, model_id)
    topics_info = (info_cached or {}).get("Topics Info", {}) or {}
    if not topics_info:
        try:
            r = requests.post(
                f"{R.API}/queries/model-info",
                json={"model_id": model_id, "config": R.RAW_CONFIG},
                timeout=60,
            )
            r.raise_for_status()
            info = r.json()
            R._cache_set(R._CACHE_MODEL_INFO, model_id, info)
            topics_info = info.get("Topics Info", {}) or {}
        except Exception as e:
            logger.warning("study-docs model-info unavailable: %s", e)

    docs = []
    topic_label_by_doc: dict[str, str] = {}
    topic_id_by_doc: dict[str, str] = {}
    for d in docs_list:
        did = str(d.get("id", ""))
        if not did:
            continue
        enriched = dict(d)
        if did in raw_text_by_id:
            enriched.setdefault("raw_text", raw_text_by_id[did])
        subject, date = _study_email_meta(enriched)
        docs.append(
            {
                "id": did,
                "display_id": str(d.get("original_id") or did),
                "subject": subject,
                "date": date,
            }
        )
        theta_row = doc_thetas.get(did) or {}
        tid = _al2_dominant_topic_id(theta_row)
        if tid:
            topic_id_by_doc[did] = str(tid)
        label = _al2_dominant_topic_label(theta_row, topics_info)
        if label:
            topic_label_by_doc[did] = str(label)

    R._cache_set(
        R._CACHE_MODEL_INFO,
        cache_key,
        {"docs": docs, "topic_label": topic_label_by_doc, "topic_id": topic_id_by_doc},
    )
    return docs, topic_label_by_doc, topic_id_by_doc, None


@login_required
def study_docs(request):
    """Themes and documents for the post-study citation picker.

    Themes are per participant. Precedence for a document's theme, highest first:
    their own active-learning label, their rename of the document's dominant
    topic, then the model's default topic label.
    """
    session = _study_session(request)
    model_id = session.model_id or get_study_model_id(request.user) or ""
    if not model_id:
        return JsonResponse({"themes": [], "emails": []})

    docs, topic_label_by_doc, topic_id_by_doc, err = _study_base_documents(model_id)
    if err is not None:
        return err

    renames = dict(
        TopicRename.objects.filter(user=request.user, model_id=model_id).values_list(
            "topic_id", "label"
        )
    )

    human_labels: dict = {}
    if user_has_feature(request.user, FEATURE_ACTIVE_LEARNING):
        try:
            lr = requests.get(
                f"{R.API}/active-learning/{model_id}/labels",
                params={"user_id": str(request.user.pk)},
                timeout=(3.05, 20),
            )
            lr.raise_for_status()
            body = lr.json() if lr.content else {}
            if isinstance(body, dict):
                human_labels = body.get("labels") or {}
        except requests.RequestException as e:
            logger.warning("study-docs could not fetch human labels: %s", e)

    emails = []
    seen_themes = []
    for doc in docs:
        did = doc["id"]
        theme = human_labels.get(did)
        if not theme:
            tid = topic_id_by_doc.get(did)
            if tid and tid in renames:
                theme = renames[tid]
        if not theme:
            theme = topic_label_by_doc.get(did)
        theme = str(theme).strip() if theme else study_flow.UNCLASSIFIED_THEME

        if theme not in seen_themes:
            seen_themes.append(theme)
        emails.append({**doc, "theme": theme})

    # Keep Unclassified last so it never leads the picker.
    themes = [t for t in seen_themes if t != study_flow.UNCLASSIFIED_THEME]
    themes.sort(key=str.lower)
    if study_flow.UNCLASSIFIED_THEME in seen_themes:
        themes.append(study_flow.UNCLASSIFIED_THEME)

    return JsonResponse({"themes": themes, "emails": emails})


def _study_parse_citations(raw: str) -> list:
    """Parse the hidden citation field (comma-separated document ids)."""
    if not raw:
        return []
    out, seen = [], set()
    for part in str(raw).split(","):
        val = part.strip()
        if val and val not in seen:
            seen.add(val)
            out.append(val)
    return out[:50]


@login_required
def study_post(request):
    """Render and accept the post-study questionnaire."""
    session = _study_session(request)

    if study_flow.has_submitted(request.user):
        return redirect(reverse("web:study_done"))

    if request.method != "POST":
        return render(
            request,
            "post_study.html",
            {"session": session, "errors": {}, "submitted": {}},
        )

    post = request.POST
    errors: dict[str, str] = {}

    tf_answers = {}
    for key in study_flow.TF_KEY:
        val = (post.get(key) or "").strip().lower()
        if not val:
            errors[key] = "Please choose an answer."
        elif val not in study_flow.TF_VALID_VALUES:
            errors[key] = "Invalid answer."
        else:
            tf_answers[key] = val

    open_answers = {}
    for key in study_flow.OPEN_QUESTIONS:
        text = (post.get(key) or "").strip()
        if not text:
            errors[key] = "Please answer this question."
        open_answers[key] = {
            "text": text,
            "citations": _study_parse_citations(post.get(f"{key}_citations", "")),
        }

    if errors:
        # Re-render with what they typed rather than losing eight answers.
        return render(
            request,
            "post_study.html",
            {
                "session": session,
                "errors": errors,
                "submitted": post.dict(),
                "submitted_citations": {
                    k: post.get(f"{k}_citations", "") for k in study_flow.OPEN_QUESTIONS
                },
            },
            status=400,
        )

    try:
        form_duration_ms = int(post.get("form_duration_ms") or 0) or None
    except (TypeError, ValueError):
        form_duration_ms = None

    StudyResponse.objects.update_or_create(
        user=request.user,
        defaults={
            "study_group": session.study_group,
            "model_id": session.model_id,
            "tf_answers": tf_answers,
            "tf_score": study_flow.score_tf(tf_answers),
            "open_answers": open_answers,
            "payload": post.dict(),
            "task_duration_s": session.task_duration_s,
            "form_duration_ms": form_duration_ms,
        },
    )
    study_flow.complete_study(session)
    _log_audit(request, "study_submit", "study", str(request.user.pk), session.model_id)
    return redirect(reverse("web:study_done"))


# ------------------------------------------------------------------------------
# Interaction telemetry
# ------------------------------------------------------------------------------
_EVENT_BATCH_LIMIT = 200
_EVENT_ACTIONS = {
    "click", "dblclick", "contextmenu",
    "key", "shortcut", "search",
    "change", "submit",
    "copy", "paste", "select",
    "modal_open", "modal_close",
    "scroll", "resize", "focus", "blur", "page",
}


@login_required
def study_events(request):
    """Accept a batch of interaction events from a participant's browser.

    Batched and fire-and-forget: the client keeps a queue and flushes on a timer
    or at page unload, so a slow or failed write never blocks the task.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    user = request.user
    if not is_study_participant(user):
        # Nothing to measure for staff; accept and discard so the client's
        # beacon never sees an error.
        return JsonResponse({"stored": 0})

    payload = _get_json(request) or {}
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        return JsonResponse({"error": "events must be a list"}, status=400)

    session = study_flow.get_or_create_session(user)
    rows = []
    for item in raw_events[:_EVENT_BATCH_LIMIT]:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").strip()[:32]
        if action not in _EVENT_ACTIONS:
            continue
        modality = str(item.get("modality") or "").strip()
        if modality not in {
            InteractionEvent.MOUSE,
            InteractionEvent.KEYBOARD,
            InteractionEvent.OTHER,
        }:
            modality = InteractionEvent.OTHER
        detail = item.get("detail")
        if not isinstance(detail, dict):
            detail = {}
        try:
            client_ms = max(0, int(item.get("t") or 0))
        except (TypeError, ValueError):
            client_ms = 0

        rows.append(
            InteractionEvent(
                user=user,
                study_group=session.study_group,
                model_id=session.model_id,
                stage=session.stage,
                page=str(item.get("page") or "")[:64],
                action=action,
                modality=modality,
                target=str(item.get("target") or "")[:255],
                detail=detail,
                client_ms=client_ms,
            )
        )

    if rows:
        InteractionEvent.objects.bulk_create(rows, batch_size=200)
    return JsonResponse({"stored": len(rows)})
