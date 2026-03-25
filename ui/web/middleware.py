"""Session user context for the web UI (``request.tova_user``)."""

from __future__ import annotations

import logging

from django.utils.functional import SimpleLazyObject

logger = logging.getLogger(__name__)


def _get_tova_user(request):
    from django.contrib.auth.models import AnonymousUser

    from web.models import User

    u = getattr(request, "user", None)
    if not u or isinstance(u, AnonymousUser) or not u.is_authenticated:
        return None
    if not isinstance(u, User):
        logger.warning("request.user is not web.User: %s", type(u))
        return None
    admin_emails = set()
    raw = __import__("os").environ.get("TOVA_ADMIN_EMAILS", "").strip()
    if raw:
        admin_emails = {e.strip().lower() for e in raw.split(",") if e.strip()}
    # Staff tools are an "admin console" for this app.
    # Treat Django staff/superuser users as admins too, so `/staff` works
    # even when `is_admin` was not set on older accounts.
    is_admin = bool(
        getattr(u, "is_admin", False)
        or getattr(u, "is_staff", False)
        or getattr(u, "is_superuser", False)
        or (u.email and u.email.lower() in admin_emails)
    )
    return {
        "id": str(u.pk),
        "email": u.email,
        "name": u.name or u.email,
        "is_admin": is_admin,
    }


class TovaUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        match = getattr(request, "resolver_match", None)
        request.endpoint = match.url_name if match else None
        request.tova_user = SimpleLazyObject(lambda: _get_tova_user(request))
        response = self.get_response(request)
        return response
