"""Application-wide admin / privilege checks (TOVA web UI).

Effective "app admin" is granted if any of the following hold for an authenticated
:class:`web.models.User`:

- ``user.is_admin`` (database flag; can be toggled by other admins)
- ``user.is_staff`` or ``user.is_superuser`` (Django admin semantics)
- Email appears in ``TOVA_ADMIN_EMAILS`` / ``DJANGO_ADMIN_EMAILS`` (env allowlist)

Env-listed users are also promoted on login via
:func:`web.admin_emails.promote_env_listed_user_to_staff` so privileges persist in the DB.

Django's built-in admin and the TOVA console under ``/admin/tova/`` use
``@staff_member_required`` / ``is_staff``; this module aligns *session* checks
(``request.tova_user``) with the same rules so ``/staff/...`` API routes match
what operators see in the UI.
"""

from __future__ import annotations

import logging

from django.contrib.auth.models import AnonymousUser

from web.admin_emails import admin_emails_from_env
from web.models import User

logger = logging.getLogger(__name__)


def user_has_tova_app_admin_access(user) -> bool:
    """Return True if *user* may use TOVA app-level admin features (staff APIs, toggles).

    Anonymous users or non-:class:`User` principals never qualify.
    """
    if not user or not user.is_authenticated or isinstance(user, AnonymousUser):
        return False
    if not isinstance(user, User):
        logger.warning("user_has_tova_app_admin_access: expected web.User, got %s", type(user))
        return False
    if getattr(user, "is_admin", False):
        return True
    if getattr(user, "is_staff", False) or getattr(user, "is_superuser", False):
        return True
    email = (getattr(user, "email", None) or "").strip().lower()
    return bool(email and email in admin_emails_from_env())


def get_tova_user_session_dict(request):
    """Build the dict exposed as ``request.tova_user`` (or ``None`` if not logged in)."""
    u = getattr(request, "user", None)
    if not u or isinstance(u, AnonymousUser) or not u.is_authenticated:
        return None
    if not isinstance(u, User):
        logger.warning("request.user is not web.User: %s", type(u))
        return None
    is_admin = user_has_tova_app_admin_access(u)
    return {
        "id": str(u.pk),
        "email": u.email,
        "name": u.name or u.email,
        "is_admin": is_admin,
    }
