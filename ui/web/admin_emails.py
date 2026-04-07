"""Resolve admin allowlist from environment (TOVA_ADMIN_EMAILS, legacy DJANGO_ADMIN_EMAILS).

Emails in this list get **effective** app admin access on every request (see ``web.authz``) and,
on login, are persisted with ``is_staff``, ``is_superuser``, and ``is_admin`` via
:func:`promote_env_listed_user_to_staff` so Django admin and ``/staff/*`` APIs stay aligned.

The database ``is_admin`` flag alone can also grant access; admins may toggle ``is_admin`` for
other users via ``/staff/users/<id>/toggle-admin`` subject to "last admin" safeguards.
"""

from __future__ import annotations

import os

from web.models import User


def admin_emails_from_env() -> set[str]:
    raw = os.getenv("TOVA_ADMIN_EMAILS", "").strip()
    if not raw:
        raw = os.getenv("DJANGO_ADMIN_EMAILS", "").strip()
    if not raw:
        return set()
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def promote_env_listed_user_to_staff(user: User) -> bool:
    """If the user's email is in the env admin list, persist staff flags (Django admin access).

    Matches behaviour of ``sync_superuser`` for env-listed accounts. Returns True if the user was updated.
    """
    emails = admin_emails_from_env()
    if not emails or not getattr(user, "email", None):
        return False
    if user.email.lower() not in emails:
        return False

    needs_save = False
    if not user.is_staff:
        user.is_staff = True
        needs_save = True
    if not user.is_superuser:
        user.is_superuser = True
        needs_save = True
    if not user.is_admin:
        user.is_admin = True
        needs_save = True
    if needs_save:
        user.save()
    return needs_save
