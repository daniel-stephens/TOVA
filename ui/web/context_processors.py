"""Template context: CSRF token for meta tags and forms."""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.middleware.csrf import get_token

from web.study_features import get_study_model_id, get_user_features, is_study_participant


def _tova_admin_css_version() -> str:
    """Mtime cache-bust for admin CSS (avoids stale WhiteNoise/browser cache after edits)."""
    for directory in getattr(settings, "STATICFILES_DIRS", []):
        path = Path(directory) / "admin" / "css" / "tova_admin.css"
        if path.is_file():
            return str(int(path.stat().st_mtime))
    root = getattr(settings, "STATIC_ROOT", None)
    if root:
        path = Path(root) / "admin" / "css" / "tova_admin.css"
        if path.is_file():
            return str(int(path.stat().st_mtime))
    return ""


def _study_stage(user):
    """Current stage and remaining task seconds, without creating a session row.

    Read-only: the session is created by the flow views and the middleware, not
    by rendering a template.
    """
    from web.models import ParticipantSession

    session = ParticipantSession.objects.filter(pk=user.pk).only(
        "stage", "task_ends_at"
    ).first()
    if session is None:
        return None, 0
    return session.stage, session.seconds_remaining


def _task_seconds() -> int:
    from web.study_flow import task_minutes

    return task_minutes() * 60


def template_extras(request):
    user = getattr(request, "user", None)
    authenticated = bool(user and user.is_authenticated)

    stage, remaining = (None, 0)
    if authenticated and is_study_participant(user):
        try:
            stage, remaining = _study_stage(user)
        except Exception:
            stage, remaining = (None, 0)

    return {
        "csrf_token": get_token(request),
        "tova_admin_css_version": _tova_admin_css_version(),
        "study_features": get_user_features(user) if authenticated else {},
        "study_model_id": get_study_model_id(user) if authenticated else None,
        "is_study_participant": is_study_participant(user) if authenticated else False,
        "study_stage": stage,
        "study_seconds_remaining": remaining,
        # Denominator for the timer's depleting meter.
        "study_task_seconds": _task_seconds() if stage else 0,
    }
