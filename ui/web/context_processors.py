"""Template context: CSRF token for meta tags and forms."""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.middleware.csrf import get_token


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


def template_extras(request):
    return {
        "csrf_token": get_token(request),
        "tova_admin_css_version": _tova_admin_css_version(),
    }
