"""Template context: CSRF token for meta tags and forms."""

from __future__ import annotations

from django.middleware.csrf import get_token


def template_extras(request):
    return {
        "csrf_token": get_token(request),
    }
