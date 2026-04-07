"""View decorators for the TOVA web UI."""

from __future__ import annotations

from functools import wraps

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import redirect
from django.urls import reverse


def require_tova_app_admin(*, json_response: bool = True, forbidden_payload: dict | None = None):
    """Require ``request.tova_user["is_admin"]`` (see :mod:`web.authz`).

    Use **below** ``@login_required`` so unauthenticated users are handled first::

        @login_required
        @require_tova_app_admin(json_response=False)
        def admin_page(request): ...

    :param json_response: If True, return 403 JSON. If False, flash a message and redirect home.
    :param forbidden_payload: Optional JSON body for 403 responses (default ``{"error": "Forbidden"}``).
    """

    def decorator(view_func):
        @wraps(view_func)
        def _wrapped(request, *args, **kwargs):
            tu = getattr(request, "tova_user", None)
            if not tu or not tu.get("is_admin"):
                if json_response:
                    payload = forbidden_payload if forbidden_payload is not None else {"error": "Forbidden"}
                    return JsonResponse(payload, status=403)
                messages.error(request, "You do not have access to the admin area.")
                return redirect(reverse("web:home"))
            return view_func(request, *args, **kwargs)

        return _wrapped

    return decorator
