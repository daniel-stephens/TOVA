"""Session user context for the web UI (``request.tova_user``)."""

from __future__ import annotations

from django.utils.functional import SimpleLazyObject

from web.authz import get_tova_user_session_dict


class TovaUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        match = getattr(request, "resolver_match", None)
        request.endpoint = match.url_name if match else None
        request.tova_user = SimpleLazyObject(lambda: get_tova_user_session_dict(request))
        response = self.get_response(request)
        return response
