"""Session user context (``request.tova_user``) and study flow enforcement."""

from __future__ import annotations

from django.http import JsonResponse
from django.shortcuts import redirect
from django.utils.functional import SimpleLazyObject

from web.authz import get_tova_user_session_dict


class TovaUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.endpoint = None
        request.tova_user = SimpleLazyObject(lambda: get_tova_user_session_dict(request))
        response = self.get_response(request)
        return response

    def process_view(self, request, view_func, view_args, view_kwargs):
        # resolver_match is not populated until after URL resolution, so this
        # has to happen in process_view rather than __call__.
        match = getattr(request, "resolver_match", None)
        request.endpoint = match.url_name if match else None
        return None


#: Paths whose responses are consumed by fetch()/XHR and must never receive an
#: HTML redirect.
_JSON_PATH_PREFIXES = ("/api/", "/staff/")
_JSON_PATHS = {
    "/get-dashboard-data",
    "/text-info",
    "/infer-text",
    "/get-trained-models",
    "/get-models-names",
    "/getAllCorpora",
    "/save-settings",
}


def _wants_json(request) -> bool:
    path = request.path
    if path.startswith(_JSON_PATH_PREFIXES) or path in _JSON_PATHS:
        return True
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return True
    accept = request.headers.get("Accept") or ""
    return "application/json" in accept and "text/html" not in accept


class StudyFlowMiddleware:
    """Pin study participants to their current stage of the flow.

    Implemented as ``process_view`` so ``request.resolver_match`` is available.

    This is deliberately fail-closed: a route that is not listed for the
    participant's current stage is refused. Adding a view without thinking about
    the study therefore blocks participants rather than silently letting them
    skip a stage, which is the failure mode that corrupts study data.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_view(self, request, view_func, view_args, view_kwargs):
        # Imported lazily: this module is imported from settings, before the app
        # registry is ready.
        from web.models import ParticipantSession
        from web.study_flow import (
            ALWAYS_ALLOWED,
            allowed_url_names,
            finish_task,
            get_or_create_session,
            stage_redirect_url,
        )

        user = getattr(request, "user", None)
        if user is None or not user.is_authenticated:
            return None
        if user.is_staff or user.is_superuser or getattr(user, "is_admin", False):
            return None
        if not getattr(user, "study_group_id", None):
            return None

        match = getattr(request, "resolver_match", None)
        if match is None:
            return None
        if match.namespace == "admin":
            return None
        url_name = match.url_name
        if url_name in ALWAYS_ALLOWED:
            return None

        session = get_or_create_session(user)

        # The deadline is enforced here, not in the browser. A participant who
        # disables JavaScript or kills the tab still cannot keep exploring.
        if session.stage == ParticipantSession.STAGE_TASK and session.expired:
            finish_task(session, early=False)

        if url_name in allowed_url_names(session, user):
            return None

        target = stage_redirect_url(session)
        if _wants_json(request):
            return JsonResponse(
                {
                    "error": "Not available at this stage of the study.",
                    "stage": session.stage,
                    "redirect": target,
                },
                status=409,
            )
        return redirect(target)
