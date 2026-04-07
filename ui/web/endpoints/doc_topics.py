"""Endpoint that proxies document topic suggestions to the FastAPI backend."""

from __future__ import annotations

import logging

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, JsonResponse

from web import views as views_monolith
from web.services import runtime as R

logger = logging.getLogger(__name__)


@login_required
def suggest_doc_topics(request: HttpRequest):
    """Proxy to FastAPI ``/queries/suggest-doc-topics`` which uses the Prompter class."""
    import requests

    payload = views_monolith._get_json(request) or {}
    doc_text = (payload.get("text") or "").strip()

    if not doc_text:
        return JsonResponse({"error": "text is required"}, status=400)

    try:
        resp = requests.post(
            f"{R.API}/queries/suggest-doc-topics",
            json={"text": doc_text},
            timeout=(5, 120),
        )
        return JsonResponse(resp.json(), status=resp.status_code)
    except requests.RequestException as e:
        logger.exception("suggest-doc-topics proxy error: %s", e)
        return JsonResponse({"error": f"Backend error: {e}"}, status=502)
