"""Dashboard theme tools.

``suggest_theme_label`` is a **stub for the backend team**. The suggestion
itself will be generated upstream; this exists so the frontend has something to
call while that is built, and so the request and response shapes are settled.
See the note on the function for what to replace.

``set_document_theme`` and ``get_document_themes`` are real: they store a
per-user correction to one document's theme as an overlay, without touching the
model.
"""

from __future__ import annotations

import logging

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, JsonResponse

from web.models import DocumentTheme
from web.services import runtime as R

logger = logging.getLogger(__name__)

#: Placeholder labels, cycled so pressing "Suggest again" visibly returns
#: something different and the frontend loop can be exercised end to end.
#: Delete these with the stub.
_STUB_LABELS = [
    "Approval oversight",
    "Assay anomalies",
    "Regulatory correspondence",
    "Supplier change",
    "Settlement discussions",
    "Routine operations",
]


@login_required
def suggest_theme_label(request: HttpRequest):
    """Return a suggested label for one theme.

    **STUB.** Replace the body below with a call to the backend's suggestion
    endpoint. The contract the frontend relies on, and which should not change:

    Request  (POST, JSON):  ``{"model_id": str, "theme_id": str|int}``
    Response (200, JSON):   ``{"label": str}``
    Errors:                 any non-200 with ``{"error": str}``; the frontend
                            surfaces the message and leaves the field untouched.

    Nothing is stored here. The label is shown to the user, who may ask for
    another, apply it through the existing rename endpoint, or close the dialog
    and leave no trace.
    """
    from web import views as V

    payload = V._get_json(request) or {}
    model_id = str(payload.get("model_id") or "").strip()
    theme_id = str(payload.get("theme_id") or "").strip()

    if not model_id:
        return JsonResponse({"error": "model_id is required"}, status=400)
    if not theme_id:
        return JsonResponse({"error": "theme_id is required"}, status=400)

    owner_id = V._request_user_id(request)
    if not owner_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    if not R._verify_model_ownership(model_id, owner_id):
        return JsonResponse({"error": "Access denied: You do not own this model"}, status=403)

    # --- replace from here -------------------------------------------------
    # e.g.
    #   up = requests.post(f"{R.API}/queries/suggest-theme-label",
    #                      json={"model_id": model_id, "theme_id": theme_id},
    #                      timeout=(5, 120))
    #   return JsonResponse(up.json(), status=up.status_code)
    import random

    label = random.choice(_STUB_LABELS)
    logger.info("suggest_theme_label stub: model=%s theme=%s", model_id, theme_id)
    return JsonResponse({"label": label, "stub": True})
    # --- to here -----------------------------------------------------------


@login_required
def set_document_theme(request: HttpRequest):
    """Move one document to a different theme, for this user only.

    An overlay: the model's own assignment is untouched, so this can be undone
    by clearing it and the underlying model is never silently edited.
    """
    from web import views as V

    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    payload = V._get_json(request) or {}
    model_id = str(payload.get("model_id") or "").strip()
    doc_id = str(payload.get("doc_id") or "").strip()
    theme = str(payload.get("theme") or "").strip()
    previous = str(payload.get("previous_theme") or "").strip()

    if not model_id or not doc_id:
        return JsonResponse({"error": "model_id and doc_id are required"}, status=400)

    owner_id = V._request_user_id(request)
    if not owner_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    if not R._verify_model_ownership(model_id, owner_id):
        return JsonResponse({"error": "Access denied: You do not own this model"}, status=403)

    if not theme:
        # Empty clears the override and returns the document to the model's own
        # assignment, rather than storing a blank theme.
        DocumentTheme.objects.filter(
            user=request.user, model_id=model_id, doc_id=doc_id
        ).delete()
        return JsonResponse({"doc_id": doc_id, "theme": None, "cleared": True})

    DocumentTheme.objects.update_or_create(
        user=request.user,
        model_id=model_id,
        doc_id=doc_id,
        defaults={"theme": theme[:255], "previous_theme": previous[:255]},
    )
    return JsonResponse({"doc_id": doc_id, "theme": theme, "cleared": False})


@login_required
def get_document_themes(request: HttpRequest):
    """This user's theme overrides for one model, as ``{doc_id: theme}``."""
    model_id = (request.GET.get("model_id") or "").strip()
    if not model_id:
        return JsonResponse({"error": "model_id is required"}, status=400)
    rows = DocumentTheme.objects.filter(
        user=request.user, model_id=model_id
    ).values_list("doc_id", "theme")
    return JsonResponse({"themes": dict(rows)})
