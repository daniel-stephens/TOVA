"""Dashboard tools for working with themes: LLM label suggestions and overrides.

Both are per-user overlays on top of the model. Neither writes to the backend:
the model keeps its own labels and assignments, and these sit on top of them
when the dashboard renders.
"""

from __future__ import annotations

import logging
import random

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, JsonResponse

from web.models import DocumentTheme
from web.services import runtime as R
from web.services.llm_client import chat_llm, get_chat_llm_defaults

logger = logging.getLogger(__name__)

#: How many of a theme's documents to show the model. Enough to characterise the
#: theme, few enough to keep the prompt small and the call quick.
SAMPLE_SIZE = 8

#: Characters of each sampled document to include. Topic-bearing words cluster
#: early in a document, and whole documents would blow the context for no gain.
SNIPPET_CHARS = 400

SYSTEM_PROMPT = (
    "You name themes in a collection of documents. Given some example documents "
    "and keywords from one theme, reply with a single short label for that "
    "theme: two to four words, plain language, no quotes, no trailing period, "
    "no explanation. Reply with the label and nothing else."
)


def _clean_label(text: str) -> str:
    """First line, stripped of the decoration models like to add."""
    label = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    label = label.strip().strip('"').strip("'").strip()
    for prefix in ("label:", "theme:", "topic:"):
        if label.lower().startswith(prefix):
            label = label[len(prefix):].strip()
    return label.rstrip(".").strip()[:120]


@login_required
def suggest_theme_label(request: HttpRequest):
    """Ask the configured LLM for a label for one theme.

    The sample is drawn at random each time rather than always taking the
    highest-scoring documents, so pressing the button again gives a genuinely
    different view of the theme instead of the same answer. Suggests only --
    applying the label is a separate, deliberate act.
    """
    from web import views as V

    payload = V._get_json(request) or {}
    model_id = str(payload.get("model_id") or "").strip()
    documents = payload.get("documents")
    keywords = str(payload.get("keywords") or "").strip()
    current = str(payload.get("current_label") or "").strip()

    if not model_id:
        return JsonResponse({"error": "model_id is required"}, status=400)
    if not isinstance(documents, list) or not documents:
        return JsonResponse(
            {"error": "No documents in this theme to describe."}, status=400
        )

    owner_id = V._request_user_id(request)
    if not owner_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)
    if not R._verify_model_ownership(model_id, owner_id):
        return JsonResponse({"error": "Access denied: You do not own this model"}, status=403)

    texts = [str(d or "").strip() for d in documents if str(d or "").strip()]
    if not texts:
        return JsonResponse(
            {"error": "No documents in this theme to describe."}, status=400
        )
    sample = random.sample(texts, min(SAMPLE_SIZE, len(texts)))

    parts = []
    if keywords:
        parts.append(f"Keywords for this theme: {keywords}")
    if current:
        parts.append(f"Its current label is \"{current}\". Suggest a better one.")
    parts.append("Example documents:")
    parts.extend(
        f"{i}. {t[:SNIPPET_CHARS]}" for i, t in enumerate(sample, start=1)
    )

    # get_chat_llm_defaults returns a tuple, not a mapping:
    # (provider, model, host, ollama_models, gpt_models, ollama_host)
    provider, model, host = get_chat_llm_defaults()[:3]
    reply, error = chat_llm(
        provider=str(payload.get("provider") or provider or ""),
        model=str(payload.get("model") or model or ""),
        system_content=SYSTEM_PROMPT,
        user_content="\n\n".join(parts),
        host=host,
    )
    if error:
        return JsonResponse({"error": error}, status=502)

    label = _clean_label(reply or "")
    if not label:
        return JsonResponse({"error": "The model returned an empty label."}, status=502)

    return JsonResponse({"label": label, "sampled": len(sample), "of": len(texts)})


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
