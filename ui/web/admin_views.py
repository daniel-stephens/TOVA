from __future__ import annotations

import logging
from typing import Any

import requests
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib import messages
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse

from tova.api.models.data_schemas import DraftType
from tova.core import drafts

from web.models import User
from web.services import runtime as R

logger = logging.getLogger(__name__)


def _owner_name_email_map(user_ids: set[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for uid in user_ids:
        if not uid:
            continue
        u = User.objects.filter(pk=uid).first()
        out[uid] = {
            "email": (u.email if u else None) or "—",
            "name": (u.name if u else None) or "",
        }
    return out


@staff_member_required
def tova_console_root(request: HttpRequest) -> HttpResponse:
    return render(request, "admin/tova_index.html")


@staff_member_required
def tova_corpora(request: HttpRequest) -> HttpResponse:
    try:
        resp = requests.get(f"{R.API}/data/corpora", timeout=15)
        resp.raise_for_status()
        corpora = resp.json()
    except Exception as e:
        logger.exception("tova_corpora fetch failed: %s", e)
        return JsonResponse({"error": "Failed to fetch corpora from backend"}, status=502)

    if not isinstance(corpora, list):
        corpora = []

    user_ids = {
        c.get("owner_id") or (c.get("metadata") or {}).get("owner_id")
        for c in corpora
        if isinstance(c, dict)
    }
    user_ids = {str(x) for x in user_ids if x}
    owners = _owner_name_email_map(user_ids)

    rows: list[dict[str, Any]] = []
    for c in corpora:
        if not isinstance(c, dict):
            continue
        oid = c.get("owner_id") or (c.get("metadata") or {}).get("owner_id")
        owner_email = owners.get(str(oid), {}).get("email") if oid else "—"
        owner_name = owners.get(str(oid), {}).get("name") if oid else ""
        rows.append(
            {
                "id": c.get("id"),
                "name": (c.get("metadata") or {}).get("name") or c.get("name") or "—",
                "owner_email": owner_email or "—",
                "owner_name": owner_name or "",
                "models_count": len(c.get("models") or []),
                "is_draft": (c.get("location") != "database") if c.get("location") else False,
            }
        )

    context = {
        "rows": rows,
    }
    return render(request, "admin/tova_corpora.html", context)


@staff_member_required
def tova_dashboards(request: HttpRequest) -> HttpResponse:
    # This admin view lists models across all users and links to the normal
    # dashboard viewer (which will still render the model's data).
    try:
        models_drafts = drafts.list_drafts(type=DraftType.model)
    except Exception as e:
        logger.exception("tova_dashboards list drafts failed: %s", e)
        return JsonResponse({"error": "Failed to list model drafts"}, status=500)

    user_ids: set[str] = set()
    models_list: list[dict[str, Any]] = []
    for draft in models_drafts:
        owner_id = draft.owner_id or (draft.metadata.get("owner_id") if draft.metadata else None)
        if owner_id:
            user_ids.add(str(owner_id))

        try:
            model = drafts.draft_to_model(draft)
            if not model:
                continue
            meta = (draft.metadata or {}).copy()
            name = model.name or (meta.get("tr_params") or {}).get("model_name") or draft.id
            models_list.append(
                {
                    "id": model.id,
                    "name": name,
                    "owner_id": str(model.owner_id) if getattr(model, "owner_id", None) is not None else (str(owner_id) if owner_id else None),
                    "owner_email": None,  # filled later
                    "corpus_id": getattr(model, "corpus_id", None) or meta.get("corpus_id"),
                    "created_at": model.created_at or meta.get("created_at"),
                }
            )
        except Exception:
            continue

    owners = _owner_name_email_map(user_ids)
    for m in models_list:
        oid = m.get("owner_id")
        m["owner_email"] = owners.get(str(oid), {}).get("email") if oid else "—"

    context = {
        "models": models_list,
        "dashboard_url_base": reverse("web:dashboard"),
    }
    return render(request, "admin/tova_dashboards.html", context)


@staff_member_required
def tova_delete_corpus(request: HttpRequest, corpus_id: str) -> HttpResponse:
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        up = requests.get(f"{R.API}/data/corpora/{corpus_id}", timeout=10)
        if up.status_code == 404:
            messages.error(request, "Corpus not found.")
            return redirect(reverse("tova_admin_corpora"))
        up.raise_for_status()
        corpus = up.json() or {}
        corpus_name = (corpus.get("metadata") or {}).get("name") or corpus_id

        # Delete associated models (best-effort)
        for model_id in corpus.get("models") or []:
            try:
                requests.delete(f"{R.API}/data/models/{model_id}", timeout=(3, 30))
            except Exception:
                pass

        del_resp = requests.delete(f"{R.API}/data/corpora/{corpus_id}", timeout=(3, 30))
        if del_resp.status_code in (200, 202, 204):
            messages.success(request, f"Deleted corpus: {corpus_name}")
        else:
            messages.error(request, f"Failed deleting corpus (upstream status {del_resp.status_code})")
    except Exception as e:
        logger.exception("tova_delete_corpus failed: %s", e)
        messages.error(request, f"Failed deleting corpus: {str(e)}")

    return redirect(reverse("tova_admin_corpora"))


@staff_member_required
def tova_delete_dashboard(request: HttpRequest, model_id: str) -> HttpResponse:
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        up = requests.get(f"{R.API}/data/models/{model_id}", timeout=10)
        if up.status_code == 404:
            messages.error(request, "Model not found.")
            return redirect(reverse("tova_admin_dashboards"))
        up.raise_for_status()
        model_meta = up.json() or {}
        corpus_id = model_meta.get("corpus_id")
        model_name = model_meta.get("name") or model_id

        if corpus_id:
            try:
                requests.post(
                    f"{R.API}/data/corpora/{corpus_id}/delete_model",
                    params={"model_id": model_id},
                    timeout=10,
                )
            except Exception:
                pass

        del_resp = requests.delete(f"{R.API}/data/models/{model_id}", timeout=(3, 30))
        if del_resp.status_code in (200, 202, 204):
            messages.success(request, f"Deleted model/dashboard: {model_name}")
        else:
            messages.error(request, f"Failed deleting model (upstream status {del_resp.status_code})")
    except Exception as e:
        logger.exception("tova_delete_dashboard failed: %s", e)
        messages.error(request, f"Failed deleting dashboard/model: {str(e)}")

    return redirect(reverse("tova_admin_dashboards"))

