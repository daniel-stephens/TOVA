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

from web.models import StudyGroup, User
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
def tova_preflight(request: HttpRequest) -> HttpResponse:
    """Is this study actually ready to run?

    Every dashboard a participant is assigned to, with the things that go wrong
    silently: no participants, a corpus that cannot be read, a backend that does
    not answer. The failure this prevents is an admin discovering a broken
    dashboard because a paid participant hit it.

    Degrades rather than erroring -- an unreachable backend is exactly the
    condition this page exists to report, so it must still render.
    """
    from web.models import ParticipantSession, User
    from web.study_features import FEATURE_ACTIVE_LEARNING, user_has_feature
    from web.admin import available_dashboards, corpus_for_model

    labels = dict(available_dashboards())

    assigned = (
        User.objects.filter(is_staff=False, is_superuser=False)
        .exclude(study_model_id="")
        .values_list("study_model_id", flat=True)
    )
    counts: dict[str, int] = {}
    for model_id in assigned:
        counts[model_id] = counts.get(model_id, 0) + 1

    # Group defaults count too: those participants inherit the dashboard.
    for group in StudyGroup.objects.exclude(default_model_id=""):
        n = group.users.filter(
            is_staff=False, is_superuser=False, study_model_id=""
        ).count()
        if n:
            counts[group.default_model_id] = counts.get(group.default_model_id, 0) + n

    rows: list[dict[str, Any]] = []
    for model_id in sorted(set(counts) | set(labels)):
        n_participants = counts.get(model_id, 0)
        label = labels.get(model_id, model_id).split("  (")[0]
        known = model_id in labels

        corpus_size = None
        error = None
        if known:
            try:
                from web.views import _study_corpus_size

                corpus_size = _study_corpus_size(model_id)
                if corpus_size is None:
                    error = "Corpus size unavailable"
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("preflight: corpus size failed for %s: %s", model_id, e)
                error = "Backend unreachable"
        else:
            error = "Not found in the model store"

        al_states = {
            user_has_feature(u, FEATURE_ACTIVE_LEARNING)
            for u in User.objects.filter(study_model_id=model_id)
        }
        if not al_states:
            active_learning = "—"
        elif al_states == {True}:
            active_learning = "On"
        elif al_states == {False}:
            active_learning = "Off"
        else:
            active_learning = "Mixed"

        rows.append(
            {
                "model_id": model_id,
                "label": label,
                "participants": n_participants,
                "corpus_size": corpus_size,
                "corpus_id": corpus_for_model(model_id),
                "active_learning": active_learning,
                "error": error,
                "ready": bool(known and not error and n_participants),
            }
        )

    unassigned = User.objects.filter(
        is_staff=False, is_superuser=False, study_model_id="", study_group__isnull=True
    ).count()
    started = ParticipantSession.objects.exclude(task_started_at=None).count()

    # The study varies the model backbone over ONE corpus. If two conditions are
    # built on different corpora, the comparison is between corpora as well as
    # between models, and the result means nothing -- so this is checked here
    # rather than discovered during analysis.
    corpora_in_use = {
        r["corpus_id"] for r in rows if r["participants"] and r["corpus_id"]
    }
    mixed_corpora = sorted(corpora_in_use) if len(corpora_in_use) > 1 else []

    return render(
        request,
        "admin/tova_preflight.html",
        {
            "rows": rows,
            "unassigned": unassigned,
            "started": started,
            "mixed_corpora": mixed_corpora,
            "any_problems": any(not r["ready"] for r in rows) or bool(mixed_corpora),
        },
    )


@staff_member_required
def tova_progress(request: HttpRequest) -> HttpResponse:
    """How far along is collection, per condition?

    Answers "are we done yet", which the row-by-row session changelist does not.
    """
    from django.db.models import Avg, Count, Q

    from web.models import ParticipantSession, StudyResponse
    from web.admin import available_dashboards

    labels = dict(available_dashboards())
    S = ParticipantSession

    per_condition = (
        S.objects.values("model_id")
        .annotate(
            total=Count("user"),
            consented=Count("user", filter=Q(consented_at__isnull=False)),
            started=Count("user", filter=Q(task_started_at__isnull=False)),
            in_questions=Count("user", filter=Q(stage=S.STAGE_POST)),
            complete=Count("user", filter=Q(stage=S.STAGE_DONE)),
            moderated=Count("user", filter=Q(moderated=True)),
        )
        .order_by("model_id")
    )

    scores = {
        r["model_id"]: r
        for r in StudyResponse.objects.values("model_id").annotate(
            mean_score=Avg("tf_score"),
            mean_answered=Avg("tf_answered"),
            mean_duration=Avg("task_duration_s"),
            n=Count("user"),
        )
    }

    rows = []
    for row in per_condition:
        model_id = row["model_id"] or ""
        stats = scores.get(model_id, {})
        rows.append(
            {
                **row,
                "label": labels.get(model_id, model_id or "—").split("  (")[0],
                "mean_score": stats.get("mean_score"),
                "mean_answered": stats.get("mean_answered"),
                "mean_duration": stats.get("mean_duration"),
                "submitted": stats.get("n", 0),
            }
        )

    totals = {
        "total": S.objects.count(),
        "consented": S.objects.exclude(consented_at=None).count(),
        "started": S.objects.exclude(task_started_at=None).count(),
        "complete": S.objects.filter(stage=S.STAGE_DONE).count(),
        "submitted": StudyResponse.objects.count(),
    }

    return render(
        request,
        "admin/tova_progress.html",
        {"rows": rows, "totals": totals},
    )


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

