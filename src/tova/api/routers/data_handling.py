import logging
import os
from pathlib import Path
from typing import List, Optional
import shutil
from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, Body
from tova.api.logger import logger
from tova.api.models.data_schemas import Corpus, Draft, DraftType, Model, PromoteDraftResponse
from tova.api.jobs.store import job_store
from tova.api.jobs.domain import JobStatus, JobType
from tova.api.jobs.tokens import cancellation_tokens
from tova.utils.cancel import CancellationToken, CancelledError
from tova.core import drafts as drafts
from tova.core import indexing as indexing
from tova.core import corpora as corpora
from tova.core import models as models
import json
import uuid
from typing import Any, Dict
import re
import unicodedata
from datetime import datetime
from fastapi.responses import JSONResponse

# the user trains a model. Once the training is done, we ask the user whether he wants to save the model. If yes, we create a new model entry in the database.
# in the meanwhile, he can inspect the model because it is stored temporarily.
# once the model has been indexed succesfully, the model folder is deleted and all the information about such a model is removed from the temporary store.

router = APIRouter(tags=["Data Handling"])

DRAFTS_SAVE = Path(os.getenv("DRAFTS_SAVE", "/data/drafts"))


def _infer_draft_type(draft_id: str) -> DraftType:
    if draft_id.startswith("m_"):
        return DraftType.model
    if draft_id.startswith("c_"):
        return DraftType.corpus
    if draft_id.startswith("d_"):
        return DraftType.dataset
    raise HTTPException(
        status_code=400, detail="Invalid draft id prefix; expected m_ or c_")


# ---- Helpers to build a filesystem-safe, unique draft id from corpus name ----
def _slugify(value: str, max_len: int = 80) -> str:
    """
    Turn an arbitrary string into a safe, lowercase slug for filesystem use.
    - Normalize unicode to ASCII where possible
    - Replace non-alphanumerics with "-"
    - Collapse repeated dashes, trim, and truncate
    """
    if not value:
        return ""
    # Normalize unicode and strip accents
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    # Lowercase and replace non-alnum with dashes
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    if not value:
        value = "unnamed"
    return value[:max_len].rstrip("-")

def _next_unique_draft_id_from_name(name: str) -> str:
    """
    Build a draft id like 'c_<slug>' and, if it already exists on disk,
    append '-2', '-3', ... until unique.
    """
    base = f"d_{_slugify(name)}" if name else f"d_{uuid.uuid4().hex[:8]}"
    candidate = base
    counter = 2
    while (DRAFTS_SAVE / candidate).exists():
        candidate = f"{base}-{counter}"
        counter += 1
    return candidate
# -----------------------
# API Routes
# -----------------------

###########
# DRAFTS  #
###########
@router.get("/drafts", response_model=List[Draft])
def list_drafts(type: Optional[DraftType] = None):
    """
    Drafts are directories under the drafts root:
      - m_XXXX/ (model drafts created during training)
      - c_XXXX/ (corpus drafts created during validation)
      - d_XXXX/ (dataset drafts created during validation)
    """
    return drafts.list_drafts(type)


@router.get("/drafts/{draft_id}", response_model=Draft)
def get_draft_metadata(draft_id: str):
    draft = drafts.get_draft_metadata(draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    return draft


@router.delete("/drafts/{draft_id}", status_code=204)
def delete_draft(draft_id: str):
    kind = _infer_draft_type(draft_id)
    deleted = drafts.delete_draft(kind, draft_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Draft not found")
    return


@router.post("/drafts/dataset/save", status_code=201)
def _save_dataset_draft(
    *,
    metadata: Dict[str, Any] = Body(...),
    data: Dict[str, Any] = Body(...),
    owner_id: Optional[str] = Body(None),
) -> Dict[str, str]:
    """
    Create a 'c_<slug-of-corpus-name>' draft directory (unique) under DRAFTS_SAVE and write:
      - metadata.json
      - corpus.json
    Returns {"draft_id": "..."} for the frontend.
    """

    # Pull a human name from metadata (fallback to UUID if missing)
    dataset_name = (metadata or {}).get("name") or (metadata or {}).get("title") or ""
    draft_id = _next_unique_draft_id_from_name(dataset_name)
    draft_dir = DRAFTS_SAVE / draft_id

    # Basic validation of expected payload shape
    docs = (data or {}).get("documents")
    if not isinstance(docs, list) or not docs:
        raise HTTPException(status_code=400, detail="'data.documents' must be a non-empty list.")
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            raise HTTPException(status_code=400, detail=f"'data.documents[{i}]' must be an object.")
        for k in ("id", "text", "sourcefile"):
            if k not in doc:
                raise HTTPException(status_code=400, detail=f"'data.documents[{i}].{k}' is required.")
        # label is optional; if you want to force null presence:
        # doc.setdefault("label", None)

    # Enrich metadata for UI/listing
    meta = dict(metadata or {})
    meta.setdefault("name", dataset_name or draft_id)  # keep name populated
    meta.setdefault("draft_id", draft_id)
    meta.setdefault("type", getattr(DraftType, "dataset", "dataset"))
    if owner_id:
        meta.setdefault("owner_id", owner_id)

    try:
        draft_dir.mkdir(parents=True, exist_ok=False)
        _write_json_atomic(draft_dir / "metadata.json", meta)
        _write_json_atomic(draft_dir / "dataset.json", data)
        logger.info("Saved corpus draft %s at %s", draft_id, draft_dir)
        return {"draft_id": draft_id}
    except Exception as e:
        logger.exception("Failed to save corpus draft")
        # best-effort cleanup
        try:
            shutil.rmtree(draft_dir, ignore_errors=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Failed to persist corpus draft") from e


async def commit_draft(draft_id: str, bg: BackgroundTasks, response: Response):
    """
    Promote a draft (model or corpus) to the permanent resource. If promotion works: the draft folder is deleted.
    """
    type = _infer_draft_type(draft_id)
    draft = drafts.get_draft_metadata(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="Draft not found")

    job_type = JobType.index_model if type == DraftType.model else JobType.index_corpus
    job = await job_store.create(type=job_type, owner_id=draft.owner_id)

    token = CancellationToken()
    cancellation_tokens[job.id] = token

    bg.add_task(
        _run_commit_job,
        job_id=job.id,
        draft_id=draft_id,
        kind=type,
        cancel=token,
        logger=logger
    )
    response.headers["Location"] = f"/status/jobs/{job.id}"
    return PromoteDraftResponse(job_id=job.id, draft_id=draft_id)


async def _run_commit_job(
    job_id: str,
    draft_id: str,
    kind: DraftType,
    cancel: CancellationToken,
    logger: logging.Logger,
) -> None:
    """
    Promote a draft to database.
    - Reloads draft (defensive).
    - Calls indexing service with payload (async HTTP).
    - On success: deletes draft and marks job succeeded.
    - On cancel/fail: keeps draft; marks job accordingly; deletes leftover resources in database.
    """

    try:
        await job_store.update(job_id, status=JobStatus.running, progress=0.0, message="Starting promotion")

        # Reload draft (defensive)
        draft = drafts.get_draft_metadata(draft_id)  # this just gets the meta
        if draft is None:
            raise RuntimeError("Draft missing at execution time")
        draft.data = drafts.get_draft_data(draft_id, kind)  # actual data

        # Index
        await job_store.update(job_id, progress=0.10, message="Indexing resource")

        resource_id = f"{draft.id}_i"

        if kind == DraftType.model:
            resp = await indexing.index_model(
                desired_id=resource_id,
                metadata=draft.metadata or {},
                data=draft.data or {},
                logger=logger,
                cancel=cancel,
            )
        else:
            resp = await indexing.index_corpus(
                desired_id=resource_id,
                metadata=draft.metadata or {},
                data=draft.data or {},
                logger=logger,
                cancel=cancel,
            )

        # Verify indexing worked
        await job_store.update(job_id, progress=0.80, message="Verifying indexing")
        # TODO: query index/DB for counts, sanity checks

        # Cleanup draft
        await job_store.update(job_id, progress=0.90, message="Cleaning up temporary artifacts")
        drafts.delete_draft(kind, draft_id)

        # Done
        await job_store.update(
            job_id,
            status=JobStatus.succeeded,
            progress=1.0,
            message="Promotion completed",
            result={"resource_id": resource_id, "draft_id": draft_id},
        )

    except CancelledError as ce:
        try:
            if resp:  # probably this needs update to read the error code
                await indexing.try_delete_resource(resource_id, kind, logger=logger)
        except Exception:
            logger.warning(
                "Compensation delete failed after cancel", exc_info=True)

        await job_store.update(job_id, status=JobStatus.cancelled, message=str(ce))

    except Exception as e:
        try:
            if resp:  # probably this needs update to read the error code
                await indexing.try_delete_resource(resource_id, kind, logger=logger)
        except Exception:
            logger.warning(
                "Compensation delete failed after error", exc_info=True)

        logger.exception("Commit job failed")
        await job_store.update(job_id, status=JobStatus.failed, error=str(e), message="Commit job failed")

    finally:
        cancellation_tokens.pop(job_id, None)



def _write_json_atomic(path: Path, payload: Any) -> None:
    """Safely write JSON to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)





@router.post("/drafts/corpus/save", status_code=201)
def save_corpus_draft(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Receive a payload with model, save_name, training_params, and datasets[],
    merge the referenced dataset drafts (d_*) into a new corpus draft (c_*),
    persist it under DRAFTS_SAVE, and return a training-config structure.
    Also persists the incoming payload as training_payload.json for reproducibility.
    """
    # ---- Validate incoming payload ----
    datasets = payload.get("datasets") or []
    if not datasets:
        raise HTTPException(status_code=400, detail="No datasets provided")

    model_name      = payload.get("model_name")
    model_type      = payload.get("model") or ""
    training_params = payload.get("training_params") or {}

    # ---- Merge documents from dataset drafts ----
    merged_docs: List[Dict[str, Any]] = []
    used_dataset_ids: List[str] = []

    for ds in datasets:
        ds_id = (ds or {}).get("id")
        if not ds_id or not ds_id.startswith("d_"):
            raise HTTPException(status_code=400, detail=f"Invalid dataset id: {ds_id}")

        ds_dir  = DRAFTS_SAVE / ds_id
        ds_file = ds_dir / "dataset.json"
        if not ds_file.exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {ds_id}")

        try:
            with ds_file.open("r", encoding="utf-8") as f:
                ds_data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read dataset {ds_id}: {e}")

        docs = ds_data.get("documents", [])
        for d in docs:
            merged_docs.append({
                "id": str(uuid.uuid4()),
                "text": d["text"],
                "label": d.get("label"),
                "sourcefile": d["sourcefile"],
                "original_id": d["id"],
            })

        used_dataset_ids.append(ds_id)

    # ---- Create a new corpus draft directory and persist files ----
    corpus_id  = f"c_{uuid.uuid4().hex[:8]}"
    corpus_dir = DRAFTS_SAVE / corpus_id
    try:
        corpus_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        corpus_id  = f"c_{uuid.uuid4().hex[:8]}"
        corpus_dir = DRAFTS_SAVE / corpus_id
        corpus_dir.mkdir(parents=True, exist_ok=False)

    created_at_iso = datetime.utcnow().isoformat() + "Z"

    metadata = {
        "id": corpus_id,
        "name": model_name,                   # human-friendly name
        "model": model_type,
        "training_params": training_params,
        "datasets": used_dataset_ids,
        "created_at": created_at_iso,
        "type": "corpus_draft",
    }
    _write_json_atomic(corpus_dir / "metadata.json", metadata)

    corpus_content = {"documents": merged_docs}
    _write_json_atomic(corpus_dir / "corpus.json", corpus_content)

    # ---- Save the incoming payload for reproducibility (THIS IS THE NEW PART) ----
    # You can enrich with server-side context if helpful.
    payload_to_save = {
        "received_at": created_at_iso,
        "corpus_id": corpus_id,
        **payload,  # original client payload as-is
    }
    # _write_json_atomic(corpus_dir / "training_payload.json", payload_to_save)

    logger.info("Created corpus draft %s with %d documents", corpus_id, len(merged_docs))

    # ---- Build the training-config structure we return (and optionally also persist) ----
    training_config = {
        "config_path": "static/config/config.yaml",
        "data": [{"id": doc["id"], "raw_text": doc["text"]} for doc in merged_docs],
        "do_preprocess": False,
        "id_col": "id",
        "model": model_type,
        "model_name": model_name,
        "text_col": "raw_text",
        "training_params": training_params,
    }

    # (Optional) save the exact training_config we returned, for auditing
    _write_json_atomic(corpus_dir / "training_config.json", {
        "draft_id": corpus_id,
        "training_config": training_config,
        "saved_at": created_at_iso,
    })

    return JSONResponse(
        content={
            "draft_id": corpus_id,
            "training_config": training_config,
        },
        status_code=201,
    )




###########
# CORPORA #
###########
@router.get("/corpora", response_model=List[Corpus])
def list_corpora():
    return corpora.list_corpora()


@router.post("/corpora", response_model=Corpus, status_code=201)
def create_corpus(corpus: Corpus):
    return corpora.create_corpus(corpus)


@router.get("/corpora/{corpus_id}", response_model=Corpus)
def get_corpus(corpus_id: str):
    c = corpora.get_corpus(corpus_id)
    if c is None:
        raise HTTPException(404, "Corpus not found")
    return c


@router.delete("/corpora/{corpus_id}", status_code=204)
def delete_corpus(corpus_id: str):
    ok = corpora.delete_corpus(corpus_id)
    if not ok:
        raise HTTPException(404, "Corpus not found")
    return

###########
#  MODELS #
###########
@router.get("/corpora/{corpus_id}/models", response_model=List[Model])
def list_corpus_models(corpus_id: str):
    return models.list_models(corpus_id)


@router.get("/corpora/{corpus_id}/models/{model_id}", response_model=Model)
def get_model(corpus_id: str, model_id: str):
    m = models.get_model(corpus_id, model_id)
    if m is None:
        raise HTTPException(404, "Model not found")
    return m


@router.delete("/corpora/{corpus_id}/models/{model_id}", status_code=204)
def delete_model(corpus_id: str, model_id: str):
    ok = models.delete_model(corpus_id, model_id)
    if not ok:
        raise HTTPException(404, "Model not found")
    return