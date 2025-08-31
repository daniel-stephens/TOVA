import logging
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response  # type: ignore

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
    raise HTTPException(
        status_code=400, detail="Invalid draft id prefix; expected m_ or c_")

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
