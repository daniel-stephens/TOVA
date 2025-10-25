import logging
from typing import List
from fastapi import APIRouter, BackgroundTasks, HTTPException, Response
from tova.api.logger import logger
from tova.api.models.data_schemas import Corpus, Draft, DraftType, Model, PromoteDraftResponse, Dataset, StorageType, DraftCreatedResponse
from tova.api.jobs.store import job_store
from tova.api.jobs.domain import JobStatus, JobType
from tova.api.jobs.tokens import cancellation_tokens
from tova.utils.cancel import CancellationToken, CancelledError
from tova.core import drafts as drafts
from tova.core import indexing as indexing
from tova.core import corpora as corpora
from tova.core import models as models
from tova.core import datasets as datasets

router = APIRouter(tags=["Data Handling"])


# the user trains a model. Once the training is done, we ask the user whether he wants to save the model. If yes, we create a new model entry in the database.
# in the meanwhile, he can inspect the model because it is stored temporarily.
# once the model has been indexed succesfully, the model folder is deleted and all the information about such a model is removed from the temporary store.


def _infer_draft_type(draft_id: str) -> DraftType:
    if draft_id.startswith("m_"):
        return DraftType.model
    if draft_id.startswith("c_"):
        return DraftType.corpus
    if draft_id.startswith("d_"):
        return DraftType.dataset
    raise HTTPException(
        status_code=400, detail="Invalid draft id prefix; expected m_ or c_")

# -----------------------
# API Routes
# -----------------------

###########
# DRAFTS  #
###########
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

        # Reload draft
        draft = drafts.get_draft_metadata(draft_id)
        if draft is None:
            raise RuntimeError("Draft missing at execution time")
        draft.data = drafts.get_draft(draft_id, kind)  # actual data

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
# DATASETS #
###########
@router.get("/datasets", response_model=List[Dataset])
def list_datasets():
    """List all available datasets."""
    return datasets.list_datasets()


@router.post("/datasets", response_model=DraftCreatedResponse, status_code=201)
def create_dataset(dataset: Dataset) -> DraftCreatedResponse:
    """
    Create a new dataset. The dataset is temporarily stored and indexed asynchronously.
    """
    try:
        result = datasets.create_dataset(dataset)  # this is a DraftCreatedResponse
        if result.status_code != 201:
            raise HTTPException(
                status_code=result.status_code, detail="Failed to create dataset")
        logger.info("Created dataset %s", result.draft_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to create dataset")
        raise HTTPException(
            status_code=500, detail="Failed to create dataset") from e


@router.get("/datasets/{dataset_id}", response_model=Dataset)
def get_dataset(dataset_id: str):
    """Get a dataset by ID."""
    dataset = datasets.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(404, "Dataset not found")
    return dataset


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str):
    """Delete a dataset by ID."""
    ok = datasets.delete_dataset(dataset_id)
    if not ok:
        raise HTTPException(404, "Dataset not found")
    return

###########
# CORPORA #
###########
@router.get("/corpora", response_model=List[Corpus])
def list_corpora():
    """List all available corpora."""
    return corpora.list_corpora()


@router.post("/corpora", response_model=DraftCreatedResponse, status_code=201)
def create_corpus(corpus: Corpus) -> DraftCreatedResponse:
    """
    Create a new corpus. The corpus is temporarily stored and indexed asynchronously.
    """
    try:
        result = corpora.create_corpus(corpus)  # this is a DraftCreatedResponse
        # result should be true or false if true then read metaa ? and convert into DraftCreatedResponse
        if result.status_code != 201:
            raise HTTPException(
                status_code=result.status_code, detail="Failed to create corpus")
        logger.info("Created corpus %s", result.draft_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to create corpus")
        raise HTTPException(
            status_code=500, detail="Failed to create corpus") from e


@router.get("/corpora/{corpus_id}", response_model=Corpus)
def get_corpus(corpus_id: str):
    """Get a corpus by ID."""
    c = corpora.get_corpus(corpus_id)
    if c is None:
        raise HTTPException(404, "Corpus not found")
    return c


@router.delete("/corpora/{corpus_id}", status_code=204)
def delete_corpus(corpus_id: str):
    """Delete a corpus by ID."""
    ok = corpora.delete_corpus(corpus_id)
    if not ok:
        raise HTTPException(404, "Corpus not found")
    return

###########
#  MODELS #
###########
@router.get("/corpora/{corpus_id}/models", response_model=List[Model])
def list_corpus_models(corpus_id: str):
    """List all models for a corpus."""
    return models.list_models(corpus_id)


@router.get("/models/{model_id}", response_model=Model)
def get_model(model_id: str):
    """Get a model by ID."""
    m = models.get_model(model_id)
    if m is None:
        raise HTTPException(404, "Model not found")
    return m


@router.delete("/models/{model_id}", status_code=204)
def delete_model(model_id: str):
    """Delete a model by ID."""
    ok = models.delete_model(model_id)
    if not ok:
        raise HTTPException(404, "Model not found")
    return