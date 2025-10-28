import asyncio
import json
import logging
from typing import Dict, List

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, status  # type: ignore

from tova.api.jobs.domain import JobStatus, JobType
from tova.api.jobs.store import job_store
from tova.api.jobs.tokens import cancellation_tokens
from tova.api.logger import logger
from tova.api.models.infer_schemas import InferRequest, InferResponse
from tova.core.dispatchers import infer_model_dispatch
from tova.utils.cancel import CancellationToken, CancelledError
from tova.utils.tm_utils import normalize_json_data

router = APIRouter(tags=["Inference"])

# -----------------------
# Job runner
# -----------------------
async def _run_inference_job(
    *,
    job_id: str,
    model_path: str,
    data: List[Dict],
    config_path: str,
    logger: logging.Logger,
    cancel: CancellationToken,
) -> None:
    """
    Executes the (sync) inference pipeline in a background thread and updates job status.
    """
    try:
        await job_store.update(job_id, status=JobStatus.running, message="Starting inference", progress=0.0)
        loop = asyncio.get_running_loop()


        def progress_callback(progress: float, msg: str):
            corou = asyncio.run_coroutine_threadsafe(
                job_store.update(job_id, progress=progress, message=msg), loop
            )
            try:
                corou.result(timeout=2)
            except Exception as e:
                logger.error(f"Failed to update inference progress: {e}")

        # Run blocking inference on thread pool
        thetas, duration = await loop.run_in_executor(
            None,
            lambda: infer_model_dispatch(
                model_path=model_path,
                data=data,
                config_path=config_path,
                logger=logger,
                progress_callback=progress_callback,
                cancel=cancel,
            ),
        )

        await job_store.update(
            job_id,
            status=JobStatus.succeeded,
            progress=1.0,
            message="Inference completed",
            result={"thetas": thetas, "duration": duration},
        )

    except CancelledError as ce:
        await job_store.update(job_id, status=JobStatus.cancelled, message=str(ce))
    except Exception as e:
        logger.exception("Inference failed")
        await job_store.update(job_id, status=JobStatus.failed, error=str(e), message="Inference failed")

# -----------------------
# Enqueue job helper
# -----------------------
async def _enqueue_inference_job(
    *,
    model_path: str,
    data: List[Dict],
    config_path: str,
    bg: BackgroundTasks,
) -> InferResponse:

    job = await job_store.create(
        type=JobType.inference,
    )

    token = CancellationToken()
    cancellation_tokens[job.id] = token

    bg.add_task(
        _run_inference_job,
        job_id=job.id,
        cancel=token,
        model_path=model_path,
        data=data,
        config_path=config_path,
        logger=logger,
    )

    return InferResponse(job_id=job.id)

# -----------------------
# API Route
# -----------------------
@router.post("/json", response_model=InferResponse, status_code=status.HTTP_202_ACCEPTED)
async def infer_from_json(req: InferRequest, bg: BackgroundTasks, response: Response):
    """
    Start inference from JSON payload. Returns a job you can poll for status.
    """
    try:
        normalized_data = normalize_json_data(
            raw_data=json.dumps([record.model_dump() for record in req.data]),
            id_col=req.id_col if req.id_col else None,
            text_col=req.text_col,
            logger=logger
        )

        infer_response: InferResponse = await _enqueue_inference_job(
            model_path=req.model_path,
            data=normalized_data,
            config_path=req.config_path,
            bg=bg,
        )

        # Expose job status endpoint in Location header for clients to poll
        response.headers["Location"] = f"/status/jobs/{infer_response.job_id}"
        return infer_response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
