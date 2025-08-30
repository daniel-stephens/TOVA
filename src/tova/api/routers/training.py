import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional

from fastapi import (APIRouter, BackgroundTasks, HTTPException,  # type: ignore
                     Response, status)

from tova.api.jobs.domain import JobStatus, JobType
from tova.api.jobs.store import job_store
from tova.api.jobs.tokens import cancellation_tokens
from tova.api.logger import logger
from tova.api.models.train_schemas import TrainRequest, TrainResponse
from tova.core.dispatchers import train_model_dispatch
from tova.utils.cancel import CancellationToken, CancelledError
from tova.utils.tm_utils import normalize_json_data

router = APIRouter(tags=["Training"])


async def _run_training_job(
    *,
    job_id: str,
    model: str,
    data: List[Dict],
    output: str,
    config_path: str,
    do_preprocess: bool,
    training_params: Optional[Dict],
    logger: logging.Logger,
    cancel: CancellationToken,
) -> None:
    """
    Executes the (sync) training pipeline in a background thread and updates job status.
    """
    try:
        await job_store.update(job_id, status=JobStatus.running, message="Starting training", progress=0.0)
        loop = asyncio.get_running_loop()

        def progress_callback(progress: float, msg: str):
            corou = asyncio.run_coroutine_threadsafe(
                job_store.update(job_id, progress=progress, message=msg), loop)
            try:
                # wait for completion to catch exceptions
                corou.result(timeout=2)
            except Exception as e:
                logger.error(f"Failed to update job progress: {e}")
                pass

        duration = await loop.run_in_executor(
            None,
            lambda: train_model_dispatch(
                model=model,
                data=data,
                output=output,
                config_path=config_path,
                logger=logger,
                do_preprocess=do_preprocess,
                tr_params=training_params,
                progress_callback=progress_callback,
                cancel=cancel
            ),
        )

        await job_store.update(
            job_id,
            status=JobStatus.succeeded,
            progress=1.0,
            message="Topic modeling training completed",
            result={"duration": duration},
        )
    except CancelledError as ce:
        await job_store.update(job_id, status=JobStatus.cancelled, message=str(ce))
    except Exception as e:
        logger.exception("Topic modeling training failed")
        await job_store.update(job_id, status=JobStatus.failed, error=str(e), message="Topic modeling training failed")


async def _enqueue_training_job(
    *,
    model: str,
    data: List[Dict],
    output: str,
    config_path: str,
    do_preprocess: bool,
    training_params: Optional[Dict],
    bg: BackgroundTasks,
) -> TrainResponse:
    model_id = f"m_{uuid.uuid4().hex[:8]}"

    job = await job_store.create(
        type=JobType.train_model,
        model_id=model_id,
        # TODO: add user_id
    )

    token = CancellationToken()
    cancellation_tokens[job.id] = token

    bg.add_task(
        _run_training_job,
        job_id=job.id,
        cancel=token,
        model=model,
        data=data,
        output=output,
        config_path=config_path,
        do_preprocess=do_preprocess,
        training_params=training_params,
        logger=logger,
    )

    return TrainResponse(job_id=job.id, model_id=model_id)

# -----------------------
# API Routes
# -----------------------
@router.post("/json", response_model=TrainResponse, status_code=status.HTTP_202_ACCEPTED)
async def train_model_from_json(req: TrainRequest, bg: BackgroundTasks, response: Response):
    """
    Start training from JSON payload. Returns a job you can poll for status.
    """
    try:
        normalized_data = normalize_json_data(
            raw_data=json.dumps([record.model_dump() for record in req.data]),
            id_col=req.id_col if req.id_col else None,
            text_col=req.text_col,
            logger=logger,
        )

        train_response: TrainResponse = await _enqueue_training_job(
            model=req.model,
            data=normalized_data,
            output=req.output,
            config_path=req.config_path,
            do_preprocess=req.do_preprocess,
            training_params=req.training_params,
            bg=bg,
        )  # this is the response body

        # add location to the response header
        response.headers["Location"] = f"/status/jobs/{train_response.job_id}"

        return train_response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))