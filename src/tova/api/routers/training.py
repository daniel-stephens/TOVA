import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import (APIRouter, BackgroundTasks, Body,  # type: ignore
                     HTTPException, Response, status)

from tova.api.jobs.domain import JobStatus, JobType
from tova.api.jobs.store import job_store
from tova.api.jobs.tokens import cancellation_tokens
from tova.api.logger import logger
from tova.api.models.train_schemas import TrainRequest, TrainResponse
from tova.core.dispatchers import train_model_dispatch
from tova.core.tfidf_lsi import *
from tova.core.tfidf_lsi import _save_training_payload
from tova.utils.cancel import CancellationToken, CancelledError
from tova.utils.common import get_unique_id
from tova.utils.tm_utils import normalize_json_data

router = APIRouter(tags=["Training"])

# paths to temporary storage
DRAFTS_SAVE = Path(os.getenv("DRAFTS_SAVE", "/data/drafts"))


async def _run_training_job(
    *,
    job_id: str,
    model: str,
    data: List[Dict],
    output: str,
    model_name: str,
    corpus_id: str,
    id: str,  # this is the model id
    config_path: str,
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
                model_name=model_name,
                corpus_id=corpus_id,
                id=id,
                config_path=config_path,
                logger=logger,
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
    model_name: str,
    corpus_id: str,
    config_path: str,
    training_params: Optional[Dict],
    bg: BackgroundTasks,
) -> TrainResponse:
    model_id = get_unique_id(prefix="m_")

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
        output=DRAFTS_SAVE.joinpath(model_id).as_posix(),
        model_name=model_name,
        corpus_id=corpus_id,
        id=model_id,
        config_path=config_path,
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
            model_name=req.model_name,
            corpus_id=req.corpus_id,
            config_path=req.config_path,
            training_params=req.training_params,
            bg=bg,
        )

        # add location to the response header
        response.headers["Location"] = f"/status/jobs/{train_response.job_id}"

        return train_response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/corpus/{corpus_id}/tfidf/")
def analyze_corpus_draft_endpoint(
    corpus_id: str,
    payload: Optional[Dict[str, Any]] = Body(None),
) -> Dict[str, Any]:
    """
    Analyze a saved corpus draft folder (DRAFTS_SAVE / corpus_id) and return:
      - metrics
      - tfidf_details (per-doc terms)
      - documents (with cluster, score, pca, keywords)

    Body (optional):
      { "n_clusters": <int, default 15 and must be >= 2> }
    """
    try:
        if not corpus_id or not corpus_id.startswith("c_"):
            raise ValueError(
                "Invalid corpus_id; expected an id starting with 'c_'")

        n_clusters_raw = (payload or {}).get("n_clusters", 15)
        n_clusters = int(n_clusters_raw)
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")

        logger.info(
            "Starting TF-IDF/cluster analysis for corpus %s (k=%d)", corpus_id, n_clusters)

        # Run analysis
        result = analyze_corpus_draft_folder(
            DRAFTS_SAVE, corpus_id, n_clusters=n_clusters, documents=payload.get("documents", []))

        # Save training payload for reproducibility
        #corpus_dir = DRAFTS_SAVE / corpus_id
        #_save_training_payload(corpus_dir, payload or {
        #                       "n_clusters": n_clusters})

        logger.info("Completed analysis for corpus %s", corpus_id)
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Analysis failed for corpus %s", corpus_id)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@router.get("/corpus/{corpus_id}/tfidf/", summary="Return saved dashboard output for a corpus")
def get_corpus(corpus_id: str) -> Dict[str, Any]:
    """
    Reads <DRAFTS_SAVE>/<corpus_id>/analysis_output.json and returns it.
    This file is the minimal payload the dashboard expects:
      {
        "documents": [...],
        "per_cluster": {...},
        "global": {...}
      }
    """
    if not corpus_id or not corpus_id.startswith("c_"):
        raise HTTPException(
            status_code=400, detail="Invalid corpus_id; must start with 'c_'")

    corpus_dir = DRAFTS_SAVE / corpus_id
    if not corpus_dir.exists():
        raise HTTPException(
            status_code=404, detail=f"Corpus folder not found: {corpus_id}")

    out_file = corpus_dir / "analysis_output.json"
    if not out_file.exists():
        raise HTTPException(
            status_code=404, detail="analysis_output.json not found; run analysis first")

    try:
        data = json.loads(out_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to read analysis_output.json: {e}")

    # Optional: quick shape validation
    if not isinstance(data, dict) or "documents" not in data or "per_cluster" not in data or "global" not in data:
        raise HTTPException(
            status_code=500, detail="analysis_output.json has unexpected structure")

    return data
