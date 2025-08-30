import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore

from tova.api.jobs.domain import JobStatus
from tova.api.jobs.store import job_store
from tova.api.jobs.tokens import cancellation_tokens
from tova.api.models.job_schemas import JobDTO
from tova.api.routers import inference, queries, training, validating
from tova.utils.common import init_logger

# Centralized logger initialization
logger = init_logger("static/config/config.yaml", name="TOVA_API")
logger.info("Main application logger 'TOVA_API' initialized.")

# Queue to hold log messages for SSE clients
log_queue: asyncio.Queue = asyncio.Queue()

class SSELogHandler(logging.Handler):
    def emit(self, record):
        try:
            log_entry = {
                "level": record.levelname,
                "name": record.name,
                "timestamp": self.formatter.formatTime(record, datefmt="%Y-%m-%d %H:%M:%S,%f")[:-3],
                "message": record.message,
                "pathname": record.pathname,
                "lineno": record.lineno
            }
            log_queue.put_nowait(json.dumps(log_entry))
        except Exception as e:
            import sys
            print(f"Error in SSELogHandler: {e}", file=sys.stderr)

sse_handler = SSELogHandler()
sse_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sse_handler.setFormatter(sse_formatter)
logger.addHandler(sse_handler)

app = FastAPI(
    title="TOVA API",
    version="1.0.0",
    swagger_ui_parameters={"syntaxHighlight": {"theme": "obsidian"}, "deepLink": True},
)

################################################################################
#                               STATUS ENDPOINTS                               #
################################################################################
async def log_stream_generator() -> AsyncGenerator[str, None]:
    """Generate log messages for SSE clients.
    """
    try:
        while True:
            # Get a log message from the queue
            log_message_json = await log_queue.get()
            yield f"data: {log_message_json}\n\n"
    except asyncio.CancelledError:
        logger.info("SSE client disconnected.")
    except Exception as e:
        logger.error(f"Error in SSE log stream: {e}", exc_info=True)

@app.get("/status/sse_logs") 
async def sse_endpoint(request: Request):
    """Stream server-sent events (SSE) for log messages.
    """
    logger.info("New SSE connection established.")
    if await request.is_disconnected():
        logger.info("SSE client disconnected on handshake.")
        return StreamingResponse(log_stream_generator(), media_type="text/event-stream")

    return StreamingResponse(log_stream_generator(), media_type="text/event-stream")

@app.get("/status/jobs/{job_id}", response_model=JobDTO)
async def get_job(job_id: str):
    try:
        job = await job_store.get(job_id)
        return JobDTO(**job.__dict__)
    except KeyError:
        raise HTTPException(404, "Job not found")


@app.delete("/status/jobs/{job_id}", status_code=status.HTTP_202_ACCEPTED)
async def cancel_job(job_id: str):
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    await job_store.update(job_id, status=JobStatus.cancelled, message="Cancellation requested")

    # Signal the running worker (if token exists)
    token = cancellation_tokens.get(job_id)
    if token:
        token.cancel()

    return {"job_id": job_id, "status": "cancelled"}


################################################################################
#                               HEALTH ENDPOINTS                               #
################################################################################
@app.get("/health")
async def health_check():
    """Check the health of the API.
    """
    return {"status": "healthy"}


################################################################################
#                            TOVA API ENDPOINTS                                #
################################################################################
app.include_router(training.router, prefix="/train", tags=["Training"])
app.include_router(inference.router, prefix="/infer", tags=["Inference"])
app.include_router(queries.router, prefix="/queries", tags=["Queries"])
app.include_router(validating.router, prefix="/validate", tags=["Validation"])