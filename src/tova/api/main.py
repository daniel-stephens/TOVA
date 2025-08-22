from fastapi import FastAPI, Request # type: ignore
from fastapi.responses import StreamingResponse # type: ignore
from tova.api.routers import training, inference, queries, validating
from tova.utils.common import init_logger
import logging
import asyncio
import json
from typing import AsyncGenerator



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


# Function to generate log messages for SSE clients
async def log_stream_generator() -> AsyncGenerator[str, None]:
    try:
        while True:
            # Get a log message from the queue
            log_message_json = await log_queue.get()
            yield f"data: {log_message_json}\n\n"
    except asyncio.CancelledError:
        logger.info("SSE client disconnected.")
    except Exception as e:
        logger.error(f"Error in SSE log stream: {e}", exc_info=True)


#--------------------------#
# Endpoint to stream logs
#--------------------------#
@app.get("/sse/logs") 
async def sse_endpoint(request: Request):
    logger.info("New SSE connection established.")
    if await request.is_disconnected():
        logger.info("SSE client disconnected on handshake.")
        return StreamingResponse(log_stream_generator(), media_type="text/event-stream")

    return StreamingResponse(log_stream_generator(), media_type="text/event-stream")


#--------------------------#
# Endpoint to check health
#--------------------------#
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

#--------------------------#
# TOVA API Routers
#--------------------------#
app.include_router(training.router, prefix="/train", tags=["Training"])
app.include_router(inference.router, prefix="/infer", tags=["Inference"])
app.include_router(queries.router, prefix="/queries", tags=["Queries"])
app.include_router(validating.router, prefix="/validate", tags=["Validation"])