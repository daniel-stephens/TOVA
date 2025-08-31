import asyncio
import logging
import os
import httpx
from typing import Optional, Dict, Any

from tova.api.models.data_schemas import DraftType
from tova.utils.cancel import CancellationToken, CancelledError


INDEX_API_BASE = os.getenv("API_SOLR_URL")
TIMEOUT_SEC = 30.0
RETRIES = 3
BACKOFF_BASE = 0.5  # seconds

async def index_corpus(
    corpus_name: str,
    data: Dict[str, Any], *,
    logger: logging.Logger,
    cancel: Optional[CancellationToken] = None
) -> Dict[str, Any]:
    """
    Calls the external indexing API:
      POST /corpora/indexCorpus
      JSON body: { corpus_name, data }
    """

    async with httpx.AsyncClient(
        base_url=INDEX_API_BASE,
        timeout=TIMEOUT_SEC,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    ) as client:

        payload = {"corpus_name": corpus_name, "data": data}

        for attempt in range(1, RETRIES + 1):
            if cancel and cancel.is_cancelled():
                raise CancelledError("Indexing cancelled before request")

            try:
                resp = await client.post("/corpora/indexCorpus", json=payload)
                resp.raise_for_status()
                if resp.headers.get("content-type", "").startswith("application/json"):
                    return resp.json()
                return {"status": "ok"}
            except httpx.HTTPError as e:
                if attempt >= RETRIES:
                    logger.error(f"Index corpus failed (final): {e}")
                    raise
                backoff = BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    f"Index corpus attempt {attempt} failed: {e}. Retrying in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)
            
            
async def index_model(
    model_name: str,
    metadata: Dict[str, Any],
    data: Dict[str, Any], *,
    logger: logging.Logger,
    cancel: Optional[CancellationToken] = None
) -> Dict[str, Any]:
    """
    Calls the external indexing API:
      POST /models/indexModel
      JSON body: { model_name, metadata, data }
    """

    async with httpx.AsyncClient(
        base_url=INDEX_API_BASE,
        timeout=TIMEOUT_SEC,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    ) as client:

        payload = {"model_name": model_name, "metadata": metadata, "data": data}

        for attempt in range(1, RETRIES + 1):
            if cancel and cancel.is_cancelled():
                raise CancelledError("Indexing cancelled before request")

            try:
                resp = await client.post("/models/indexModel", json=payload)
                resp.raise_for_status()
                if resp.headers.get("content-type", "").startswith("application/json"):
                    return resp.json()
                return {"status": "ok"}
            except httpx.HTTPError as e:
                if attempt >= RETRIES:
                    logger.error(f"Index model failed (final): {e}")
                    raise
                backoff = BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    f"Index model attempt {attempt} failed: {e}. Retrying in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)

def try_delete_resource(resource_id: str, kind: DraftType, logger: logging.Logger) -> None:
    """
    Attempts to delete a resource from the index. This is needed when a draft promotion is cancelled or failed.
    """
    pass    