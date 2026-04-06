import os
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException # type: ignore

from tova.api.models.query_requests import ModelInfoRequest, ThetasByDocsIdsRequest, TopicInfoRequest
from tova.api.models.rag_schemas import RAGRetrieveRequest, RAGRetrieveResponse, RAGTopic, RAGDocument
from tova.core.dispatchers import get_model_info_dispatch, get_thetas_documents_by_id_dispatch, get_topic_info_dispatch
from tova.core.topic_retriever import get_retriever
from tova.api.logger import logger

router = APIRouter()

# paths to temporary storage
DRAFTS_SAVE = Path(os.getenv("DRAFTS_SAVE", "/data/drafts"))

# -----------------------
# API Route
# -----------------------
@router.post("/model-info", tags=["Queries"])
def get_model_info(req: ModelInfoRequest) -> Dict[str, Any]:
    """
    Returns topic-level metadata for a trained topic model.
    """
    if not os.path.isdir(DRAFTS_SAVE.joinpath(req.model_id).as_posix()):
        raise HTTPException(status_code=400, detail="Model path not found or not a directory.")

    try:
        model_info = get_model_info_dispatch(
            model_path= DRAFTS_SAVE.joinpath(req.model_id).as_posix(),
            config_path=req.config_path,
            logger=logger
        )
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/topic-info", tags=["Queries"])
def get_topic_info(req: TopicInfoRequest) -> Dict[str, Any]:
    """
    Returns topic-level metadata for a specific topic in a trained topic model.
    """
    if not os.path.isdir(DRAFTS_SAVE.joinpath(req.model_id).as_posix()):
        raise HTTPException(status_code=400, detail="Model path not found or not a directory.")

    try:
        topic_info = get_topic_info_dispatch(
            model_path=DRAFTS_SAVE.joinpath(req.model_id).as_posix(),
            config_path=req.config_path,
            topic_id=req.topic_id,
            logger=logger
        )
        return topic_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/thetas-by-docs-ids", tags=["Queries"])
def get_thetas_by_docs_ids(req: ThetasByDocsIdsRequest) -> Dict[str, Any]:
    """
    Returns topic weights for specific documents by their IDs from a trained topic model.
    """
    if not os.path.isdir(DRAFTS_SAVE.joinpath(req.model_id).as_posix()):
        raise HTTPException(status_code=400, detail="Model path not found or not a directory.")

    try:
        thetas_info = get_thetas_documents_by_id_dispatch(
            model_path=DRAFTS_SAVE.joinpath(req.model_id).as_posix(),
            docs_ids=req.docs_ids.split(","),
            config_path=req.config_path,
            logger=logger
        )
        return thetas_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag-retrieve", response_model=RAGRetrieveResponse, tags=["Queries"])
def rag_retrieve(req: RAGRetrieveRequest):
    """Retrieve the most relevant topics and documents for a user query.

    Uses sentence-transformer embeddings to rank topics by semantic
    similarity to the query, then returns the top representative
    documents from those topics.
    """
    if not DRAFTS_SAVE.joinpath(req.model_id).is_dir():
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        retriever = get_retriever(req.model_id)
        raw = retriever.retrieve(
            query=req.query,
            top_k_topics=req.top_k_topics,
            top_n_docs=req.top_n_docs,
        )
        topics = [
            RAGTopic(
                topic_id=t["topic_id"],
                label=t["label"],
                keywords=t["keywords"],
                summary=t["summary"],
                similarity=t["similarity"],
                documents=[
                    RAGDocument(doc_id=d["doc_id"], text=d["text"], score=d["score"])
                    for d in t["documents"]
                ],
            )
            for t in raw
        ]
        return RAGRetrieveResponse(topics=topics)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("RAG retrieval failed for model %s", req.model_id)
        raise HTTPException(status_code=500, detail=str(e))