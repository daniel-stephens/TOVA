import os
import json
from typing import Dict, Any

from fastapi import APIRouter, HTTPException # type: ignore

from tova.api.models.query_requests import ModelInfoRequest, ThetasByDocsIdsRequest, TopicInfoRequest
from tova.core.dispatchers import get_model_info_dispatch, get_thetas_documents_by_id_dispatch, get_topic_info_dispatch
from tova.api.logger import logger

router = APIRouter()

# -----------------------
# API Route
# -----------------------
@router.post("/dashboard-data", tags=["Queries"])
def get_dashboard_data(req: ModelInfoRequest) -> Dict[str, Any]:
    """
    Returns topic model data formatted for dashboard visualization.
    """
    if not os.path.isdir(req.model_path):
        raise HTTPException(status_code=400, detail="Model path not found or not a directory.")

    try:
        model_info = get_model_info_dispatch(
            model_path=req.model_path,
            config_path=req.config_path,
            model_metadata=req.model_metadata,
            model_training_corpus=req.model_training_corpus,
            logger=logger,
            output="dashboard",
        )
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model-info", tags=["Queries"])
def get_model_info(req: ModelInfoRequest) -> Dict[str, Any]:
    """
    Returns topic-level metadata for a trained topic model.
    """
    if not os.path.isdir(req.model_path):
        raise HTTPException(status_code=400, detail="Model path not found or not a directory.")

    try:
        model_info = get_model_info_dispatch(
            model_path=req.model_path,
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
    if not os.path.isdir(req.model_path):
        raise HTTPException(status_code=400, detail="Model path not found or not a directory.")

    try:
        topic_info = get_topic_info_dispatch(
            model_path=req.model_path,
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
    if not os.path.isdir(req.model_path):
        raise HTTPException(status_code=400, detail="Model path not found or not a directory.")

    try:
        thetas_info = get_thetas_documents_by_id_dispatch(
            model_path=req.model_path,
            docs_ids=req.docs_ids.split(","),
            config_path=req.config_path,
            logger=logger
        )
        return thetas_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))