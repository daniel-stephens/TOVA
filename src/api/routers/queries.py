import os
import json
from typing import Dict

from fastapi import APIRouter, HTTPException # type: ignore

from src.api.models.query_requests import ModelInfoRequest
from src.core.dispatchers import get_model_info_dispatch
from src.utils.common import init_logger

router = APIRouter()

# -----------------------
# API Route
# -----------------------

@router.post("/model-info", tags=["Queries"])
def get_model_info(req: ModelInfoRequest) -> Dict[str, Dict]:
    """
    Returns topic-level metadata for a trained topic model.
    """
    if not os.path.isdir(req.model_path):
        raise HTTPException(status_code=400, detail="Model path not found or not a directory.")

    try:
        logger = init_logger(config_file=req.config_path)
        model_info = get_model_info_dispatch(
            model_path=req.model_path,
            config_path=req.config_path,
            logger=logger
        )
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
