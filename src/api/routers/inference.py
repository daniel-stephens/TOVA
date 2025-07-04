import logging
import os
import json
from typing import List, Dict

from fastapi import APIRouter, HTTPException # type: ignore

from src.api.models.infer_requests import InferRequest, InferFileRequest
from src.utils.tm_utils import prepare_training_data, normalize_json_data
from src.api.logger import logger
from src.core.dispatchers import infer_model_dispatch

router = APIRouter()

# -----------------------
# Helpers
# -----------------------

def _infer_via_api(
    model_path: str,
    data: List[Dict],
    logger: logging.Logger,
    config_path: str = "static/config/config.yaml"
) -> Dict:
    """
    Shared logic for JSON and file-based training routes.
    """
    try:

        thetas, duration = infer_model_dispatch(
            model_path=model_path,
            data=data,
            config_path=config_path,
            logger=logger
        )

        return {
            "thetas": thetas,
            "duration": duration
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------
# API Routes
# -----------------------

@router.post("/json", tags=["Inference"])
def infer_from_json(req: InferRequest):
        
    normalized_data = normalize_json_data(
        raw_data = json.dumps([record.model_dump() for record in req.data]),
        id_col=req.id_col if req.id_col else None,
        text_col=req.text_col,
        logger=logger
        
    )
    return _infer_via_api(
        model_path=req.model_path,
        data=normalized_data,
        logger=logger,
        config_path=req.config_path
    )


@router.post("/file", tags=["Inference"])
def infer_from_file(req: InferFileRequest):
    if not os.path.isfile(req.data_path):
        raise HTTPException(status_code=400, detail="Data file not found")
    
    normalized_data = prepare_training_data(
        path=req.data_path,
        logger=logger,
        text_col=req.text_col,
        id_col=req.id_col if req.id_col else None,
        get_embeddings=True
    )
    
    return _infer_via_api(
        model_path=req.model_path,
        data=normalized_data,
        logger=logger,
        config_path=req.config_path
    )