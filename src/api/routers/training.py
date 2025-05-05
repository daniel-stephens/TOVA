import logging
import os
import json
from typing import List, Dict

from fastapi import APIRouter, HTTPException # type: ignore

from src.api.models.train_requests import TrainRequest, TrainFileRequest
from src.utils.tm_utils import prepare_training_data, normalize_json_data
from src.utils.common import init_logger
from src.core.dispatchers import train_model_dispatch

router = APIRouter()

# -----------------------
# Helpers
# -----------------------

def _train_via_api(
    model: str,
    data: List[Dict],
    output: str,
    logger: logging.Logger,
    do_preprocess: bool = False,
    training_params: Dict = None,
    config_path: str = "static/config/config.yaml"
) -> Dict:
    """
    Shared logic for JSON and file-based training routes.
    """
    try:
        duration = train_model_dispatch(
            model=model,
            data=data,
            output=output,
            config_path=config_path,
            logger=logger,
            do_preprocess=do_preprocess,
            tr_params=training_params
        )

        return {
            "status": "Training completed",
            "duration": duration
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------
# API Routes
# -----------------------

@router.post("/json", tags=["Training"])
def train_model_from_json(req: TrainRequest):
    
    logger = init_logger(req.config_path)
    
    normalized_data = normalize_json_data(
        raw_data = json.dumps([record.model_dump() for record in req.data]),
        id_col=req.id_col if req.id_col else None,
        text_col=req.text_col,
        logger=logger
    )
    return _train_via_api(
        model=req.model,
        data=normalized_data,
        output=req.output,
        logger=logger,
        config_path=req.config_path,
        do_preprocess=req.do_preprocess,
        training_params=req.training_params
    )


@router.post("/file", tags=["Training"])
def train_model_from_file(req: TrainFileRequest):
    if not os.path.isfile(req.data_path):
        raise HTTPException(status_code=400, detail="Data file not found")
    
    logger = init_logger(req.config_path)

    normalized_data = prepare_training_data(
        path=req.data_path,
        logger=logger,
        text_col=req.text_col,
        id_col=req.id_col if req.id_col else None,
        get_embeddings=True
    )
    return _train_via_api(
        model=req.model,
        data=normalized_data,
        output=req.output,
        logger=logger,
        config_path=req.config_path,
        do_preprocess=req.do_preprocess,
        training_params=req.training_params
    )
