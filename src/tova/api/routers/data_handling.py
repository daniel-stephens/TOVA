import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import APIRouter, HTTPException # type: ignore

from tova.api.logger import logger
from tova.api.models.data_schemas import Corpus, Draft, DraftType, ModelMeta
from tova.api.jobs.store import job_store
from tova.api.jobs.domain import JobStatus, JobType
from tova.api.jobs.tokens import cancellation_tokens
from tova.utils.cancel import CancellationToken, CancelledError
from tova.core import drafts as drafts_dao

# the user trains a model. Once the training is done, we ask the user whether he wants to save the model. If yes, we create a new model entry in the database.
# in the meanwhile, he can inspect the model because it is stored temporarily.
# once the model has been indexed succesfully, the model folder is deleted and all the information about such a model is removed from the temporary store.

router = APIRouter(tags=["Data Handling"])

DRAFTS_SAVE = Path(os.getenv("DRAFTS_SAVE", "/data/drafts"))

def _infer_draft_type(draft_id: str) -> DraftType:
    if draft_id.startswith("m_"):
        return DraftType.model
    if draft_id.startswith("c_"):
        return DraftType.corpus
    raise HTTPException(status_code=400, detail="Invalid draft id prefix; expected m_ or c_")

# -----------------------
# API Routes
# -----------------------
@router.get("/drafts", response_model=List[Draft])
def list_drafts(type: Optional[DraftType] = None):
    """
    Drafts are directories under the drafts root:
      - m_XXXX/ (model drafts created during training)
      - c_XXXX/ (corpus drafts created during validation)
    """
    return drafts_dao.list_drafts(type)

@router.get("/drafts/{draft_id}", response_model=Draft)
def get_draft(draft_id: str):
    draft = drafts_dao.get_draft(draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    return draft

@router.delete("/drafts/{draft_id}", status_code=204)
def delete_draft(draft_id: str):
    kind = _infer_draft_type(draft_id)
    deleted = drafts_dao.delete_draft(kind, draft_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Draft not found")
    return