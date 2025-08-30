import logging
import os
import json
from typing import List, Dict

from fastapi import APIRouter, HTTPException # type: ignore

from tova.api.logger import logger
from tova.api.models.data_schemas import Corpus, Draft, ModelMeta

router = APIRouter()

# -----------------------
# API Routes
# -----------------------
# the user trains a model. Once the training is done, we ask the user whether he wants to save the model. If yes, we create a new model entry in the database.
# in the meanwhile, he can inspect the model because it is stored temporarily.
# once the model has been indexed succesfully, the model folder is deleted and all the information about such a model is removed from the temporary store.

# same applies for the corpus

# --- Corpora ---
@router.get("/corpora", response_model=List[Corpus], tags=["Corpora"])
def list_corpora():
    """List all corpora."""
    pass

@router.post("/corpora", response_model=Corpus, status_code=201, tags=["Corpora"])
def create_corpus(corpus: Corpus):
    """Create a new corpus."""
    pass
    
@router.get("/corpora/{corpus_id}", response_model=Corpus, tags=["Corpora"])
def get_corpus(corpus_id: str):
    """Get a specific corpus by ID."""
    pass

@router.delete("/corpora/{corpus_id}", status_code=204, tags=["Corpora"])
def delete_corpus(corpus_id: str):
    """Delete a specific corpus by ID."""
    pass

# --- Models (per corpus) ---
@router.get("/corpora/{corpus_id}/models", response_model=List[ModelMeta], tags=["Models"])
def list_corpus_models(corpus_id: str):
    """List all models for a specific corpus."""
    pass

@router.get("/corpora/{corpus_id}/models/{model_id}", response_model=ModelMeta, tags=["Models"])
def get_model(corpus_id: str, model_id: str):
    """Get a specific model by ID."""
    pass

@router.delete("/corpora/{corpus_id}/models/{model_id}", status_code=204, tags=["Models"])
def delete_model(corpus_id: str, model_id: str):
    """Delete a specific model by ID."""
    pass

# --- Drafts (temporary store) ---
@router.get("/drafts", response_model=List[Draft], tags=["Drafts"])
def list_drafts(): ...

@router.post("/drafts", response_model=Draft, status_code=201, tags=["Drafts"])
def create_draft(draft: Draft): ...

@router.delete("/drafts/{draft_id}", status_code=204, tags=["Drafts"])
def delete_draft(draft_id: str): ...

@router.post("/drafts/{draft_id}:commit", status_code=202, tags=["Drafts"])
def commit_draft(draft_id: str, corpus_id: str): ...