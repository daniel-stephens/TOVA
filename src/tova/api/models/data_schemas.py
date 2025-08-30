from typing import Any, Dict, Optional
from pydantic import BaseModel, Field # type: ignore
from enum import Enum

class DataRecord(BaseModel):
    id: Optional[str] = Field(None, description="Unique identifier for the record", example="112-HR-4219")
    raw_text: str = Field(..., description="Raw or preprocessed input text", example="climate change is accelerating")
    embeddings: Optional[str] = Field(
        None,
        description="Optional precomputed embeddings (e.g. stringified list or vector reference)",
        example="[0.1, 0.2, 0.3]",
    )
    tfidf: Optional[str] = Field(
        None,
        description="Optional TF-IDF representation of the document",
        example="[0.1, 0.2, 0.3]",
    )
    bow: Optional[str] = Field(
        None,
        description="Optional Bag-of-Words representation of the document",
        example="[0, 1, 5]",
    )
    
class Corpus(BaseModel):
    id: str
    name: str
    description: Optional[str] = None


class ModelStatus(str, Enum):
    queued = "queued"
    training = "training"
    ready = "ready"
    failed = "failed"
    cancelled = "cancelled"
    
class ModelMeta(BaseModel):
    id: str # system-assigned, unique
    owner_id: Optional[str] = None
    name: Optional[str] = None
    corpus_id: str # name of the corpus with which it was trained
    created_at: str
    status: str  # queued|training|ready|failed @TODO: check if keep (e.g., for listing models being trained)
    training_params: Dict[str, Any]
    
class Draft(BaseModel):
    id: str
    payload: dict