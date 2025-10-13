from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field  # type: ignore


class DataRecord(BaseModel):
    id: Optional[str] = Field(
        None, description="Unique identifier for the record", example="112-HR-4219")
    raw_text: str = Field(..., description="Raw or preprocessed input text",
                          example="climate change is accelerating")
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


class TopicRecord(BaseModel):
    id: Optional[str] = Field(
        None,
        description="Unique identifier for the topic",
        example=0
    )  # TODO: Complete this based on what Daniel needs for visualization


class StorageType(str, Enum):
    database = "database"
    temporal = "temporal"


class Corpus(BaseModel):
    id: str  # system-assigned, unique
    owner_id: Optional[str] = None
    name: Optional[str] = None  # user-assigned
    description: Optional[str] = None
    created_at: str
    location: StorageType = StorageType.database
    metadata: Dict[str, Any] = None
    # this can be None since we can create a corpus and just add its metadata
    documents: List[DataRecord] = None


class Model(BaseModel):
    id: str  # system-assigned, unique
    owner_id: Optional[str] = None
    name: Optional[str] = None
    corpus_id: str  # ID of the corpus with which it was trained
    created_at: str
    location: StorageType = StorageType.database
    metadata: Dict[str, Any] = None
    # this can be None since we can create a model and just add its metadata
    topics: List[TopicRecord] = None


class DraftType(str, Enum):
    model = "model"
    corpus = "corpus"
    dataset = "dataset"


class Draft(BaseModel):
    id: str
    type: DraftType
    owner_id: str | None = None
    metadata: Dict[str, Any] | None = None
    data: Dict[str, Any] | None = None


class PromoteDraftResponse(BaseModel):
    job_id: str
    draft_id: str
