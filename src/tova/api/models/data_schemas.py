from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field  # type: ignore


class DataRecord(BaseModel):
    id: Optional[str] = Field(
        None, description="Unique identifier for the record", example="112-HR-4219")
    original_id: Optional[str] = Field(
        None, description="Original identifier for the record, if any", example="orig-5678")
    text: str = Field(..., description="Raw or preprocessed input text",
                          example="climate change is accelerating")
    sourcefile: Optional[str] = Field(
        None, description="Optional source file name or identifier",
        example="document1.txt",
    )
    label: Optional[str] = Field(
        None, description="Optional label for the document", example="environment"
    )
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


class Dataset(BaseModel):
    id: Optional[str] = None            
    owner_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[str] = None
    location: StorageType = StorageType.database
    metadata: Dict[str, Any] = Field(default_factory=dict)   
    documents: List[DataRecord] = Field(default_factory=list) 


class Corpus(BaseModel):
    id: Optional[str] = None           
    owner_id: Optional[str] = None
    name: Optional[str] = None  # user-assigned
    description: Optional[str] = None
    created_at: Optional[str] = None
    location: StorageType = StorageType.database
    metadata: Dict[str, Any] = Field(default_factory=dict)  
    datasets: Optional[List[Dataset]] = None
    documents: Optional[List[DataRecord]] = None

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
    data: List[DataRecord] | None = None


class PromoteDraftResponse(BaseModel):
    job_id: str
    draft_id: str
    
class DraftCreatedResponse(BaseModel):
    draft_id: str
    status_code: int
