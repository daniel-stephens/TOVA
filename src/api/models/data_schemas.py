from typing import Optional
from pydantic import BaseModel, Field # type: ignore

class DataRecord(BaseModel):
    id: Optional[str] = Field(None, description="Unique identifier for the record", example="112-HR-4219")
    raw_text: str = Field(..., description="Raw or preprocessed input text", example="climate change is accelerating")
    embeddings: Optional[str] = Field(
        None,
        description="Optional precomputed embeddings (e.g. stringified list or vector reference)",
        example="[0.1, 0.2, 0.3]"
    )