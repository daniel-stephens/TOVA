from pydantic import BaseModel, Field # type: ignore
from typing import Optional

from src.api.docs.examples import MODEL_INFO_REQUEST_EXAMPLE, TOPIC_INFO_REQUEST_EXAMPLE

class ModelInfoRequest(BaseModel):
    model_path: str = Field(..., description="Path to the trained model directory", example="data/models/tomotopy_run1")
    config_path: Optional[str] = Field("static/config/config.yaml", description="Path to the YAML configuration file", example="static/config/config.yaml")

    class Config:
        json_schema_extra = {
            "example": MODEL_INFO_REQUEST_EXAMPLE
        }

class TopicInfoRequest(BaseModel):
    topic_id: int = Field(..., description="ID of the topic to query", example=0)
    model_path: str = Field(..., description="Path to the trained model directory", example="data/models/tomotopy_run1")
    config_path: Optional[str] = Field("static/config/config.yaml", description="Path to the YAML configuration file", example="static/config/config.yaml")

    class Config:
        json_schema_extra = {
            "example": TOPIC_INFO_REQUEST_EXAMPLE
        }