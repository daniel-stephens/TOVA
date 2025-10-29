from typing import List, Optional
from pydantic import BaseModel, Field # type: ignore

from tova.api.docs.examples import INFER_REQUEST_EXAMPLE
from tova.api.models.data_schemas import DataRecord

class InferRequest(BaseModel):
    #model_path: str = Field(..., description="Path to the model directory that is going to be used for inference", example="data/models/tomotopy_run1")
    model_id: str = Field(..., description="ID of the model to be used for inference", example="m_cbf054a0b4f44581b6c2e56c71836458")
    data: List[DataRecord] = Field(..., description="List of input records to make inference on", example=[{"id": 1, "raw_text": "This is a sample text."}])
    id_col: Optional[str] = Field("id", description="Name of the ID column", example="id")
    text_col: str = Field("raw_text", description="Column containing the text", example="raw_text")
    config_path: Optional[str] = Field("static/config/config.yaml", description="Path to YAML config file", example="static/config/config.yaml")
    
    class Config:
        json_schema_extra = {
            "example": INFER_REQUEST_EXAMPLE
        }

class InferResponse(BaseModel):
    job_id: str