from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field # type: ignore

from tova.api.docs.examples import TRAIN_REQUEST_EXAMPLE
from tova.api.models.data_schemas import DataRecord

class TrainRequest(BaseModel):
    model: str = Field(..., description="Model key to load from the registry", example="tomotopyLDA")
    data: List[DataRecord] = Field(..., description="List of input records to train the model")
    id_col: Optional[str] = Field("id", description="Name of the ID column", example="id")
    text_col: str = Field("raw_text", description="Column containing the text", example="raw_text")
    training_params: Optional[Dict[str, Any]] = Field(
        None, description="Custom model hyperparameters", example={"num_topics": 50, "alpha": 0.1}
    )
    config_path: Optional[str] = Field("static/config/config.yaml", description="Path to YAML config file", example="static/config/config.yaml")
    model_name: Optional[str] = Field("tomotopyLDA", description="Name of the model", example="tomotopyLDA")
    corpus_id: Optional[str] = Field(None, description="Name of the corpus used for training", example="c_cbf054a0b4f44581b6c2e56c71836458")
    class Config:
        json_schema_extra = {
            "example": TRAIN_REQUEST_EXAMPLE
        }

class TrainResponse(BaseModel):
    job_id: str
    model_id: str