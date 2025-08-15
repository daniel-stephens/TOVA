from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field # type: ignore

from tova.api.docs.examples import TRAIN_REQUEST_EXAMPLE, TRAIN_FILE_REQUEST_EXAMPLE
from tova.api.models.data_schemas import DataRecord

class TrainRequest(BaseModel):
    model: str = Field(..., description="Model key to load from the registry", example="tomotopyLDA")
    data: List[DataRecord] = Field(..., description="List of input records to train the model")
    id_col: Optional[str] = Field("id", description="Name of the ID column", example="id")
    text_col: str = Field("raw_text", description="Column containing the text", example="raw_text")
    do_preprocess: bool = Field(False, description="Whether to apply preprocessing", example=False)
    training_params: Optional[Dict[str, Any]] = Field(
        None, description="Custom model hyperparameters", example={"num_topics": 50, "alpha": 0.1}
    )
    config_path: Optional[str] = Field("static/config/config.yaml", description="Path to YAML config file", example="static/config/config.yaml")
    output: str = Field(..., description="Directory where model results will be saved", example="data/models/tomotopy_run1")

    class Config:
        json_schema_extra = {
            "example": TRAIN_REQUEST_EXAMPLE
        }

class TrainFileRequest(BaseModel):
    model: str = Field(..., description="Model key to load from the registry", example="tomotopyLDA")
    data_path: str = Field(..., description="Path to the input data file", example="data/bills_sample_100.csv")
    id_col: Optional[str] = Field("id", description="Name of the ID column", example="id")
    text_col: str = Field("raw_text", description="Column containing the text", example="raw_text")
    do_preprocess: bool = Field(False, description="Whether to apply preprocessing", example=False)
    training_params: Optional[Dict[str, Any]] = Field(
        None, description="Custom model hyperparameters", example={"num_topics": 50, "alpha": 0.1}
    )
    config_path: Optional[str] = Field("static/config/config.yaml", description="Path to YAML config file", example="static/config/config.yaml")
    output: str = Field(..., description="Directory where model results will be saved", example="data/models/tomotopy_run1")

    class Config:
        json_schema_extra = {
            "example": TRAIN_FILE_REQUEST_EXAMPLE
        }