from typing import List, Optional
from pydantic import BaseModel, Field # type: ignore
from tova.api.models.data_schemas import DataRecord # type: ignore

class CorpusFileMeta(BaseModel):
    corpus_name: str = Field(..., description="Name of the corpus", example="climate_corpus")
    corpus_path: Optional[str] = Field(..., description="Path to the corpus files", example="/path/to/corpus") #optional because when we delete (from solr) we do not need the path to where it was located before indexing)
    description: Optional[str] = Field(None, description="Optional description of the corpus")

class CorpusJSONRequest(BaseModel):
    corpus_name: str = Field(..., description="Name of the corpus", example="climate_corpus")
    description: Optional[str] = Field(None, description="Optional description of the corpus")
    records: List[DataRecord] = Field(..., description="List of records to index into the corpus")