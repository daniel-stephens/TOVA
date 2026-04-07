from typing import List

from pydantic import BaseModel, Field


class RAGRetrieveRequest(BaseModel):
    query: str = Field(..., description="User question to retrieve context for")
    model_id: str = Field(..., description="Trained topic model ID")
    top_k_topics: int = Field(3, ge=1, le=50, description="Number of topics to retrieve")
    top_n_docs: int = Field(5, ge=1, le=50, description="Documents per topic")


class RAGDocument(BaseModel):
    doc_id: str
    text: str
    score: float = Field(..., description="Document-topic probability")


class RAGTopic(BaseModel):
    topic_id: int
    label: str
    keywords: str
    summary: str
    similarity: float = Field(..., description="Cosine similarity to query")
    documents: List[RAGDocument]


class RAGRetrieveResponse(BaseModel):
    topics: List[RAGTopic]
