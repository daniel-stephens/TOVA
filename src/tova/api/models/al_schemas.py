from typing import Dict, Optional

from pydantic import BaseModel, Field


class ALLabelRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID to label")
    label: str = Field(..., description="Human-assigned label for the document")


class ALRecommendation(BaseModel):
    doc_id: str = Field(..., description="Recommended document ID to label next")
    text: str = Field(..., description="Document text for display")
    current_labels: Dict[str, str] = Field(
        default_factory=dict,
        description="All current human labels: {doc_id: label}",
    )
    classifier_trained: bool = Field(
        False, description="Whether the recommendation is classifier-driven or random"
    )


class ALStatusResponse(BaseModel):
    model_id: str
    num_labeled: int
    num_unlabeled: int
    num_total: int
    class_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Number of labeled documents per class",
    )
    classifier_trained: bool


class ALLabelListResponse(BaseModel):
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="All current labels: {doc_id: label}",
    )
