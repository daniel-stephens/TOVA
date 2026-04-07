"""Active-learning API router.

Provides endpoints to initialise per-model active-learning sessions,
submit / remove human labels, and receive document recommendations based
on classifier uncertainty (entropy sampling).
"""

from fastapi import APIRouter, HTTPException

from tova.api.models.al_schemas import (
    ALLabelListResponse,
    ALLabelRequest,
    ALRecommendation,
    ALStatusResponse,
)
from tova.core import active_learning as al

router = APIRouter(tags=["Active Learning"])


@router.get(
    "/{model_id}/recommend",
    response_model=ALRecommendation,
    summary="Get the next recommended document to label",
)
def recommend(model_id: str):
    try:
        session = al.get_session(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        doc_id, text, classifier_driven = session.recommend()
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return ALRecommendation(
        doc_id=doc_id,
        text=text,
        current_labels=session.get_labels(),
        classifier_trained=classifier_driven,
    )


@router.post(
    "/{model_id}/label",
    response_model=ALRecommendation,
    summary="Submit a human label and get the next recommendation",
)
def add_label(model_id: str, req: ALLabelRequest):
    try:
        session = al.get_session(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        session.add_label(req.doc_id, req.label)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        doc_id, text, classifier_driven = session.recommend()
    except ValueError:
        return ALRecommendation(
            doc_id="",
            text="All documents have been labeled.",
            current_labels=session.get_labels(),
            classifier_trained=session._classifier is not None,
        )

    return ALRecommendation(
        doc_id=doc_id,
        text=text,
        current_labels=session.get_labels(),
        classifier_trained=classifier_driven,
    )


@router.delete(
    "/{model_id}/label/{doc_id}",
    response_model=ALRecommendation,
    summary="Remove a label and get an updated recommendation",
)
def remove_label(model_id: str, doc_id: str):
    try:
        session = al.get_session(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        session.remove_label(doc_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        rec_id, text, classifier_driven = session.recommend()
    except ValueError:
        return ALRecommendation(
            doc_id="",
            text="All documents have been labeled.",
            current_labels=session.get_labels(),
            classifier_trained=session._classifier is not None,
        )

    return ALRecommendation(
        doc_id=rec_id,
        text=text,
        current_labels=session.get_labels(),
        classifier_trained=classifier_driven,
    )


@router.get(
    "/{model_id}/status",
    response_model=ALStatusResponse,
    summary="Get active-learning session status",
)
def get_status(model_id: str):
    try:
        session = al.get_session(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ALStatusResponse(**session.get_status())


@router.get(
    "/{model_id}/labels",
    response_model=ALLabelListResponse,
    summary="Get all current human labels",
)
def get_labels(model_id: str):
    try:
        session = al.get_session(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ALLabelListResponse(labels=session.get_labels())
