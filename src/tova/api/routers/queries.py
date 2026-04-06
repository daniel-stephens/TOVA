import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException  # type: ignore
from pydantic import BaseModel, Field

from tova.api.models.query_requests import ModelInfoRequest, ThetasByDocsIdsRequest, TopicInfoRequest
from tova.api.models.rag_schemas import RAGRetrieveRequest, RAGRetrieveResponse, RAGTopic, RAGDocument
from tova.core.dispatchers import get_model_info_dispatch, get_thetas_documents_by_id_dispatch, get_topic_info_dispatch
from tova.core.topic_retriever import get_retriever
from tova.api.logger import logger
from tova.utils.common import load_yaml_config_file

router = APIRouter()

# paths to temporary storage
DRAFTS_SAVE = Path(os.getenv("DRAFTS_SAVE", "/data/drafts"))

# -----------------------
# API Route
# -----------------------
@router.post("/model-info", tags=["Queries"])
def get_model_info(req: ModelInfoRequest) -> Dict[str, Any]:
    """
    Returns topic-level metadata for a trained topic model.
    """
    if not os.path.isdir(DRAFTS_SAVE.joinpath(req.model_id).as_posix()):
        raise HTTPException(status_code=400, detail="Model path not found or not a directory.")

    try:
        model_info = get_model_info_dispatch(
            model_path= DRAFTS_SAVE.joinpath(req.model_id).as_posix(),
            config_path=req.config_path,
            logger=logger
        )
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/topic-info", tags=["Queries"])
def get_topic_info(req: TopicInfoRequest) -> Dict[str, Any]:
    """
    Returns topic-level metadata for a specific topic in a trained topic model.
    """
    if not os.path.isdir(DRAFTS_SAVE.joinpath(req.model_id).as_posix()):
        raise HTTPException(status_code=400, detail="Model path not found or not a directory.")

    try:
        topic_info = get_topic_info_dispatch(
            model_path=DRAFTS_SAVE.joinpath(req.model_id).as_posix(),
            config_path=req.config_path,
            topic_id=req.topic_id,
            logger=logger
        )
        return topic_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/thetas-by-docs-ids", tags=["Queries"])
def get_thetas_by_docs_ids(req: ThetasByDocsIdsRequest) -> Dict[str, Any]:
    """
    Returns topic weights for specific documents by their IDs from a trained topic model.
    """
    if not os.path.isdir(DRAFTS_SAVE.joinpath(req.model_id).as_posix()):
        raise HTTPException(status_code=400, detail="Model path not found or not a directory.")

    try:
        thetas_info = get_thetas_documents_by_id_dispatch(
            model_path=DRAFTS_SAVE.joinpath(req.model_id).as_posix(),
            docs_ids=req.docs_ids.split(","),
            config_path=req.config_path,
            logger=logger
        )
        return thetas_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag-retrieve", response_model=RAGRetrieveResponse, tags=["Queries"])
def rag_retrieve(req: RAGRetrieveRequest):
    """Retrieve the most relevant topics and documents for a user query.

    Uses sentence-transformer embeddings to rank topics by semantic
    similarity to the query, then returns the top representative
    documents from those topics.
    """
    if not DRAFTS_SAVE.joinpath(req.model_id).is_dir():
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        retriever = get_retriever(req.model_id)
        raw = retriever.retrieve(
            query=req.query,
            top_k_topics=req.top_k_topics,
            top_n_docs=req.top_n_docs,
        )
        topics = [
            RAGTopic(
                topic_id=t["topic_id"],
                label=t["label"],
                keywords=t["keywords"],
                summary=t["summary"],
                similarity=t["similarity"],
                documents=[
                    RAGDocument(doc_id=d["doc_id"], text=d["text"], score=d["score"])
                    for d in t["documents"]
                ],
            )
            for t in raw
        ]
        return RAGRetrieveResponse(topics=topics)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("RAG retrieval failed for model %s", req.model_id)
        raise HTTPException(status_code=500, detail=str(e))


# --------------- Document topic suggestion ---------------

_DOC_TOPIC_PROMPT_PATH = Path("src/tova/prompter/prompts/doc_topic_suggestion.txt")
_MAX_DOC_CHARS = 4000


class DocTopicSuggestionRequest(BaseModel):
    text: str = Field(..., description="Document text to analyze")
    config_path: Optional[str] = Field(
        "static/config/config.yaml",
        description="Path to YAML config file",
    )


class TopicSuggestion(BaseModel):
    label: str
    description: str = ""


class DocTopicSuggestionResponse(BaseModel):
    suggestions: List[TopicSuggestion]


def _parse_suggestion_json(raw: str) -> list[dict]:
    """Best-effort extraction of a JSON array from LLM output."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(parsed, list):
        return []
    return [
        {"label": str(item["label"]).strip(), "description": str(item.get("description", "")).strip()}
        for item in parsed[:3]
        if isinstance(item, dict) and "label" in item
    ]


@router.post(
    "/suggest-doc-topics",
    response_model=DocTopicSuggestionResponse,
    tags=["Queries"],
)
def suggest_doc_topics(req: DocTopicSuggestionRequest):
    """Use the configured LLM (via Prompter) to suggest 3 candidate topic labels for a document."""
    from tova.prompter.prompter import Prompter

    config_path = Path(req.config_path or "static/config/config.yaml")
    try:
        tm_cfg = load_yaml_config_file(str(config_path), "topic_modeling", logger)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config error: {e}")

    general = tm_cfg.get("general", {})
    model_type = general.get("llm_model_type", "gemma3:4b")
    llm_server = general.get("llm_server")
    llm_provider = general.get("llm_provider")

    try:
        prompter = Prompter(
            config_path=config_path,
            model_type=model_type,
            llm_server=llm_server,
            llm_provider=llm_provider,
            logger=logger,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM not available: {e}")

    prompt_template = _DOC_TOPIC_PROMPT_PATH.read_text()
    question = prompt_template.format(doc_text=req.text[:_MAX_DOC_CHARS])

    try:
        result_text, _ = prompter.prompt(
            system_prompt_template_path=None,
            question=question,
        )
    except Exception as e:
        logger.exception("Prompter call failed for suggest-doc-topics")
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    suggestions = _parse_suggestion_json(result_text)
    if not suggestions:
        logger.warning("Could not parse LLM topic suggestions: %s", result_text[:300])
        suggestions = [{"label": result_text[:80].strip(), "description": ""}]

    return DocTopicSuggestionResponse(
        suggestions=[TopicSuggestion(**s) for s in suggestions]
    )