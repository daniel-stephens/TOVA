
import os
from datetime import datetime
from pathlib import Path
import shutil
from typing import List, Optional

from tova.api.models.data_schemas import (Corpus, DataRecord, Draft,
                                          DraftCreatedResponse, DraftType,
                                          StorageType, Model)
from tova.core import drafts as drafts
from tova.core import models
from tova.utils.common import get_unique_id

# Import constants from drafts module
DRAFTS_SAVE = Path(os.getenv("DRAFTS_SAVE", "/data/drafts"))
METADATA_FILENAME = "metadata.json"
DATA_FILENAME = "data.json"


def _corpus_owner_id(corpus: Corpus) -> Optional[str]:
    return corpus.owner_id or (corpus.metadata or {}).get("owner_id")


def _corpus_display_name(corpus: Corpus) -> str:
    name = corpus.name or (corpus.metadata or {}).get("name") or ""
    return str(name).strip()


def owner_has_corpus_name(
    owner_id: Optional[str],
    name: str,
    *,
    exclude_id: Optional[str] = None,
) -> bool:
    """Return True if the owner already has a corpus with this name (case-insensitive)."""
    key = str(name or "").strip().lower()
    if not key:
        return False

    for corpus in list_corpora():
        if exclude_id and corpus.id == exclude_id:
            continue
        if _corpus_owner_id(corpus) != owner_id:
            continue
        if _corpus_display_name(corpus).lower() == key:
            return True
    return False


def list_corpora() -> List[Corpus]:
    """
    List all corpora, including both the "temporary" and the ones indexed in the database, if any.
    """
    # list drafts and transform to Corpus objects
    corpora_drafts = drafts.list_drafts(type=DraftType.corpus)
    corpora_lst = [drafts.draft_to_corpus(draft) for draft in corpora_drafts]
    # do not list the documents inside each corpus for listing purposes
    #  remove the documents field
    for corpus in corpora_lst:
        corpus.documents = None

    # TODO: Implement actual database query and merge with drafts
    return corpora_lst

def list_corpus_models(corpus_id: str) -> List[Model]:
    """
    List all model objects associated with a given corpus, including both the "temporary" and the ones indexed in the database, if any.
    """
    
    corpus = get_corpus(corpus_id)
    if not corpus:
        return []
    
    models_lst = []
    if corpus.models:  # Check if corpus.models exists and is not None
        for model_id in corpus.models:
            model = models.get_model(model_id)
            if model:  # Only add if model exists
                models_lst.append(model)
    
    return models_lst


def create_corpus(corpus: Corpus) -> DraftCreatedResponse:
    """
    Create a new corpus by creating a draft on disk and returning corpus info.
    The draft will be later indexed in the persistent DB asynchronously if the user decides to do so.
    """

    corpus_name = (corpus.name or "").strip()
    if not corpus_name:
        raise ValueError("Corpus name is required.")

    owner_id = _corpus_owner_id(corpus)
    if owner_has_corpus_name(owner_id, corpus_name):
        raise ValueError(f"A corpus named '{corpus_name}' already exists.")

    corpus.id = get_unique_id(prefix="c_")
    corpus.created_at = datetime.utcnow().isoformat() + "Z"
    corpus.location = StorageType.temporal
    corpus.metadata = dict(corpus.metadata or {})
    corpus.name = corpus_name

    documents_data = []
    if not corpus.datasets:
        raise ValueError(
            "At least one dataset must be provided to create a corpus.")

    for dataset in corpus.datasets:
        if not dataset.documents:
            raise ValueError(
                "Datasets must contain documents to create a corpus.")

        for doc in dataset.documents:
            if not isinstance(doc.text, str) or not doc.text.strip():
                raise ValueError(
                    "Every document must include a non-empty 'text' field.")

            documents_data.append(DataRecord(
                id=get_unique_id(prefix="doc_"),
                text=doc.text,
                # Carry the original document through. ``text`` is preprocessed
                # (lemmatised, stopped, bigram-joined) and is what the models
                # train on; ``raw_text`` is what a human has to read. Dropping it
                # here left readers with nothing legible to display.
                raw_text=doc.raw_text or doc.text,
                sourcefile=doc.sourcefile or "api_upload",
                label=doc.label,
                original_id=doc.original_id or doc.id
            ))

    draft = Draft(
        id=corpus.id,
        type=DraftType.corpus,
        owner_id=corpus.owner_id,
        metadata={
            "name": corpus.name,
            "description": corpus.description,
            "created_at": corpus.created_at,
            "owner_id": corpus.owner_id,
        },
        data=documents_data
    )
    return drafts.save_draft(draft)


def get_corpus(corpus_id: str) -> Optional[Corpus]:
    """
    Get a specific corpus by its ID.
    """
    # list drafts datasets
    corpora_drafts = drafts.list_drafts(type=DraftType.corpus)
    if corpus_id in [d.id for d in corpora_drafts]:
        #  convert draft to corpus
        corpus = drafts.draft_to_corpus(
            drafts.get_draft(corpus_id, DraftType.corpus))
        return corpus
    else:  # search in database
        # TODO: Implement actual database retrieval
        pass
    return None


def delete_corpus(corpus_id: str) -> bool:
    """
    Delete a corpus by ID.
    """
    
    # list drafts datasets
    corpora_drafts = drafts.list_drafts(type=DraftType.corpus)
    if corpus_id in [d.id for d in corpora_drafts]:
        #  convert draft to corpus
        corpus = drafts.draft_to_corpus(
            drafts.get_draft(corpus_id, DraftType.corpus))
    else:  # search in database
        # TODO: Implement actual database deletion
        pass

    path_corpus = DRAFTS_SAVE / corpus.id
    
    if path_corpus.exists() and path_corpus.is_dir():
        shutil.rmtree(path_corpus)
        return True
    return False
