
import os
from datetime import datetime
from pathlib import Path
import shutil
from typing import List, Optional

from tova.api.models.data_schemas import (Corpus, DataRecord, Draft,
                                          DraftCreatedResponse, DraftType,
                                          StorageType)
from tova.core import drafts as drafts
from tova.utils.common import get_unique_id

# Import constants from drafts module
DRAFTS_SAVE = Path(os.getenv("DRAFTS_SAVE", "/data/drafts"))
METADATA_FILENAME = "metadata.json"
DATA_FILENAME = "data.json"


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


def create_corpus(corpus: Corpus) -> DraftCreatedResponse:
    """
    Create a new corpus by creating a draft on disk and returning corpus info.
    The draft will be later indexed in the persistent DB asynchronously if the user decides to do so.
    """

    corpus.id = get_unique_id(prefix="c_")
    corpus.created_at = datetime.utcnow().isoformat() + "Z"
    corpus.location = StorageType.temporal
    corpus.metadata = dict(corpus.metadata or {})

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
                sourcefile=doc.sourcefile or "api_upload",
                label=doc.label,
                original_id=doc.id
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
