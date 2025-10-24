import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil
import logging

from tova.api.models.data_schemas import Corpus, Draft, DraftCreatedResponse, DraftType, DataRecord, Model, Dataset, StorageType
from tova.utils.common import get_unique_id, write_json_atomic

DRAFTS_SAVE = Path(os.getenv("DRAFTS_SAVE", "/data/drafts"))
METADATA_FILENAME = "metadata.json"
DATA_FILENAME = "data.json"

# temp topic models get saved through the TMmodel class directly in the draft path under a "m_XXXX" folder
# temp corpora get saved through the validation class directly in the draft path under a "c_XXXX" folder

logger = logging.getLogger(__name__)


def list_drafts(type: Optional[DraftType] = None) -> List[Draft]:
    """
    List drafts by type if provided, otherwise return both (models and corpora).
    Drafts are stored as directories under DRAFTS_SAVE:
      - m_xxxx for model drafts
      - c_xxxx for corpus drafts
    """
    drafts = []

    for dir in DRAFTS_SAVE.iterdir():
        if not dir.is_dir():
            continue

        if dir.name.startswith("m_"):
            draft_type = DraftType.model
        elif dir.name.startswith("c_"):
            draft_type = DraftType.corpus

        elif dir.name.startswith("d_"):
            draft_type = DraftType.dataset
        else:
            continue

        if type and draft_type != type:
            continue

        
        draft = get_draft(dir.name, draft_type)
        if draft:
            drafts.append(draft)
            
    return drafts


def get_draft(draft_id: str, kind: Optional[DraftType] = None) -> Optional[Draft]:
    """
    Load the draft metadata and data by ID. If kind is provided, validate the type.
    """
    draft_dir = DRAFTS_SAVE / draft_id
    if not draft_dir.exists():
        return None

    # Load metadata
    meta_path = draft_dir / METADATA_FILENAME
    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Determine draft type from ID prefix
    if draft_id.startswith("m_"):
        draft_type = DraftType.model
    elif draft_id.startswith("c_"):
        draft_type = DraftType.corpus
    elif draft_id.startswith("d_"):
        draft_type = DraftType.dataset
    else:
        return None

    if kind and draft_type != kind:
        return None

    # Load data
    data = None
    data_path = draft_dir / DATA_FILENAME
    if data_path.exists():
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass

    return Draft(
        id=draft_id,
        type=draft_type,
        path=str(draft_dir),
        metadata=metadata,
        data=data,
    )


def delete_draft(kind: DraftType, draft_id: str) -> bool:
    """
    Delete a draft by ID. Returns True if deleted, False if not found.
    """
    path_draft = DRAFTS_SAVE / draft_id
    if path_draft.exists() and path_draft.is_dir():
        shutil.rmtree(path_draft, ignore_errors=True)
        return True
    return False


def save_draft(draft: Draft) -> bool:

    try:
        logger.info("Saving draft: %s", draft.dict())
        draft_dir = DRAFTS_SAVE / draft.id
        draft_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(draft_dir / METADATA_FILENAME, draft.metadata)
        data_serialized = [record.dict() for record in draft.data]
        write_json_atomic(draft_dir / DATA_FILENAME, data_serialized)
        logger.info("Draft saved successfully: %s", draft.id)
        return DraftCreatedResponse(
            draft_id=draft.id,
            status_code=201
        )

    except Exception as e:
        logger.exception("Failed to save draft: %s", e)
        try:
            shutil.rmtree(draft_dir, ignore_errors=True)
        except Exception as cleanup_error:
            logger.exception("Failed to clean up draft directory: %s", cleanup_error)
        return DraftCreatedResponse(
            draft_id="",
            status_code=500
        )


def draft_to_data(draft: Draft) -> List[DataRecord]:
    """
    Convert a Draft to a list of DataRecord objects.
    """
    if draft.data is None:
        return []
    return [record if isinstance(record, DataRecord) else DataRecord(**record) for record in draft.data]

def draft_to_model(draft: Draft) -> Model:
    """
    Convert a Draft to a Model object.
    """
    if draft.type != DraftType.model:
        raise ValueError("Draft type must be 'model' to convert to Model.")

    return Model(
        id=draft.id,
        owner_id=draft.owner_id,
        name=draft.metadata.get("name"),
        corpus_id=draft.metadata.get("corpus_id"),
        created_at=draft.metadata.get("created_at"),
        location=StorageType.temporal,
        metadata=draft.metadata,
        topics=draft.data  # Assuming topics are stored in draft.data
    )


def draft_to_dataset(draft: Draft) -> Dataset:
    """
    Convert a Draft to a Dataset object.
    """
    if draft.type != DraftType.dataset:
        raise ValueError("Draft type must be 'dataset' to convert to Dataset.")

    return Dataset(
        id=draft.id,
        owner_id=draft.owner_id,
        name=draft.metadata.get("name"),
        description=draft.metadata.get("description"),
        created_at=draft.metadata.get("created_at"),
        location=StorageType.temporal,
        metadata=draft.metadata,
        documents=draft_to_data(draft)
    )

def draft_to_corpus(draft: Draft) -> Corpus:
    """
    Convert a Draft to a Corpus object.
    """
    if draft.type != DraftType.corpus:
        raise ValueError("Draft type must be 'corpus' to convert to Corpus.")

    return Corpus(
        id=draft.id,
        owner_id=draft.owner_id,
        name=draft.metadata.get("name"),
        description=draft.metadata.get("description"),
        created_at=draft.metadata.get("created_at"),
        location=StorageType.temporal,
        metadata=draft.metadata,
        documents=draft_to_data(draft)
    )