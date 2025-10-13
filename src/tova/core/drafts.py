import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tova.api.models.data_schemas import Draft, DraftType

DRAFTS_SAVE = Path(os.getenv("DRAFTS_SAVE", "/data/drafts"))
# temp topic models get saved through the TMmodel class directly in the draft path under a "m_XXXX" folder
# temp corpora get saved through the validation class directly in the draft path under a "c_XXXX" folder


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

        meta_path = dir / "metadata.json"
        metadata = {}
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        drafts.append(
            Draft(
                id=dir.name,
                type=draft_type,
                path=str(dir),
                metadata=metadata,
            )
        )
    return drafts


def get_draft_metadata(draft_id: str) -> Optional[Draft]:
    """
    Load a single draft by ID, or None if not found.
    """

    meta_path = DRAFTS_SAVE / draft_id / "metadata.json"
    
    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return Draft(
            id=draft_id,
            type=DraftType.model if draft_id.startswith("m_") else DraftType.corpus,
            path=str(dir),
            metadata=metadata,
        )

def get_draft_data(draft_id: str, kind: DraftType) -> Optional[Dict[str, Any]]:
    """
    Load the draft data from a draft by ID and kind, or None if not found.
    """
    if kind == DraftType.corpus:
        return get_corpus_draft_data(draft_id)
    
    elif kind == DraftType.dataset:
        return get_dataset_draft_data(draft_id)

    else:
        return get_model_draft_data(draft_id)

def get_corpus_draft_data(draft_id: str) -> Optional[Dict[str, Any]]:
    """
    Load the corpus data from a draft by ID, or None if not found.
    """
    path_draft = DRAFTS_SAVE / draft_id / "data.json"
    if not path_draft.exists():
        return None
    with open(path_draft, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_dataset_draft_data(draft_id: str) -> Optional[Dict[str, Any]]:
    """
    Load the corpus data from a draft by ID, or None if not found.
    """
    path_draft = DRAFTS_SAVE / draft_id / "dataset.json"
    if not path_draft.exists():
        return None
    with open(path_draft, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_model_draft_data(draft_id: str):
    """
    Load the model data from a draft by ID, or None if not found.
    """
    # TODO
    pass

def delete_draft(draft_id: str) -> bool:
    """
    Delete a draft by ID. Returns True if deleted, False if not found.
    """
    path_draft = DRAFTS_SAVE / draft_id
    if path_draft.exists():
        path_draft.unlink()
        return True
    return False
