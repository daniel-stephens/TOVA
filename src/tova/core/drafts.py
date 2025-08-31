import json
import os
from pathlib import Path
from typing import List, Optional

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


def get_draft(draft_id: str) -> Optional[Draft]:
    """
    Load a single draft by ID, or None if not found.
    """
    path_draft = DRAFTS_SAVE / draft_id
    if not path_draft.exists():
        return None
    with open(path_draft, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Draft(**data)


def delete_draft(draft_id: str) -> bool:
    """
    Delete a draft by ID. Returns True if deleted, False if not found.
    """
    path_draft = DRAFTS_SAVE / draft_id
    if path_draft.exists():
        path_draft.unlink()
        return True
    return False
