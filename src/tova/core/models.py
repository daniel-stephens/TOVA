
import shutil
from typing import List, Optional
import os
from pathlib import Path
from tova.api.models.data_schemas import DraftType, Model
from tova.core import drafts as drafts

# Import constants from drafts module
DRAFTS_SAVE = Path(os.getenv("DRAFTS_SAVE", "/data/drafts"))
METADATA_FILENAME = "metadata.json"
DATA_FILENAME = "data.json"


def list_models(corpus_id: str) -> List[Model]:
    """
    List all models for a given corpus, including both the "temporary" and the ones indexed in the database, if any
    """
    # TODO: the association between models and corpora needs to be implemented in drafts
    # list drafts and transform to Model objects
    models_drafts = drafts.list_drafts(type=DraftType.model)
    models_lst = [drafts.draft_to_model(draft) for draft in models_drafts]

    # TODO: Implement actual database query and merge with drafts
    return models_lst


def get_model(model_id: str) -> Optional[Model]:
    """
    Get a specific model by its ID.
    """
    model_drafts = drafts.list_drafts(type=DraftType.model)
    if model_id in [d.id for d in model_drafts]:
        #  convert draft to model
        model = drafts.draft_to_model(
            drafts.get_draft(model_id, DraftType.model))
        return model
    return None


def delete_model(model_id: str) -> bool:
    """
    Delete a model by ID within a corpus.
    """
    # list drafts datasets
    models_drafts = drafts.list_drafts(type=DraftType.model)
    if model_id in [d.id for d in models_drafts]:
        #  convert draft to model
        model = drafts.draft_to_model(
            drafts.get_draft(model_id, DraftType.model))
    else:  # search in database
        # TODO: Implement actual database deletion
        pass

    path_model = DRAFTS_SAVE / model.id

    if path_model.exists() and path_model.is_dir():
        shutil.rmtree(path_model)
        return True
    return False
