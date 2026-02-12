
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
    Delete a model by ID. Removes the model directory from disk.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Use model_id directly to construct the path
    path_model = DRAFTS_SAVE / model_id

    if not path_model.exists():
        logger.warning(f"Model directory does not exist: {path_model}")
        return False
    
    if not path_model.is_dir():
        logger.warning(f"Model path exists but is not a directory: {path_model}")
        return False

    try:
        # Try to delete directly
        shutil.rmtree(path_model)
        logger.info(f"Successfully deleted model directory: {path_model}")
        return True
    except PermissionError as e:
        # Permission denied - likely owned by root
        logger.error(
            f"Permission denied deleting model directory {path_model}. "
            f"Directory may be owned by root. Error: {e}. "
            f"Try running: sudo rm -rf {path_model}"
        )
        # Re-raise with more context for the API to handle
        raise PermissionError(
            f"Cannot delete model {model_id}: permission denied. "
            f"The model directory is likely owned by root. "
            f"Please fix file ownership or delete manually."
        ) from e
    except OSError as e:
        # Other OS errors (e.g., file in use, read-only filesystem)
        logger.error(f"OS error deleting model directory {path_model}: {e}")
        raise
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error deleting model directory {path_model}: {e}")
        raise
