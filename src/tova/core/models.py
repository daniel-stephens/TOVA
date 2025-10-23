
from typing import List, Optional

from tova.api.models.data_schemas import DraftType, Model
from tova.core import drafts


def list_models(corpus_id: str) -> List[Model]:
    """
    List all models for a given corpus, including both the "temporary" and the ones indexed in the database, if any
    """
    # TODO: Implement actual database query
    # For now, return empty list - this should query the persistent storage
    return drafts.list_drafts(type=DraftType.model)


def get_model(corpus_id: str, model_id: str) -> Optional[Model]:
    """
    Get a specific model by its ID within a corpus.
    """
    # TODO: Implement actual database query
    # Should check both permanent storage and drafts
    return None


def delete_model(corpus_id: str, model_id: str) -> bool:
    """
    Delete a model by ID within a corpus.
    """
    # TODO: Implement actual database deletion
    # Should handle both permanent storage and drafts
    return False
