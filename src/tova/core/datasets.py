import logging
from datetime import datetime
from typing import List, Optional

from tova.api.models.data_schemas import (Dataset, Draft, DraftCreatedResponse,
                                          DraftType, StorageType)
from tova.core import drafts
from tova.utils.common import get_unique_id

# @TODO: make this logging centralized
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def list_datasets() -> List[Dataset]:
    """
    List all available datasets.
    """
    # TODO: Implement actual database query
    # For now, return empty list - this should query the persistent storage
    return drafts.list_drafts(type=DraftType.dataset)


def create_dataset(dataset: Dataset) -> DraftCreatedResponse:
    """
    Create a new dataset by creating a draft internally and returning dataset info.
    The draft will be indexed asynchronously in the background.
    """

    dataset.id = get_unique_id(prefix="d_")
    dataset.created_at = datetime.utcnow().isoformat() + "Z"
    dataset.location = StorageType.temporal
    dataset.metadata = dict(dataset.metadata or {})

    if not dataset.documents:
        raise ValueError("Datasets must contain documents to create a corpus.")

    try:
        logger.info("Creating draft with dataset: %s", dataset.dict())
        draft = Draft(
            id=dataset.id,
            type=DraftType.dataset,
            owner_id=dataset.owner_id,
            metadata={
                "name": dataset.name,
                "description": dataset.description,
                "created_at": dataset.created_at,
                "owner_id": dataset.owner_id,
            },
            data=dataset.documents
        )
        logger.info("Draft created: %s", draft.dict())
        return drafts.save_draft(draft)
    except Exception as e:
        logger.exception("Failed to save draft: %s", e)
        raise


def get_dataset(dataset_id: str) -> Optional[Dataset]:
    """
    Get a dataset by ID.
    """

    # list drafts datasets
    dataset_drafts = drafts.list_drafts(type=DraftType.dataset)
    if dataset_id in [d.id for d in dataset_drafts]:
        #  convert draft to dataset
        dtset = drafts.draft_to_dataset(
            drafts.get_draft(dataset_id, DraftType.dataset))
        return dtset
    else:  # search in database
        # TODO: Implement actual database retrieval
        pass


def delete_dataset(dataset_id: str) -> bool:
    """
    Delete a dataset by ID.
    """
    # TODO: Implement actual database deletion
    # Should handle both permanent storage and drafts
    return False
