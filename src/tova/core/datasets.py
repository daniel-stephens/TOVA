import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from tova.api.models.data_schemas import (Dataset, Draft, DraftCreatedResponse,
                                          DraftType, StorageType)
from tova.core import drafts as drafts
from tova.utils.common import get_unique_id

# Import constants from drafts module
DRAFTS_SAVE = Path(os.getenv("DRAFTS_SAVE", "/data/drafts"))
METADATA_FILENAME = "metadata.json"
DATA_FILENAME = "data.json"


def list_datasets() -> List[Dataset]:
    """
    List all available datasets, including both the "temporary" and the ones indexed in the database, if any.
    """
    # list drafts and transform to Dataset objects
    dtset_drafts = drafts.list_drafts(type=DraftType.dataset)
    datasets_lst = [drafts.draft_to_dataset(draft) for draft in dtset_drafts]
    # do not list the documents inside each dataset for listing purposes
    #  remove the documents field
    for dataset in datasets_lst:
        dataset.documents = None

    # TODO: Implement actual database query and merge with drafts
    return datasets_lst


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
        return drafts.save_draft(draft)
    except Exception as e:
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
    # list drafts datasets
    dataset_drafts = drafts.list_drafts(type=DraftType.dataset)
    if dataset_id in [d.id for d in dataset_drafts]:
        #  convert draft to dataset
        dataset = drafts.draft_to_dataset(
            drafts.get_draft(dataset_id, DraftType.dataset))
    else:  # search in database
        # TODO: Implement actual database deletion
        pass

    path_dataset = DRAFTS_SAVE / dataset.id

    if path_dataset.exists() and path_dataset.is_dir():
        shutil.rmtree(path_dataset)
        return True
    return False
