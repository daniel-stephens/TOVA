from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time, uuid

class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"

class JobType(str, Enum):
    train_model = "train_model"
    inference = "inference"

@dataclass
class Job:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: JobType = JobType.train_model
    owner_id: Optional[str] = None
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    model_id: Optional[str] = None