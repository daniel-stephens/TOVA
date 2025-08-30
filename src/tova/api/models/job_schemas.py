from pydantic import BaseModel
from typing import Optional, Dict, Any
from tova.api.jobs.domain import JobStatus

class JobDTO(BaseModel):
    id: str
    type: str
    status: JobStatus
    progress: float
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float
    updated_at: float
