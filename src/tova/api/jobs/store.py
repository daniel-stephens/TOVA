import time
import uuid
import asyncio
from typing import Dict, Iterable, Optional, Any

from .domain import Job, JobStatus, JobType

class JobStore:
    async def create(
        self,
        type: JobType,
        owner_id: Optional[str] = None,
        model_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Job: ...
    
    async def get(self, job_id: str) -> Job: ...
    
    async def list(
        self,
        owner_id: Optional[str] = None,
        type: Optional[JobType] = None
    ) -> Iterable[Job]: ...
    
    async def update(self, job_id: str, **fields) -> Job: ...


class InMemoryJobStore(JobStore):
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        type: JobType,
        owner_id: Optional[str] = None,
        model_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Job:
        job = Job(
            id=str(uuid.uuid4()),
            type=type,
            owner_id=owner_id,
            model_id=model_id,
            status=JobStatus.queued,
            progress=0.0,
            message="Job created",
            result=metadata or {}
        )
        async with self._lock:
            self._jobs[job.id] = job
        return job

    async def get(self, job_id: str) -> Job:
        async with self._lock:
            return self._jobs[job_id]

    async def list(
        self,
        owner_id: Optional[str] = None,
        type: Optional[JobType] = None
    ):
        async with self._lock:
            return [
                j for j in self._jobs.values()
                if (owner_id is None or j.owner_id == owner_id)
                and (type is None or j.type == type)
            ]

    async def update(self, job_id: str, **fields) -> Job:
        async with self._lock:
            j = self._jobs[job_id]
            for k, v in fields.items():
                setattr(j, k, v)
            j.updated_at = time.time()
            return j


job_store = InMemoryJobStore()