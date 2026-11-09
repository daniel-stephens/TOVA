import json
import sqlite3

from fastmcp import FastMCP

from corpus_state.db import get_instance, list_instances


def register(mcp: FastMCP, db: sqlite3.Connection) -> None:
    @mcp.resource("instance://{instance_id}/status")
    def instance_status(instance_id: str) -> str:
        try:
            iid = int(instance_id)
        except ValueError:
            raise ValueError(f"instance_id must be an integer, got: {instance_id!r}")
        inst = get_instance(db, iid)
        if inst is None:
            raise ValueError(f"Instance not found: {iid}")
        return json.dumps(inst.model_dump(), indent=2)

    @mcp.resource("instance://{corpus_id}/summary")
    def instance_summary(corpus_id: str) -> str:
        instances = list_instances(db, corpus_id=corpus_id)
        summary = {
            "corpus_id": corpus_id,
            "total": len(instances),
            "completed": sum(1 for i in instances if i.status == "completed"),
            "in_progress": sum(1 for i in instances if i.status == "in_progress"),
            "pending": sum(1 for i in instances if i.status == "pending"),
            "instances": [i.model_dump() for i in instances],
        }
        return json.dumps(summary, indent=2)
