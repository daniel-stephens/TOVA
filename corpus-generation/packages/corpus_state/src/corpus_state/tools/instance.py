import json
import sqlite3

from fastmcp import FastMCP

from corpus_state.db import create_instance, get_instance, complete_instance, list_instances


def register(mcp: FastMCP, db: sqlite3.Connection) -> None:
    @mcp.tool()
    def create_instance_tool(instance_id: int, corpus_id: str = "default", metadata: dict | None = None) -> str:
        """Register a new generation instance for tracking.

        Args:
            instance_id: Unique integer ID for this instance (used in mid namespacing).
            corpus_id: Corpus this instance belongs to (e.g. "wire-corp-v1").
            metadata: Optional dict of instance metadata (e.g., story name, blur level).
        """
        try:
            inst = create_instance(db, corpus_id, instance_id, metadata)
            return json.dumps(inst.model_dump(), indent=2)
        except sqlite3.Error as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def complete_instance_tool(instance_id: int) -> str:
        """Mark an instance as completed and record its final email count.

        Args:
            instance_id: The instance ID to complete.
        """
        try:
            complete_instance(db, instance_id)
            inst = get_instance(db, instance_id)
            if inst is None:
                return json.dumps({"error": f"Instance not found: {instance_id}"})
            return json.dumps(inst.model_dump(), indent=2)
        except sqlite3.Error as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def list_instances_tool(status: str | None = None, corpus_id: str | None = None) -> str:
        """List all generation instances, optionally filtered by status and/or corpus.

        Args:
            status: Filter by status: 'pending', 'in_progress', or 'completed'. Omit for all.
            corpus_id: Filter by corpus ID. Omit for all corpora.
        """
        try:
            instances = list_instances(db, status, corpus_id)
            return json.dumps([i.model_dump() for i in instances], indent=2)
        except sqlite3.Error as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def reset_instance_tool(instance_id: int) -> str:
        """Delete all emails for an instance and reset it to in_progress.

        Use this to re-generate a failed or unsatisfactory instance.

        Args:
            instance_id: The instance ID to reset.
        """
        try:
            with db:
                cursor = db.execute(
                    "DELETE FROM emails WHERE instance_id = ?", (instance_id,)
                )
                deleted = cursor.rowcount
                db.execute(
                    "UPDATE instances SET status = 'in_progress', num_emails = 0, completed_at = NULL WHERE instance_id = ?",
                    (instance_id,),
                )
            return json.dumps({"ok": True, "instance_id": instance_id, "deleted_emails": deleted})
        except sqlite3.Error as e:
            return json.dumps({"error": str(e)})
