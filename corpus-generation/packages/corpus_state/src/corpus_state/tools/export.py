import json
import sqlite3

from fastmcp import FastMCP

from corpus_state.db import export_instance_emails


def register(mcp: FastMCP, db: sqlite3.Connection) -> None:
    @mcp.tool()
    def export_corpus_tool(instance_id: int | None = None) -> str:
        """Export emails from the database as a list of email objects.

        Returns all emails for the given instance, or all emails across all
        instances if instance_id is omitted. Use this to produce the final
        corpus JSONL output from the authoritative database state.

        Args:
            instance_id: Export only this instance's emails. Omit for all.
        """
        try:
            emails = export_instance_emails(db, instance_id)
            return json.dumps({
                "count": len(emails),
                "instance_id": instance_id,
                "emails": [e.model_dump() for e in emails],
            }, indent=2)
        except sqlite3.Error as e:
            return json.dumps({"error": str(e)})
