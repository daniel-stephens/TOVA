import json
import sqlite3

from fastmcp import FastMCP

from corpus_state.db import get_email_thread


def register(mcp: FastMCP, db: sqlite3.Connection) -> None:
    @mcp.resource("thread://{email_id}/history")
    def thread_history(email_id: str) -> str:
        try:
            mid = int(email_id)
        except ValueError:
            raise ValueError(f"email_id must be an integer, got: {email_id!r}")
        thread = get_email_thread(db, mid)
        if not thread:
            raise ValueError(f"No thread found for email mid: {mid}")
        return json.dumps([e.model_dump() for e in thread], indent=2)
