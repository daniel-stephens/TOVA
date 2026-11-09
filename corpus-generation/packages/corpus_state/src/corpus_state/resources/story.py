import json
import sqlite3

from fastmcp import FastMCP

from corpus_state.db import get_story_progress


def register(mcp: FastMCP, db: sqlite3.Connection) -> None:
    @mcp.resource("story://{corpus_id}/current_progress")
    def story_progress(corpus_id: str) -> str:
        progress = get_story_progress(db, corpus_id)
        return json.dumps(progress.model_dump(), indent=2)
