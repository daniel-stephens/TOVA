import json
import sqlite3

from fastmcp import FastMCP

from corpus_state.db import get_character


def register(mcp: FastMCP, db: sqlite3.Connection) -> None:
    @mcp.resource("character://{corpus_id}/{id}/profile")
    def character_profile(corpus_id: str, id: str) -> str:
        char = get_character(db, corpus_id, id)
        if char is None:
            raise ValueError(f"Character not found: {corpus_id}/{id}")
        return json.dumps(char.model_dump(), indent=2)
