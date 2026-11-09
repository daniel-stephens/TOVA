import json
import sqlite3

from fastmcp import FastMCP

from corpus_state.db import get_relationship


def register(mcp: FastMCP, db: sqlite3.Connection) -> None:
    @mcp.resource("relationship://{corpus_id}/{char_a}/{char_b}")
    def relationship_metadata(corpus_id: str, char_a: str, char_b: str) -> str:
        rel = get_relationship(db, corpus_id, char_a, char_b)
        if rel is None:
            raise ValueError(f"Relationship not found: {corpus_id}/{char_a}↔{char_b}")
        return json.dumps(rel.model_dump(), indent=2)
