import json
import sqlite3

from fastmcp import FastMCP

from corpus_state.db import log_plot_revelation


def register(mcp: FastMCP, db: sqlite3.Connection) -> None:
    @mcp.tool()
    def log_plot_revelation_tool(
        corpus_id: str,
        plot_point_id: str,
        description: str,
        email_mid: int,
    ) -> str:
        """Mark a story beat as revealed in a specific email.

        Args:
            corpus_id: Corpus this plot point belongs to.
            plot_point_id: Unique identifier for the plot point (e.g. "pager_clone").
            description: Human-readable description of the story beat.
            email_mid: The mid of the email where this beat was first revealed.
        """
        try:
            log_plot_revelation(db, corpus_id, plot_point_id, description, email_mid)
            return json.dumps({"ok": True, "corpus_id": corpus_id, "plot_point_id": plot_point_id, "email_mid": email_mid})
        except sqlite3.Error as e:
            return json.dumps({"error": str(e)})
