import os
import sqlite3

from fastmcp import FastMCP

from corpus_state.db import create_database
from corpus_state.resources import character as char_res
from corpus_state.resources import thread as thread_res
from corpus_state.resources import story as story_res
from corpus_state.resources import relationship as rel_res
from corpus_state.resources import instance as instance_res
from corpus_state.tools import character as char_tools
from corpus_state.tools import email as email_tools
from corpus_state.tools import story as story_tools
from corpus_state.tools import instance as instance_tools
from corpus_state.tools import export as export_tools

mcp = FastMCP("corpus-state")


def main() -> None:
    db_path = os.environ.get("CORPUS_DB_PATH")
    db: sqlite3.Connection = create_database(db_path)
    try:
        char_res.register(mcp, db)
        thread_res.register(mcp, db)
        story_res.register(mcp, db)
        rel_res.register(mcp, db)
        instance_res.register(mcp, db)
        char_tools.register(mcp, db)
        email_tools.register(mcp, db)
        story_tools.register(mcp, db)
        instance_tools.register(mcp, db)
        export_tools.register(mcp, db)

        mcp.run()
    finally:
        db.close()


if __name__ == "__main__":
    main()
