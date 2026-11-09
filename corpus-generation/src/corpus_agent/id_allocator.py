from __future__ import annotations

import sqlite3


class InstanceIdAllocator:
    def __init__(self, conn: sqlite3.Connection, corpus_id: str, count: int):
        row = conn.execute(
            "SELECT COALESCE(MAX(instance_id), 0) FROM instances",
        ).fetchone()
        start = row[0] + 1
        self._ids = iter(range(start, start + count))

    def next_id(self) -> int:
        return next(self._ids)

