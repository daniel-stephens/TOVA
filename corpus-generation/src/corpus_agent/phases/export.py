from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from corpus_state.db import export_instance_emails, list_instances

logger = logging.getLogger(__name__)


def export_corpus(conn: sqlite3.Connection, export_path: str, corpus_id: str) -> Path:
    out = Path(export_path)
    out.mkdir(parents=True, exist_ok=True)

    instances = list_instances(conn, status="completed", corpus_id=corpus_id)
    all_emails = []

    instances_dir = out / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    for inst in instances:
        emails = export_instance_emails(conn, inst.instance_id)
        all_emails.extend(emails)

        instance_file = instances_dir / f"instance_{inst.instance_id:04d}.jsonl"
        with open(instance_file, "w", encoding="utf-8") as f:
            for e in emails:
                f.write(json.dumps(e.model_dump(exclude_none=True)) + "\n")
        logger.info(
            "Exported instance %d: %d emails -> %s",
            inst.instance_id,
            len(emails),
            instance_file,
        )

    corpus_file = out / "email_corpus.jsonl"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for e in all_emails:
            f.write(json.dumps(e.model_dump(exclude_none=True)) + "\n")
    logger.info(
        "Exported combined corpus: %d emails -> %s", len(all_emails), corpus_file
    )

    return out

