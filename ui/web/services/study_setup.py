"""Normalize local document exports without dropping records or source IDs."""
from __future__ import annotations

import json
from pathlib import Path


def load_documents(path: Path, limit: int | None = None) -> tuple[list[dict], dict]:
    raw = path.read_text(encoding="utf-8-sig")
    decoder = json.JSONDecoder()
    values, position = [], 0
    # Some email exports put multiple JSON objects on one line. Decode the
    # stream rather than assuming that every physical line is one document.
    while position < len(raw):
        while position < len(raw) and raw[position] in " \r\n\t,":
            position += 1
        if position == len(raw):
            break
        value, position = decoder.raw_decode(raw, position)
        values.append(value)
    if len(values) == 1 and isinstance(values[0], list):
        rows = values[0]
    elif len(values) == 1 and isinstance(values[0], dict) and "documents" in values[0]:
        rows = values[0]["documents"]
    else:
        rows = values
    if not isinstance(rows, list) or not rows:
        raise ValueError("Source contains no documents.")
    total = len(rows)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive.")
        rows = rows[:limit]
    documents, source_ids = [], []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index} must be a JSON object.")
        original_id = str(next((row[key] for key in ("original_id", "id", "mid")
                                if row.get(key) is not None and row.get(key) != ""), index))
        source_ids.append(original_id)
        body = str(row.get("text") or row.get("body") or "").replace("\\n", "\n").strip()
        if not body:
            raise ValueError(f"Record {index} has no text or body.")
        subject = str(row.get("subject") or "").strip()
        text = f"{subject}\n\n{body}" if subject and "text" not in row else body
        headers = [f"Email-ID: {original_id}"]
        for key, label in (("sender", "From"), ("recipient", "To"), ("date", "Date"), ("subject", "Subject")):
            if row.get(key):
                headers.append(f"{label}: {row[key]}")
        raw_text = row.get("raw_text") or ("\n".join(headers) + "\n\n" + body if "body" in row else text)
        documents.append({
            "id": f"source_{index:06d}", "original_id": original_id,
            "text": text, "raw_text": str(raw_text), "sourcefile": path.name,
        })
    return documents, {
        "source_records": total, "imported_records": len(documents),
        "repeated_source_ids": len(source_ids) - len(set(source_ids)),
        "policy": "Preserve every selected record; allocate unique document IDs; retain original IDs.",
    }
