from __future__ import annotations

from pathlib import Path


def read_story_files(story_path: str) -> str:
    story_dir = Path(story_path)
    parts = []
    for f in sorted(story_dir.glob("*.md")):
        parts.append(f.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(parts)
