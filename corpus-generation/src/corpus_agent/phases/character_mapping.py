from __future__ import annotations

import logging
import json
import sqlite3

from corpus_state.db import upsert_character, upsert_relationship
from corpus_state.types import Character, Relationship

from corpus_agent.config import PlotPoint, SyntheticSetting
from corpus_agent.llm import LLMClient
from corpus_agent.utils import read_story_files

logger = logging.getLogger(__name__)


def _normalize_blur_rules(value: object) -> dict[str, str]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"notes": value}
    return {}


def _characters_exist(conn: sqlite3.Connection, corpus_id: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM characters WHERE corpus_id = ?", (corpus_id,)
    ).fetchone()
    return row[0] > 0


def build_mapping_prompt(
    story_text: str,
    setting_name: str,
    setting_org: str,
    setting_domain: str,
    blur_level: str,
    plot_outline: list[PlotPoint],
) -> str:
    outline_text = "\n".join(f"- {p.id}: {p.description}" for p in plot_outline)
    return (
        f"## Source Story\n\n{story_text}\n\n"
        f"## Synthetic Setting\n\n"
        f"- Name: {setting_name}\n"
        f"- Organization: {setting_org}\n"
        f"- Domain: {setting_domain}\n\n"
        f"## Blur Level: {blur_level}\n\n"
        f"## Plot Outline\n\n{outline_text}\n\n"
        "Map each character from the source story to the synthetic setting. "
        "Return a JSON object with two keys:\n"
        '- "characters": array of character objects, each with: '
        "id, wire_character, synthetic_character, email_address, role, "
        "organization, personality, voice_anchor, blur_rules\n"
        '- "relationships": array of relationship objects, each with: '
        "char_a, char_b, strength (1-10), status\n"
    )


async def run_character_mapping(
    conn: sqlite3.Connection,
    llm: LLMClient,
    model: str,
    corpus_id: str,
    source_story_path: str,
    setting: SyntheticSetting,
    blur_level: str,
    plot_outline: list[PlotPoint],
    skill_content: str,
) -> list[Character]:
    if _characters_exist(conn, corpus_id):
        logger.info(
            "Characters already exist for corpus %s, skipping mapping", corpus_id
        )
        rows = conn.execute(
            "SELECT * FROM characters WHERE corpus_id = ?", (corpus_id,)
        ).fetchall()
        return [Character(**dict(r)) for r in rows]

    story_text = read_story_files(source_story_path)
    user_prompt = build_mapping_prompt(
        story_text,
        setting.name,
        setting.organization,
        setting.domain,
        blur_level,
        plot_outline,
    )

    result: dict = {}
    characters_data: list[dict] | None = None
    relationships_data: list[dict] = []
    for attempt in range(3):
        result = await llm.complete_json(
            model=model,
            system=skill_content,
            user=user_prompt,
            temperature=0.7,
        )
        maybe_chars = result.get("characters")
        if isinstance(maybe_chars, list):
            characters_data = maybe_chars
            maybe_rels = result.get("relationships", [])
            if isinstance(maybe_rels, list):
                relationships_data = maybe_rels
            break
        logger.warning(
            "Character mapping response missing list key 'characters' on attempt %d/3; keys=%s",
            attempt + 1,
            sorted(result.keys()),
        )
    if characters_data is None:
        raise ValueError(
            "Character mapping response missing required list key 'characters'"
        )

    characters = []
    for c in characters_data:
        c = dict(c)
        c["blur_rules"] = _normalize_blur_rules(c.get("blur_rules"))
        char = Character(corpus_id=corpus_id, **c)
        upsert_character(conn, char)
        characters.append(char)
        logger.info(
            "Mapped character: %s -> %s", char.wire_character, char.synthetic_character
        )

    for r in relationships_data:
        rel = Relationship(corpus_id=corpus_id, **r)
        upsert_relationship(conn, rel)

    return characters
