from __future__ import annotations

import logging
import sqlite3

from corpus_state.db import get_story_mapping, save_story_mapping
from corpus_state.types import ResolvedStoryMapping

from corpus_agent.config import PlotPoint, StoryMapping
from corpus_agent.llm import LLMClient
from corpus_agent.utils import read_story_files

logger = logging.getLogger(__name__)

MAX_AUTO_RETRIES = 3


def build_outline_prompt(
    story_text: str,
    setting_name: str,
    setting_org: str,
    setting_domain: str,
    blur_level: str,
) -> str:
    return (
        f"## Source Story\n\n{story_text}\n\n"
        f"## Synthetic Setting\n\n"
        f"- Name: {setting_name}\n"
        f"- Organization: {setting_org}\n"
        f"- Domain: {setting_domain}\n\n"
        f"## Blur Level: {blur_level}\n\n"
        "Analyze the source story and generate a plot outline adapted to the "
        "synthetic setting. Apply the blur level to determine how closely the "
        "outline mirrors the source.\n\n"
        "Return a JSON object with a single key:\n"
        '- "plot_points": array of objects, each with "id" (snake_case) and '
        '"description" (1-2 sentences in the synthetic setting context)\n\n'
        "Produce between 5 and 12 plot points, ordered chronologically.\n"
    )


def _validate_auto_response(result: dict) -> list[dict] | None:
    points = result.get("plot_points")
    if not isinstance(points, list) or len(points) == 0:
        return None
    for p in points:
        if not isinstance(p, dict) or "id" not in p or "description" not in p:
            return None
        if not isinstance(p["id"], str) or not isinstance(p["description"], str):
            return None
    return points


async def resolve_story_mapping(
    conn: sqlite3.Connection,
    corpus_id: str,
    story_mapping: StoryMapping,
    source_story_path: str,
    blur_level: str,
    llm: LLMClient | None,
    model: str | None,
    skill_content: str | None,
) -> list[PlotPoint]:
    existing = get_story_mapping(conn, corpus_id)
    if existing is not None:
        logger.info(
            "Story mapping already resolved for corpus %s (mode=%s), reusing. "
            "To regenerate, use a new corpus_id or delete the story_mappings row.",
            corpus_id,
            existing.mode,
        )
        return [PlotPoint(**p) for p in existing.plot_points]

    setting = story_mapping.synthetic_setting
    outline_cfg = story_mapping.plot_outline

    if outline_cfg.mode == "manual":
        plot_points = outline_cfg.points
        resolved = ResolvedStoryMapping(
            corpus_id=corpus_id,
            setting_name=setting.name,
            setting_organization=setting.organization,
            setting_domain=setting.domain,
            blur_level=blur_level,
            mode="manual",
            plot_points=[p.model_dump() for p in plot_points],
        )
        save_story_mapping(conn, resolved)
        logger.info(
            "Story mapping resolved (manual): %d plot points persisted",
            len(plot_points),
        )
        return plot_points

    if llm is None or model is None or skill_content is None:
        raise ValueError(
            "llm, model, and skill_content are required for auto outline generation"
        )

    story_text = read_story_files(source_story_path)
    user_prompt = build_outline_prompt(
        story_text=story_text,
        setting_name=setting.name,
        setting_org=setting.organization,
        setting_domain=setting.domain,
        blur_level=blur_level,
    )

    points_data = None
    for attempt in range(MAX_AUTO_RETRIES):
        result = await llm.complete_json(
            model=model,
            system=skill_content,
            user=user_prompt,
            temperature=0.7,
        )
        points_data = _validate_auto_response(result)
        if points_data is not None:
            break
        logger.warning(
            "Auto outline response malformed on attempt %d/%d; keys=%s",
            attempt + 1,
            MAX_AUTO_RETRIES,
            sorted(result.keys()) if isinstance(result, dict) else type(result).__name__,
        )

    if points_data is None:
        raise ValueError(
            f"Failed to generate valid plot outline after {MAX_AUTO_RETRIES} attempts"
        )

    plot_points = [PlotPoint(**p) for p in points_data]
    resolved = ResolvedStoryMapping(
        corpus_id=corpus_id,
        setting_name=setting.name,
        setting_organization=setting.organization,
        setting_domain=setting.domain,
        blur_level=blur_level,
        mode="auto",
        plot_points=points_data,
    )
    save_story_mapping(conn, resolved)
    logger.info(
        "Story mapping resolved (auto): %d plot points generated and persisted",
        len(plot_points),
    )
    return plot_points
