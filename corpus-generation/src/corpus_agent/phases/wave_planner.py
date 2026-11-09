from __future__ import annotations

import logging

from pydantic import BaseModel

from corpus_state.types import Character

from corpus_agent.config import PlotPoint
from corpus_agent.llm import LLMClient

logger = logging.getLogger(__name__)

MAX_WAVE_RETRIES = 3


class EmailPlan(BaseModel):
    sender_id: str
    recipient_ids: list[str]
    is_reply: bool = False
    parent_mid: int | None = None
    plot_point_to_reveal: str | None = None
    is_noise: bool = False
    intent: str
    evidence_slot: str | None = None


class WavePlan(BaseModel):
    emails: list[EmailPlan]
    wave_rationale: str


def build_planner_prompt(
    characters: list[Character],
    wave_number: int,
    total_waves: int,
    emails_per_wave: int,
    plot_outline: list[PlotPoint],
    revealed_plot_points: list[str],
    blur_level: str,
    content_style: str = "",
    question_brief: str = "",
) -> str:
    char_block = "\n".join(
        f"- {c.id}: {c.synthetic_character} ({c.role}, {c.organization}) - {c.personality}"
        for c in characters
    )
    outline_block = "\n".join(
        f"- {p.id}: {p.description}"
        + (" [REVEALED]" if p.id in revealed_plot_points else "")
        for p in plot_outline
    )
    unrevealed = [p for p in plot_outline if p.id not in revealed_plot_points]
    return (
        f"## Wave {wave_number} of {total_waves}\n\n"
        f"Target emails this wave: {emails_per_wave} (you may vary between 2-6)\n\n"
        f"## Characters\n\n{char_block}\n\n"
        f"## Plot Outline (with revelation status)\n\n{outline_block}\n\n"
        f"Unrevealed plot points remaining: {len(unrevealed)}\n"
        f"Blur level: {blur_level}\n\n"
        + (
            f"## Document Style\n\n{content_style.strip()}\n\n"
            if content_style.strip()
            else ""
        )
        + (
            f"## Fixed Study Questions\n\n{question_brief.strip()}\n\n"
            if question_brief.strip()
            else ""
        )
        + "Produce a wave plan as JSON with keys:\n"
        '- "emails": array of objects with sender_id, recipient_ids, is_reply, parent_mid, '
        "plot_point_to_reveal (null if none), is_noise, intent\n"
        '- "wave_rationale": why this wave shape was chosen\n\n'
        "Rules:\n"
        "- No single email should contain a complete answer to any investigative question\n"
        "- Distribute plot points gradually; at most 1-2 revelations per wave\n"
        "- Include noise emails proportional to blur level\n"
        "- Replies must reference an existing parent_mid from a prior wave\n"
        "\nReturn only the JSON object, no other text, markdown fences, or explanation.\n"
    )


def _validate_wave_response(result: dict) -> WavePlan | None:
    if not isinstance(result, dict):
        return None
    if "emails" not in result or "wave_rationale" not in result:
        return None
    try:
        return WavePlan(**result)
    except Exception:
        return None


async def plan_wave(
    llm: LLMClient,
    model: str,
    characters: list[Character],
    wave_number: int,
    total_waves: int,
    emails_per_wave: int,
    plot_outline: list[PlotPoint],
    revealed_plot_points: list[str],
    blur_level: str,
    skill_content: str,
    content_style: str = "",
    question_brief: str = "",
) -> WavePlan:
    user_prompt = build_planner_prompt(
        characters,
        wave_number,
        total_waves,
        emails_per_wave,
        plot_outline,
        revealed_plot_points,
        blur_level,
        content_style,
        question_brief,
    )
    plan = None
    for attempt in range(MAX_WAVE_RETRIES):
        result = await llm.complete_json(
            model=model,
            system=skill_content,
            user=user_prompt,
        )
        plan = _validate_wave_response(result)
        if plan is not None:
            break
        logger.warning(
            "Wave plan response malformed on attempt %d/%d; keys=%s",
            attempt + 1,
            MAX_WAVE_RETRIES,
            sorted(result.keys()) if isinstance(result, dict) else type(result).__name__,
        )

    if plan is None:
        raise ValueError(
            f"Failed to generate valid wave plan after {MAX_WAVE_RETRIES} attempts"
        )
    return plan
