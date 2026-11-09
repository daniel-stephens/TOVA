from __future__ import annotations

import logging
import uuid

from corpus_state.types import Character, Email

from corpus_agent.llm import LLMClient
from corpus_agent.phases.wave_planner import EmailPlan

logger = logging.getLogger(__name__)

MAX_EMAIL_RETRIES = 3
_REQUIRED_EMAIL_FIELDS = {"sender", "recipient", "date", "message_id", "subject", "body"}


def build_generator_prompt(
    plan: EmailPlan,
    sender: Character,
    recipients: list[Character],
    thread_history: list[Email],
    plot_description: str | None,
    blur_level: str,
    content_style: str = "",
    question_brief: str = "",
) -> str:
    recip_block = "\n".join(
        f"- {r.synthetic_character} ({r.role}, {r.organization})" for r in recipients
    )
    thread_block = ""
    if thread_history:
        thread_block = "## Thread History (oldest first)\n\n"
        for e in thread_history:
            thread_block += (
                f"From: {e.sender}\nTo: {e.recipient}\nSubject: {e.subject}\n\n"
                f"{e.body}\n\n---\n\n"
            )

    plot_block = ""
    if plot_description:
        plot_block = (
            "## Plot Point to Weave In\n\n"
            f"{plot_description}\n\n"
            "Weave this information naturally into the email. Do NOT state it explicitly - "
            "it should be inferable but require the reader to connect dots across emails.\n\n"
        )

    return (
        "## Sender Profile\n\n"
        f"Name: {sender.synthetic_character}\n"
        f"Role: {sender.role}, {sender.organization}\n"
        f"Personality: {sender.personality}\n"
        f"Voice: {sender.voice_anchor}\n"
        f"Emotional state: {sender.emotional_state}\n\n"
        f"## Recipients\n\n{recip_block}\n\n"
        f"{thread_block}"
        f"{plot_block}"
        f"## Email Intent\n\n{plan.intent}\n\n"
        f"Blur level: {blur_level}\n"
        f"Is noise email: {plan.is_noise}\n\n"
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
        + "Generate a single email as JSON with keys: "
        "sender, recipient, date, message_id, subject, body, folder.\n"
        "- sender and recipient must be email addresses\n"
        "- date must be ISO 8601\n"
        "- message_id must be a unique angle-bracketed ID\n"
        "- folder should be \\Sent for the sender's copy\n"
        "- Do NOT include mid or parent_mid in the output\n"
        "\nReturn only the JSON object, no other text, markdown fences, or explanation.\n"
    )


async def generate_email(
    llm: LLMClient,
    model: str,
    plan: EmailPlan,
    sender: Character,
    recipients: list[Character],
    thread_history: list[Email],
    plot_description: str | None,
    blur_level: str,
    skill_content: str,
    content_style: str = "",
    question_brief: str = "",
) -> Email:
    user_prompt = build_generator_prompt(
        plan,
        sender,
        recipients,
        thread_history,
        plot_description,
        blur_level,
        content_style,
        question_brief,
    )
    result = None
    for attempt in range(MAX_EMAIL_RETRIES):
        candidate = await llm.complete_json(
            model=model,
            system=skill_content,
            user=user_prompt,
        )
        if isinstance(candidate, dict) and _REQUIRED_EMAIL_FIELDS.issubset(candidate):
            result = candidate
            break
        logger.warning(
            "Email response malformed on attempt %d/%d; keys=%s",
            attempt + 1,
            MAX_EMAIL_RETRIES,
            sorted(candidate.keys()) if isinstance(candidate, dict) else type(candidate).__name__,
        )

    if result is None:
        raise ValueError(
            f"Failed to generate valid email after {MAX_EMAIL_RETRIES} attempts"
        )
    # Guarantee message_id uniqueness regardless of LLM output by appending a UUID.
    # LLMs produce timestamp-based IDs that collide across concurrent instances.
    raw_mid = result["message_id"].strip().strip("<>")
    unique_mid = f"<{raw_mid}.{uuid.uuid4().hex[:8]}>"
    return Email(
        sender=result["sender"],
        recipient=result["recipient"],
        date=result["date"],
        message_id=unique_mid,
        subject=result["subject"],
        body=result["body"],
        folder=result.get("folder", "\\Sent"),
        parent_mid=plan.parent_mid,
    )

