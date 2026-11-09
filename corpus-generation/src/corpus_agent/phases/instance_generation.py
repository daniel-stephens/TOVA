from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Awaitable, Callable

from corpus_state.db import (
    bulk_save_emails,
    complete_instance,
    create_instance,
    export_instance_emails,
    get_instance,
    log_plot_revelation,
    mark_email_read,
)
from corpus_state.types import Character, Email

from corpus_agent.config import PlotPoint
from corpus_agent.phases.question_planning import QuestionPlan
from corpus_agent.phases.wave_planner import EmailPlan, WavePlan

logger = logging.getLogger(__name__)

PlanWaveFn = Callable[..., Awaitable[WavePlan]]
GenerateEmailFn = Callable[..., Awaitable[Email]]


def _get_revealed_plot_points(conn: sqlite3.Connection, corpus_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT id FROM plot_points WHERE corpus_id = ?", (corpus_id,)
    ).fetchall()
    return [r[0] for r in rows]


def _mark_instance_failed(
    conn: sqlite3.Connection, instance_id: int, wave_num: int, error: str
) -> None:
    inst = get_instance(conn, instance_id)
    if inst:
        meta = inst.metadata.copy()
        meta["failed_wave"] = wave_num
        meta["error"] = error
        conn.execute(
            "UPDATE instances SET metadata = ? WHERE instance_id = ?",
            (json.dumps(meta), instance_id),
        )
        conn.commit()


def _parent_mid_exists(conn: sqlite3.Connection, mid: int) -> bool:
    row = conn.execute("SELECT 1 FROM emails WHERE mid = ?", (mid,)).fetchone()
    return row is not None


async def run_instance(
    conn: sqlite3.Connection,
    instance_id: int,
    corpus_id: str,
    characters: list[Character],
    plot_outline: list[PlotPoint],
    waves_per_instance: int,
    emails_per_wave: int,
    emails_per_instance: int,
    blur_level: str,
    plan_wave_fn: PlanWaveFn,
    generate_email_fn: GenerateEmailFn,
    wave_pause_seconds: float = 1.0,
    question_plan: QuestionPlan | None = None,
    export_path: str | None = None,
) -> None:
    inst = get_instance(conn, instance_id)
    if inst is None:
        create_instance(
            conn, corpus_id, instance_id, metadata={"waves_target": waves_per_instance}
        )
    elif inst.status == "completed":
        logger.info("Instance %d already completed, skipping", instance_id)
        return

    char_map = {c.id: c for c in characters}
    if question_plan is not None:
        await _seed_question_evidence(
            conn=conn,
            instance_id=instance_id,
            corpus_id=corpus_id,
            characters=characters,
            plot_outline=plot_outline,
            question_plan=question_plan,
            emails_per_wave=emails_per_wave,
            generate_email_fn=generate_email_fn,
            blur_level=blur_level,
        )
    existing_emails = export_instance_emails(conn, instance_id)
    start_wave = len(existing_emails) // max(emails_per_wave, 1)
    revealed = _get_revealed_plot_points(conn, corpus_id)
    total_emails = len(existing_emails)

    for wave_num in range(start_wave + 1, waves_per_instance + 1):
        if total_emails >= emails_per_instance:
            logger.info(
                "Instance %d reached email cap (%d), stopping",
                instance_id,
                emails_per_instance,
            )
            break

        try:
            wave_plan = await plan_wave_fn(
                characters=characters,
                wave_number=wave_num,
                total_waves=waves_per_instance,
                emails_per_wave=emails_per_wave,
                plot_outline=plot_outline,
                revealed_plot_points=revealed,
                blur_level=blur_level,
            )
        except Exception as e:
            logger.error(
                "Instance %d wave %d: plan_wave failed (%s), skipping instance",
                instance_id, wave_num, e,
            )
            _mark_instance_failed(conn, instance_id, wave_num, str(e))
            return

        wave_emails: list[Email] = []
        for email_plan in wave_plan.emails:
            sender = char_map[email_plan.sender_id]
            recipients = [char_map[rid] for rid in email_plan.recipient_ids]

            plot_desc = None
            if email_plan.plot_point_to_reveal:
                point = next(
                    (
                        p
                        for p in plot_outline
                        if p.id == email_plan.plot_point_to_reveal
                    ),
                    None,
                )
                if point is not None:
                    plot_desc = point.description

            try:
                email = await generate_email_fn(
                    plan=email_plan,
                    sender=sender,
                    recipients=recipients,
                    thread_history=[],
                    plot_description=plot_desc,
                    blur_level=blur_level,
                )
            except Exception as e:
                logger.error(
                    "Instance %d wave %d: generate_email failed (%s), skipping instance",
                    instance_id, wave_num, e,
                )
                _mark_instance_failed(conn, instance_id, wave_num, str(e))
                return
            if email.parent_mid is not None and not _parent_mid_exists(
                conn, email.parent_mid
            ):
                logger.warning(
                    "Instance %d wave %d: dropping invalid parent_mid=%s",
                    instance_id,
                    wave_num,
                    email.parent_mid,
                )
                email.parent_mid = None
            wave_emails.append(email)

        result = bulk_save_emails(conn, instance_id, wave_emails)
        if "error" in result:
            logger.error(
                "Failed to save wave %d for instance %d: %s",
                wave_num,
                instance_id,
                result["error"],
            )
            _mark_instance_failed(conn, instance_id, wave_num, result["error"])
            return

        mids = result["mids"]
        for i, email_plan in enumerate(wave_plan.emails):
            for rid in email_plan.recipient_ids:
                mark_email_read(conn, mids[i], rid)
            if email_plan.plot_point_to_reveal:
                point = next(
                    (
                        p
                        for p in plot_outline
                        if p.id == email_plan.plot_point_to_reveal
                    ),
                    None,
                )
                if point is not None:
                    log_plot_revelation(
                        conn, corpus_id, point.id, point.description, mids[i]
                    )
                    revealed.append(point.id)

        total_emails += len(wave_emails)
        logger.info(
            "Instance %d: wave %d complete (%d emails total)",
            instance_id,
            wave_num,
            total_emails,
        )

        if wave_num < waves_per_instance and wave_pause_seconds > 0:
            await asyncio.sleep(wave_pause_seconds)

    complete_instance(conn, instance_id)
    if question_plan is not None and export_path:
        _write_instance_questions(conn, instance_id, question_plan, export_path)
    logger.info("Instance %d completed with %d emails", instance_id, total_emails)


def _sender_for_slot(characters: list[Character], writer_role: str, index: int) -> Character:
    role = writer_role.lower()
    for character in characters:
        haystack = f"{character.role} {character.synthetic_character}".lower()
        if role and (role in haystack or any(word in haystack for word in role.split() if len(word) > 4)):
            return character
    return characters[index % len(characters)]


def _update_instance_metadata(conn: sqlite3.Connection, instance_id: int, **fields: object) -> None:
    inst = get_instance(conn, instance_id)
    meta = dict(inst.metadata) if inst else {}
    meta.update(fields)
    conn.execute(
        "UPDATE instances SET metadata = ? WHERE instance_id = ?",
        (json.dumps(meta), instance_id),
    )
    conn.commit()


async def _seed_question_evidence(
    conn: sqlite3.Connection,
    instance_id: int,
    corpus_id: str,
    characters: list[Character],
    plot_outline: list[PlotPoint],
    question_plan: QuestionPlan,
    emails_per_wave: int,
    generate_email_fn: GenerateEmailFn,
    blur_level: str,
) -> None:
    """Write one document per evidence slot before any filler snippets."""
    inst = get_instance(conn, instance_id)
    bindings = dict((inst.metadata if inst else {}).get("evidence_bindings") or {})
    slots = [
        slot
        for question in question_plan.questions
        for slot in question.evidence_slots
        if slot.slot_id not in bindings
    ]
    if not slots:
        return
    if not characters:
        raise ValueError("Cannot seed question evidence without characters")

    topics = {point.id: point for point in plot_outline}
    chunk_size = max(emails_per_wave, 1)
    for start in range(0, len(slots), chunk_size):
        chunk = slots[start : start + chunk_size]
        emails: list[Email] = []
        plans: list[EmailPlan] = []
        for offset, slot in enumerate(chunk):
            sender = _sender_for_slot(characters, slot.writer_role, start + offset)
            recipient = characters[(characters.index(sender) + 1) % len(characters)]
            plan = EmailPlan(
                sender_id=sender.id,
                recipient_ids=[recipient.id],
                plot_point_to_reveal=slot.topic_id,
                is_noise=False,
                evidence_slot=slot.slot_id,
                intent=(
                    "This document is evidence for a study question that was written first. "
                    "State only the fact below. Do not state the connection, the answer, "
                    f"or any other topic. Fact: {slot.fact}"
                ),
            )
            email = await generate_email_fn(
                plan=plan,
                sender=sender,
                recipients=[recipient],
                thread_history=[],
                plot_description=slot.fact,
                blur_level=blur_level,
            )
            emails.append(email)
            plans.append(plan)
        result = bulk_save_emails(conn, instance_id, emails)
        if "error" in result:
            raise ValueError(result["error"])
        for plan, email, mid in zip(plans, emails, result["mids"]):
            bindings[plan.evidence_slot] = email.message_id
            point = topics.get(plan.plot_point_to_reveal or "")
            if point is not None:
                log_plot_revelation(conn, corpus_id, point.id, point.description, mid)
            for recipient_id in plan.recipient_ids:
                mark_email_read(conn, mid, recipient_id)
        _update_instance_metadata(conn, instance_id, evidence_bindings=bindings)
        logger.info(
            "Instance %d: seeded %d/%d question evidence documents",
            instance_id,
            len(bindings),
            sum(len(question.evidence_slots) for question in question_plan.questions),
        )
    _update_instance_metadata(conn, instance_id, questions_seeded=True, evidence_bindings=bindings)


def _write_instance_questions(
    conn: sqlite3.Connection,
    instance_id: int,
    question_plan: QuestionPlan,
    export_path: str,
) -> None:
    inst = get_instance(conn, instance_id)
    bindings = (inst.metadata if inst else {}).get("evidence_bindings") or {}
    payload = {
        "instance_id": instance_id,
        "corpus_id": question_plan.corpus_id,
        "questions": [],
    }
    for question in question_plan.questions:
        evidence = [
            bindings[slot.slot_id]
            for slot in question.evidence_slots
            if slot.slot_id in bindings
        ]
        payload["questions"].append(
            {
                **question.model_dump(),
                "evidence_emails": evidence,
            }
        )
    out = Path(export_path) / "instances"
    out.mkdir(parents=True, exist_ok=True)
    file_path = out / f"questions_{instance_id:04d}.json"
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
