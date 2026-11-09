"""Write the study questions before any corpus documents exist."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from corpus_state.types import Character

from corpus_agent.config import PlotPoint
from corpus_agent.phases.question_extraction import QUESTION_TYPES

logger = logging.getLogger(__name__)

CONNECTION_TYPES = (
    "document_topic_connection",
    "topic_topic_connection",
    "document_document_connection",
)


class EvidenceSlot(BaseModel):
    slot_id: str
    topic_id: str
    writer_role: str
    fact: str = Field(min_length=1)


class PlannedQuestion(BaseModel):
    question: str
    question_type: str
    difficulty: int = Field(ge=1, le=5)
    required_plot_points: list[str] = Field(min_length=1)
    answer_summary: str
    evidence_slots: list[EvidenceSlot] = Field(min_length=1)


class QuestionPlan(BaseModel):
    corpus_id: str
    questions: list[PlannedQuestion]


def build_question_plan_prompt(
    characters: list[Character],
    plot_outline: list[PlotPoint],
    content_style: str = "",
) -> str:
    char_block = "\n".join(
        f"- {c.synthetic_character}: {c.role}, {c.organization}" for c in characters
    )
    topic_block = "\n".join(f"- {p.id}: {p.description}" for p in plot_outline)
    type_block = "\n".join(f"- {name}: {desc}" for name, desc in QUESTION_TYPES.items())
    style = f"\n\n## Document style\n\n{content_style.strip()}" if content_style.strip() else ""
    return (
        "## Characters\n\n"
        f"{char_block}\n\n"
        "## Topics\n\n"
        "These plot points are the only topics. Use their ids exactly.\n\n"
        f"{topic_block}\n\n"
        "## Question types\n\n"
        f"{type_block}"
        f"{style}\n\n"
        "## Instructions\n\n"
        "Write the questions a user will be asked after reading the corpus. "
        "Do this before any document is written. Each question must be answerable "
        "only by connecting at least two later documents, and the answer must follow "
        "from the topics above.\n"
        "Include every question type, and at least two questions for each of "
        "document_topic_connection, topic_topic_connection, and document_document_connection.\n"
        "Every question must have 2 or 3 evidence_slots. One slot is invalid. "
        "Each slot states one fact and one topic. No single fact may contain the full answer.\n"
        "Return a JSON object with key \"questions\". Each question has:\n"
        "- question (str)\n"
        "- question_type (str)\n"
        "- difficulty (int, 1-5)\n"
        "- required_plot_points (list of topic ids)\n"
        "- answer_summary (str)\n"
        "- evidence_slots (list of objects with slot_id, topic_id, writer_role, fact)\n"
        "slot_id values must be unique. topic_id must be one of required_plot_points.\n"
        "Return only the JSON object.\n"
    )


def validate_question_plan(
    plan: QuestionPlan,
    topic_ids: set[str],
) -> list[str]:
    """Return a list of problems. An empty list means the plan is usable."""
    problems: list[str] = []
    present = {q.question_type for q in plan.questions}
    for required in CONNECTION_TYPES:
        if required not in present:
            problems.append(f"missing question type {required}")
    seen_slots: set[str] = set()
    for i, question in enumerate(plan.questions):
        if question.question_type not in QUESTION_TYPES:
            problems.append(f"question {i} has unknown type {question.question_type}")
        unknown = [topic for topic in question.required_plot_points if topic not in topic_ids]
        if unknown:
            problems.append(f"question {i} uses unknown topics {unknown}")
        if len(question.evidence_slots) < 2:
            problems.append(
                f"question {i} needs at least 2 evidence_slots, found {len(question.evidence_slots)}"
            )
        for slot in question.evidence_slots:
            if slot.slot_id in seen_slots:
                problems.append(f"duplicate slot_id {slot.slot_id}")
            seen_slots.add(slot.slot_id)
            if slot.topic_id not in question.required_plot_points:
                problems.append(
                    f"slot {slot.slot_id} topic {slot.topic_id} is not on its question"
                )
    return problems


def normalize_question_topics(raw_questions: list[dict], topic_ids: set[str]) -> list[dict]:
    """Keep a slot's topic on the question when the model forgot to list it."""
    for question in raw_questions:
        topics = [
            topic
            for topic in (question.get("required_plot_points") or [])
            if topic in topic_ids
        ]
        for slot in question.get("evidence_slots") or []:
            topic_id = slot.get("topic_id")
            if topic_id in topic_ids and topic_id not in topics:
                topics.append(topic_id)
        question["required_plot_points"] = topics
    return raw_questions


def assign_slot_ids(raw_questions: list[dict]) -> list[dict]:
    """Fill any missing slot ids so the model can omit them."""
    used: set[str] = set()
    for q_index, question in enumerate(raw_questions):
        slots = question.get("evidence_slots") or []
        for s_index, slot in enumerate(slots):
            slot_id = str(slot.get("slot_id") or "").strip()
            if not slot_id or slot_id in used:
                slot_id = f"q{q_index + 1}-d{s_index + 1}"
            while slot_id in used:
                slot_id = f"{slot_id}b"
            slot["slot_id"] = slot_id
            used.add(slot_id)
    return raw_questions


def save_question_plan(plan: QuestionPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")


def load_question_plan(path: Path) -> QuestionPlan | None:
    if not path.exists():
        return None
    return QuestionPlan.model_validate(json.loads(path.read_text(encoding="utf-8")))


def format_plan_for_generation(plan: QuestionPlan) -> str:
    lines = [
        "The study questions are already fixed. Do not invent new ones.",
        "No document may contain a complete answer. Each document may carry only one evidence fact.",
    ]
    for question in plan.questions:
        topics = ", ".join(question.required_plot_points)
        lines.append(
            f"- ({question.question_type}) {question.question} [topics: {topics}]"
        )
    return "\n".join(lines)


async def plan_study_questions(
    llm,
    model: str,
    corpus_id: str,
    characters: list[Character],
    plot_outline: list[PlotPoint],
    content_style: str = "",
) -> QuestionPlan:
    topic_ids = {point.id for point in plot_outline}
    prompt = build_question_plan_prompt(characters, plot_outline, content_style)
    last_problems: list[str] = []
    last_plan: QuestionPlan | None = None
    for attempt in range(4):
        result = await llm.complete_json(
            model=model,
            system=(
                "You write study questions before a corpus exists. "
                "Questions ask how documents and topics connect. Return only JSON."
            ),
            user=prompt if attempt == 0 else (
                prompt
                + "\n\nThe previous plan was rejected:\n"
                + "\n".join(f"- {problem}" for problem in last_problems)
                + "\nReturn a corrected JSON object."
            ),
            temperature=0.4,
            max_tokens=8000,
        )
        raw_questions = result.get("questions") if isinstance(result, dict) else None
        if not isinstance(raw_questions, list):
            last_problems = ["response did not contain a questions list"]
            continue
        try:
            plan = QuestionPlan.model_validate(
                {
                    "corpus_id": corpus_id,
                    "questions": normalize_question_topics(
                        assign_slot_ids(raw_questions),
                        topic_ids,
                    ),
                }
            )
        except Exception as exc:
            last_problems = [str(exc)]
            continue
        last_plan = plan
        last_problems = validate_question_plan(plan, topic_ids)
        if not last_problems:
            logger.info("Planned %d study questions", len(plan.questions))
            return plan
        logger.warning("Question plan rejected on attempt %d: %s", attempt + 1, last_problems)
    if last_plan is not None:
        kept = [
            question
            for question in last_plan.questions
            if len(question.evidence_slots) >= 2
            and question.question_type in QUESTION_TYPES
            and all(topic in topic_ids for topic in question.required_plot_points)
            and all(slot.topic_id in question.required_plot_points for slot in question.evidence_slots)
        ]
        last_plan.questions = kept
        last_problems = validate_question_plan(last_plan, topic_ids)
        if not last_problems:
            logger.info("Planned %d study questions after dropping short ones", len(kept))
            return last_plan
    raise ValueError("Failed to plan study questions: " + "; ".join(last_problems))
