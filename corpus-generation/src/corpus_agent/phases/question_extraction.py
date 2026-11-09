from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from corpus_state.db import export_instance_emails, get_story_progress, list_instances

logger = logging.getLogger(__name__)

MAX_EXTRACTION_RETRIES = 3

QUESTION_TYPES: dict[str, str] = {
    "attribution": (
        "Who performed a specific action or made a specific decision? "
        "Requires identifying the actor behind an event by synthesizing clues "
        "distributed across multiple emails."
    ),
    "timeline": (
        "What is the chronological sequence of events? "
        "Requires ordering events that are revealed out of sequence across "
        "different emails and characters."
    ),
    "relationship_mapping": (
        "What is the relationship or connection between two or more entities? "
        "Requires piecing together interactions, alliances, or conflicts from "
        "separate email threads."
    ),
    "information_asymmetry": (
        "What does one character know that another does not? "
        "Requires comparing information available to different characters "
        "based on which emails they sent, received, or were excluded from."
    ),
    "motive_intent": (
        "Why did a character take a particular action? "
        "Requires inferring motivation from behavioral patterns, stated "
        "concerns, and contextual clues across multiple emails."
    ),
    "coordination_detection": (
        "Are two or more characters coordinating their actions? "
        "Requires detecting synchronized behavior, shared language, or "
        "complementary actions visible only when comparing separate email threads."
    ),
    "document_topic_connection": (
        "How does a document connect to a topic? Plot points are the topics and "
        "emails are the documents. The answer must name the topic and the documents "
        "that carry it. The link is not stated in any single document."
    ),
    "topic_topic_connection": (
        "How are two topics connected? Requires documents that establish each topic "
        "separately and at least one document that bridges them. Plot point ids are "
        "the topic ids."
    ),
    "document_document_connection": (
        "How are two documents connected? The connection is a shared topic, a reply "
        "thread, or a fact that appears only when both documents are read."
    ),
}

QuestionType = Literal[
    "attribution",
    "timeline",
    "relationship_mapping",
    "information_asymmetry",
    "motive_intent",
    "coordination_detection",
    "document_topic_connection",
    "topic_topic_connection",
    "document_document_connection",
]


class SensemakingQuestion(BaseModel):
    question: str
    question_type: QuestionType
    difficulty: int = Field(ge=1, le=5)
    required_plot_points: list[str]
    answer_summary: str
    evidence_emails: list[str] = Field(min_length=1)


class QuestionSet(BaseModel):
    instance_id: int
    corpus_id: str
    questions: list[SensemakingQuestion]

    def missing_types(self) -> list[str]:
        present = {q.question_type for q in self.questions}
        return [t for t in QUESTION_TYPES if t not in present]


def build_extraction_prompt(
    emails: list,
    characters: list,
    plot_outline: list,
    revelations: dict[str, int],
    content_style: str = "",
) -> str:
    """Build the LLM prompt for question extraction.
    
    Args:
        emails: List of Email objects
        characters: List of Character objects
        plot_outline: List of PlotPoint objects
        revelations: Dict mapping plot_point.id to email mid
    
    Returns:
        Formatted prompt string for the LLM
    """
    char_block = "\n".join(
        f"- {c.id}: {c.synthetic_character} ({c.role}, {c.organization}) <{c.email_address}>"
        for c in characters
    )

    outline_block = "\n".join(
        f"- {p.id}: {p.description}"
        + (f" [revealed in email mid={revelations[p.id]}]" if p.id in revelations else " [NOT revealed]")
        for p in plot_outline
    )

    email_block = "\n\n---\n\n".join(
        f"message_id: {e.message_id}\n"
        f"mid: {e.mid}\n"
        f"From: {e.sender}\n"
        f"To: {e.recipient}\n"
        f"Date: {e.date}\n"
        f"Subject: {e.subject}\n"
        f"Parent: {e.parent_mid or 'none'}\n\n"
        f"{e.body}"
        for e in emails
    )

    taxonomy_block = "\n".join(
        f"- {qtype}: {desc}" for qtype, desc in QUESTION_TYPES.items()
    )

    style_block = ""
    if content_style.strip():
        style_block = f"## Document Style\n\n{content_style.strip()}\n\n"

    return (
        "## Characters\n\n"
        f"{char_block}\n\n"
        "## Topics (plot points, with revelation status)\n\n"
        "Each plot point is a topic. Use its id when a question is about that topic.\n\n"
        f"{outline_block}\n\n"
        "## Documents\n\n"
        "Each email is one document. Cite documents by message_id.\n\n"
        f"{email_block}\n\n"
        f"{style_block}"
        "## Question Taxonomy\n\n"
        f"{taxonomy_block}\n\n"
        "## Instructions\n\n"
        "You have privileged access to the full generation context for this corpus. "
        "Extract sensemaking questions that:\n"
        "1. Provably require cross-document synthesis (answering requires reading 2+ documents)\n"
        "2. Come with answer keys citing specific evidence documents by message_id\n"
        "3. Are rated by difficulty (1-5) based on number of documents required and directness of evidence\n"
        "4. Cover every question type in the taxonomy above\n"
        "5. For document_topic_connection, topic_topic_connection, and "
        "document_document_connection, ask about the link itself: which documents "
        "carry a topic, how two topics depend on each other, or how two documents "
        "connect through a shared topic. Put the topic ids in required_plot_points.\n\n"
        "A question FAILS validation if:\n"
        "- It can be answered from a single email\n"
        "- The cited evidence_emails do not actually contain the relevant evidence\n"
        "- The required_plot_points are not actually revealed in the corpus\n\n"
        "Return a JSON object with a single key:\n"
        '- "questions": array of objects, each with:\n'
        "  - question (str): the sensemaking question\n"
        "  - question_type (str): one of the taxonomy types\n"
        "  - difficulty (int): 1-5\n"
        "  - required_plot_points (list[str]): plot point IDs needed to answer\n"
        "  - answer_summary (str): concise answer with reasoning\n"
        "  - evidence_emails (list[str]): message_ids of emails containing evidence\n\n"
        "Generate at least one question per taxonomy type if the corpus supports it. "
        "Aim for 8-15 questions total.\n"
    )


EXTRACTION_TEMPERATURE = 0.4
EXTRACTION_MAX_TOKENS = 8000


def validate_questions(
    questions: list[SensemakingQuestion],
    corpus_message_ids: set[str],
    revealed_plot_points: set[str],
) -> tuple[list[SensemakingQuestion], list[str]]:
    """Validate a list of SensemakingQuestions against corpus constraints.

    A question passes only if:
    - It cites at least 2 evidence emails (cross-document requirement).
    - Every cited evidence email exists in the corpus by message_id.
    - Every required plot point has been revealed in this instance.

    Returns:
        (passed, warnings) where passed is the filtered list and warnings
        contains a human-readable rejection reason for each rejected question.
    """
    passed: list[SensemakingQuestion] = []
    warnings: list[str] = []
    for i, q in enumerate(questions):
        failed = False
        if len(q.evidence_emails) < 2:
            warnings.append(
                f"Question {i} ({q.question_type}): REJECTED — answerable from single email "
                f"(only {len(q.evidence_emails)} evidence email cited)"
            )
            failed = True
        for mid in q.evidence_emails:
            if mid not in corpus_message_ids:
                warnings.append(
                    f"Question {i} ({q.question_type}): REJECTED — evidence email {mid} "
                    f"not found in corpus"
                )
                failed = True
        for pp in q.required_plot_points:
            if pp not in revealed_plot_points:
                warnings.append(
                    f"Question {i} ({q.question_type}): REJECTED — required plot point "
                    f"'{pp}' not revealed in this instance"
                )
                failed = True
        if not failed:
            passed.append(q)
    return passed, warnings


def _validate_extraction_response(result: dict) -> list[dict] | None:
    """Return the questions list from an LLM response dict, or None if malformed."""
    questions = result.get("questions")
    if not isinstance(questions, list) or len(questions) == 0:
        return None
    for q in questions:
        if not isinstance(q, dict):
            return None
        required = {
            "question", "question_type", "difficulty", "required_plot_points",
            "answer_summary", "evidence_emails",
        }
        if not required.issubset(q.keys()):
            return None
    return questions


def _build_coverage_retry_prompt(missing_types: list[str]) -> str:
    """Build a targeted follow-up prompt requesting questions for missing types."""
    types_block = "\n".join(
        f"- {t}: {QUESTION_TYPES[t]}" for t in missing_types
    )
    return (
        "The previous extraction was missing questions for these types:\n\n"
        f"{types_block}\n\n"
        "Generate one question per missing type, using the same corpus context "
        "provided earlier. Return a JSON object with a single key "
        '"questions" containing the new questions only.\n'
    )


async def extract_instance_questions(
    conn: sqlite3.Connection,
    llm,
    model: str,
    instance_id: int,
    corpus_id: str,
    characters: list,
    plot_outline: list,
    skill_content: str,
    content_style: str = "",
) -> QuestionSet:
    """Extract and validate sensemaking questions for a completed instance.

    Performs an initial LLM extraction pass, validates all questions, then fires
    a single coverage-gap retry if any question types are missing.

    Args:
        conn: Active SQLite connection (corpus_state database).
        llm: LLMClient (or compatible AsyncMock in tests).
        model: Model identifier string passed to llm.complete_json.
        instance_id: The instance whose emails to analyse.
        corpus_id: Corpus identifier for story-progress lookup.
        characters: List of Character objects for prompt context.
        plot_outline: List of PlotPoint objects for prompt context.
        skill_content: System-prompt text for the extractor LLM call.

    Returns:
        QuestionSet containing all validated questions.
    """
    emails = export_instance_emails(conn, instance_id)
    progress = get_story_progress(conn, corpus_id)
    revelations = {
        pp.id: pp.first_email_mid
        for pp in progress.plot_points
        if pp.first_email_mid is not None
    }

    user_prompt = build_extraction_prompt(
        emails=emails,
        characters=characters,
        plot_outline=plot_outline,
        revelations=revelations,
        content_style=content_style,
    )

    # Initial extraction with retry loop for malformed responses.
    questions_data = None
    for attempt in range(MAX_EXTRACTION_RETRIES):
        result = await llm.complete_json(
            model=model,
            system=skill_content,
            user=user_prompt,
            temperature=EXTRACTION_TEMPERATURE,
            max_tokens=EXTRACTION_MAX_TOKENS,
        )
        questions_data = _validate_extraction_response(result)
        if questions_data is not None:
            break
        logger.warning(
            "Question extraction response malformed on attempt %d/%d; keys=%s",
            attempt + 1,
            MAX_EXTRACTION_RETRIES,
            sorted(result.keys()) if isinstance(result, dict) else type(result).__name__,
        )

    if questions_data is None:
        raise ValueError(
            f"Failed to extract valid questions after {MAX_EXTRACTION_RETRIES} attempts"
        )

    # Parse and skip unknown / malformed question entries.
    parsed: list[SensemakingQuestion] = []
    for qd in questions_data:
        if qd.get("question_type") not in QUESTION_TYPES:
            logger.warning("Skipping question with unknown type: %s", qd.get("question_type"))
            continue
        try:
            parsed.append(SensemakingQuestion(**qd))
        except Exception as e:
            logger.warning("Skipping malformed question: %s", e)

    corpus_mids = {e.message_id for e in emails}
    revealed_points = set(revelations.keys())
    passed, validation_warnings = validate_questions(parsed, corpus_mids, revealed_points)
    for w in validation_warnings:
        logger.warning("Validation: %s", w)

    # Coverage-gap retry: fire once for missing types.
    temp_set = QuestionSet(instance_id=instance_id, corpus_id=corpus_id, questions=passed)
    missing = temp_set.missing_types()
    if missing:
        logger.info(
            "Instance %d: missing %d types, attempting coverage retry: %s",
            instance_id, len(missing), missing,
        )
        retry_prompt = _build_coverage_retry_prompt(missing)
        retry_result = await llm.complete_json(
            model=model,
            system=skill_content,
            user=user_prompt + "\n\n" + retry_prompt,
            temperature=EXTRACTION_TEMPERATURE,
            max_tokens=EXTRACTION_MAX_TOKENS,
        )
        retry_data = _validate_extraction_response(retry_result)
        missing_set = set(missing)
        if retry_data:
            for qd in retry_data:
                qtype = qd.get("question_type")
                if qtype not in missing_set:
                    continue
                try:
                    parsed_q = SensemakingQuestion(**qd)
                    retry_passed, retry_warns = validate_questions(
                        [parsed_q], corpus_mids, revealed_points
                    )
                    for w in retry_warns:
                        logger.warning("Retry validation: %s", w)
                    passed.extend(retry_passed)
                except Exception as e:
                    logger.warning("Skipping malformed retry question: %s", e)

    question_set = QuestionSet(
        instance_id=instance_id,
        corpus_id=corpus_id,
        questions=passed,
    )

    still_missing = question_set.missing_types()
    if still_missing:
        logger.warning(
            "Instance %d: still missing question types after retry: %s",
            instance_id, still_missing,
        )

    return question_set


def export_question_set(question_set: QuestionSet, export_path: str) -> Path:
    out = Path(export_path)
    instances_dir = out / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "instance_id": question_set.instance_id,
        "corpus_id": question_set.corpus_id,
        "questions": [q.model_dump() for q in question_set.questions],
        "coverage": {
            "total_questions": len(question_set.questions),
            "types_covered": list(
                {q.question_type for q in question_set.questions}
            ),
            "missing_types": question_set.missing_types(),
        },
    }

    file_path = instances_dir / f"questions_{question_set.instance_id:04d}.json"
    file_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(
        "Exported %d questions for instance %d -> %s",
        len(question_set.questions),
        question_set.instance_id,
        file_path,
    )
    return file_path


async def run_question_extraction(
    conn: sqlite3.Connection,
    llm,
    model: str,
    corpus_id: str,
    characters: list,
    plot_outline: list,
    skill_content: str,
    export_path: str,
    max_concurrent: int = 5,
    content_style: str = "",
) -> list[QuestionSet]:
    instances = list_instances(conn, status="completed", corpus_id=corpus_id)
    if not instances:
        logger.warning("No completed instances found for corpus %s", corpus_id)
        return []

    sem = asyncio.Semaphore(max_concurrent)

    async def _extract_one(inst_id: int) -> QuestionSet | None:
        async with sem:
            logger.info("Extracting questions for instance %d", inst_id)
            try:
                question_set = await extract_instance_questions(
                    conn=conn,
                    llm=llm,
                    model=model,
                    instance_id=inst_id,
                    corpus_id=corpus_id,
                    characters=characters,
                    plot_outline=plot_outline,
                    skill_content=skill_content,
                    content_style=content_style,
                )
                export_question_set(question_set, export_path)
                return question_set
            except Exception as e:
                logger.error(
                    "Failed to extract questions for instance %d: %s",
                    inst_id,
                    e,
                )
                return None

    raw_results = await asyncio.gather(
        *[_extract_one(inst.instance_id) for inst in instances]
    )
    results = [r for r in raw_results if r is not None]

    if results:
        combined_path = Path(export_path) / "questions.json"
        combined = [
            {
                "instance_id": qs.instance_id,
                "corpus_id": qs.corpus_id,
                "questions": [q.model_dump() for q in qs.questions],
                "coverage": {
                    "total_questions": len(qs.questions),
                    "types_covered": list(
                        {q.question_type for q in qs.questions}
                    ),
                    "missing_types": qs.missing_types(),
                },
            }
            for qs in results
        ]
        combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        logger.info(
            "Exported combined questions: %d instances -> %s",
            len(results),
            combined_path,
        )

    return results
