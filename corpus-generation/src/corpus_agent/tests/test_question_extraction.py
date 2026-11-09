import pytest
from pydantic import ValidationError

from corpus_agent.phases.question_extraction import (
    QUESTION_TYPES,
    SensemakingQuestion,
    QuestionSet,
)


class TestQuestionTypes:
    def test_taxonomy_has_connection_types(self):
        assert len(QUESTION_TYPES) == 9
        expected = {
            "attribution",
            "timeline",
            "relationship_mapping",
            "information_asymmetry",
            "motive_intent",
            "coordination_detection",
            "document_topic_connection",
            "topic_topic_connection",
            "document_document_connection",
        }
        assert set(QUESTION_TYPES.keys()) == expected

    def test_each_type_has_description(self):
        for key, desc in QUESTION_TYPES.items():
            assert isinstance(desc, str)
            assert len(desc) > 10


class TestSensemakingQuestion:
    def test_valid_question(self):
        q = SensemakingQuestion(
            question="Who leaked the lab results to the press?",
            question_type="attribution",
            difficulty=3,
            required_plot_points=["leak_detected", "press_coverage"],
            answer_summary="Alice Chen leaked the results via her personal email on Jan 15.",
            evidence_emails=["<msg-42.abc@novabio.com>", "<msg-58.def@novabio.com>"],
        )
        assert q.question_type == "attribution"
        assert q.difficulty == 3
        assert len(q.evidence_emails) == 2

    def test_rejects_invalid_type(self):
        with pytest.raises(ValidationError):
            SensemakingQuestion(
                question="test?",
                question_type="vibes",
                difficulty=1,
                required_plot_points=["p1"],
                answer_summary="answer",
                evidence_emails=["<1@t>"],
            )

    def test_rejects_difficulty_out_of_range(self):
        with pytest.raises(ValidationError):
            SensemakingQuestion(
                question="test?",
                question_type="attribution",
                difficulty=6,
                required_plot_points=["p1"],
                answer_summary="answer",
                evidence_emails=["<1@t>"],
            )

    def test_rejects_empty_evidence(self):
        with pytest.raises(ValidationError):
            SensemakingQuestion(
                question="test?",
                question_type="attribution",
                difficulty=1,
                required_plot_points=["p1"],
                answer_summary="answer",
                evidence_emails=[],
            )


class TestQuestionSet:
    def test_valid_question_set(self):
        qs = QuestionSet(
            instance_id=1,
            corpus_id="test",
            questions=[
                SensemakingQuestion(
                    question="Who?",
                    question_type="attribution",
                    difficulty=2,
                    required_plot_points=["p1"],
                    answer_summary="Alice",
                    evidence_emails=["<1@t>"],
                ),
                SensemakingQuestion(
                    question="When?",
                    question_type="timeline",
                    difficulty=3,
                    required_plot_points=["p1", "p2"],
                    answer_summary="Jan 15 before Jan 20",
                    evidence_emails=["<1@t>", "<2@t>"],
                ),
            ],
        )
        assert qs.instance_id == 1
        assert len(qs.questions) == 2

    def test_coverage_check(self):
        qs = QuestionSet(
            instance_id=1,
            corpus_id="test",
            questions=[
                SensemakingQuestion(
                    question="q1?",
                    question_type="attribution",
                    difficulty=1,
                    required_plot_points=["p1"],
                    answer_summary="a1",
                    evidence_emails=["<1@t>"],
                ),
            ],
        )
        missing = qs.missing_types()
        assert "attribution" not in missing
        assert "timeline" in missing
        assert "document_topic_connection" in missing
        assert len(missing) == 8


from corpus_state.types import Character, Email

from corpus_agent.config import PlotPoint
from corpus_agent.phases.question_extraction import build_extraction_prompt


CHARACTERS = [
    Character(
        id="alice",
        corpus_id="test",
        wire_character="A",
        synthetic_character="Alice Chen",
        email_address="a@b.com",
        role="Lead Researcher",
        organization="NovaBio",
        personality="curious, determined",
        voice_anchor="Formal but warm",
    ),
    Character(
        id="bob",
        corpus_id="test",
        wire_character="B",
        synthetic_character="Robert Drake",
        email_address="r@b.com",
        role="CFO",
        organization="NovaBio",
        personality="calculating",
        voice_anchor="Terse",
    ),
]

EMAILS = [
    Email(
        mid=1,
        sender="a@b.com",
        recipient="r@b.com",
        date="2026-01-15T09:00:00Z",
        message_id="<msg-1@t>",
        subject="Lab Results",
        body="Robert, I found anomalies in batch 7.",
        instance_id=1,
    ),
    Email(
        mid=2,
        sender="r@b.com",
        recipient="a@b.com",
        date="2026-01-15T10:00:00Z",
        message_id="<msg-2@t>",
        subject="Re: Lab Results",
        body="Keep this between us for now.",
        parent_mid=1,
        instance_id=1,
    ),
]

PLOT_OUTLINE = [
    PlotPoint(id="anomaly_detected", description="Anomaly found in batch 7"),
    PlotPoint(id="cover_up", description="CFO suppresses findings"),
]

REVELATIONS = {"anomaly_detected": 1, "cover_up": 2}


class TestBuildExtractionPrompt:
    def test_includes_all_context(self):
        prompt = build_extraction_prompt(
            emails=EMAILS,
            characters=CHARACTERS,
            plot_outline=PLOT_OUTLINE,
            revelations=REVELATIONS,
        )
        assert "Alice Chen" in prompt
        assert "Robert Drake" in prompt
        assert "anomaly_detected" in prompt
        assert "cover_up" in prompt
        assert "<msg-1@t>" in prompt
        assert "Lab Results" in prompt
        assert "attribution" in prompt
        assert "timeline" in prompt

    def test_includes_taxonomy(self):
        prompt = build_extraction_prompt(
            emails=EMAILS,
            characters=CHARACTERS,
            plot_outline=PLOT_OUTLINE,
            revelations=REVELATIONS,
        )
        for qtype in QUESTION_TYPES:
            assert qtype in prompt


from unittest.mock import AsyncMock

from corpus_state.db import (
    bulk_save_emails,
    complete_instance,
    create_database,
    create_instance,
    log_plot_revelation,
    upsert_character,
)

from corpus_agent.phases.question_extraction import (
    extract_instance_questions,
    validate_questions,
)


def _setup_instance_db():
    conn = create_database(":memory:")
    for c in CHARACTERS:
        upsert_character(conn, c)
    create_instance(conn, "test", 1)
    bulk_save_emails(conn, 1, EMAILS)
    complete_instance(conn, 1)
    log_plot_revelation(conn, "test", "anomaly_detected", "Anomaly found", 1)
    log_plot_revelation(conn, "test", "cover_up", "CFO suppresses", 2)
    return conn


MOCK_LLM_RESPONSE = {
    "questions": [
        {
            "question": "Who first discovered the anomaly in batch 7?",
            "question_type": "attribution",
            "difficulty": 2,
            "required_plot_points": ["anomaly_detected"],
            "answer_summary": "Alice Chen found it and reported to Robert Drake.",
            "evidence_emails": ["<msg-1@t>", "<msg-2@t>"],
        },
        {
            "question": "In what order did the discovery and cover-up occur?",
            "question_type": "timeline",
            "difficulty": 3,
            "required_plot_points": ["anomaly_detected", "cover_up"],
            "answer_summary": "Discovery at 9am, cover-up directive at 10am.",
            "evidence_emails": ["<msg-1@t>", "<msg-2@t>"],
        },
        {
            "question": "What is the professional relationship between Alice and Robert?",
            "question_type": "relationship_mapping",
            "difficulty": 2,
            "required_plot_points": ["anomaly_detected"],
            "answer_summary": "Researcher reporting to CFO, becomes complicit.",
            "evidence_emails": ["<msg-1@t>", "<msg-2@t>"],
        },
        {
            "question": "What does Alice know that other employees do not?",
            "question_type": "information_asymmetry",
            "difficulty": 3,
            "required_plot_points": ["anomaly_detected", "cover_up"],
            "answer_summary": "Alice knows about the anomaly AND that Robert wants it hidden.",
            "evidence_emails": ["<msg-1@t>", "<msg-2@t>"],
        },
        {
            "question": "Why did Robert ask Alice to keep the results quiet?",
            "question_type": "motive_intent",
            "difficulty": 4,
            "required_plot_points": ["cover_up"],
            "answer_summary": "To protect stock price and avoid regulatory scrutiny.",
            "evidence_emails": ["<msg-2@t>"],
        },
        {
            "question": "Are Alice and Robert coordinating to hide the anomaly?",
            "question_type": "coordination_detection",
            "difficulty": 3,
            "required_plot_points": ["anomaly_detected", "cover_up"],
            "answer_summary": "Robert directs, Alice complies by not escalating.",
            "evidence_emails": ["<msg-1@t>", "<msg-2@t>"],
        },
    ],
}


class TestValidateQuestions:
    def test_rejects_single_email_questions(self):
        bad_q = SensemakingQuestion(
            question="Why silence?",
            question_type="motive_intent",
            difficulty=4,
            required_plot_points=["cover_up"],
            answer_summary="To protect stock price.",
            evidence_emails=["<msg-2@t>"],
        )
        corpus_mids = {"<msg-1@t>", "<msg-2@t>"}
        revealed_points = {"anomaly_detected", "cover_up"}
        passed, warnings = validate_questions([bad_q], corpus_mids, revealed_points)
        assert len(passed) == 0
        assert any("single email" in w.lower() for w in warnings)

    def test_rejects_missing_evidence_email(self):
        bad_q = SensemakingQuestion(
            question="Who?",
            question_type="attribution",
            difficulty=2,
            required_plot_points=["anomaly_detected"],
            answer_summary="Alice",
            evidence_emails=["<msg-1@t>", "<nonexistent@t>"],
        )
        corpus_mids = {"<msg-1@t>", "<msg-2@t>"}
        revealed_points = {"anomaly_detected"}
        passed, warnings = validate_questions([bad_q], corpus_mids, revealed_points)
        assert len(passed) == 0
        assert any("not found" in w.lower() for w in warnings)

    def test_rejects_unrevealed_plot_point(self):
        bad_q = SensemakingQuestion(
            question="What happened with sabotage?",
            question_type="attribution",
            difficulty=2,
            required_plot_points=["sabotage"],
            answer_summary="Nothing yet",
            evidence_emails=["<msg-1@t>", "<msg-2@t>"],
        )
        corpus_mids = {"<msg-1@t>", "<msg-2@t>"}
        revealed_points = {"anomaly_detected"}
        passed, warnings = validate_questions([bad_q], corpus_mids, revealed_points)
        assert len(passed) == 0
        assert any("not revealed" in w.lower() for w in warnings)

    def test_passes_valid_question(self):
        good_q = SensemakingQuestion(
            question="Who found it and what happened next?",
            question_type="attribution",
            difficulty=2,
            required_plot_points=["anomaly_detected"],
            answer_summary="Alice found it, Robert suppressed it.",
            evidence_emails=["<msg-1@t>", "<msg-2@t>"],
        )
        corpus_mids = {"<msg-1@t>", "<msg-2@t>"}
        revealed_points = {"anomaly_detected"}
        passed, warnings = validate_questions([good_q], corpus_mids, revealed_points)
        assert len(passed) == 1
        assert passed[0].question == good_q.question
        assert len(warnings) == 0

    def test_mixed_valid_and_invalid(self):
        good_q = SensemakingQuestion(
            question="Who found it?",
            question_type="attribution",
            difficulty=2,
            required_plot_points=["anomaly_detected"],
            answer_summary="Alice",
            evidence_emails=["<msg-1@t>", "<msg-2@t>"],
        )
        bad_q = SensemakingQuestion(
            question="Why?",
            question_type="motive_intent",
            difficulty=4,
            required_plot_points=["cover_up"],
            answer_summary="Greed",
            evidence_emails=["<msg-2@t>"],  # single email
        )
        corpus_mids = {"<msg-1@t>", "<msg-2@t>"}
        revealed_points = {"anomaly_detected", "cover_up"}
        passed, warnings = validate_questions([good_q, bad_q], corpus_mids, revealed_points)
        assert len(passed) == 1
        assert passed[0].question_type == "attribution"
        assert len(warnings) == 1


class TestExtractInstanceQuestions:
    @pytest.mark.asyncio
    async def test_extracts_and_filters_questions(self):
        conn = _setup_instance_db()
        mock_llm = AsyncMock()
        mock_llm.complete_json = AsyncMock(return_value=MOCK_LLM_RESPONSE)

        result = await extract_instance_questions(
            conn=conn,
            llm=mock_llm,
            model="test-model",
            instance_id=1,
            corpus_id="test",
            characters=CHARACTERS,
            plot_outline=PLOT_OUTLINE,
            skill_content="You are a question extractor.",
        )

        assert isinstance(result, QuestionSet)
        assert result.instance_id == 1
        # The motive_intent question has only 1 evidence email and gets filtered out.
        # Coverage retry fires, and the mock returns the same response, so the retry
        # motive_intent also gets filtered. Result: 5 questions, missing motive_intent.
        assert len(result.questions) == 5
        assert "motive_intent" in result.missing_types()

    @pytest.mark.asyncio
    async def test_coverage_retry_fills_gaps(self):
        conn = _setup_instance_db()
        # First call returns only attribution; retry fills timeline
        first_response = {
            "questions": [MOCK_LLM_RESPONSE["questions"][0]],
        }
        retry_response = {
            "questions": [
                {
                    "question": "When did the cover-up start relative to the discovery?",
                    "question_type": "timeline",
                    "difficulty": 3,
                    "required_plot_points": ["anomaly_detected", "cover_up"],
                    "answer_summary": "Discovery at 9am, cover-up at 10am same day.",
                    "evidence_emails": ["<msg-1@t>", "<msg-2@t>"],
                },
            ],
        }
        mock_llm = AsyncMock()
        mock_llm.complete_json = AsyncMock(side_effect=[first_response, retry_response])

        result = await extract_instance_questions(
            conn=conn,
            llm=mock_llm,
            model="test-model",
            instance_id=1,
            corpus_id="test",
            characters=CHARACTERS,
            plot_outline=PLOT_OUTLINE,
            skill_content="You are a question extractor.",
        )

        types_covered = {q.question_type for q in result.questions}
        assert "attribution" in types_covered
        assert "timeline" in types_covered
        assert len(result.questions) == 2


import json
from pathlib import Path

from corpus_agent.phases.question_extraction import (
    export_question_set,
    run_question_extraction,
)


class TestExportQuestionSet:
    def test_writes_json_file(self, tmp_path):
        qs = QuestionSet(
            instance_id=1,
            corpus_id="test",
            questions=[
                SensemakingQuestion(
                    question="Who?",
                    question_type="attribution",
                    difficulty=2,
                    required_plot_points=["p1"],
                    answer_summary="Alice",
                    evidence_emails=["<1@t>", "<2@t>"],
                ),
            ],
        )
        out_path = export_question_set(qs, str(tmp_path))
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["instance_id"] == 1
        assert data["corpus_id"] == "test"
        assert len(data["questions"]) == 1
        assert data["questions"][0]["question_type"] == "attribution"
        assert data["coverage"]["missing_types"] == [
            "timeline",
            "relationship_mapping",
            "information_asymmetry",
            "motive_intent",
            "coordination_detection",
            "document_topic_connection",
            "topic_topic_connection",
            "document_document_connection",
        ]
        assert data["coverage"]["total_questions"] == 1

    def test_writes_to_instances_subdir(self, tmp_path):
        qs = QuestionSet(
            instance_id=3,
            corpus_id="test",
            questions=[
                SensemakingQuestion(
                    question="q?",
                    question_type="timeline",
                    difficulty=1,
                    required_plot_points=["p1"],
                    answer_summary="a",
                    evidence_emails=["<1@t>", "<2@t>"],
                ),
            ],
        )
        out_path = export_question_set(qs, str(tmp_path))
        assert out_path == tmp_path / "instances" / "questions_0003.json"


class TestRunQuestionExtraction:
    @pytest.mark.asyncio
    async def test_extracts_for_all_completed_instances(self, tmp_path):
        conn = _setup_instance_db()
        mock_llm = AsyncMock()
        mock_llm.complete_json = AsyncMock(return_value=MOCK_LLM_RESPONSE)

        results = await run_question_extraction(
            conn=conn,
            llm=mock_llm,
            model="test-model",
            corpus_id="test",
            characters=CHARACTERS,
            plot_outline=PLOT_OUTLINE,
            skill_content="You are a question extractor.",
            export_path=str(tmp_path),
        )

        assert len(results) == 1
        assert results[0].instance_id == 1
        assert (tmp_path / "instances" / "questions_0001.json").exists()

        combined = tmp_path / "questions.json"
        assert combined.exists()
        data = json.loads(combined.read_text())
        assert len(data) == 1
        assert data[0]["instance_id"] == 1
