from unittest.mock import AsyncMock

import pytest

from corpus_state.types import Character, Email
from corpus_agent.phases.email_generator import build_generator_prompt, generate_email
from corpus_agent.phases.wave_planner import EmailPlan


ALICE = Character(
    id="alice",
    corpus_id="test",
    wire_character="Alice",
    synthetic_character="Alice Chen",
    email_address="a.chen@novabio.com",
    role="Lead Researcher",
    organization="NovaBio",
    personality="curious, determined",
    voice_anchor="Formal but warm. Uses technical terms freely.",
)
BOB = Character(
    id="bob",
    corpus_id="test",
    wire_character="Bob",
    synthetic_character="Robert Drake",
    email_address="r.drake@novabio.com",
    role="CFO",
    organization="NovaBio",
    personality="calculating, ambitious",
    voice_anchor="Terse. Bottom-line focused.",
)

PLAN = EmailPlan(
    sender_id="alice",
    recipient_ids=["bob"],
    is_reply=False,
    parent_mid=None,
    plot_point_to_reveal="anomaly_detected",
    is_noise=False,
    intent="Alice reports anomaly in lab results to Bob",
)

MOCK_EMAIL_RESPONSE = {
    "sender": "a.chen@novabio.com",
    "recipient": "r.drake@novabio.com",
    "date": "2026-05-15T09:30:00Z",
    "message_id": "<abc123@novabio.com>",
    "subject": "Anomalous Results in Batch 7",
    "body": "Robert,\n\nI need to flag something...",
    "folder": "\\Sent",
}


class TestBuildGeneratorPrompt:
    def test_includes_sender_profile(self):
        prompt = build_generator_prompt(
            plan=PLAN,
            sender=ALICE,
            recipients=[BOB],
            thread_history=[],
            plot_description="Anomaly detected in lab batch",
            blur_level="medium",
        )
        assert "Alice Chen" in prompt
        assert "Formal but warm" in prompt
        assert "anomaly" in prompt.lower()


class TestGenerateEmail:
    @pytest.mark.asyncio
    async def test_retries_on_wrapped_response(self):
        mock_llm = AsyncMock()
        wrapped = {"email": MOCK_EMAIL_RESPONSE, "instance_id": 1}
        mock_llm.complete_json = AsyncMock(
            side_effect=[wrapped, wrapped, MOCK_EMAIL_RESPONSE]
        )

        email = await generate_email(
            llm=mock_llm,
            model="test",
            plan=PLAN,
            sender=ALICE,
            recipients=[BOB],
            thread_history=[],
            plot_description=None,
            blur_level="medium",
            skill_content="You are an email generator.",
        )

        assert isinstance(email, Email)
        assert mock_llm.complete_json.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        mock_llm = AsyncMock()
        mock_llm.complete_json = AsyncMock(return_value={"wrong": "shape"})

        with pytest.raises(ValueError, match="email"):
            await generate_email(
                llm=mock_llm,
                model="test",
                plan=PLAN,
                sender=ALICE,
                recipients=[BOB],
                thread_history=[],
                plot_description=None,
                blur_level="medium",
                skill_content="You are an email generator.",
            )

    @pytest.mark.asyncio
    async def test_returns_email(self):
        mock_llm = AsyncMock()
        mock_llm.complete_json = AsyncMock(return_value=MOCK_EMAIL_RESPONSE)

        email = await generate_email(
            llm=mock_llm,
            model="test",
            plan=PLAN,
            sender=ALICE,
            recipients=[BOB],
            thread_history=[],
            plot_description="Anomaly detected",
            blur_level="medium",
            skill_content="You are an email generator.",
        )

        assert isinstance(email, Email)
        assert email.sender == "a.chen@novabio.com"
        assert email.recipient == "r.drake@novabio.com"
        assert email.subject == "Anomalous Results in Batch 7"
        assert email.mid is None
        assert email.parent_mid is None
        # message_id should be uniquified with a UUID suffix
        assert email.message_id.startswith("<abc123@novabio.com.")
        assert email.message_id.endswith(">")

