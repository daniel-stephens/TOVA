from unittest.mock import AsyncMock

import pytest

from corpus_state.types import Character
from corpus_agent.config import PlotPoint
from corpus_agent.phases.wave_planner import WavePlan, build_planner_prompt, plan_wave


CHARACTERS = [
    Character(
        id="alice",
        corpus_id="test",
        wire_character="Alice",
        synthetic_character="Alice Chen",
        email_address="a@b.com",
        role="Lead",
        organization="X",
        personality="driven",
        voice_anchor="formal",
    ),
    Character(
        id="bob",
        corpus_id="test",
        wire_character="Bob",
        synthetic_character="Bob Drake",
        email_address="b@b.com",
        role="CFO",
        organization="X",
        personality="calculating",
        voice_anchor="terse",
    ),
]

MOCK_WAVE_PLAN = {
    "emails": [
        {
            "sender_id": "alice",
            "recipient_ids": ["bob"],
            "is_reply": False,
            "parent_mid": None,
            "plot_point_to_reveal": "inciting_incident",
            "is_noise": False,
            "intent": "Alice reports anomaly to Bob",
        },
        {
            "sender_id": "bob",
            "recipient_ids": ["alice"],
            "is_reply": False,
            "parent_mid": None,
            "plot_point_to_reveal": None,
            "is_noise": True,
            "intent": "Bob sends routine budget update",
        },
    ],
    "wave_rationale": "Introduce inciting incident with noise cover",
}


class TestBuildPlannerPrompt:
    def test_includes_characters_and_plot(self):
        prompt = build_planner_prompt(
            characters=CHARACTERS,
            wave_number=1,
            total_waves=10,
            emails_per_wave=4,
            plot_outline=[PlotPoint(id="p1", description="d1")],
            revealed_plot_points=[],
            blur_level="medium",
        )
        assert "Alice Chen" in prompt
        assert "Bob Drake" in prompt
        assert "wave 1 of 10" in prompt.lower() or "Wave 1" in prompt


class TestPlanWave:
    @pytest.mark.asyncio
    async def test_retries_on_wrapped_response(self):
        mock_llm = AsyncMock()
        wrapped = {"instance_id": 1, "corpus_id": "x", "wave_plan": MOCK_WAVE_PLAN}
        mock_llm.complete_json = AsyncMock(
            side_effect=[wrapped, wrapped, MOCK_WAVE_PLAN]
        )

        plan = await plan_wave(
            llm=mock_llm,
            model="test",
            characters=CHARACTERS,
            wave_number=1,
            total_waves=10,
            emails_per_wave=4,
            plot_outline=[PlotPoint(id="inciting_incident", description="d1")],
            revealed_plot_points=[],
            blur_level="medium",
            skill_content="You are a planner.",
        )

        assert isinstance(plan, WavePlan)
        assert mock_llm.complete_json.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        mock_llm = AsyncMock()
        mock_llm.complete_json = AsyncMock(
            return_value={"wrong": "shape"}
        )

        with pytest.raises(ValueError, match="wave plan"):
            await plan_wave(
                llm=mock_llm,
                model="test",
                characters=CHARACTERS,
                wave_number=1,
                total_waves=10,
                emails_per_wave=4,
                plot_outline=[PlotPoint(id="p1", description="d1")],
                revealed_plot_points=[],
                blur_level="medium",
                skill_content="You are a planner.",
            )

    @pytest.mark.asyncio
    async def test_returns_wave_plan(self):
        mock_llm = AsyncMock()
        mock_llm.complete_json = AsyncMock(return_value=MOCK_WAVE_PLAN)

        plan = await plan_wave(
            llm=mock_llm,
            model="test",
            characters=CHARACTERS,
            wave_number=1,
            total_waves=10,
            emails_per_wave=4,
            plot_outline=[PlotPoint(id="inciting_incident", description="d1")],
            revealed_plot_points=[],
            blur_level="medium",
            skill_content="You are a planner.",
        )

        assert isinstance(plan, WavePlan)
        assert len(plan.emails) == 2
        assert plan.emails[0].sender_id == "alice"
        assert plan.emails[0].plot_point_to_reveal == "inciting_incident"
        assert plan.emails[1].is_noise is True
        assert plan.wave_rationale == "Introduce inciting incident with noise cover"

