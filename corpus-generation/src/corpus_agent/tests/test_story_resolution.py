from unittest.mock import AsyncMock

import pytest

from corpus_state.db import create_database, get_story_mapping
from corpus_agent.config import (
    PlotPoint,
    PlotOutlineConfig,
    StoryMapping,
    SyntheticSetting,
)
from corpus_agent.phases.story_resolution import (
    resolve_story_mapping,
    build_outline_prompt,
)


SETTING = SyntheticSetting(
    name="biotech startup",
    organization="NovaBio Inc.",
    domain="genetic research",
)

MANUAL_OUTLINE = PlotOutlineConfig(
    mode="manual",
    points=[
        PlotPoint(id="p1", description="Handler killed during specimen transfer"),
        PlotPoint(id="p2", description="Rival bribes employee to steal samples"),
    ],
)

AUTO_OUTLINE = PlotOutlineConfig(mode="auto")

MOCK_STORY = "# Jurassic Park\n\nA theme park with cloned dinosaurs goes wrong."

MOCK_LLM_RESPONSE = {
    "plot_points": [
        {"id": "inciting_incident", "description": "Lab accident triggers safety review"},
        {"id": "espionage", "description": "Corporate spy recruited from research team"},
        {"id": "containment_failure", "description": "Specimens escape containment"},
    ],
}


class TestBuildOutlinePrompt:
    def test_includes_story_and_setting(self):
        prompt = build_outline_prompt(
            story_text=MOCK_STORY,
            setting_name="biotech startup",
            setting_org="NovaBio Inc.",
            setting_domain="genetic research",
            blur_level="medium",
        )
        assert "biotech startup" in prompt
        assert "NovaBio" in prompt
        assert "genetic research" in prompt
        assert "medium" in prompt
        assert "Jurassic Park" in prompt


class TestResolveStoryMapping:
    @pytest.mark.asyncio
    async def test_manual_mode_persists_and_returns(self, tmp_path):
        story_dir = tmp_path / "story"
        story_dir.mkdir()
        (story_dir / "story.md").write_text(MOCK_STORY)

        conn = create_database(":memory:")
        mapping = StoryMapping(synthetic_setting=SETTING, plot_outline=MANUAL_OUTLINE)

        result = await resolve_story_mapping(
            conn=conn,
            corpus_id="test-manual",
            story_mapping=mapping,
            source_story_path=str(story_dir),
            blur_level="medium",
            llm=None,
            model=None,
            skill_content=None,
        )

        assert len(result) == 2
        assert result[0].id == "p1"
        assert result[1].id == "p2"

        stored = get_story_mapping(conn, "test-manual")
        assert stored is not None
        assert stored.mode == "manual"
        assert len(stored.plot_points) == 2

    @pytest.mark.asyncio
    async def test_auto_mode_calls_llm_and_persists(self, tmp_path):
        story_dir = tmp_path / "story"
        story_dir.mkdir()
        (story_dir / "story.md").write_text(MOCK_STORY)

        conn = create_database(":memory:")
        mapping = StoryMapping(synthetic_setting=SETTING, plot_outline=AUTO_OUTLINE)

        mock_llm = AsyncMock()
        mock_llm.complete_json = AsyncMock(return_value=MOCK_LLM_RESPONSE)

        result = await resolve_story_mapping(
            conn=conn,
            corpus_id="test-auto",
            story_mapping=mapping,
            source_story_path=str(story_dir),
            blur_level="medium",
            llm=mock_llm,
            model="test-model",
            skill_content="You are an outline generator.",
        )

        assert len(result) == 3
        assert result[0].id == "inciting_incident"
        mock_llm.complete_json.assert_called_once()

        stored = get_story_mapping(conn, "test-auto")
        assert stored is not None
        assert stored.mode == "auto"
        assert len(stored.plot_points) == 3

    @pytest.mark.asyncio
    async def test_resume_reuses_manual_mapping(self, tmp_path):
        story_dir = tmp_path / "story"
        story_dir.mkdir()
        (story_dir / "story.md").write_text(MOCK_STORY)

        conn = create_database(":memory:")
        mapping = StoryMapping(synthetic_setting=SETTING, plot_outline=MANUAL_OUTLINE)

        await resolve_story_mapping(
            conn=conn, corpus_id="test-resume", story_mapping=mapping,
            source_story_path=str(story_dir), blur_level="medium",
            llm=None, model=None, skill_content=None,
        )

        mock_llm = AsyncMock()
        result = await resolve_story_mapping(
            conn=conn, corpus_id="test-resume", story_mapping=mapping,
            source_story_path=str(story_dir), blur_level="medium",
            llm=mock_llm, model="m", skill_content="x",
        )

        assert len(result) == 2
        assert result[0].id == "p1"
        mock_llm.complete_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_resume_reuses_auto_generated_mapping(self, tmp_path):
        story_dir = tmp_path / "story"
        story_dir.mkdir()
        (story_dir / "story.md").write_text(MOCK_STORY)

        conn = create_database(":memory:")
        mapping = StoryMapping(synthetic_setting=SETTING, plot_outline=AUTO_OUTLINE)

        mock_llm = AsyncMock()
        mock_llm.complete_json = AsyncMock(return_value=MOCK_LLM_RESPONSE)

        first_result = await resolve_story_mapping(
            conn=conn, corpus_id="test-auto-resume", story_mapping=mapping,
            source_story_path=str(story_dir), blur_level="medium",
            llm=mock_llm, model="test-model", skill_content="You are an outline generator.",
        )
        assert mock_llm.complete_json.call_count == 1

        mock_llm_2 = AsyncMock()
        second_result = await resolve_story_mapping(
            conn=conn, corpus_id="test-auto-resume", story_mapping=mapping,
            source_story_path=str(story_dir), blur_level="medium",
            llm=mock_llm_2, model="test-model", skill_content="You are an outline generator.",
        )
        mock_llm_2.complete_json.assert_not_called()
        assert [p.id for p in first_result] == [p.id for p in second_result]

    @pytest.mark.asyncio
    async def test_auto_mode_retries_on_malformed_response(self, tmp_path):
        story_dir = tmp_path / "story"
        story_dir.mkdir()
        (story_dir / "story.md").write_text(MOCK_STORY)

        conn = create_database(":memory:")
        mapping = StoryMapping(synthetic_setting=SETTING, plot_outline=AUTO_OUTLINE)

        mock_llm = AsyncMock()
        mock_llm.complete_json = AsyncMock(
            side_effect=[{"wrong_key": "no plot_points"}, {"wrong_key": "still bad"}, MOCK_LLM_RESPONSE]
        )

        result = await resolve_story_mapping(
            conn=conn, corpus_id="test-retry", story_mapping=mapping,
            source_story_path=str(story_dir), blur_level="medium",
            llm=mock_llm, model="test-model", skill_content="You are an outline generator.",
        )

        assert len(result) == 3
        assert mock_llm.complete_json.call_count == 3

    @pytest.mark.asyncio
    async def test_auto_mode_raises_after_max_retries(self, tmp_path):
        story_dir = tmp_path / "story"
        story_dir.mkdir()
        (story_dir / "story.md").write_text(MOCK_STORY)

        conn = create_database(":memory:")
        mapping = StoryMapping(synthetic_setting=SETTING, plot_outline=AUTO_OUTLINE)

        mock_llm = AsyncMock()
        mock_llm.complete_json = AsyncMock(return_value={"wrong": "format"})

        with pytest.raises(ValueError, match="plot outline"):
            await resolve_story_mapping(
                conn=conn, corpus_id="test-fail", story_mapping=mapping,
                source_story_path=str(story_dir), blur_level="medium",
                llm=mock_llm, model="test-model", skill_content="You are an outline generator.",
            )

    @pytest.mark.asyncio
    async def test_auto_mode_raises_without_llm(self, tmp_path):
        story_dir = tmp_path / "story"
        story_dir.mkdir()
        (story_dir / "story.md").write_text(MOCK_STORY)

        conn = create_database(":memory:")
        mapping = StoryMapping(synthetic_setting=SETTING, plot_outline=AUTO_OUTLINE)

        with pytest.raises(ValueError, match="required for auto"):
            await resolve_story_mapping(
                conn=conn, corpus_id="test-no-llm", story_mapping=mapping,
                source_story_path=str(story_dir), blur_level="medium",
                llm=None, model=None, skill_content=None,
            )
