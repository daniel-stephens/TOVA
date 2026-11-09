from unittest.mock import AsyncMock

import pytest

from corpus_state.db import create_database, get_character, get_relationship
from corpus_agent.config import PlotPoint, SyntheticSetting
from corpus_agent.phases.character_mapping import (
    build_mapping_prompt,
    run_character_mapping,
)

MOCK_STORY = "# Test Story\n\n## Characters\n- Alice: protagonist\n- Bob: antagonist"

MOCK_LLM_RESPONSE = {
    "characters": [
        {
            "id": "alice",
            "wire_character": "Alice",
            "synthetic_character": "Alice Chen",
            "email_address": "a.chen@novabio.com",
            "role": "Lead Researcher",
            "organization": "NovaBio",
            "personality": "curious, determined",
            "voice_anchor": "Formal but warm. Uses technical terms freely.",
            "blur_rules": {"protagonist": "lead researcher"},
        },
        {
            "id": "bob",
            "wire_character": "Bob",
            "synthetic_character": "Robert Drake",
            "email_address": "r.drake@novabio.com",
            "role": "CFO",
            "organization": "NovaBio",
            "personality": "calculating, ambitious",
            "voice_anchor": "Terse. Bottom-line focused.",
            "blur_rules": {"antagonist": "executive adversary"},
        },
    ],
    "relationships": [
        {
            "char_a": "alice",
            "char_b": "bob",
            "strength": 4,
            "status": "adversarial",
        }
    ],
}


class TestBuildMappingPrompt:
    def test_includes_story_and_setting(self):
        prompt = build_mapping_prompt(
            story_text=MOCK_STORY,
            setting_name="biotech startup",
            setting_org="NovaBio",
            setting_domain="genetics",
            blur_level="medium",
            plot_outline=[PlotPoint(id="p1", description="first event")],
        )
        assert "biotech startup" in prompt
        assert "NovaBio" in prompt
        assert "Alice" in prompt
        assert "medium" in prompt
        assert "first event" in prompt


class TestRunCharacterMapping:
    @pytest.mark.asyncio
    async def test_creates_characters_and_relationships(self, tmp_path):
        story_dir = tmp_path / "story"
        story_dir.mkdir()
        (story_dir / "story.md").write_text(MOCK_STORY)

        conn = create_database(":memory:")

        mock_llm = AsyncMock()
        mock_llm.complete_json = AsyncMock(return_value=MOCK_LLM_RESPONSE)

        await run_character_mapping(
            conn=conn,
            llm=mock_llm,
            model="test-model",
            corpus_id="test-corpus",
            source_story_path=str(story_dir),
            setting=SyntheticSetting(
                name="biotech startup",
                organization="NovaBio",
                domain="genetics",
            ),
            blur_level="medium",
            plot_outline=[PlotPoint(id="p1", description="first event")],
            skill_content="You are a character mapper.",
        )

        alice = get_character(conn, "test-corpus", "alice")
        assert alice is not None
        assert alice.synthetic_character == "Alice Chen"
        assert alice.email_address == "a.chen@novabio.com"

        bob = get_character(conn, "test-corpus", "bob")
        assert bob is not None

        rel = get_relationship(conn, "test-corpus", "alice", "bob")
        assert rel is not None
        assert rel.status == "adversarial"
        assert rel.strength == 4

    @pytest.mark.asyncio
    async def test_skips_if_characters_exist(self, tmp_path):
        story_dir = tmp_path / "story"
        story_dir.mkdir()
        (story_dir / "story.md").write_text(MOCK_STORY)

        conn = create_database(":memory:")
        from corpus_state.db import upsert_character
        from corpus_state.types import Character

        upsert_character(
            conn,
            Character(
                id="alice",
                corpus_id="test-corpus",
                wire_character="Alice",
                synthetic_character="Alice Chen",
                email_address="a@b.com",
                role="x",
                organization="y",
                personality="z",
                voice_anchor="w",
            ),
        )

        mock_llm = AsyncMock()
        await run_character_mapping(
            conn=conn,
            llm=mock_llm,
            model="m",
            corpus_id="test-corpus",
            source_story_path=str(story_dir),
            setting=SyntheticSetting(name="s", organization="o", domain="d"),
            blur_level="light",
            plot_outline=[PlotPoint(id="p1", description="d1")],
            skill_content="x",
        )
        mock_llm.complete_json.assert_not_called()

