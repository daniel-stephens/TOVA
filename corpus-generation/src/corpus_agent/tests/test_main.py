import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corpus_agent.main import run_pipeline


MINIMAL_YAML = textwrap.dedent(
    """\
    corpus_id: integration-test

    source_story:
      path: {story_dir}

    story_mapping:
      synthetic_setting:
        name: biotech startup
        organization: NovaBio Inc.
        domain: genetic research
      plot_outline:
        mode: manual
        points:
          - id: p1
            description: "First event"
          - id: p2
            description: "Second event"

    corpus_size:
      num_instances: 2
      waves_per_instance: 2
      emails_per_wave: 1
      emails_per_instance: 5

    blur_level: light

    models:
      character_mapping: test-model
      outline_generator: test-model
      planner: test-model
      content_generator: test-model

    llm:
      api_key_env_var: TEST_RCHAT_API_KEY
      base_url: https://rchat.nist.gov/proxy
      max_tokens: 128

    rate_limits:
      max_concurrent_instances: 2
      requests_per_minute: 1000
      wave_pause_seconds: 0

    output:
      db_path: {db_path}
      export_path: {export_path}
"""
)


CHARACTER_RESPONSE = {
    "characters": [
        {
            "id": "alice",
            "wire_character": "A",
            "synthetic_character": "Alice",
            "email_address": "a@b.com",
            "role": "Lead",
            "organization": "X",
            "personality": "driven",
            "voice_anchor": "formal",
            "blur_rules": {},
        },
    ],
    "relationships": [],
}


class TestRunPipeline:
    @pytest.mark.asyncio
    async def test_end_to_end(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_RCHAT_API_KEY", "test-key")
        story_dir = tmp_path / "story"
        story_dir.mkdir()
        (story_dir / "story.md").write_text("# Test\nA story.")

        db_path = str(tmp_path / "test.db")
        export_path = str(tmp_path / "export")

        config_text = MINIMAL_YAML.format(
            story_dir=str(story_dir),
            db_path=db_path,
            export_path=export_path,
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_text)

        skills_dir = tmp_path / "skills"
        for name in ("character-mapping", "email-generation", "orchestration", "outline-generation", "question-extraction"):
            d = skills_dir / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"You are the {name} agent.")

        email_counter = {"n": 0}

        async def mock_complete_json(model, system, user, temperature=0.7, max_tokens=None):
            if "Map each character" in user:
                return CHARACTER_RESPONSE
            if "## Wave" in user:
                return {
                    "emails": [
                        {
                            "sender_id": "alice",
                            "recipient_ids": ["alice"],
                            "is_reply": False,
                            "parent_mid": None,
                            "plot_point_to_reveal": None,
                            "is_noise": True,
                            "intent": "test",
                        }
                    ],
                    "wave_rationale": "test",
                }
            email_counter["n"] += 1
            return {
                "sender": "a@b.com",
                "recipient": "a@b.com",
                "date": f"2026-01-{email_counter['n']:02d}",
                "message_id": f"<{email_counter['n']}@t>",
                "subject": f"Email {email_counter['n']}",
                "body": f"Body {email_counter['n']}",
                "folder": "\\Sent",
            }

        mock_client = AsyncMock()
        mock_llm = MagicMock()
        mock_llm.complete_json = mock_complete_json
        mock_llm.complete = AsyncMock(return_value="")

        with (
            patch("corpus_agent.main.LLMClient", return_value=mock_llm),
            patch("corpus_agent.main.AsyncOpenAI", return_value=mock_client) as mock_async_openai,
        ):
            await run_pipeline(str(config_file), resume=False, skills_root=str(skills_dir))

        mock_async_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://rchat.nist.gov/proxy",
        )
        assert (Path(export_path) / "email_corpus.jsonl").exists()
        lines = (Path(export_path) / "email_corpus.jsonl").read_text().strip().split("\n")
        assert len(lines) == 4

    @pytest.mark.asyncio
    async def test_end_to_end_auto_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_RCHAT_API_KEY", "test-key")
        story_dir = tmp_path / "story"
        story_dir.mkdir()
        (story_dir / "story.md").write_text("# Test\nA story about a lab.")

        db_path = str(tmp_path / "test.db")
        export_path = str(tmp_path / "export")

        yaml_text = textwrap.dedent(f"""\
            corpus_id: auto-integration
            source_story:
              path: {story_dir}
            story_mapping:
              synthetic_setting:
                name: biotech startup
                organization: NovaBio Inc.
                domain: genetic research
              plot_outline:
                mode: auto
            corpus_size:
              num_instances: 1
            blur_level: medium
            models:
              character_mapping: test-model
              outline_generator: test-model
              planner: test-model
              content_generator: test-model
            llm:
              api_key_env_var: TEST_RCHAT_API_KEY
              base_url: https://rchat.nist.gov/proxy
              max_tokens: 128
            rate_limits:
              max_concurrent_instances: 1
              requests_per_minute: 1000
              wave_pause_seconds: 0
            output:
              db_path: {db_path}
              export_path: {export_path}
        """)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_text)

        skills_dir = tmp_path / "skills"
        for name in ("character-mapping", "email-generation", "orchestration", "outline-generation", "question-extraction"):
            d = skills_dir / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"You are the {name} agent.")

        email_counter = {"n": 0}
        auto_outline_response = {
            "plot_points": [
                {"id": "auto_p1", "description": "Auto event 1"},
                {"id": "auto_p2", "description": "Auto event 2"},
                {"id": "auto_p3", "description": "Auto event 3"},
            ],
        }

        async def mock_complete_json(model, system, user, temperature=0.7, max_tokens=None):
            if "Analyze the source story" in user:
                return auto_outline_response
            if "Map each character" in user:
                return CHARACTER_RESPONSE
            if "## Wave" in user:
                return {
                    "emails": [
                        {
                            "sender_id": "alice",
                            "recipient_ids": ["alice"],
                            "is_reply": False,
                            "parent_mid": None,
                            "plot_point_to_reveal": None,
                            "is_noise": True,
                            "intent": "test",
                        }
                    ],
                    "wave_rationale": "test",
                }
            email_counter["n"] += 1
            return {
                "sender": "a@b.com",
                "recipient": "a@b.com",
                "date": f"2026-01-{email_counter['n']:02d}",
                "message_id": f"<auto-{email_counter['n']}@t>",
                "subject": f"Email {email_counter['n']}",
                "body": f"Body {email_counter['n']}",
                "folder": "\\Sent",
            }

        mock_client = AsyncMock()
        mock_llm = MagicMock()
        mock_llm.complete_json = mock_complete_json
        mock_llm.complete = AsyncMock(return_value="")

        with (
            patch("corpus_agent.main.LLMClient", return_value=mock_llm),
            patch("corpus_agent.main.AsyncOpenAI", return_value=mock_client),
        ):
            await run_pipeline(str(config_file), resume=False, skills_root=str(skills_dir))

        assert (Path(export_path) / "email_corpus.jsonl").exists()
        lines = (Path(export_path) / "email_corpus.jsonl").read_text().strip().split("\n")
        # Mock wave planner returns 1 email per wave.
        # Auto sizing: 3 plot points -> 3 waves. So 3 waves * 1 mock email = 3 lines.
        # This validates the loop runs the auto-derived wave count, not the default 10.
        assert len(lines) == 3


class TestRunPipelineWithQuestions:
    @pytest.mark.asyncio
    async def test_extract_questions_flag(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_RCHAT_API_KEY", "test-key")
        story_dir = tmp_path / "story"
        story_dir.mkdir()
        (story_dir / "story.md").write_text("# Test\nA story.")

        db_path = str(tmp_path / "test.db")
        export_path = str(tmp_path / "export")

        config_text = MINIMAL_YAML.format(
            story_dir=str(story_dir),
            db_path=db_path,
            export_path=export_path,
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_text)

        skills_dir = tmp_path / "skills"
        for name in (
            "character-mapping",
            "email-generation",
            "orchestration",
            "outline-generation",
            "question-extraction",
        ):
            d = skills_dir / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"You are the {name} agent.")

        email_counter = {"n": 0}

        question_response = {
            "questions": [
                {
                    "question": "Who sent the first email?",
                    "question_type": "attribution",
                    "difficulty": 1,
                    "required_plot_points": [],
                    "answer_summary": "Alice",
                    "evidence_emails": ["<1@t>", "<2@t>"],
                },
            ],
        }

        async def mock_complete_json(model, system, user, temperature=0.7, max_tokens=None):
            if "Map each character" in user:
                return CHARACTER_RESPONSE
            if "## Wave" in user:
                return {
                    "emails": [
                        {
                            "sender_id": "alice",
                            "recipient_ids": ["alice"],
                            "is_reply": False,
                            "parent_mid": None,
                            "plot_point_to_reveal": None,
                            "is_noise": True,
                            "intent": "test",
                        }
                    ],
                    "wave_rationale": "test",
                }
            if "## Question Taxonomy" in user:
                return question_response
            email_counter["n"] += 1
            return {
                "sender": "a@b.com",
                "recipient": "a@b.com",
                "date": f"2026-01-{email_counter['n']:02d}",
                "message_id": f"<{email_counter['n']}@t>",
                "subject": f"Email {email_counter['n']}",
                "body": f"Body {email_counter['n']}",
                "folder": "\\Sent",
            }

        mock_client = AsyncMock()
        mock_llm = MagicMock()
        mock_llm.complete_json = mock_complete_json
        mock_llm.complete = AsyncMock(return_value="")

        with (
            patch("corpus_agent.main.LLMClient", return_value=mock_llm),
            patch("corpus_agent.main.AsyncOpenAI", return_value=mock_client),
        ):
            await run_pipeline(
                str(config_file),
                resume=False,
                skills_root=str(skills_dir),
                extract_questions=True,
            )

        assert (Path(export_path) / "email_corpus.jsonl").exists()
        assert (Path(export_path) / "questions.json").exists()
        questions_data = json.loads(
            (Path(export_path) / "questions.json").read_text()
        )
        assert len(questions_data) == 2  # 2 instances
        for instance_data in questions_data:
            assert "instance_id" in instance_data
            assert "questions" in instance_data
            assert "coverage" in instance_data
