import textwrap
import warnings

import pytest
from pydantic import ValidationError

from corpus_agent.config import load_config


MINIMAL_YAML = textwrap.dedent(
    """\
    corpus_id: test-corpus-v1

    source_story:
      path: data/raw/synthetic-story/jurassic-park/

    synthetic_setting:
      name: biotech startup
      organization: NovaBio Inc.
      domain: genetic research

    corpus_size:
      num_instances: 5
      waves_per_instance: 3
      emails_per_wave: 4
      emails_per_instance: 30

    blur_level: medium

    plot_outline:
      - id: inciting_incident
        description: "Handler killed during specimen transfer"
      - id: espionage
        description: "Rival bribes employee to steal samples"

    models:
      character_mapping: NVIDIA-Nemotron-3-Super-120B-A12B-FP8
      planner: NVIDIA-Nemotron-3-Super-120B-A12B-FP8
      content_generator: NVIDIA-Nemotron-3-Super-120B-A12B-FP8

    llm:
      api_key_env_var: RCHAT_API_KEY
      base_url: https://rchat.nist.gov/proxy
      max_tokens: 3000

    rate_limits:
      max_concurrent_instances: 3
      requests_per_minute: 60

    output:
      db_path: data/corpus_state.db
      export_path: data/processed/synthetic/test-corpus-v1/
"""
)


def test_load_config_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(MINIMAL_YAML)
    with pytest.warns(DeprecationWarning):
        cfg = load_config(str(config_file))
    assert cfg.corpus_id == "test-corpus-v1"
    assert cfg.corpus_size.num_instances == 5
    assert cfg.blur_level == "medium"
    assert len(cfg.story_mapping.plot_outline.points) == 2
    assert cfg.story_mapping.plot_outline.points[0].id == "inciting_incident"
    assert cfg.models.planner == "NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    assert cfg.llm.api_key_env_var == "RCHAT_API_KEY"
    assert cfg.llm.base_url == "https://rchat.nist.gov/proxy"
    assert cfg.llm.max_tokens == 3000
    assert cfg.rate_limits.max_concurrent_instances == 3


def test_config_defaults(tmp_path):
    yaml_text = textwrap.dedent(
        """\
        corpus_id: minimal
        source_story:
          path: some/path
        synthetic_setting:
          name: corporate
          organization: Acme
          domain: finance
        corpus_size:
          num_instances: 1
        blur_level: light
        plot_outline:
          - id: p1
            description: first thing
        output:
          db_path: data/corpus_state.db
          export_path: out/
    """
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_text)
    with pytest.warns(DeprecationWarning):
        cfg = load_config(str(config_file))
    assert cfg.corpus_size.waves_per_instance == 10
    assert cfg.corpus_size.emails_per_wave == 4
    assert cfg.corpus_size.emails_per_instance == 100
    assert cfg.rate_limits.max_concurrent_instances == 5
    assert cfg.rate_limits.requests_per_minute == 60
    assert cfg.rate_limits.retry_max_attempts == 3
    assert cfg.rate_limits.retry_initial_delay_seconds == 2.0
    assert cfg.rate_limits.wave_pause_seconds == 1.0
    assert cfg.llm.api_key_env_var == "RCHAT_API_KEY"
    assert cfg.llm.base_url == "https://rchat.nist.gov/proxy"
    assert cfg.llm.max_tokens == 3000
    assert cfg.models.character_mapping == "NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    assert cfg.models.planner == "NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    assert cfg.models.content_generator == "NVIDIA-Nemotron-3-Super-120B-A12B-FP8"


def test_config_rejects_invalid_blur_level(tmp_path):
    yaml_text = textwrap.dedent(
        """\
        corpus_id: bad
        source_story:
          path: x
        synthetic_setting:
          name: y
          organization: z
          domain: w
        corpus_size:
          num_instances: 1
        blur_level: extreme
        plot_outline:
          - id: p1
            description: d1
        models:
          character_mapping: m1
          planner: m2
          content_generator: m3
        output:
          db_path: d
          export_path: e
    """
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_text)
    with pytest.warns(DeprecationWarning), pytest.raises(ValidationError):
        load_config(str(config_file))


def test_new_story_mapping_manual_mode(tmp_path):
    yaml_text = textwrap.dedent("""\
        corpus_id: new-format
        source_story:
          path: some/path
        story_mapping:
          synthetic_setting:
            name: biotech startup
            organization: NovaBio
            domain: genetics
          plot_outline:
            mode: manual
            points:
              - id: p1
                description: first event
              - id: p2
                description: second event
        corpus_size:
          num_instances: 1
        blur_level: medium
        output:
          db_path: d
          export_path: e
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_text)
    cfg = load_config(str(config_file))
    assert cfg.story_mapping.synthetic_setting.name == "biotech startup"
    assert cfg.story_mapping.plot_outline.mode == "manual"
    assert len(cfg.story_mapping.plot_outline.points) == 2
    assert cfg.story_mapping.plot_outline.points[0].id == "p1"


def test_new_story_mapping_auto_mode(tmp_path):
    yaml_text = textwrap.dedent("""\
        corpus_id: auto-format
        source_story:
          path: some/path
        story_mapping:
          synthetic_setting:
            name: law firm
            organization: Sterling & Associates
            domain: corporate law
          plot_outline:
            mode: auto
        corpus_size:
          num_instances: 1
        blur_level: heavy
        output:
          db_path: d
          export_path: e
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_text)
    cfg = load_config(str(config_file))
    assert cfg.story_mapping.plot_outline.mode == "auto"
    assert cfg.story_mapping.plot_outline.points == []


def test_legacy_config_migrates_to_story_mapping(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(MINIMAL_YAML)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_config(str(config_file))
    assert cfg.story_mapping.synthetic_setting.name == "biotech startup"
    assert cfg.story_mapping.plot_outline.mode == "manual"
    assert len(cfg.story_mapping.plot_outline.points) == 2
    assert any("deprecated" in str(warning.message).lower() for warning in w)


def test_auto_mode_rejects_missing_points_in_manual(tmp_path):
    yaml_text = textwrap.dedent("""\
        corpus_id: bad
        source_story:
          path: x
        story_mapping:
          synthetic_setting:
            name: y
            organization: z
            domain: w
          plot_outline:
            mode: manual
        corpus_size:
          num_instances: 1
        blur_level: light
        output:
          db_path: d
          export_path: e
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_text)
    with pytest.raises(ValidationError):
        load_config(str(config_file))


def test_story_mapping_takes_precedence_over_legacy(tmp_path):
    yaml_text = textwrap.dedent("""\
        corpus_id: both
        source_story:
          path: some/path
        story_mapping:
          synthetic_setting:
            name: new-setting
            organization: NewOrg
            domain: new-domain
          plot_outline:
            mode: manual
            points:
              - id: p1
                description: from story_mapping
        synthetic_setting:
          name: old-setting
          organization: OldOrg
          domain: old-domain
        plot_outline:
          - id: old_p1
            description: from legacy
        corpus_size:
          num_instances: 1
        blur_level: light
        output:
          db_path: d
          export_path: e
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_text)
    cfg = load_config(str(config_file))
    assert cfg.story_mapping.synthetic_setting.name == "new-setting"
    assert cfg.story_mapping.plot_outline.points[0].description == "from story_mapping"


def test_auto_mode_accepts_total_emails(tmp_path):
    yaml_text = textwrap.dedent("""\
        corpus_id: auto-total
        source_story:
          path: some/path
        story_mapping:
          synthetic_setting:
            name: biotech startup
            organization: NovaBio
            domain: genetics
          plot_outline:
            mode: auto
        corpus_size:
          total_emails: 500
        blur_level: medium
        output:
          db_path: d
          export_path: e
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_text)
    cfg = load_config(str(config_file))
    assert cfg.corpus_size.total_emails == 500
    assert cfg.corpus_size.num_instances is None


def test_manual_mode_ignores_total_emails(tmp_path):
    yaml_text = textwrap.dedent("""\
        corpus_id: manual-total
        source_story:
          path: some/path
        story_mapping:
          synthetic_setting:
            name: biotech startup
            organization: NovaBio
            domain: genetics
          plot_outline:
            mode: manual
            points:
              - id: p1
                description: first event
        corpus_size:
          num_instances: 5
          total_emails: 999
        blur_level: medium
        output:
          db_path: d
          export_path: e
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_text)
    cfg = load_config(str(config_file))
    assert cfg.corpus_size.total_emails == 999
    assert cfg.corpus_size.num_instances == 5


def test_total_emails_rejects_zero(tmp_path):
    yaml_text = textwrap.dedent("""\
        corpus_id: bad
        source_story:
          path: x
        story_mapping:
          synthetic_setting:
            name: y
            organization: z
            domain: w
          plot_outline:
            mode: auto
        corpus_size:
          total_emails: 0
        blur_level: medium
        output:
          db_path: d
          export_path: e
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_text)
    with pytest.raises(ValidationError):
        load_config(str(config_file))


def test_total_emails_rejects_negative(tmp_path):
    yaml_text = textwrap.dedent("""\
        corpus_id: bad
        source_story:
          path: x
        story_mapping:
          synthetic_setting:
            name: y
            organization: z
            domain: w
          plot_outline:
            mode: auto
        corpus_size:
          total_emails: -5
        blur_level: medium
        output:
          db_path: d
          export_path: e
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_text)
    with pytest.raises(ValidationError):
        load_config(str(config_file))


def test_config_has_question_extractor_model_default(tmp_path):
    yaml_text = textwrap.dedent("""\
        corpus_id: qe-test
        source_story:
          path: some/path
        story_mapping:
          synthetic_setting:
            name: biotech startup
            organization: NovaBio
            domain: genetics
          plot_outline:
            mode: manual
            points:
              - id: p1
                description: first event
        corpus_size:
          num_instances: 1
        blur_level: medium
        output:
          db_path: d
          export_path: e
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_text)
    cfg = load_config(str(config_file))
    assert cfg.models.question_extractor == "NVIDIA-Nemotron-3-Super-120B-A12B-FP8"


def test_config_overrides_question_extractor_model(tmp_path):
    yaml_text = textwrap.dedent("""\
        corpus_id: qe-test
        source_story:
          path: some/path
        story_mapping:
          synthetic_setting:
            name: biotech startup
            organization: NovaBio
            domain: genetics
          plot_outline:
            mode: manual
            points:
              - id: p1
                description: first event
        corpus_size:
          num_instances: 1
        blur_level: medium
        models:
          question_extractor: custom-model-v2
        output:
          db_path: d
          export_path: e
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_text)
    cfg = load_config(str(config_file))
    assert cfg.models.question_extractor == "custom-model-v2"
