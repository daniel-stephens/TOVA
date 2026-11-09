from __future__ import annotations

import warnings
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class SourceStory(BaseModel):
    path: str


class SyntheticSetting(BaseModel):
    name: str
    organization: str
    domain: str


class CorpusSize(BaseModel):
    num_instances: int | None = None
    total_emails: int | None = Field(default=None, ge=1)
    waves_per_instance: int = 10
    emails_per_wave: int = 4
    emails_per_instance: int = 100


class PlotPoint(BaseModel):
    id: str
    description: str


class PlotOutlineConfig(BaseModel):
    mode: Literal["auto", "manual"] = "manual"
    points: list[PlotPoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _manual_requires_points(self) -> PlotOutlineConfig:
        if self.mode == "manual" and not self.points:
            raise ValueError("plot_outline.points is required when mode='manual'")
        return self


class StoryMapping(BaseModel):
    synthetic_setting: SyntheticSetting
    plot_outline: PlotOutlineConfig


class Models(BaseModel):
    character_mapping: str = "NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    outline_generator: str = "NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    planner: str = "NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    content_generator: str = "NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    question_extractor: str = "NVIDIA-Nemotron-3-Super-120B-A12B-FP8"


class LLMConfig(BaseModel):
    api_key_env_var: str = "RCHAT_API_KEY"
    base_url: str = "https://rchat.nist.gov/proxy"
    max_tokens: int = 3000


class RateLimits(BaseModel):
    max_concurrent_instances: int = 5
    requests_per_minute: int = 60
    retry_max_attempts: int = 3
    retry_initial_delay_seconds: float = 2.0
    wave_pause_seconds: float = 1.0


class Output(BaseModel):
    db_path: str
    export_path: str


class CorpusConfig(BaseModel):
    corpus_id: str
    source_story: SourceStory
    story_mapping: StoryMapping
    corpus_size: CorpusSize
    blur_level: Literal["light", "medium", "heavy"]
    # Extra writing instructions for this corpus. Empty keeps the default email voice.
    content_style: str = ""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    models: Models = Field(default_factory=Models)
    rate_limits: RateLimits = Field(default_factory=RateLimits)
    output: Output

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "story_mapping" in data:
            data.pop("synthetic_setting", None)
            data.pop("plot_outline", None)
            return data
        ss = data.get("synthetic_setting")
        po = data.get("plot_outline")
        if ss is not None and po is not None:
            data.pop("synthetic_setting")
            data.pop("plot_outline")
            warnings.warn(
                "Top-level 'synthetic_setting' and 'plot_outline' are deprecated. "
                "Use 'story_mapping' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["story_mapping"] = {
                "synthetic_setting": ss,
                "plot_outline": {"mode": "manual", "points": po},
            }
            return data
        missing = []
        if ss is None:
            missing.append("'synthetic_setting'")
        if po is None:
            missing.append("'plot_outline'")
        raise ValueError(
            f"Config must provide 'story_mapping', or both 'synthetic_setting' and "
            f"'plot_outline' (deprecated). Missing: {', '.join(missing)}"
        )


def load_config(path: str) -> CorpusConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CorpusConfig(**data)
