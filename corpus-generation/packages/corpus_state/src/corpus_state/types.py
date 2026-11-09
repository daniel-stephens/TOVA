from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Character(BaseModel):
    id: str
    corpus_id: str = "default"
    wire_character: str
    synthetic_character: str
    email_address: str
    role: str
    organization: str
    personality: str
    voice_anchor: str                        # writing-style rules that prevent persona drift
    emotional_state: str = "neutral"
    blur_rules: dict[str, str] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_row(cls, data: object) -> object:
        if hasattr(data, "keys"):  # sqlite3.Row duck-type
            data = dict(data)
            blur = data.get("blur_rules") or "{}"
            data["blur_rules"] = json.loads(blur) if isinstance(blur, str) else blur
        return data


class Email(BaseModel):
    mid: Optional[int] = None
    sender: str
    recipient: str                           # may be comma-separated for multiple recipients
    date: str                                # ISO 8601
    message_id: str
    subject: str
    body: str
    folder: str = "\\Inbox"
    parent_mid: Optional[int] = None         # None = root of thread
    instance_id: Optional[int] = None


class EmailInput(BaseModel):
    """Validated input for save_email — enforces schema_email.csv format."""
    sender: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    date: str = Field(min_length=1)
    message_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)
    folder: str = "\\Inbox"
    parent_mid: Optional[int] = None
    instance_id: Optional[int] = None


class PlotPoint(BaseModel):
    id: str
    corpus_id: str = "default"
    description: str
    first_email_mid: Optional[int] = None
    revealed_at: Optional[str] = None


class DivergenceRule(BaseModel):
    id: Optional[int] = None
    corpus_id: str = "default"
    source_archetype: str                    # e.g. "Mad Scientist"
    synthetic_archetype: str                 # e.g. "Bureaucratic Perfectionist"
    rule_description: str
    created_at: Optional[str] = None


class Relationship(BaseModel):
    char_a: str
    char_b: str
    corpus_id: str = "default"
    strength: int = Field(ge=1, le=10)       # 1–10
    status: str                              # e.g. "trusted", "adversarial", "dormant"


class StoryProgress(BaseModel):
    plot_points: list[PlotPoint]
    divergence_rules: list[DivergenceRule]


class ResolvedStoryMapping(BaseModel):
    corpus_id: str
    setting_name: str
    setting_organization: str
    setting_domain: str
    blur_level: str
    mode: str  # "auto" or "manual"
    plot_points: list[dict]
    created_at: str = ""

    @model_validator(mode="after")
    def _validate_plot_points(self) -> "ResolvedStoryMapping":
        for i, p in enumerate(self.plot_points):
            if not isinstance(p, dict) or "id" not in p or "description" not in p:
                raise ValueError(
                    f"plot_points[{i}] must have 'id' and 'description' keys, "
                    f"got: {sorted(p.keys()) if isinstance(p, dict) else type(p).__name__}"
                )
            if not isinstance(p["id"], str) or not isinstance(p["description"], str):
                raise ValueError(
                    f"plot_points[{i}] 'id' and 'description' must be strings"
                )
        return self


class Instance(BaseModel):
    instance_id: int
    corpus_id: str = "default"
    status: str = "pending"  # pending | in_progress | completed
    num_emails: int = 0
    metadata: dict = Field(default_factory=dict)
    created_at: str = ""
    completed_at: Optional[str] = None


class BulkEmailInput(BaseModel):
    """Input for bulk email save — a list of emails all tagged to one instance."""
    instance_id: int
    emails: list[EmailInput]
