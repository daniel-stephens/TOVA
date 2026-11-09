from __future__ import annotations

from pathlib import Path

SKILL_NAMES = ("character-mapping", "email-generation", "orchestration", "outline-generation", "question-extraction")

_DEFAULT_SKILLS_ROOT = str(Path(__file__).parents[2] / "skills")


def load_skill(name: str, skills_root: str = _DEFAULT_SKILLS_ROOT) -> str:
    path = Path(skills_root) / name / "SKILL.md"
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_all_skills(skills_root: str = _DEFAULT_SKILLS_ROOT) -> dict[str, str]:
    return {name: load_skill(name, skills_root) for name in SKILL_NAMES}
