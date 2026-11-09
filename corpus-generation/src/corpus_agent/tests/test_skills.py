import pytest

from corpus_agent.skills import SKILL_NAMES, load_all_skills, load_skill


def test_load_skill_reads_file(tmp_path):
    skill_dir = tmp_path / "character-mapping"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Character Mapping\nMap characters.")

    content = load_skill("character-mapping", skills_root=str(tmp_path))
    assert "Map characters." in content


def test_load_skill_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_skill("nonexistent", skills_root=str(tmp_path))


def test_load_all_skills(tmp_path):
    for name in SKILL_NAMES:
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(f"# {name}\nContent for {name}.")

    skills = load_all_skills(skills_root=str(tmp_path))
    assert set(skills.keys()) == set(SKILL_NAMES)
    for name in SKILL_NAMES:
        assert f"Content for {name}." in skills[name]


def test_load_all_skills_includes_outline_generation(tmp_path):
    for name in SKILL_NAMES:
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(f"# {name}\nContent for {name}.")

    skills = load_all_skills(skills_root=str(tmp_path))
    assert "outline-generation" in skills
    assert "Content for outline-generation." in skills["outline-generation"]


def test_load_skill_empty_file_returns_empty_string(tmp_path):
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("")
    result = load_skill("test-skill", skills_root=str(tmp_path))
    assert result == ""


def test_skill_names_includes_question_extraction():
    assert "question-extraction" in SKILL_NAMES

