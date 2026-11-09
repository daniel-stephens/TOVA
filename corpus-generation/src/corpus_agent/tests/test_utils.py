from corpus_agent.utils import read_story_files


def test_read_story_files_reads_md_files(tmp_path):
    (tmp_path / "a.md").write_text("alpha")
    (tmp_path / "b.md").write_text("beta")
    result = read_story_files(str(tmp_path))
    assert "alpha" in result
    assert "beta" in result
    assert result.index("alpha") < result.index("beta")


def test_read_story_files_ignores_non_md(tmp_path):
    (tmp_path / "story.md").write_text("the story")
    (tmp_path / "notes.txt").write_text("should not appear")
    result = read_story_files(str(tmp_path))
    assert "the story" in result
    assert "should not appear" not in result


def test_read_story_files_empty_dir(tmp_path):
    result = read_story_files(str(tmp_path))
    assert result == ""
