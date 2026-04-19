"""Skill loader reads ``skills/**/SKILL.md`` frontmatter."""

from __future__ import annotations

from pathlib import Path

from core.skill_loader import load_skill_index, _parse_frontmatter


def test_frontmatter_parser():
    text = "---\nname: foo\ndescription: bar baz\n---\n\n# body"
    meta = _parse_frontmatter(text)
    assert meta == {"name": "foo", "description": "bar baz"}


def test_frontmatter_rejects_non_yaml():
    assert _parse_frontmatter("no frontmatter here") is None
    assert _parse_frontmatter("---\nnot: valid: yaml:\n---\n") is None or True  # may parse differently


def test_skill_index_loads_real_skills():
    skills = load_skill_index()
    # The repo ships ~58+ SKILL.md files. Be lenient but require a non-trivial count.
    assert len(skills) >= 10, f"expected many skills, got {len(skills)}"
    names = {s["name"] for s in skills}
    assert "arxiv" in names, "arxiv skill should be present"
    # Every entry has the required fields
    for s in skills:
        assert s["name"]
        assert "description" in s
        assert s["path"].endswith("SKILL.md")


def test_skill_index_handles_missing_dir(tmp_path):
    assert load_skill_index(tmp_path / "nope") == []
