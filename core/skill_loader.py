"""Skill discovery — scan ``skills/**/SKILL.md`` and parse frontmatter."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _parse_frontmatter(text: str) -> Dict | None:
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        data = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def load_skill_index(skills_dir: Path | None = None) -> List[Dict[str, str]]:
    """Return ``[{name, description, path}, ...]`` sorted by name.

    Scans ``SKILL.md`` files with YAML frontmatter containing at least ``name``.
    """
    root = skills_dir or SKILLS_DIR
    results: List[Dict[str, str]] = []
    if not root.exists():
        return results
    for path in root.rglob("SKILL.md"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        meta = _parse_frontmatter(text)
        if not meta:
            continue
        name = meta.get("name")
        if not name:
            continue
        description = str(meta.get("description") or "").strip()
        results.append({
            "name": str(name),
            "description": description,
            "path": str(path.relative_to(root.parent)),
        })
    results.sort(key=lambda s: s["name"])
    return results
