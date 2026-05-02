"""
skills.py — AgentSkills.io compatible skill registry.

Scans for directories containing SKILL.md files per the agentskills.io
open standard. Parses frontmatter for discovery, loads full content
for activation. Skills are portable across Claude Code, Cursor,
Gemini CLI, VS Code Copilot, and 30+ other agent tools.

Spec: https://agentskills.io/specification
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agent_harness.skills")

SKILLS_DIR = Path(__file__).parent / "skills"


def parse_frontmatter(path: Path) -> Optional[dict]:
    import yaml
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", text, re.DOTALL)
    if not match:
        return None
    fm = yaml.safe_load(match.group(1))
    if not fm or not fm.get("description"):
        return None
    return fm


class SkillRegistry:
    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self._skills: dict[str, dict] = {}
        self._add_scan_dir(skills_dir)

    def add_project_skills(self, workspace: Path) -> None:
        self._add_scan_dir(workspace / ".agents" / "skills")

    def add_user_skills(self) -> None:
        self._add_scan_dir(Path.home() / ".agents" / "skills")

    def _add_scan_dir(self, d: Path) -> None:
        if not d.exists() or not d.is_dir():
            return
        for entry in sorted(d.iterdir()):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.exists():
                continue
            fm = parse_frontmatter(skill_md)
            if fm is None:
                logger.warning("Skipping skill at %s: invalid frontmatter", entry)
                continue
            name = fm.get("name", entry.name)
            if name in self._skills:
                logger.debug("Skill '%s' already loaded, skipping %s", name, entry)
                continue
            self._skills[name] = {
                "description": fm["description"],
                "path": skill_md,
                "dir": entry,
                "allowed_tools": fm.get("allowed-tools", ""),
                "compatibility": fm.get("compatibility", ""),
                "metadata": fm.get("metadata", {}),
            }
            logger.info("Discovered skill: %s", name)

    def list_skills(self) -> list[dict]:
        return [{"name": n, "description": s["description"]}
                for n, s in self._skills.items()]

    def load_skill(self, name: str) -> Optional[str]:
        skill = self._skills.get(name)
        if not skill:
            return None
        return skill["path"].read_text(encoding="utf-8")

    def get(self, name: str) -> Optional[dict]:
        return self._skills.get(name)

    def create_from_content(self, name: str, content: str) -> None:
        skill_dir = SKILLS_DIR / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(content, encoding="utf-8")
        fm = parse_frontmatter(skill_md)
        if fm:
            self._skills[name] = {
                "description": fm["description"],
                "path": skill_md,
                "dir": skill_dir,
                "allowed_tools": fm.get("allowed-tools", ""),
                "compatibility": fm.get("compatibility", ""),
                "metadata": fm.get("metadata", {}),
            }
            logger.info("Created skill: %s", name)

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills
