# AgentSkills.io Integration Spec

## v2 — Updated after code review (9 findings addressed)

## Goal

Replace the custom YAML+JSON skill format with the agentskills.io open standard. Skills become portable across Claude Code, Cursor, Gemini CLI, VS Code Copilot, and 30+ other agent tools.

## Current State

```
skills/
  research.yaml + research.json        # custom format
  code_review.yaml + code_review.json   # custom format
  ...
```

Executed by `SkillEngine` (skills.py, ~380 lines) — declarative step engine with loops, retries, schema validation.

## Target State

```
skills/                                 # harness-local skills
  research/
    SKILL.md
  code-review/
    SKILL.md
    references/
      checklist.md
.agents/skills/                         # cross-client shared skills (agentskills.io convention)
  github-pr/
    SKILL.md
    scripts/
      diff_parser.py
```

Executed by the LLM itself — reads SKILL.md instructions, uses tools at its own discretion.

---

## SKILL.md Format (per agentskills.io spec)

```markdown
---
name: code-review
description: Reviews code files for bugs, security issues, and best practices. Use when asked to review, audit, or check code quality.
allowed-tools: read_file list_files shell
compatibility: Requires access to the project workspace
metadata:
  author: simple-harness
  version: "1.0"
  mcp-servers: ""
---

## Instructions

1. Read the target file using read_file
2. Analyze for:
   - Bugs and logic errors
   - Security vulnerabilities
   - Style and readability issues
   - Performance concerns
3. Provide a structured report with specific line references

## Output Format

Structure your review as:
- **Summary**: one paragraph overview
- **Bugs**: numbered list with severity
- **Security**: any vulnerabilities found
- **Suggestions**: actionable improvements
- **Rating**: Good / Needs Work / Critical Issues
```

### Frontmatter Fields

| Field | Required | Our Usage |
|-------|----------|-----------|
| `name` | Yes | Skill identifier. Lowercase, hyphens, 1-64 chars. Must match directory name |
| `description` | Yes | What it does + when to use it. Max 1024 chars. Used for discovery |
| `allowed-tools` | No | Space-separated tool names. Advisory — injected into context, not enforced |
| `compatibility` | No | Environment requirements. Checked before activation |
| `metadata` | No | Arbitrary key-value. Custom extension: `mcp-servers` for MCP dependencies |
| `license` | No | Passthrough, not interpreted by harness |

### Name Validation Rules (per agentskills.io spec)

- 1-64 characters
- Lowercase alphanumeric + hyphens only
- Must not start or end with hyphen
- Must not contain consecutive hyphens
- Must match parent directory name

Enforced in `create()`. Lenient in `_discover()` (warn and load anyway for third-party skills).

### Custom Metadata Keys

| Key | Purpose | Note |
|-----|---------|------|
| `mcp-servers` | Space-separated MCP server names required | Harness-specific extension. Other agentskills.io clients will ignore this field |
| `author` | Skill creator | Standard convention |
| `version` | Skill version string | Standard convention |

---

## Skill Discovery Paths

Per agentskills.io cross-client convention, scan multiple locations:

| Path | Priority | Purpose |
|------|----------|---------|
| `skills/` (relative to harness) | 1 (highest) | Harness-local skills |
| `<workspace>/.agents/skills/` | 2 | Project-level shared skills |
| `~/.agents/skills/` | 3 (lowest) | User-level shared skills |

Name collision rule: higher priority wins. Duplicates from lower-priority paths are skipped with a debug log.

---

## Progressive Disclosure

Three stages, per agentskills.io spec:

### 1. Discovery (startup)

Scan all skill paths for directories containing `SKILL.md`. Parse ONLY the frontmatter (name + description). Skip skills with empty descriptions (log warning).

Cost: ~100 tokens per skill (just name + description in the tool schema).

### 2. Activation (on use)

When the LLM calls `run_skill("code-review")` or user types `/code-review`, load the FULL SKILL.md body. Return it as the **tool result** (not system message — see Finding #1).

Cost: full SKILL.md content (recommended <5000 tokens per agentskills.io).

### 3. Execution (as needed)

The LLM follows instructions, calling tools as specified. If it needs a reference file (e.g., `references/checklist.md`), it can read it via `read_file` tool.

Scripts in `scripts/` can be executed via `shell` or `python_exec` tools.

---

## RunSkillTool (rewrite)

**Key design decision (Finding #1):** Return skill content as the tool result, NOT as an injected system message. Injecting a system message between assistant tool_call and tool result breaks the message ordering contract for OpenAI-compatible APIs.

```python
class RunSkillTool(RunnableTool):
    name = "run_skill"
    description = (
        "Activate a skill to guide your next actions. "
        "The skill provides expert instructions for a specific task. "
        "Input: skill_name (str), context (str) — optional additional context."
    )
    execution_mode = "sync"
    timeout_seconds = 5.0

    def __init__(self, registry, mcp_manager=None):
        self._registry = registry
        self._mcp = mcp_manager
        self._activated = set()    # Finding #8: track activated skills

    def run(self, skill_name="", context="", **kwargs):
        skill_name = skill_name or kwargs.get("name", "")
        context = context or kwargs.get("task", kwargs.get("input", ""))

        skill = self._registry.get(skill_name)
        if not skill:
            available = [s["name"] for s in self._registry.list_skills()]
            return f"Skill '{skill_name}' not found. Available: {available}"

        # Finding #8: skip duplicate activation
        if skill_name in self._activated:
            return f"Skill '{skill_name}' is already active. Follow the instructions previously provided."

        # Check MCP compatibility
        mcp_servers = skill.get("metadata", {}).get("mcp-servers", "")
        if mcp_servers and self._mcp:
            status = self._mcp.status()
            servers = status.get("servers", {})
            required = mcp_servers.split()
            missing = [s for s in required if s not in servers or not servers[s].get("alive")]
            if missing:
                return f"Skill requires MCP servers not connected: {missing}"

        # Load full SKILL.md
        content = self._registry.load_skill(skill_name)
        if not content:
            return f"Could not load skill '{skill_name}'"

        self._activated.add(skill_name)

        # Return as tool result (Finding #1: NOT system message)
        result = f"=== SKILL: {skill_name} ===\n\n"
        result += content
        if skill.get("allowed_tools"):
            result += f"\n\nApproved tools for this skill: {skill['allowed_tools']}"
        if context:
            result += f"\n\nUser context: {context}"
        result += "\n\nFollow these instructions now using the approved tools."

        return result
```

### Activation Reset

`self._activated` is cleared when:
- The REPL moves to the next prompt (agent._history.clear())
- New instance created per prompt

---

## User-Explicit Activation (Finding #9)

The REPL intercepts `/skill-name` commands:

```python
# In repl(), before sending to agent loop:
if prompt.startswith("/"):
    skill_name = prompt.lstrip("/").strip()
    if skill_name in registry:
        content = registry.load_skill(skill_name)
        if content:
            # Inject as user message so LLM sees it naturally
            agent._history.append({"role": "user", "content": f"Activate and follow this skill:\n\n{content}"})
            prompt = ""  # skip normal prompt, let LLM follow skill
```

This means:
- `/research` → activates the research skill
- `/code-review` → activates code review
- `/exit` → exits (existing behavior, unchanged)

---

## CreateSkillTool (rewrite)

Uses `yaml.dump()` for safe frontmatter generation (Finding #2). Validates name (Finding #3).

```python
class CreateSkillTool(RunnableTool):
    name = "create_skill"
    description = (
        "Create a new reusable skill following the agentskills.io format. "
        "Input: skill_name (str) — lowercase with hyphens, "
        "description (str) — what it does and when to use it, "
        "instructions (str) — markdown instructions the agent should follow."
    )
    execution_mode = "sync"
    timeout_seconds = 5.0

    _NAME_PATTERN = re.compile(r'^[a-z0-9]([a-z0-9-]*[a-z0-9])?$')

    def __init__(self, registry):
        self._registry = registry

    def run(self, skill_name="", description="", instructions="", **kwargs):
        skill_name = skill_name or kwargs.get("name", "")
        description = description or kwargs.get("desc", "")
        instructions = instructions or kwargs.get("body", kwargs.get("content", ""))

        if not skill_name:
            return "Error: 'skill_name' is required (lowercase, hyphens only)"
        if not description:
            return "Error: 'description' is required"
        if not instructions:
            return "Error: 'instructions' is required"

        # Finding #3: validate name
        if len(skill_name) > 64 or not self._NAME_PATTERN.match(skill_name) or '--' in skill_name:
            return f"Error: invalid skill name '{skill_name}'. Must be 1-64 chars, lowercase alphanumeric + hyphens."

        # Finding #2: safe YAML generation
        import yaml
        frontmatter = yaml.dump({
            "name": skill_name,
            "description": description,
            "metadata": {"author": "ai-generated", "version": "1.0"},
        }, default_flow_style=False, allow_unicode=True)

        content = f"---\n{frontmatter}---\n\n{instructions}\n"

        self._registry.create_from_content(skill_name, content)
        return f"Skill '{skill_name}' created at skills/{skill_name}/SKILL.md"
```

---

## SkillRegistry (rewrite in skills.py)

```python
"""
skills.py — AgentSkills.io compatible skill registry.

Scans for directories containing SKILL.md files.
Parses frontmatter for discovery, loads full content for activation.
"""

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agent_harness.skills")

SKILLS_DIR = Path(__file__).parent / "skills"


def parse_frontmatter(path: Path) -> Optional[dict]:
    """Parse YAML frontmatter from a SKILL.md file."""
    import yaml
    text = path.read_text(encoding="utf-8")
    # Finding #5: trailing newline optional
    match = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", text, re.DOTALL)
    if not match:
        return None
    fm = yaml.safe_load(match.group(1))
    if not fm or not fm.get("description"):
        return None   # Finding #5: skip skills with empty description
    return fm


class SkillRegistry:
    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self._dirs = []
        self._skills: dict[str, dict] = {}
        self._add_scan_dir(skills_dir)

    def add_project_skills(self, workspace: Path) -> None:
        """Finding #7: scan .agents/skills/ at project level."""
        self._add_scan_dir(workspace / ".agents" / "skills")

    def add_user_skills(self) -> None:
        """Finding #7: scan ~/.agents/skills/ at user level."""
        self._add_scan_dir(Path.home() / ".agents" / "skills")

    def _add_scan_dir(self, d: Path) -> None:
        if d.exists() and d.is_dir():
            self._dirs.append(d)
            self._scan_dir(d)

    def _scan_dir(self, d: Path) -> None:
        for entry in sorted(d.iterdir()):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.exists():
                continue
            fm = parse_frontmatter(skill_md)
            if fm is None:
                logger.warning("Skipping skill at %s: invalid or empty frontmatter", entry)
                continue
            name = fm.get("name", entry.name)
            if name in self._skills:
                logger.debug("Skill '%s' already loaded, skipping duplicate from %s", name, entry)
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

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills
```

---

## Context Protection (Finding #6)

Skill content returned as tool result is wrapped in structured tags to aid context management:

```
=== SKILL: code-review ===
[SKILL_INSTRUCTIONS_START]
{SKILL.md content}
[SKILL_INSTRUCTIONS_END]
```

The harness does not currently do context compaction (LivingSession handles it), but if added in the future, any content between `SKILL_INSTRUCTIONS_START` and `SKILL_INSTRUCTIONS_END` tags should be preserved.

---

## Skill Migration

### research/SKILL.md

```markdown
---
name: research
description: Search the web for a topic and provide a comprehensive summary with sources. Use when asked to research, investigate, or find information about any subject.
allowed-tools: web_search http_fetch
---

## Instructions

1. Use web_search to find relevant information about the given topic
2. Search with 2-3 different query variations for comprehensive coverage
3. If specific articles look promising, use http_fetch to get more detail
4. Summarize findings in a structured format:
   - **Key findings** (bullet points)
   - **Sources** (title + URL for each)
   - **Gaps** (note what couldn't be found or conflicting information)
```

### code-review/SKILL.md

```markdown
---
name: code-review
description: Reviews source code files for bugs, security issues, style problems, and best practices. Use when asked to review, audit, or check code quality.
allowed-tools: read_file list_files
---

## Instructions

1. Read the target file(s) using read_file
2. Analyze the code for:
   - **Bugs**: logic errors, off-by-one, null handling, race conditions
   - **Security**: injection, hardcoded secrets, unsafe operations
   - **Style**: naming, formatting, readability
   - **Performance**: unnecessary allocations, O(n²) where O(n) is possible
3. Cite specific line numbers or code snippets for each finding
4. Provide an overall rating: Good / Needs Work / Critical Issues
5. List actionable fix suggestions with priority (P0/P1/P2)
```

### file-stats/SKILL.md

```markdown
---
name: file-stats
description: Analyze a directory structure showing file counts, line counts, and project summary. Use when asked about project size, structure, or file statistics.
allowed-tools: list_files shell
---

## Instructions

1. Use list_files to get the directory contents
2. Use shell to count lines per file type
3. Summarize: total files by type, line counts, overall project size, notable patterns
```

### multi-review/SKILL.md

```markdown
---
name: multi-review
description: Read all files in a directory and summarize each file's purpose. Use when asked to review or understand an entire project or directory.
allowed-tools: shell read_file
---

## Instructions

1. Use shell to find all files: find {directory} -type f -not -name '.*'
2. Read each file using read_file
3. For each file, write 2-3 sentences describing its purpose and key components
4. Provide an overall project summary at the end
```

---

## What Gets Removed

| Component | Lines | Reason |
|-----------|-------|--------|
| `SkillEngine` class | ~120 | LLM replaces step execution |
| `StepState`, `EngineState` | ~40 | No step state needed |
| `validate_output`, `_validate_type` | ~40 | No schema validation |
| `SchemaValidationError`, `SkillNotFoundError` | ~10 | Simplified |
| `_resolve_inputs`, `_resolve_string` | ~20 | No reference resolution |
| YAML manifests + JSON sequences | 8 files | Replaced by SKILL.md |
| **Total removed** | **~230 lines + 8 files** | |

## What Gets Added

| Component | Lines | Purpose |
|-----------|-------|---------|
| `parse_frontmatter` | ~15 | YAML frontmatter parser |
| `SkillRegistry` (rewrite) | ~90 | Discovery + loading + multi-path scan |
| `RunSkillTool` (rewrite) | ~45 | Tool-result activation + dedup |
| `CreateSkillTool` (rewrite) | ~35 | Safe YAML generation + name validation |
| REPL slash command handling | ~10 | `/skill-name` activation |
| 4 migrated skills | 4 SKILL.md files | research, code-review, file-stats, multi-review |
| **Total added** | **~195 lines + 4 files** | |

**Net: -35 lines of code, -4 files, +portability across 30+ agent tools.**

---

## Validation

After implementation, verify:
1. `python harness.py` → `list_skills` shows all 4 migrated skills
2. `run_skill("research")` → LLM searches web and summarizes
3. `run_skill("code-review")` → LLM reads file and produces review
4. `create_skill` → creates valid SKILL.md in new directory with proper YAML
5. `/research` in REPL → activates skill directly
6. Duplicate `run_skill("research")` in same prompt → returns "already active"
7. Skill with `mcp-servers: github` → warns if github MCP not connected
8. Skills in `.agents/skills/` are discovered
9. Invalid skill name in `create_skill` → rejected with error
10. SKILL.md files pass `skills-ref validate ./skills/research`
