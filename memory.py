"""
memory.py — Living Document Memory & Session Context
======================================================
Two self-rewriting documents maintained by the LLM:

  LivingMemory  — persistent across sessions (memory.md)
  LivingSession — current session only (session.md, deleted on exit)

Both use the same engine: after each interaction, the LLM rewrites
the document, compressing old info and updating contradictions.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("agent_harness.memory")

MEMORY_FILE = Path("memory.md")
MEMORY_ARCHIVE = Path("memory_archive.md")
SESSION_FILE = Path("session.md")
ARCHIVE_MAX_AGE_DAYS = 90

# ---------------------------------------------------------------------------
# Base engine — shared by memory and session
# ---------------------------------------------------------------------------

class _LivingDocument:
    def __init__(
        self,
        file_path: Path,
        consolidation_prompt: str,
        system_prefix: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        model: str = "qwen3:8b",
        max_words: int = 500,
        vertex_project: Optional[str] = None,
        vertex_region: str = "global",
        vertex_model: str = "claude-sonnet-4-6",
    ):
        self._file = file_path
        self._prompt_template = consolidation_prompt
        self._system_prefix = system_prefix
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._max_words = max_words
        self._vertex_project = vertex_project
        self._vertex_region = vertex_region
        self._vertex_model = vertex_model
        self._content = self._load()

    @property
    def content(self) -> str:
        return self._content

    def system_message(self) -> Optional[dict]:
        if not self._content.strip():
            return None
        return {"role": "system", "content": self._system_prefix + self._content}

    def consolidate(self, conversation: list[dict]) -> None:
        if not conversation:
            return

        self._archive_before_rewrite()

        conv_text = self._format_conversation(conversation)
        prompt = self._prompt_template.format(
            max_words=self._max_words,
            current=self._content or "(empty — first interaction)",
            conversation=conv_text,
        )
        try:
            self._content = self._call_llm(prompt).strip()
            self._save()
            logger.info("%s consolidated (%d words)", self.__class__.__name__, len(self._content.split()))
        except Exception as exc:
            logger.warning("%s consolidation failed: %s", self.__class__.__name__, exc)

    def _archive_before_rewrite(self) -> None:
        """Override in subclasses that need archiving."""
        pass

    def clear(self) -> None:
        self._content = ""
        if self._file.exists():
            self._file.unlink()

    def _format_conversation(self, messages: list[dict]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "tool":
                name = msg.get("tool_name", "tool")
                status = msg.get("status", "")
                output = msg.get("output", msg.get("error", ""))
                lines.append(f"[tool:{name}] {status}: {output}")
            elif role == "system":
                continue
            elif content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _call_llm(self, prompt: str) -> str:
        if self._vertex_project:
            try:
                return self._call_vertex(prompt)
            except Exception as exc:
                logger.warning("Vertex consolidation failed, falling back to Ollama: %s", exc)
        return self._call_ollama(prompt)

    def _call_vertex(self, prompt: str) -> str:
        from anthropic import AnthropicVertex
        client = AnthropicVertex(project_id=self._vertex_project, region=self._vertex_region)
        response = client.messages.create(
            model=self._vertex_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text")

    def _call_ollama(self, prompt: str) -> str:
        response = requests.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.3,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _load(self) -> str:
        if self._file.exists() and self._file.stat().st_size > 0:
            return self._file.read_text(encoding="utf-8")
        return ""

    def _save(self) -> None:
        self._file.write_text(self._content, encoding="utf-8")

    def __repr__(self) -> str:
        words = len(self._content.split()) if self._content else 0
        return f"{self.__class__.__name__}({words} words)"


# ---------------------------------------------------------------------------
# Long-term memory (persists across sessions)
# ---------------------------------------------------------------------------

_MEMORY_PROMPT = """You are maintaining your own persistent memory. Below is your current memory and a conversation that just happened.

Rewrite your memory document. Rules:
- ALWAYS preserve identity facts: user's name, location, role, background — these NEVER expire
- Keep preferences, decisions, technical facts, corrections
- Update anything that changed (e.g., if the user moved, update their location)
- Drop noise: greetings, filler, things that don't matter for future conversations
- Stay under {max_words} words
- Write in second person about the user ("User's name is...")
- Include behavioral notes if relevant ("User prefers concise answers")
- Output ONLY the updated memory document, nothing else

CURRENT MEMORY:
{current}

CONVERSATION THAT JUST HAPPENED:
{conversation}

UPDATED MEMORY:"""


class LivingMemory(_LivingDocument):
    def __init__(self, base_url="http://localhost:11434/v1", api_key="ollama",
                 model="qwen3:8b", max_words=500, memory_file=MEMORY_FILE,
                 vertex_project=None, vertex_region="global", vertex_model="claude-sonnet-4-6"):
        super().__init__(
            file_path=memory_file,
            consolidation_prompt=_MEMORY_PROMPT,
            system_prefix=(
                "Your persistent memory from previous interactions "
                "(these are facts you KNOW, use them naturally):\n\n"
            ),
            base_url=base_url, api_key=api_key, model=model, max_words=max_words,
            vertex_project=vertex_project, vertex_region=vertex_region, vertex_model=vertex_model,
        )
        self._archive_file = MEMORY_ARCHIVE
        self._trim_archive()

    def _archive_before_rewrite(self) -> None:
        if not self._content.strip():
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        entry = f"\n---\n**[{timestamp}]**\n{self._content}\n"
        with open(self._archive_file, "a", encoding="utf-8") as f:
            f.write(entry)
        logger.info("Memory archived before rewrite (%d words)", len(self._content.split()))

    def _trim_archive(self) -> None:
        if not self._archive_file.exists():
            return
        try:
            raw = self._archive_file.read_text(encoding="utf-8")
            entries = raw.split("\n---\n")
            cutoff = datetime.now(timezone.utc).timestamp() - (ARCHIVE_MAX_AGE_DAYS * 86400)

            kept = []
            for entry in entries:
                ts_match = __import__("re").search(r"\*\*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\*\*", entry)
                if not ts_match:
                    if entry.strip():
                        kept.append(entry)
                    continue
                try:
                    entry_time = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                    if entry_time.timestamp() >= cutoff:
                        kept.append(entry)
                except ValueError:
                    kept.append(entry)

            trimmed = len(entries) - len(kept)
            if trimmed > 0:
                self._archive_file.write_text("\n---\n".join(kept), encoding="utf-8")
                logger.info("Archive trimmed: removed %d entries older than %d days", trimmed, ARCHIVE_MAX_AGE_DAYS)
        except Exception as exc:
            logger.warning("Archive trim failed: %s", exc)


# ---------------------------------------------------------------------------
# Session context (current session only, deleted on exit)
# ---------------------------------------------------------------------------

_SESSION_PROMPT = """You are maintaining a summary of the current conversation session. Below is your current session summary and the latest exchange.

Rewrite the session summary. Rules:
- Capture what was discussed, what was asked, what was answered
- Preserve the FLOW of conversation: what topics came up, what the user's current focus is
- Keep specific details: numbers, code snippets, lists the user referenced, key points from your responses
- If the user said "expand on point 3", note what point 3 was
- Stay under {max_words} words
- Write as a running narrative: "User asked about X. I explained Y. User then wanted Z."
- Output ONLY the updated session summary, nothing else

CURRENT SESSION SUMMARY:
{current}

LATEST EXCHANGE:
{conversation}

UPDATED SESSION SUMMARY:"""


class LivingSession(_LivingDocument):
    def __init__(self, base_url="http://localhost:11434/v1", api_key="ollama",
                 model="qwen3:8b", max_words=800, session_file=SESSION_FILE,
                 vertex_project=None, vertex_region="global", vertex_model="claude-sonnet-4-6"):
        super().__init__(
            file_path=session_file,
            consolidation_prompt=_SESSION_PROMPT,
            system_prefix=(
                "Current conversation session context "
                "(use this to maintain continuity):\n\n"
            ),
            base_url=base_url, api_key=api_key, model=model, max_words=max_words,
            vertex_project=vertex_project, vertex_region=vertex_region, vertex_model=vertex_model,
        )
        self.clear()
