import argparse
import inspect
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, get_type_hints

import requests
from pydantic import BaseModel, ValidationError, model_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agent_harness")


# ---------------------------------------------------------------------------
# Subsystem 1: Configuration
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    model: str = "qwen3:8b"
    max_retries: int = 3
    timeout_seconds: float = 30.0
    max_tokens: int = 4096
    temperature: float = 0.7
    tool_call_timeout: float = 10.0

    @model_validator(mode="after")
    def validate_ranges(self) -> "AgentConfig":
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature {self.temperature} out of [0.0, 2.0]")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
        return self


def apply_and_revalidate_overrides(base_config, overrides):
    base_dict = base_config.model_dump()
    unknown = set(overrides.keys()) - set(base_dict.keys())
    if unknown:
        raise ValueError(f"Unknown override keys: {unknown}")
    merged = {**base_dict, **overrides}
    try:
        return AgentConfig(**merged)
    except ValidationError as exc:
        raise ValueError(f"Invalid config after overrides: {exc}") from exc


# ---------------------------------------------------------------------------
# History Backend
# ---------------------------------------------------------------------------

class HistoryBackend:
    """Thread-safe conversation history with optional JSONL checkpointing.

    When checkpoint_id is provided, every append is flushed to disk.
    On startup with resume=True, replays the checkpoint file into memory.
    clear() resets in-memory history but keeps the checkpoint file intact.
    """

    CHECKPOINT_DIR = Path(".checkpoints")

    def __init__(self, checkpoint_id: Optional[str] = None, resume: bool = True):
        self._lock = threading.Lock()
        self._history: list[dict] = []
        self._seq = 0
        self._file = None
        self.checkpoint_id = checkpoint_id

        if checkpoint_id:
            self.CHECKPOINT_DIR.mkdir(exist_ok=True)
            self._path = self.CHECKPOINT_DIR / f"{checkpoint_id}.jsonl"
            if resume and self._path.exists():
                self._load()
            self._file = self._path.open("a", encoding="utf-8")

    def append(self, entry):
        with self._lock:
            self._history.append(entry)
            if self._file:
                import json as _json
                from datetime import datetime as _dt, timezone as _tz
                record = {"seq": self._seq, "ts": _dt.now(_tz.utc).isoformat(), "entry": entry}
                self._file.write(_json.dumps(record, default=str) + "\n")
                self._file.flush()
                self._seq += 1

    def snapshot(self):
        with self._lock:
            return list(self._history)

    def clear(self):
        with self._lock:
            self._history.clear()

    def __len__(self):
        with self._lock:
            return len(self._history)

    def close(self):
        with self._lock:
            if self._file:
                self._file.flush()
                self._file.close()
                self._file = None

    def _load(self):
        loaded = 0
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._history.append(record["entry"])
                    self._seq = record["seq"] + 1
                    loaded += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        if loaded:
            logger.info("Resumed checkpoint '%s' — %d turns", self.checkpoint_id, loaded)

    @classmethod
    def list_checkpoints(cls) -> list[dict]:
        if not cls.CHECKPOINT_DIR.exists():
            return []
        results = []
        for p in sorted(cls.CHECKPOINT_DIR.glob("*.jsonl"), reverse=True):
            try:
                lines = p.read_text(encoding="utf-8").strip().splitlines()
                count = len(lines)
                last_ts = json.loads(lines[-1])["ts"] if lines else None
                size_kb = p.stat().st_size / 1024
                results.append({
                    "id": p.stem, "turns": count,
                    "last_ts": last_ts, "size_kb": round(size_kb, 1),
                })
            except Exception:
                continue
        return results


# ---------------------------------------------------------------------------
# Subsystem 4: Tools
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


class RunnableTool(ABC):
    name: str = "base_tool"
    description: str = ""
    execution_mode: str = "async"
    timeout_seconds: float = 30.0

    @abstractmethod
    def run(self, **kwargs) -> Any:
        pass

    def to_openai_schema(self) -> dict:
        """
        Introspect the run() method signature and build an OpenAI function-calling
        compatible tool schema automatically.

        Supported annotation types: str, int, float, bool. Falls back to "string"
        for anything unknown or unannotated.
        """
        sig = inspect.signature(self.run)
        try:
            hints = get_type_hints(self.run)
        except Exception:
            hints = {}

        _py_to_json = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }

        properties: dict[str, dict] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            py_type = hints.get(param_name, str)
            json_type = _py_to_json.get(py_type, "string")
            properties[param_name] = {"type": json_type}

            # A parameter is required when it has no default value
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class CalculatorTool(RunnableTool):
    name = "calculator"
    description = "Evaluates a safe arithmetic expression. Input: expr (str). Supports +, -, *, /, **, (), abs(), round()."
    execution_mode = "sync"
    timeout_seconds = 2.0

    def run(self, expr: str) -> Any:
        import ast
        import operator
        _OPS = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.Mod: operator.mod,
            ast.FloorDiv: operator.floordiv, ast.USub: operator.neg,
        }
        _FUNCS = {"abs": abs, "round": round, "min": min, "max": max}

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
                return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
                return _OPS[type(node.op)](_eval(node.operand))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
                args = [_eval(a) for a in node.args]
                return _FUNCS[node.func.id](*args)
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

        tree = ast.parse(expr.strip(), mode="eval")
        return _eval(tree)


class WebSearchTool(RunnableTool):
    name = "web_search"
    description = "Search the web via DuckDuckGo. Input: query (str). Returns top results with title, url, and snippet."
    execution_mode = "async"
    timeout_seconds = 15.0

    def run(self, query: str) -> str:
        from ddgs import DDGS
        logging.getLogger("primp").setLevel(logging.WARNING)
        logging.getLogger("ddgs").setLevel(logging.WARNING)
        results = DDGS().text(query, max_results=5)
        if not results:
            return f"No results found for '{query}'"
        lines = []
        for r in results:
            lines.append(f"- {r['title']}\n  {r['href']}\n  {r['body']}")
        return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Workspace-sandboxed tools
# ---------------------------------------------------------------------------

import subprocess as _subprocess
from pathlib import Path as _Path

_WORKSPACE = _Path.cwd().resolve()
_ALLOWED_EXECUTABLES = ["ls", "cat", "grep", "find", "git", "head", "tail", "wc", "sort", "uniq", "diff", "echo"]


def _safe_path(user_path: str) -> _Path:
    resolved = (_WORKSPACE / user_path).resolve()
    try:
        resolved.relative_to(_WORKSPACE)
    except ValueError:
        raise RuntimeError(f"Path '{user_path}' is outside workspace {_WORKSPACE}")
    return resolved


class ShellTool(RunnableTool):
    name = "shell"
    description = (
        "Run a shell command. Input: command (str) — the full command string. "
        "Allowed executables: " + ", ".join(_ALLOWED_EXECUTABLES) + ". "
        "Returns stdout. No rm, sudo, python3, or destructive commands."
    )
    execution_mode = "async"
    timeout_seconds = 30.0

    def run(self, command: str) -> str:
        import shlex
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return f"Error: invalid command syntax: {exc}"
        if not parts:
            return "Error: empty command"
        executable = parts[0]
        if executable not in _ALLOWED_EXECUTABLES:
            return f"Error: '{executable}' not allowed. Allowed: {_ALLOWED_EXECUTABLES}"
        try:
            result = _subprocess.run(
                parts, capture_output=True, text=True, timeout=30,
                cwd=_WORKSPACE, shell=False,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr] {result.stderr}"
            if len(output) > 100_000:
                output = output[:100_000] + "\n... [truncated]"
            return output or "(no output)"
        except _subprocess.TimeoutExpired:
            return "Error: command timed out after 30s"
        except FileNotFoundError:
            return f"Error: executable '{executable}' not found"


class ReadFileTool(RunnableTool):
    name = "read_file"
    description = "Read a file's contents. Input: path (str) — relative to workspace."
    execution_mode = "sync"
    timeout_seconds = 5.0

    def run(self, path: str) -> str:
        try:
            safe = _safe_path(path)
            if not safe.exists():
                return f"Error: file not found: {path}"
            if safe.stat().st_size > 1_000_000:
                return f"Error: file too large ({safe.stat().st_size} bytes, max 1MB)"
            return safe.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: '{path}' is not a text file"
        except RuntimeError as exc:
            return str(exc)


class WriteFileTool(RunnableTool):
    name = "write_file"
    description = (
        "Write content to a file. Input: path (str), content (str). "
        "Creates parent directories if needed. Stays within workspace."
    )
    execution_mode = "sync"
    timeout_seconds = 5.0

    def run(self, path: str = "", content: str = "", **kwargs) -> str:
        if not path:
            path = kwargs.get("filename", kwargs.get("file_path", kwargs.get("name", "")))
        if not content:
            content = kwargs.get("code", kwargs.get("text", kwargs.get("data", "")))
        if not path:
            return "Error: 'path' parameter is required"
        if not content:
            return "Error: 'content' parameter is required — provide the text to write"
        try:
            safe = _safe_path(path)
            safe.parent.mkdir(parents=True, exist_ok=True)
            encoded = content.encode("utf-8")
            safe.write_bytes(encoded)
            return f"Wrote {len(encoded)} bytes to {path}"
        except RuntimeError as exc:
            return str(exc)


class ListFilesTool(RunnableTool):
    name = "list_files"
    description = "List files in a directory. Input: path (str) — defaults to '.'. Returns filenames."
    execution_mode = "sync"
    timeout_seconds = 5.0

    def run(self, path: str = ".") -> str:
        try:
            safe = _safe_path(path)
            if not safe.is_dir():
                return f"Error: not a directory: {path}"
            entries = sorted(safe.iterdir(), key=lambda p: (p.is_file(), p.name))
            lines = []
            for e in entries[:200]:
                rel = e.relative_to(_WORKSPACE)
                suffix = "/" if e.is_dir() else f" ({e.stat().st_size}b)"
                lines.append(f"  {rel}{suffix}")
            return "\n".join(lines) or "(empty directory)"
        except RuntimeError as exc:
            return str(exc)


class PythonExecTool(RunnableTool):
    name = "python_exec"
    description = (
        "Execute Python code and return stdout. Input: code (str). "
        "Has access to math, string operations, data processing. "
        "No filesystem access, no subprocess, no network."
    )
    execution_mode = "async"
    timeout_seconds = 30.0

    _BLOCKED_MODULES = frozenset([
        "os", "shutil", "subprocess", "pathlib", "sys",
        "socket", "http", "urllib", "requests", "ftplib",
        "ctypes", "importlib",
    ])

    def run(self, code: str) -> str:
        import ast
        import contextlib
        import io
        import threading

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"Error: syntax error: {exc}"

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif node.module:
                    names = [node.module.split(".")[0]]
                for name in names:
                    if name in self._BLOCKED_MODULES:
                        return f"Error: module '{name}' is not allowed in python_exec"

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        safe_builtins = {k: v for k, v in __builtins__.items() if isinstance(__builtins__, dict)} if isinstance(__builtins__, dict) else {k: getattr(__builtins__, k) for k in dir(__builtins__) if not k.startswith("_")}
        for name in ["open", "exec", "eval", "compile", "__import__", "exit", "quit"]:
            safe_builtins.pop(name, None)

        namespace = {"__builtins__": safe_builtins}
        exc_holder = []

        def _run():
            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    compiled = compile(code, "<python_exec>", "exec")
                    exec(compiled, namespace)
            except Exception as exc:
                exc_holder.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=30)

        if thread.is_alive():
            return "Error: code execution timed out after 30s"

        output = stdout_buf.getvalue()
        errors = stderr_buf.getvalue()

        if exc_holder:
            return f"Error: {exc_holder[0]}"
        if errors:
            output += f"\n[stderr] {errors}"
        if len(output) > 100_000:
            output = output[:100_000] + "\n... [truncated]"
        return output or "(no output)"


class HttpFetchTool(RunnableTool):
    name = "http_fetch"
    description = "Fetch a URL and return the response body. Input: url (str). Returns status code and body text."
    execution_mode = "async"
    timeout_seconds = 15.0

    def run(self, url: str) -> str:
        import urllib.request
        import urllib.error

        if not url.startswith(("http://", "https://")):
            return f"Error: only http/https URLs allowed, got: {url}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AIHarness/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read(500_000).decode("utf-8", errors="replace")
                if len(body) >= 500_000:
                    body += "\n... [truncated at 500KB]"
                return f"HTTP {resp.status}\n\n{body}"
        except urllib.error.HTTPError as exc:
            return f"HTTP {exc.code}: {exc.reason}"
        except urllib.error.URLError as exc:
            return f"Error: {exc.reason}"
        except TimeoutError:
            return "Error: request timed out"


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
        self._activated = set()

    def run(self, skill_name: str = "", context: str = "", **kwargs) -> str:
        skill_name = skill_name or kwargs.get("name", kwargs.get("skill", ""))
        context = context or kwargs.get("task", kwargs.get("input", kwargs.get("inputs_json", "")))
        if not skill_name:
            return "Error: 'skill_name' parameter is required"

        skill = self._registry.get(skill_name)
        if not skill:
            available = [s["name"] for s in self._registry.list_skills()]
            return f"Skill '{skill_name}' not found. Available: {available}"

        if skill_name in self._activated:
            return f"Skill '{skill_name}' is already active. Follow the instructions previously provided."

        mcp_servers = skill.get("metadata", {}).get("mcp-servers", "")
        if mcp_servers and self._mcp:
            status = self._mcp.status()
            servers = status.get("servers", {})
            missing = [s for s in mcp_servers.split() if s not in servers or not servers[s].get("alive")]
            if missing:
                return f"Skill requires MCP servers not connected: {missing}"

        content = self._registry.load_skill(skill_name)
        if not content:
            return f"Could not load skill '{skill_name}'"

        self._activated.add(skill_name)

        result = f"=== SKILL: {skill_name} ===\n\n"
        result += "[SKILL_INSTRUCTIONS_START]\n"
        result += content
        result += "\n[SKILL_INSTRUCTIONS_END]\n"
        if skill.get("allowed_tools"):
            result += f"\nApproved tools for this skill: {skill['allowed_tools']}"
        if context:
            result += f"\n\nUser context: {context}"
        result += "\n\nFollow these instructions now using the approved tools."
        return result


class ListSkillsTool(RunnableTool):
    name = "list_skills"
    description = "List all available skills with their descriptions."
    execution_mode = "sync"
    timeout_seconds = 2.0

    def __init__(self, registry):
        self._registry = registry

    def run(self) -> str:
        skills = self._registry.list_skills()
        if not skills:
            return "No skills available."
        lines = [f"- {s['name']}: {s['description']}" for s in skills]
        return "\n".join(lines)


class CreateSkillTool(RunnableTool):
    name = "create_skill"
    description = (
        "Create a new reusable skill (agentskills.io format). "
        "Input: skill_name (str) — lowercase with hyphens only, "
        "description (str) — what the skill does and when to use it, "
        "instructions (str) — markdown instructions for the agent to follow."
    )
    execution_mode = "sync"
    timeout_seconds = 5.0

    _NAME_PATTERN = re.compile(r'^[a-z0-9]([a-z0-9-]*[a-z0-9])?$')

    def __init__(self, registry):
        self._registry = registry

    def run(self, skill_name: str = "", description: str = "", instructions: str = "", **kwargs) -> str:
        skill_name = skill_name or kwargs.get("name", kwargs.get("skill", ""))
        description = description or kwargs.get("desc", "")
        instructions = instructions or kwargs.get("body", kwargs.get("content", kwargs.get("manifest_yaml", "")))
        if not skill_name:
            return "Error: 'skill_name' is required (lowercase, hyphens only)"
        if not description:
            return "Error: 'description' is required"
        if not instructions:
            return "Error: 'instructions' is required"

        if len(skill_name) > 64 or not self._NAME_PATTERN.match(skill_name) or '--' in skill_name:
            return f"Error: invalid skill name '{skill_name}'. Must be 1-64 chars, lowercase alphanumeric + hyphens."

        import yaml
        frontmatter = yaml.dump({
            "name": skill_name,
            "description": description,
            "metadata": {"author": "ai-generated", "version": "1.0"},
        }, default_flow_style=False, allow_unicode=True)

        content = f"---\n{frontmatter}---\n\n{instructions}\n"
        self._registry.create_from_content(skill_name, content)
        return f"Skill '{skill_name}' created at skills/{skill_name}/SKILL.md"


PLANNING_PROMPT = """You are a project planner. Given a task, decide what specialist agents are needed and create an execution plan.

AVAILABLE TOOLS that agents can use:
calculator, web_search, shell, read_file, write_file, list_files, python_exec, http_fetch

AVAILABLE MODELS:
- "qwen3:8b" — local, fast, free. Good for: research, search, simple formatting
- "claude-sonnet-4-6" — capable, moderate cost. Good for: code generation, architecture, complex tasks

RULES:
- If the task is simple (single question, one tool needed), respond with: {{"agents": [], "plan": []}}
- Maximum 5 agents, maximum 10 plan steps
- Each agent needs: name (short, lowercase), role (what it does), tools (from available list), model
- Plan steps execute sequentially — each step gets context from all previous steps
- Respond with ONLY valid JSON, nothing else

EXAMPLE for "Build a web scraper with tests":
{{"agents": [{{"name": "coder", "role": "Write Python code", "tools": ["read_file", "write_file", "python_exec"], "model": "claude-sonnet-4-6"}}, {{"name": "tester", "role": "Write and run tests", "tools": ["read_file", "write_file", "python_exec"], "model": "claude-sonnet-4-6"}}], "plan": [{{"step": 1, "agent": "coder", "task": "Implement the web scraper with requests and BeautifulSoup"}}, {{"step": 2, "agent": "tester", "task": "Write unit tests for the scraper"}}]}}

EXAMPLE for "What is 2+2?":
{{"agents": [], "plan": []}}

TASK: {task}

JSON:"""


def _create_agent_from_spec(spec, provider_config, tool_pool):
    sub_history = HistoryBackend()
    sub_tools = [tool_pool[t] for t in spec.get("tools", []) if t in tool_pool]
    sub_orch = AgentOrchestrator(tools=sub_tools, history=sub_history)

    model = spec.get("model", "qwen3:8b")

    if model.startswith("claude-") and provider_config.get("vertex_project"):
        sub_config = AgentConfig(
            model=model, max_retries=3, timeout_seconds=300.0,
            max_tokens=spec.get("max_tokens", 4096), temperature=0.7,
        )
        sub_llm = VertexClaudeClient(
            project=provider_config["vertex_project"],
            region=provider_config.get("vertex_region", "global"),
            config=sub_config,
        )
    else:
        if model.startswith("claude-"):
            model = "qwen3:8b"
        sub_config = AgentConfig(
            model=model, max_retries=3, timeout_seconds=300.0,
            max_tokens=spec.get("max_tokens", 4096), temperature=0.7,
        )
        sub_llm = LLMClient(
            base_url=provider_config.get("base_url", "http://localhost:11434/v1"),
            api_key=provider_config.get("api_key", "ollama"),
            config=sub_config,
        )

    agent_name = spec["name"]
    sub_loop = AgentLoop(
        llm_client=sub_llm, orchestrator=sub_orch,
        history=sub_history, config=sub_config,
        agent_name=agent_name,
    )

    def handler(task, _loop=sub_loop, _hist=sub_history):
        _hist.clear()
        _hist.append({
            "role": "system",
            "content": (
                "You are a specialist agent. Complete the task using your tools. "
                "When done, respond with a TEXT SUMMARY of what you did. "
                "Do NOT keep calling tools indefinitely — finish and report back."
            ),
        })
        return _loop.run(task)

    return {
        "description": spec.get("role", ""),
        "model": model,
        "handler": handler,
        "orchestrator": sub_orch,
    }


class PlanAgentsTool(RunnableTool):
    name = "plan_agents"
    description = (
        "Analyze a complex task and create specialist agents to handle it. "
        "Call this FIRST on tasks that need multiple steps or expertise areas. "
        "Input: task_description (str) — the user's full request."
    )
    execution_mode = "async"
    timeout_seconds = 120.0

    def __init__(self, provider_config, tool_pool, agent_registry):
        self._provider_config = provider_config
        self._tool_pool = tool_pool
        self._agents = agent_registry
        self._planner_llm = self._build_planner()

    def _build_planner(self):
        pc = self._provider_config
        if pc.get("vertex_project"):
            try:
                config = AgentConfig(model="claude-opus-4-6", max_retries=2,
                                     timeout_seconds=60.0, max_tokens=2048, temperature=0.3)
                return VertexClaudeClient(
                    project=pc["vertex_project"],
                    region=pc.get("vertex_region", "global"),
                    config=config,
                )
            except Exception:
                pass
        config = AgentConfig(model=pc.get("model", "qwen3:8b"), max_retries=2,
                             timeout_seconds=60.0, max_tokens=2048, temperature=0.3)
        return LLMClient(
            base_url=pc.get("base_url", "http://localhost:11434/v1"),
            api_key=pc.get("api_key", "ollama"),
            config=config,
        )

    def run(self, task_description: str = "", **kwargs) -> str:
        task_description = task_description or kwargs.get("task", kwargs.get("prompt", ""))
        if not task_description:
            return "Error: 'task_description' parameter is required"

        _spinner_detail[0] = "Planning agents (Opus)"

        prompt = PLANNING_PROMPT.format(task=task_description)
        plan = None

        for attempt in range(2):
            try:
                response = self._planner_llm.chat(
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.text or ""
                text = text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0]
                plan = json.loads(text)
                break
            except (json.JSONDecodeError, RuntimeError) as exc:
                if attempt == 0:
                    prompt += "\n\nYour previous response was not valid JSON. Respond with ONLY JSON."
                    continue
                return f"Planning failed: {exc}. Handle the task directly with your own tools."

        agents_spec = plan.get("agents", [])
        steps = plan.get("plan", [])

        if not agents_spec:
            return "No specialist agents needed — handle this task directly with your tools."

        if len(agents_spec) > 5:
            agents_spec = agents_spec[:5]
        if len(steps) > 10:
            steps = steps[:10]

        _spinner_detail[0] = "Creating agents"

        created = []
        for spec in agents_spec:
            name = spec.get("name", "")
            if not name or name in self._agents:
                continue
            try:
                agent_entry = _create_agent_from_spec(spec, self._provider_config, self._tool_pool)
                self._agents[name] = agent_entry
                created.append(f"{name} ({spec.get('role', '')})")
            except Exception as exc:
                logger.warning("Failed to create agent '%s': %s", name, exc)

        if not created:
            return "Could not create any agents. Handle the task directly with your tools."

        _spinner_detail[0] = f"Created: {', '.join(c.split(' (')[0] for c in created)}"

        plan_text = f"Created {len(created)} specialist agents:\n"
        for c in created:
            plan_text += f"  - {c}\n"
        plan_text += "\nExecution plan:\n"
        for s in steps:
            plan_text += f"  Step {s.get('step', '?')}: delegate to '{s.get('agent', '?')}' — {s.get('task', '')}\n"
        plan_text += "\nUse the 'delegate' tool to execute each step in order. "
        plan_text += "Pass the full step task description plus context from previous steps."

        return plan_text


class DelegateTool(RunnableTool):
    name = "delegate"
    description = "Delegate a task to a specialist agent. Input: agent_name (str), task (str)."
    execution_mode = "async"
    timeout_seconds = 600.0

    def __init__(self, agent_registry):
        self._agents = agent_registry
        agents_desc = ", ".join(f"{n} ({a['description']})" for n, a in self._agents.items())
        self.description = (
            f"Delegate a task to a specialist agent. Available agents: {agents_desc}. "
            "Input: agent_name (str), task (str)."
        )

    def run(self, agent_name: str = "", task: str = "", **kwargs) -> str:
        agent_name = agent_name or kwargs.get("name", kwargs.get("agent", ""))
        task = task or kwargs.get("prompt", kwargs.get("message", kwargs.get("content", "")))
        if not agent_name:
            return "Error: 'agent_name' parameter is required"
        if not task:
            return "Error: 'task' parameter is required"
        agent = self._agents.get(agent_name)
        if agent is None:
            available = list(self._agents.keys())
            return f"Agent '{agent_name}' not found. Available: {available}"

        handler = agent.get("handler")
        if handler is None:
            return f"Agent '{agent_name}' has no handler"

        _spinner_detail[0] = f"Delegating to {agent_name}"
        _set_agent_status(agent_name, "working...")
        try:
            result = handler(task)
            _set_agent_status(agent_name, "done")
            time.sleep(0.2)
            _clear_agent_status(agent_name)
            return result if isinstance(result, str) else json.dumps(result, default=str)
        except Exception as exc:
            _clear_agent_status(agent_name)
            return f"Agent '{agent_name}' failed: {exc}"


class ListAgentsTool(RunnableTool):
    name = "list_agents"
    description = "List all available specialist agents and their capabilities."
    execution_mode = "sync"
    timeout_seconds = 2.0

    def __init__(self, agent_registry):
        self._agents = agent_registry

    def run(self) -> str:
        if not self._agents:
            return "No agents available."
        lines = [f"- {name}: {a['description']} (model: {a.get('model', '?')})"
                 for name, a in self._agents.items()]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subsystem 5: Orchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    def __init__(self, tools, history, max_workers=4):
        self._tools = {t.name: t for t in tools}
        self._history = history
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit_user_message(self, content):
        self._history.append({"role": "user", "content": content})

    def run_tool(self, tool_name, tool_call_id=None, **kwargs):
        tool = self._tools.get(tool_name)
        if tool is None:
            return self._record_failure(tool_name, f"Unknown tool: {tool_name!r}", elapsed=0.0, tool_call_id=tool_call_id)
        if tool.execution_mode == "sync":
            result = self._run_sync(tool, **kwargs)
        else:
            result = self._run_async(tool, **kwargs)
        self._record_result(result, tool_call_id=tool_call_id)
        return result

    def context_snapshot(self):
        return self._history.snapshot()

    def tool_schemas(self) -> list[dict]:
        """Return OpenAI-compatible tool schemas for all registered tools."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def shutdown(self):
        self._executor.shutdown(wait=False)

    def _run_sync(self, tool, **kwargs):
        start = time.perf_counter()
        try:
            output = tool.run(**kwargs)
            elapsed = time.perf_counter() - start
            return ToolResult(tool_name=tool.name, success=True, output=output, elapsed_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return ToolResult(tool_name=tool.name, success=False, output=None, error=str(exc), elapsed_seconds=elapsed)

    def _run_async(self, tool, **kwargs):
        start = time.perf_counter()
        future = self._executor.submit(tool.run, **kwargs)
        try:
            output = future.result(timeout=tool.timeout_seconds)
            elapsed = time.perf_counter() - start
            return ToolResult(tool_name=tool.name, success=True, output=output, elapsed_seconds=elapsed)
        except FuturesTimeoutError:
            future.cancel()
            elapsed = time.perf_counter() - start
            return ToolResult(tool_name=tool.name, success=False, output=None,
                              error=f"Timed out after {tool.timeout_seconds}s", elapsed_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return ToolResult(tool_name=tool.name, success=False, output=None, error=str(exc), elapsed_seconds=elapsed)

    def _record_result(self, result, tool_call_id=None):
        if result.success:
            entry = {"role": "tool", "tool_name": result.tool_name, "status": "success",
                     "output": result.output, "elapsed_seconds": round(result.elapsed_seconds, 4)}
        else:
            entry = {"role": "tool", "tool_name": result.tool_name, "status": "failure",
                     "error": result.error, "elapsed_seconds": round(result.elapsed_seconds, 4),
                     "agent_hint": f"Tool '{result.tool_name}' failed. Consider an alternative approach."}
        if tool_call_id:
            entry["tool_call_id"] = tool_call_id
        self._history.append(dict(entry))

    def _record_failure(self, tool_name, error, elapsed, tool_call_id=None):
        result = ToolResult(tool_name=tool_name, success=False, output=None, error=error, elapsed_seconds=elapsed)
        self._record_result(result, tool_call_id=tool_call_id)
        return result


# ---------------------------------------------------------------------------
# Subsystem 6: LLM Client
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Structured response from the LLM.

    Exactly one of `text` or `tool_call` will be populated on a successful
    parse; both may be None on an unexpected response shape (treated as an
    empty assistant turn by the agent loop).
    """
    text: Optional[str]
    tool_call: Optional[dict]   # {"id": str, "name": str, "arguments": dict}
    raw: dict                   # The full parsed JSON body for debugging

    @property
    def has_tool_call(self) -> bool:
        return self.tool_call is not None

    @property
    def has_text(self) -> bool:
        return bool(self.text)


# HTTP status codes that are worth retrying
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMClient:
    """
    Thin HTTP client for any OpenAI-compatible chat-completion endpoint.

    Supports:
    - OpenAI (https://api.openai.com/v1)
    - Ollama  (http://localhost:11434/v1)
    - vLLM   (http://localhost:8000/v1)
    - LiteLLM proxy
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        config: AgentConfig,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._config = config
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        })
        logger.info(
            "LLMClient initialised — endpoint=%s model=%s",
            self._base_url,
            self._config.model,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> LLMResponse:
        """
        Send a chat-completion request and return a structured LLMResponse.

        Retries on transient HTTP errors with exponential back-off.
        Raises RuntimeError after all retries are exhausted.
        """
        payload = self._build_payload(messages, tools)
        url = f"{self._base_url}/chat/completions"

        last_exc: Optional[Exception] = None
        for attempt in range(self._config.max_retries + 1):
            if attempt > 0:
                wait = 2 ** (attempt - 1)          # 1 s, 2 s, 4 s …
                logger.warning(
                    "LLM request retry %d/%d — waiting %ds",
                    attempt, self._config.max_retries, wait,
                )
                time.sleep(wait)

            try:
                response = self._session.post(
                    url,
                    json=payload,
                    timeout=self._config.timeout_seconds,
                )
            except requests.exceptions.Timeout as exc:
                logger.warning("LLM request timed out (attempt %d): %s", attempt + 1, exc)
                last_exc = exc
                continue
            except requests.exceptions.RequestException as exc:
                logger.warning("LLM request network error (attempt %d): %s", attempt + 1, exc)
                last_exc = exc
                continue

            if response.status_code == 200:
                return self._parse_response(response)

            if response.status_code in _RETRYABLE_STATUS:
                logger.warning(
                    "LLM returned retryable HTTP %d (attempt %d): %s",
                    response.status_code, attempt + 1, response.text[:200],
                )
                last_exc = RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
                continue

            # Non-retryable HTTP error — fail immediately
            raise RuntimeError(
                f"LLM request failed with HTTP {response.status_code}: {response.text[:500]}"
            )

        raise RuntimeError(
            f"LLM request failed after {self._config.max_retries + 1} attempts. "
            f"Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_payload(self, messages: list[dict], tools: Optional[list[dict]]) -> dict:
        """
        Construct the request body.

        The history stored in HistoryBackend may contain internal bookkeeping
        keys (tool_name, status, elapsed_seconds, agent_hint) that the OpenAI
        API does not accept.  We normalise every message here.
        """
        normalised = [self._normalise_message(m) for m in messages]

        payload: dict = {
            "model": self._config.model,
            "messages": normalised,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        return payload

    @staticmethod
    def _normalise_message(msg: dict) -> dict:
        """
        Convert an internal history entry into a valid OpenAI message dict.

        OpenAI accepted roles: system, user, assistant, tool
        """
        role = msg.get("role", "user")

        if role == "tool":
            # OpenAI tool result format
            return {
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", "unknown"),
                "content": json.dumps(msg.get("output") if msg.get("status") == "success"
                                      else {"error": msg.get("error")}),
            }

        if role == "assistant":
            out: dict = {"role": "assistant", "content": msg.get("content")}
            if msg.get("tool_calls"):
                out["tool_calls"] = msg["tool_calls"]
            return out

        # user / system
        return {"role": role, "content": msg.get("content", "")}

    @staticmethod
    def _parse_response(response: requests.Response) -> LLMResponse:
        """
        Decode a successful HTTP 200 body into an LLMResponse.

        Handles:
        - Standard text reply  (choices[0].message.content)
        - Tool-call reply      (choices[0].message.tool_calls[0])
        """
        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(f"LLM returned non-JSON body: {response.text[:200]}") from exc

        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError(f"LLM response has no choices: {body}")

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls")

        if tool_calls:
            # Take the first tool call (parallel tool calls are uncommon but valid;
            # the agent loop can be extended to handle multiple later).
            tc = tool_calls[0]
            raw_args = tc.get("function", {}).get("arguments", "{}")
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                logger.warning("Could not parse tool call arguments as JSON: %r", raw_args)
                arguments = {"_raw": raw_args}

            tool_call = {
                "id": tc.get("id", "tc_unknown"),
                "name": tc.get("function", {}).get("name", ""),
                "arguments": arguments,
                # Preserve the raw OpenAI format for history replay
                "_raw_tool_calls": tool_calls,
            }
            logger.info("LLM requested tool call: %s(%s)", tool_call["name"], arguments)
            return LLMResponse(text=None, tool_call=tool_call, raw=body)

        content = message.get("content") or ""
        logger.info("LLM returned text reply (%d chars)", len(content))
        return LLMResponse(text=content, tool_call=None, raw=body)


class VertexClaudeClient:
    """
    LLM client for Claude models via Vertex AI using the Anthropic SDK.

    Implements the same .chat() interface as LLMClient so the AgentLoop
    can use either interchangeably.
    """

    def __init__(self, project: str, region: str, config: AgentConfig):
        from anthropic import AnthropicVertex
        self._client = AnthropicVertex(project_id=project, region=region)
        self._config = config
        logger.info(
            "VertexClaudeClient initialised — project=%s region=%s model=%s",
            project, region, self._config.model,
        )

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> LLMResponse:
        system_parts = []
        api_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                system_parts.append(msg.get("content", ""))
            elif role == "tool":
                api_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", "unknown"),
                        "content": json.dumps(
                            msg.get("output") if msg.get("status") == "success"
                            else {"error": msg.get("error")}
                        ),
                    }],
                })
            elif role == "assistant":
                content = []
                if msg.get("tool_calls"):
                    tc = msg["tool_calls"][0]
                    fn = tc.get("function", {})
                    raw_args = fn.get("arguments", "{}")
                    content.append({
                        "type": "tool_use",
                        "id": tc.get("id", "tc_unknown"),
                        "name": fn.get("name", ""),
                        "input": json.loads(raw_args) if isinstance(raw_args, str) else raw_args,
                    })
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                if content:
                    api_messages.append({"role": "assistant", "content": content})
            else:
                api_messages.append({"role": "user", "content": msg.get("content", "")})

        if not api_messages:
            raise RuntimeError("No messages to send")

        kwargs: dict = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "messages": api_messages,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        if tools:
            kwargs["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"].get("parameters", {}),
                }
                for t in tools
            ]

        last_exc: Optional[Exception] = None
        for attempt in range(self._config.max_retries + 1):
            if attempt > 0:
                wait = 2 ** (attempt - 1)
                logger.warning("Claude retry %d/%d — waiting %ds", attempt, self._config.max_retries, wait)
                time.sleep(wait)
            try:
                response = self._client.messages.create(**kwargs)
                return self._parse_response(response)
            except Exception as exc:
                logger.warning("Claude request failed (attempt %d): %s", attempt + 1, exc)
                last_exc = exc

        raise RuntimeError(f"Claude request failed after {self._config.max_retries + 1} attempts: {last_exc}")

    @staticmethod
    def _parse_response(response) -> LLMResponse:
        raw = {"id": response.id, "model": response.model, "stop_reason": response.stop_reason}

        for block in response.content:
            if block.type == "tool_use":
                tool_call = {
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                    "_raw_tool_calls": [{
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }],
                }
                logger.info("Claude requested tool call: %s(%s)", block.name, block.input)
                return LLMResponse(text=None, tool_call=tool_call, raw=raw)

        text = "".join(b.text for b in response.content if b.type == "text")
        logger.info("Claude returned text reply (%d chars)", len(text))
        return LLMResponse(text=text or None, tool_call=None, raw=raw)


# ---------------------------------------------------------------------------
# Subsystem 7: Agent Loop
# ---------------------------------------------------------------------------

class AgentLoop:
    """
    Autonomous ReAct-style agent loop.

    Flow:
      1. Append user message to history.
      2. Call LLM with full history + tool schemas.
      3a. If the LLM returns a tool_call → execute it, append result, goto 2.
      3b. If the LLM returns text → that is the final answer; return it.
      4. If max_iterations is reached without a final answer, return whatever
         the last text response was (or a timeout message).
    """

    # We use max_retries from AgentConfig as the maximum number of LLM turns
    # (not just HTTP retries).  A separate ceiling prevents infinite loops.
    _HARD_ITERATION_CEILING = 30

    def __init__(
        self,
        llm_client: LLMClient,
        orchestrator: AgentOrchestrator,
        history: HistoryBackend,
        config: AgentConfig,
        memory: Optional[LivingMemory] = None,
        session: Optional[LivingSession] = None,
        mcp_manager=None,
        agent_name: Optional[str] = None,
    ):
        self._llm = llm_client
        self._orch = orchestrator
        self._history = history
        self._agent_name = agent_name
        self._config = config
        self._memory = memory
        self._session = session
        self._mcp = mcp_manager
        # Max agent iterations: respect config but never exceed hard ceiling
        self._max_iterations = self._HARD_ITERATION_CEILING

    def run(self, user_prompt: str) -> str:
        """
        Execute the agent loop for a single user prompt.
        Returns the final answer string.
        """
        logger.info("Agent loop starting — prompt: %r", user_prompt[:120])

        if self._memory is not None:
            sys_msg = self._memory.system_message()
            if sys_msg:
                self._history.append(sys_msg)

        if self._session is not None:
            ses_msg = self._session.system_message()
            if ses_msg:
                self._history.append(ses_msg)

        self._orch.submit_user_message(user_prompt)
        tool_schemas = self._orch.tool_schemas()

        if self._mcp is not None:
            results = self._mcp.search_tools(user_prompt, top_k=5)
            for r in results:
                mcp_tool = self._mcp.get_tool(r.tool.qualified_name)
                if mcp_tool and mcp_tool.name not in self._orch._tools:
                    self._orch._tools[mcp_tool.name] = mcp_tool
                    tool_schemas.append(mcp_tool.to_openai_schema())
                    _spinner_detail[0] = f"+ {r.tool.qualified_name}"

        for iteration in range(1, self._max_iterations + 1):
            logger.info("--- Agent iteration %d/%d ---", iteration, self._max_iterations)

            messages = self._history.snapshot()
            response = self._llm.chat(messages=messages, tools=tool_schemas)

            # ----------------------------------------------------------------
            # Branch A: tool call
            # ----------------------------------------------------------------
            if response.has_tool_call:
                tc = response.tool_call
                tool_name = tc["name"]
                arguments = tc["arguments"]
                tool_call_id = tc["id"]

                # Append the assistant's tool-call turn to history so the LLM
                # sees its own decision on the next turn.
                self._history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tc["_raw_tool_calls"],
                })

                if self._agent_name:
                    _set_agent_status(self._agent_name, f"{tool_name}")
                else:
                    _spinner_detail[0] = f"Running {tool_name}"
                logger.info(
                    "Executing tool '%s' with args: %s", tool_name, arguments
                )
                result = self._orch.run_tool(tool_name, tool_call_id=tool_call_id, **arguments)

                if result.success:
                    logger.info(
                        "Tool '%s' succeeded: %s", tool_name, str(result.output)[:200]
                    )
                else:
                    logger.warning(
                        "Tool '%s' failed: %s", tool_name, result.error
                    )

                if self._agent_name:
                    _set_agent_status(self._agent_name, "thinking...")
                else:
                    _spinner_detail[0] = ""
                continue

            # ----------------------------------------------------------------
            # Branch B: final text answer
            # ----------------------------------------------------------------
            if response.has_text:
                self._history.append({
                    "role": "assistant",
                    "content": response.text,
                })
                snapshot = self._history.snapshot()

                def _consolidate_bg(snap, session, memory):
                    if session is not None:
                        session.consolidate(snap)
                    if memory is not None:
                        memory.consolidate(snap)

                t = threading.Thread(
                    target=_consolidate_bg,
                    args=(snapshot, self._session, self._memory),
                    daemon=True,
                )
                t.start()

                logger.info("Agent loop complete after %d iteration(s).", iteration)
                return response.text

            # ----------------------------------------------------------------
            # Branch C: empty response (shouldn't happen, but be defensive)
            # ----------------------------------------------------------------
            logger.warning(
                "LLM returned neither text nor tool_call on iteration %d. "
                "Treating as empty answer.",
                iteration,
            )
            return "(No response from model)"

        # Exhausted all iterations
        msg = (
            f"Agent loop reached maximum iterations ({self._max_iterations}) "
            "without producing a final answer."
        )
        logger.error(msg)
        return msg


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness.py",
        description="LLM orchestration harness with autonomous tool-use agent loop.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="The user prompt to send to the agent (required unless --demo is set).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="Base URL for an OpenAI-compatible API endpoint. "
             "Default: http://localhost:11434/v1 (Ollama local)",
    )
    parser.add_argument(
        "--api-key",
        default="ollama",
        help="API key (use 'ollama' for local Ollama, or your OpenAI key). "
             "Default: ollama",
    )
    parser.add_argument(
        "--model",
        default="qwen3:8b",
        help="Model name. Default: qwen3:8b",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0–2.0). Default: 0.7",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens in the LLM response. Default: 4096",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum HTTP retries on transient errors. Default: 3",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request HTTP timeout in seconds. Default: 60.0",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Working directory for file operations. Creates it if needed. Default: current directory.",
    )
    parser.add_argument(
        "--checkpoint-id",
        default=None,
        help="Resume a previous conversation by checkpoint ID.",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="Print saved checkpoints and exit.",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "vertex-claude"],
        help="LLM provider. 'openai' for any OpenAI-compatible API (Ollama, OpenAI, vLLM). "
             "'vertex-claude' for Claude on Vertex AI. Default: openai",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Google Cloud project ID (required for vertex-claude). "
             "Falls back to GOOGLE_CLOUD_PROJECT env var.",
    )
    parser.add_argument(
        "--region",
        default="global",
        help="Vertex AI region. Default: global",
    )
    parser.add_argument(
        "--mcp-config",
        default=None,
        help="Path to MCP server config JSON file. Servers are started and "
             "tools are discovered automatically.",
    )
    parser.add_argument(
        "--agents",
        default=None,
        help="Path to agents config JSON defining specialist sub-agents.",
    )
    parser.add_argument(
        "--channels",
        default=None,
        help="Path to channels config JSON (telegram, slack, etc.).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a built-in demo without calling a real LLM endpoint.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set log level to DEBUG.",
    )
    return parser


def run_demo():
    """
    Execute the original demo from the existing harness without needing a live
    LLM endpoint.  Useful for verifying that the orchestrator / tools still work.
    """
    print("=" * 60)
    print("DEMO MODE — no LLM endpoint required")
    print("=" * 60)
    history = HistoryBackend()
    orchestrator = AgentOrchestrator(
        tools=[CalculatorTool(), WebSearchTool()],
        history=history,
    )
    try:
        orchestrator.submit_user_message("What is 17 * 4 + 9?")
        result = orchestrator.run_tool("calculator", expr="17 * 4 + 9")
        print(f"\n[calculator] success={result.success}  output={result.output}")

        print("\n--- Tool schemas (OpenAI format) ---")
        for schema in orchestrator.tool_schemas():
            print(json.dumps(schema, indent=2))

        print("\n--- Conversation history ---")
        for entry in orchestrator.context_snapshot():
            print(entry)
    finally:
        orchestrator.shutdown()


from rich.console import Console
from rich.markdown import Markdown

from memory import LivingMemory, LivingSession
from skills import SkillRegistry

_console = Console()

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"


CLEAR_LINE = "\033[2K\r"
MAGENTA = "\033[35m"


def _separator():
    return f"{DIM}{'─' * shutil.get_terminal_size().columns}{RESET}"


_spinner_detail = [""]
_agent_status = {}
_agent_status_lock = threading.Lock()

GREEN_FG = "\033[32m"
YELLOW_FG = "\033[33m"
UP_LINE = "\033[A"


def _set_agent_status(name: str, status: str):
    with _agent_status_lock:
        _agent_status[name] = status


def _clear_agent_status(name: str):
    with _agent_status_lock:
        _agent_status.pop(name, None)


def _run_with_spinner(fn, label="Thinking"):
    _spinner_detail[0] = ""
    with _agent_status_lock:
        _agent_status.clear()
    result_box = [None, None]

    def worker():
        try:
            result_box[0] = fn()
        except Exception as e:
            result_box[1] = e

    t = threading.Thread(target=worker)
    t.start()
    start = time.time()
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    idx = 0
    prev_lines = 0

    cols = shutil.get_terminal_size().columns

    while t.is_alive():
        elapsed = int(time.time() - start)
        frame = frames[idx % len(frames)]
        detail = _spinner_detail[0]

        # Build main line
        main = f"{frame} {label}"
        if detail:
            main += f" · {detail}"
        main += f" · {elapsed}s"

        # Agent status lines
        with _agent_status_lock:
            agents = dict(_agent_status)

        agent_lines = []
        for name, status in agents.items():
            agent_lines.append(f"  ● {name:12s} · {status}")

        # Clear previous output: move up and clear each line
        if prev_lines > 0:
            for _ in range(prev_lines):
                sys.stdout.write(f"\033[A\033[2K")
        sys.stdout.write(f"\r\033[2K")

        # Write new output
        padded = main.ljust(cols)[:cols]
        sys.stdout.write(f"\r{MAGENTA}{padded}{RESET}")

        for al in agent_lines:
            padded_al = al.ljust(cols)[:cols]
            sys.stdout.write(f"\n{GREEN_FG}{padded_al}{RESET}")

        sys.stdout.flush()
        prev_lines = len(agent_lines)
        idx += 1
        t.join(timeout=0.1)

    # Final cleanup
    if prev_lines > 0:
        for _ in range(prev_lines):
            sys.stdout.write(f"\033[A\033[2K")
    sys.stdout.write(f"\r\033[2K")
    sys.stdout.flush()

    if result_box[1]:
        raise result_box[1]
    return result_box[0]


def repl(agent: AgentLoop, config: AgentConfig, registry=None, channel_names=None) -> None:
    tool_count = len(agent._orch._tools)
    mem_words = len(agent._memory.content.split()) if agent._memory and agent._memory.content else 0
    mem_str = f"{mem_words} words" if mem_words else "empty"

    info_parts = [f"Tools: {tool_count}", f"Memory: {mem_str}"]

    if agent._mcp is not None:
        mcp_status = agent._mcp.status()
        servers = mcp_status.get("servers", {})
        alive = [name for name, s in servers.items() if s.get("alive")]
        if alive:
            info_parts.append(f"MCP: {', '.join(alive)}")

    if channel_names:
        info_parts.append(f"Channels: {', '.join(channel_names)}")

    if agent._history.checkpoint_id:
        info_parts.append(f"Session: {agent._history.checkpoint_id}")

    print(_separator())
    print(f"{BOLD}AI Harness{RESET} · {config.model}")
    print(f"{DIM}{' · '.join(info_parts)}{RESET}")
    print(_separator())
    print()

    while True:
        try:
            print(_separator())
            line = input(f"{CYAN}{BOLD}❯{RESET} ")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        if line.strip().startswith('"""') or line.strip().startswith("'''"):
            marker = line.strip()[:3]
            lines = [line.strip()[3:]]
            print(f"{DIM}  ... (paste text, end with {marker}){RESET}")
            try:
                while True:
                    more = input(f"{DIM}  ...{RESET} ")
                    if marker in more:
                        lines.append(more.split(marker)[0])
                        break
                    lines.append(more)
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            prompt = "\n".join(lines).strip()
        else:
            prompt = line.strip()

        print(_separator())

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "/exit", "/quit"):
            print(f"{DIM}Goodbye.{RESET}")
            break

        # Slash command: /skill-name activates a skill directly
        if prompt.startswith("/") and not prompt.startswith("//"):
            skill_name = prompt.lstrip("/").strip()
            from skills import SkillRegistry as _SR
            if skill_name in registry:
                content = registry.load_skill(skill_name)
                if content:
                    agent._history.clear()
                    agent._history.append({"role": "user", "content": f"Activate and follow this skill:\n\n{content}"})
                    print()
                    try:
                        answer = _run_with_spinner(lambda: agent.run("Follow the skill instructions."))
                    except RuntimeError as exc:
                        print(f"\n{BOLD}Error:{RESET} {exc}\n")
                        continue
                    print()
                    _console.print(Markdown(answer))
                    print()
                    continue
                else:
                    print(f"{DIM}Skill '{skill_name}' could not be loaded.{RESET}")
                    continue

        agent._history.clear()
        print()
        try:
            answer = _run_with_spinner(lambda: agent.run(prompt))
        except RuntimeError as exc:
            print(f"\n{BOLD}Error:{RESET} {exc}\n")
            continue
        print()
        _console.print(Markdown(answer))
        print()

    agent._orch.shutdown()
    agent._history.close()
    if agent._session is not None:
        agent._session.clear()
    if agent._mcp is not None:
        agent._mcp.stop()


def _build_agent(args) -> tuple[AgentLoop, AgentConfig]:
    model = args.model
    if args.provider == "vertex-claude" and model == "qwen3:8b":
        model = "claude-sonnet-4-6"

    try:
        config = AgentConfig(
            model=model,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except ValidationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    checkpoint_id = getattr(args, "checkpoint_id", None)
    if not checkpoint_id:
        from datetime import datetime as _dt, timezone as _tz
        checkpoint_id = _dt.now(_tz.utc).strftime("%Y%m%dT%H%M%S")
    history = HistoryBackend(checkpoint_id=checkpoint_id, resume=True)

    registry = SkillRegistry()
    project = None

    if args.provider == "vertex-claude":
        project = args.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            print("Error: --project or GOOGLE_CLOUD_PROJECT required for vertex-claude", file=sys.stderr)
            sys.exit(1)
        llm_client = VertexClaudeClient(
            project=project,
            region=args.region,
            config=config,
        )
        memory = LivingMemory(
            base_url=args.base_url,
            api_key=args.api_key,
            model="qwen3:8b",
            vertex_project=project,
            vertex_region=args.region,
        )
    else:
        llm_client = LLMClient(
            base_url=args.base_url,
            api_key=args.api_key,
            config=config,
        )
        memory = LivingMemory(
            base_url=args.base_url,
            api_key=args.api_key,
            model=model,
        )

    orchestrator = AgentOrchestrator(
        tools=[
            CalculatorTool(),
            WebSearchTool(),
            ShellTool(),
            ReadFileTool(),
            WriteFileTool(),
            ListFilesTool(),
            PythonExecTool(),
            HttpFetchTool(),
            ListSkillsTool(registry),
        ],
        history=history,
    )

    mcp_manager = None
    if args.mcp_config:
        try:
            from mcp_client import MCPClientManager, load_mcp_config
            mcp_config = load_mcp_config(args.mcp_config)
            mcp_manager = MCPClientManager.from_config(mcp_config)
            mcp_manager.start()
        except Exception as exc:
            print(f"{DIM}MCP: failed to start — {exc}{RESET}")

    run_skill = RunSkillTool(registry, mcp_manager=mcp_manager)
    create_skill = CreateSkillTool(registry)
    orchestrator._tools[run_skill.name] = run_skill
    orchestrator._tools[create_skill.name] = create_skill

    # Shared agent registry (used by plan_agents, delegate, list_agents)
    agent_registry = {}

    tool_pool = {t.name: t for t in [
        CalculatorTool(), WebSearchTool(), ShellTool(), ReadFileTool(),
        WriteFileTool(), ListFilesTool(), PythonExecTool(), HttpFetchTool(),
    ]}

    provider_config = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": model,
        "vertex_project": project if args.provider == "vertex-claude" else os.environ.get("GOOGLE_CLOUD_PROJECT"),
        "vertex_region": args.region if args.provider == "vertex-claude" else "global",
    }

    # Pre-defined agents from --agents
    if args.agents:
        try:
            with open(args.agents) as f:
                agents_config = json.load(f)
            for aname, acfg in agents_config.get("agents", {}).items():
                spec = {
                    "name": aname,
                    "role": acfg.get("description", ""),
                    "tools": acfg.get("tools", []),
                    "model": acfg.get("model", "qwen3:8b"),
                }
                agent_registry[aname] = _create_agent_from_spec(spec, provider_config, tool_pool)
            print(f"{DIM}Agents: {', '.join(agent_registry.keys())}{RESET}")
        except Exception as exc:
            print(f"{DIM}Agents: failed to load — {exc}{RESET}")

    # Always register planning + delegation tools
    plan_tool = PlanAgentsTool(provider_config, tool_pool, agent_registry)
    delegate_tool = DelegateTool(agent_registry)
    list_agents_tool = ListAgentsTool(agent_registry)
    orchestrator._tools[plan_tool.name] = plan_tool
    orchestrator._tools[delegate_tool.name] = delegate_tool
    orchestrator._tools[list_agents_tool.name] = list_agents_tool

    vertex_proj = project if args.provider == "vertex-claude" else None
    session = LivingSession(
        base_url=args.base_url,
        api_key=args.api_key,
        model="qwen3:8b" if args.provider == "vertex-claude" else model,
        vertex_project=vertex_proj,
        vertex_region=args.region if vertex_proj else "global",
    )

    agent = AgentLoop(
        llm_client=llm_client,
        orchestrator=orchestrator,
        history=history,
        config=config,
        memory=memory,
        session=session,
        mcp_manager=mcp_manager,
    )
    return agent, config, registry


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger("agent_harness").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)

    if args.list_checkpoints:
        rows = HistoryBackend.list_checkpoints()
        if not rows:
            print("No checkpoints found.")
        else:
            print(f"{'ID':30s}  {'Turns':6s}  {'Size':8s}  Last activity")
            print("-" * 75)
            for r in rows:
                print(f"{r['id']:30s}  {r['turns']:>5d}   {r['size_kb']:>6.1f}KB  {r.get('last_ts', '?')}")
        return

    if args.workspace:
        import pathlib
        ws = pathlib.Path(args.workspace).resolve()
        ws.mkdir(parents=True, exist_ok=True)
        os.chdir(ws)
        global _WORKSPACE
        _WORKSPACE = ws
        print(f"{DIM}Workspace: {ws}{RESET}")

    if args.demo:
        run_demo()
        return

    agent, config, registry = _build_agent(args)

    # Start messaging channels (telegram, etc.)
    channel_router = None
    if args.channels:
        try:
            from channels import load_channels, ChannelRouter

            channels = load_channels(args.channels)
            if channels:
                def agent_factory():
                    sub_history = HistoryBackend()
                    sub_orch = AgentOrchestrator(
                        tools=[t for t in agent._orch._tools.values()],
                        history=sub_history,
                    )
                    sub_loop = AgentLoop(
                        llm_client=agent._llm,
                        orchestrator=sub_orch,
                        history=sub_history,
                        config=agent._config,
                        memory=agent._memory,
                    )
                    return sub_loop, sub_history

                channel_router = ChannelRouter(channels, agent_factory)
                channel_router.start()
        except Exception as exc:
            print(f"{DIM}Channels: failed — {exc}{RESET}")

    if args.prompt:
        print(f"\n{BOLD}User:{RESET} {args.prompt}\n")
        try:
            answer = _run_with_spinner(lambda: agent.run(args.prompt))
        except KeyboardInterrupt:
            print("\n[Interrupted]")
            sys.exit(130)
        except RuntimeError as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            sys.exit(1)
        finally:
            agent._orch.shutdown()
        _console.print(Markdown(answer))
    else:
        ch_names = [c.name for c in channel_router._channels.values()] if channel_router else None
        repl(agent, config, registry, channel_names=ch_names)

    if channel_router:
        channel_router.stop()


if __name__ == "__main__":
    main()
