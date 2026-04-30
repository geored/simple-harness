import argparse
import hashlib
import inspect
import json
import logging
import os
import shutil
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
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
# Subsystem 2: Checkpointing
# ---------------------------------------------------------------------------

class BackendType(str, Enum):
    REDIS = "redis"
    POSTGRES = "postgres"
    MEMORY = "memory"


class CheckpointV1(BaseModel):
    schema_version: int = 1
    step: int
    messages: list[dict]
    tool_results: dict[str, Any]


class CheckpointV2(BaseModel):
    schema_version: int = 2
    phase: int
    messages: list[dict]
    tool_results: dict[str, Any]
    backend_type: str
    written_at: str
    field_fingerprint: str

    @classmethod
    def field_names(cls):
        return frozenset(cls.model_fields.keys())

    @classmethod
    def compute_fingerprint(cls):
        canonical = json.dumps(sorted(cls.field_names()), separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


MIGRATIONS = {
    (1, 2): lambda old: {
        **{k: v for k, v in old.items() if k != "step"},
        "phase": old["step"],
        "schema_version": 2,
        "backend_type": old.get("backend_type", BackendType.MEMORY.value),
        "written_at": datetime.now(timezone.utc).isoformat(),
        "field_fingerprint": CheckpointV2.compute_fingerprint(),
    },
}


# ---------------------------------------------------------------------------
# Subsystem 3: History Backend
# ---------------------------------------------------------------------------

class HistoryBackend:
    def __init__(self):
        self._lock = threading.Lock()
        self._history = []

    def append(self, entry):
        with self._lock:
            self._history.append(entry)

    def snapshot(self):
        with self._lock:
            return list(self._history)

    def clear(self):
        with self._lock:
            self._history.clear()

    def __len__(self):
        with self._lock:
            return len(self._history)


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

    def run(self, path: str, content: str) -> str:
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
        "Execute a multi-step skill by name. Available skills can be listed "
        "with list_skills. Input: skill_name (str), inputs_json (str) — "
        "JSON object with the skill's required inputs."
    )
    execution_mode = "async"
    timeout_seconds = 120.0

    def __init__(self, registry, orchestrator, llm_client, mcp_manager=None):
        self._registry = registry
        self._orchestrator = orchestrator
        self._llm_client = llm_client
        self._mcp = mcp_manager

    def run(self, skill_name: str, inputs_json: str) -> str:
        from skills import SkillEngine
        skill_data = self._registry.get(skill_name)
        if skill_data is None:
            available = [s["name"] for s in self._registry.list_skills()]
            return f"Skill '{skill_name}' not found. Available: {available}"

        try:
            inputs = json.loads(inputs_json) if inputs_json else {}
        except json.JSONDecodeError:
            return f"Invalid JSON for inputs: {inputs_json}"

        def tool_runner(tool_name, **kwargs):
            tool = self._orchestrator._tools.get(tool_name)
            if tool is None and self._mcp is not None:
                tool = self._mcp.get_tool(tool_name)
            if tool is None:
                raise RuntimeError(f"Unknown tool: {tool_name}")
            return tool.run(**kwargs)

        def llm_runner(prompt):
            messages = [{"role": "user", "content": prompt}]
            response = self._llm_client.chat(messages=messages)
            return response.text or ""

        def on_step(num, total, step_id, label):
            _spinner_detail[0] = f"Step {num}/{total} · {label}"

        engine = SkillEngine(
            manifest=skill_data["manifest"],
            sequence=skill_data["sequence"],
            tool_runner=tool_runner,
            llm_runner=llm_runner,
            on_step=on_step,
        )

        try:
            state = engine.run(initial_inputs=inputs)
            sys.stdout.write(CLEAR_LINE)
            sys.stdout.flush()
            last_step = list(state.steps.values())[-1] if state.steps else None
            if last_step:
                return json.dumps(last_step.output, default=str)
            return "Skill completed with no output"
        except Exception as exc:
            sys.stdout.write(CLEAR_LINE)
            sys.stdout.flush()
            return f"Skill '{skill_name}' failed: {exc}"


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
        "Create a new reusable multi-step skill. Inputs: skill_name (str), "
        "manifest_yaml (str) — YAML with skills list defining tool schemas, "
        "sequence_json (str) — JSON execution sequence. "
        "Steps reference prior outputs with $step_id or $step_id.field. "
        "Inputs from the caller use $input.field_name. "
        "Steps can be tool calls (skill: tool_name, inputs: {...}) or "
        "LLM calls (type: llm, prompt: text with $refs). "
        "Example sequence: {\"sequence_id\": \"my_skill\", \"steps\": ["
        "{\"step_id\": \"s1\", \"skill\": \"read_file\", \"inputs\": {\"path\": \"$input.file_path\"}, "
        "\"retry\": {\"count\": 1, \"delay_seconds\": 0}, \"on_error\": \"abort\"}, "
        "{\"step_id\": \"s2\", \"type\": \"llm\", \"prompt\": \"Analyze: $s1\", \"on_error\": \"abort\"}]}"
    )
    execution_mode = "sync"
    timeout_seconds = 5.0

    def __init__(self, registry):
        self._registry = registry

    def run(self, skill_name: str, manifest_yaml: str, sequence_json: str) -> str:
        import yaml as _yaml
        try:
            manifest = _yaml.safe_load(manifest_yaml)
            sequence = json.loads(sequence_json)
        except Exception as exc:
            return f"Invalid skill definition: {exc}"

        self._registry.register(skill_name, manifest, sequence)
        return f"Skill '{skill_name}' created and saved to disk."


class DelegateTool(RunnableTool):
    name = "delegate"
    description = "Delegate a task to a specialist agent. Input: agent_name (str), task (str)."
    execution_mode = "async"
    timeout_seconds = 120.0

    def __init__(self, agent_registry):
        self._agents = agent_registry
        agents_desc = ", ".join(f"{n} ({a['description']})" for n, a in self._agents.items())
        self.description = (
            f"Delegate a task to a specialist agent. Available agents: {agents_desc}. "
            "Input: agent_name (str), task (str)."
        )

    def run(self, agent_name: str, task: str) -> str:
        agent = self._agents.get(agent_name)
        if agent is None:
            available = list(self._agents.keys())
            return f"Agent '{agent_name}' not found. Available: {available}"

        handler = agent.get("handler")
        if handler is None:
            return f"Agent '{agent_name}' has no handler"

        _spinner_detail[0] = f"Delegating to {agent_name}"
        try:
            result = handler(task)
            return result if isinstance(result, str) else json.dumps(result, default=str)
        except Exception as exc:
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
    ):
        self._llm = llm_client
        self._orch = orchestrator
        self._history = history
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
                if self._session is not None:
                    self._session.consolidate(snapshot)
                if self._memory is not None:
                    self._memory.consolidate(snapshot)
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


def _run_with_spinner(fn, label="Thinking"):
    _spinner_detail[0] = ""
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
    while t.is_alive():
        elapsed = int(time.time() - start)
        frame = frames[idx % len(frames)]
        detail = _spinner_detail[0]
        line = f"{MAGENTA}{frame}{RESET} {DIM}{label}"
        if detail:
            line += f" · {detail}"
        line += f" · {elapsed}s{RESET}"
        sys.stdout.write(f"{CLEAR_LINE}{line}")
        sys.stdout.flush()
        idx += 1
        t.join(timeout=0.1)

    sys.stdout.write(CLEAR_LINE)
    sys.stdout.flush()

    if result_box[1]:
        raise result_box[1]
    return result_box[0]


def repl(agent: AgentLoop, config: AgentConfig) -> None:
    print(_separator())
    print(f"{BOLD}AI Harness{RESET} · {config.model}")
    print(f"{DIM}Tools: {', '.join(agent._orch._tools.keys())}{RESET}")
    if agent._mcp is not None:
        mcp_status = agent._mcp.status()
        servers = mcp_status.get("servers", {})
        parts = [f"{name} ({s.get('tools', 0)} tools)" for name, s in servers.items() if s.get("alive")]
        if parts:
            print(f"{DIM}MCP: {' · '.join(parts)}{RESET}")
    if agent._memory is not None:
        words = len(agent._memory.content.split()) if agent._memory.content else 0
        status = f"{words} words" if words else "empty"
        print(f"{DIM}Memory: {status}{RESET}")
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

    history = HistoryBackend()
    registry = SkillRegistry()

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

    run_skill = RunSkillTool(registry, orchestrator, llm_client, mcp_manager=mcp_manager)
    create_skill = CreateSkillTool(registry)
    orchestrator._tools[run_skill.name] = run_skill
    orchestrator._tools[create_skill.name] = create_skill

    # Sub-agents
    if args.agents:
        try:
            with open(args.agents) as f:
                agents_config = json.load(f)

            all_tools_map = {t.name: t for t in [
                CalculatorTool(), WebSearchTool(), ShellTool(), ReadFileTool(),
                WriteFileTool(), ListFilesTool(), PythonExecTool(), HttpFetchTool(),
            ]}

            agent_registry = {}
            for agent_name, acfg in agents_config.get("agents", {}).items():
                sub_history = HistoryBackend()
                sub_tools = [all_tools_map[t] for t in acfg.get("tools", []) if t in all_tools_map]
                sub_orch = AgentOrchestrator(tools=sub_tools, history=sub_history)

                if acfg.get("provider") == "vertex-claude":
                    sub_proj = acfg.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT")
                    sub_config = AgentConfig(
                        model=acfg.get("model", "claude-sonnet-4-6"),
                        max_retries=3, timeout_seconds=60.0,
                        max_tokens=acfg.get("max_tokens", 4096), temperature=0.7,
                    )
                    sub_llm = VertexClaudeClient(project=sub_proj, region=acfg.get("region", "global"), config=sub_config)
                else:
                    sub_config = AgentConfig(
                        model=acfg.get("model", "qwen3:8b"),
                        max_retries=3, timeout_seconds=60.0,
                        max_tokens=acfg.get("max_tokens", 4096), temperature=0.7,
                    )
                    sub_llm = LLMClient(
                        base_url=acfg.get("base_url", args.base_url),
                        api_key=acfg.get("api_key", args.api_key),
                        config=sub_config,
                    )

                sub_loop = AgentLoop(
                    llm_client=sub_llm, orchestrator=sub_orch,
                    history=sub_history, config=sub_config,
                )

                def make_handler(loop, hist):
                    def handler(task):
                        hist.clear()
                        return loop.run(task)
                    return handler

                agent_registry[agent_name] = {
                    "description": acfg.get("description", ""),
                    "model": acfg.get("model", ""),
                    "handler": make_handler(sub_loop, sub_history),
                }

            delegate_tool = DelegateTool(agent_registry)
            list_agents_tool = ListAgentsTool(agent_registry)
            orchestrator._tools[delegate_tool.name] = delegate_tool
            orchestrator._tools[list_agents_tool.name] = list_agents_tool
            print(f"{DIM}Agents: {', '.join(agent_registry.keys())}{RESET}")
        except Exception as exc:
            print(f"{DIM}Agents: failed to load — {exc}{RESET}")

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
    return agent, config


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger("agent_harness").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)

    if args.demo:
        run_demo()
        return

    agent, config = _build_agent(args)

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
        repl(agent, config)


if __name__ == "__main__":
    main()
