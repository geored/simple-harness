"""
mcp_client.py — Complete MCP Client for AI Agent Harness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implements MCP (Model Context Protocol) client with:
  - STDIO and HTTP/SSE transports
  - Dynamic tool discovery + registration as RunnableTool
  - BM25-based semantic tool search (no embeddings, no API calls)
  - Multi-server management with lifecycle control
  - Resilient reconnection and health checks

Dependencies: Python stdlib + requests only.
"""

from __future__ import annotations

import abc
import contextlib
import json
import logging
import math
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. RunnableTool ABC (stub — replace with your harness import)
# ─────────────────────────────────────────────────────────────────────────────

class RunnableTool(abc.ABC):
    """Abstract base class every tool in the harness must implement."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def description(self) -> str: ...

    @property
    @abc.abstractmethod
    def parameters_schema(self) -> Dict[str, Any]: ...

    @abc.abstractmethod
    def run(self, **kwargs) -> Any: ...

    # Optional: called by orchestrator to get OpenAI-style function spec
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MCPToolSchema:
    """Raw tool schema as returned by tools/list."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str

    # Usage stats for ranking
    call_count: int = 0
    success_count: int = 0
    last_used: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.call_count, 1)

    @property
    def qualified_name(self) -> str:
        """server::tool — avoids cross-server name conflicts."""
        return f"{self.server_name}::{self.name}"


@dataclass
class ServerConfig:
    name: str
    transport: str                      # "stdio" | "http"
    command: Optional[List[str]] = None # stdio only
    env: Optional[Dict[str, str]] = None
    url: Optional[str] = None          # http only
    timeout: float = 30.0
    max_retries: int = 3


@dataclass
class SearchResult:
    tool: MCPToolSchema
    score: float
    match_reason: str


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transport layer
# ─────────────────────────────────────────────────────────────────────────────

class MCPTransport(abc.ABC):
    """Abstract transport — send a JSON-RPC request, get a response."""

    @abc.abstractmethod
    def send(self, method: str, params: Dict, timeout: float) -> Dict: ...

    @abc.abstractmethod
    def is_alive(self) -> bool: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class STDIOTransport(MCPTransport):
    """
    Manages a subprocess MCP server over stdin/stdout.

    Protocol: newline-delimited JSON-RPC 2.0.
    Each request is written as a single JSON line; each response arrives
    as a single JSON line. We use a background reader thread + per-request
    Event so multiple callers can share the connection safely.
    """

    def __init__(self, config: ServerConfig):
        self._config = config
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._pending: Dict[str, threading.Event] = {}
        self._results: Dict[str, Dict] = {}
        self._reader_thread: Optional[threading.Thread] = None
        self._alive = False
        self._start()

    # ── startup ──────────────────────────────────────────────────────────────

    def _start(self) -> None:
        env = {**os.environ, **(self._config.env or {})}
        self._proc = subprocess.Popen(
            self._config.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,              # line-buffered
        )
        self._alive = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name=f"mcp-reader-{self._config.name}",
        )
        self._reader_thread.start()

        # MCP initialize handshake
        self._initialize()

    def _initialize(self) -> None:
        resp = self.send(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agent-harness", "version": "1.0"},
            },
            timeout=self._config.timeout,
        )
        if "error" in resp:
            raise RuntimeError(f"MCP init error: {resp['error']}")
        # Send initialized notification (no response expected)
        self._write_raw({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    # ── reader thread ─────────────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        """Continuously read lines from subprocess stdout."""
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("STDIO bad JSON: %s", line[:200])
                    continue
                req_id = str(msg.get("id", ""))
                if req_id in self._pending:
                    self._results[req_id] = msg
                    self._pending[req_id].set()
        except Exception as exc:
            logger.error("STDIO reader crashed (%s): %s", self._config.name, exc)
        finally:
            self._alive = False
            # Unblock all waiting callers with an error sentinel
            for ev in self._pending.values():
                ev.set()

    # ── public API ────────────────────────────────────────────────────────────

    def send(self, method: str, params: Dict, timeout: float = 30.0) -> Dict:
        req_id = str(uuid.uuid4())
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}

        ev = threading.Event()
        with self._lock:
            self._pending[req_id] = ev

        try:
            self._write_raw(payload)
            if not ev.wait(timeout=timeout):
                raise TimeoutError(f"MCP timeout on {method} ({self._config.name})")
            result = self._results.pop(req_id, {})
            if not result:
                raise ConnectionError(f"Transport closed while waiting for {method}")
            return result
        finally:
            with self._lock:
                self._pending.pop(req_id, None)

    def _write_raw(self, payload: Dict) -> None:
        line = json.dumps(payload) + "\n"
        with self._lock:
            if self._proc and self._proc.stdin:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()

    def is_alive(self) -> bool:
        return self._alive and self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        self._alive = False
        if self._proc:
            with contextlib.suppress(Exception):
                self._proc.stdin.close()
            with contextlib.suppress(Exception):
                self._proc.terminate()
                self._proc.wait(timeout=5)
        self._proc = None


class HTTPTransport(MCPTransport):
    """
    HTTP + SSE transport for remote MCP servers.

    Uses plain POST for request/response; SSE endpoint for streaming
    responses (falls back to polling if server doesn't support SSE).
    """

    def __init__(self, config: ServerConfig):
        self._config = config
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"
        self._session.headers["Accept"] = "application/json, text/event-stream"
        self._alive = False
        self._initialize()

    def _initialize(self) -> None:
        resp = self.send(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agent-harness", "version": "1.0"},
            },
            timeout=self._config.timeout,
        )
        if "error" in resp:
            raise RuntimeError(f"MCP HTTP init error: {resp['error']}")
        self._alive = True
        # Send initialized notification
        self._post_fire_forget({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def send(self, method: str, params: Dict, timeout: float = 30.0) -> Dict:
        req_id = str(uuid.uuid4())
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        try:
            r = self._session.post(
                self._config.url,
                json=payload,
                timeout=timeout,
                stream=False,
            )
            r.raise_for_status()
            ct = r.headers.get("Content-Type", "")
            if "text/event-stream" in ct:
                return self._parse_sse(r)
            return r.json()
        except requests.Timeout:
            raise TimeoutError(f"HTTP timeout on {method} ({self._config.name})")
        except requests.RequestException as exc:
            self._alive = False
            raise ConnectionError(f"HTTP error on {method}: {exc}") from exc

    def _post_fire_forget(self, payload: Dict) -> None:
        with contextlib.suppress(Exception):
            self._session.post(self._config.url, json=payload, timeout=5)

    @staticmethod
    def _parse_sse(response: requests.Response) -> Dict:
        """Parse a single JSON-RPC response from an SSE stream."""
        data_lines: List[str] = []
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                if data_lines:
                    break
                continue
            if raw_line.startswith("data:"):
                data_lines.append(raw_line[5:].strip())
        blob = "\n".join(data_lines)
        return json.loads(blob) if blob else {}

    def is_alive(self) -> bool:
        return self._alive

    def close(self) -> None:
        self._alive = False
        self._session.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. MCPClient — single-server client
# ─────────────────────────────────────────────────────────────────────────────

class MCPClient:
    """
    Represents a connection to one MCP server.
    Handles tool/resource discovery and invocation.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self._transport: Optional[MCPTransport] = None
        self._tools: Dict[str, MCPToolSchema] = {}          # name → schema
        self._tool_cache_ts: float = 0.0
        self._cache_ttl: float = 300.0                       # 5 min cache
        self._connect()

    # ── connection ────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        if self.config.transport == "stdio":
            self._transport = STDIOTransport(self.config)
        elif self.config.transport == "http":
            self._transport = HTTPTransport(self.config)
        else:
            raise ValueError(f"Unknown transport: {self.config.transport}")
        logger.info("Connected to MCP server: %s", self.config.name)

    def reconnect(self) -> bool:
        """Attempt to reconnect. Returns True on success."""
        try:
            if self._transport:
                self._transport.close()
            self._connect()
            self._tools = {}            # invalidate cache
            self._tool_cache_ts = 0.0
            return True
        except Exception as exc:
            logger.error("Reconnect failed (%s): %s", self.config.name, exc)
            return False

    def is_alive(self) -> bool:
        return self._transport is not None and self._transport.is_alive()

    def close(self) -> None:
        if self._transport:
            self._transport.close()

    # ── JSON-RPC helper ───────────────────────────────────────────────────────

    def _rpc(self, method: str, params: Dict | None = None, timeout: float | None = None) -> Any:
        if not self.is_alive():
            raise ConnectionError(f"Server {self.config.name} is not connected")
        response = self._transport.send(
            method,
            params or {},
            timeout=timeout or self.config.timeout,
        )
        if "error" in response:
            err = response["error"]
            raise RuntimeError(f"MCP error [{err.get('code')}]: {err.get('message')}")
        return response.get("result", {})

    # ── tool discovery ────────────────────────────────────────────────────────

    def list_tools(self, force_refresh: bool = False) -> List[MCPToolSchema]:
        now = time.time()
        if not force_refresh and (now - self._tool_cache_ts) < self._cache_ttl and self._tools:
            return list(self._tools.values())

        result = self._rpc("tools/list")
        raw_tools = result.get("tools", [])
        new_tools: Dict[str, MCPToolSchema] = {}

        for t in raw_tools:
            name = t["name"]
            # Preserve stats from previous version
            existing = self._tools.get(name)
            schema = MCPToolSchema(
                name=name,
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
                server_name=self.config.name,
                call_count=existing.call_count if existing else 0,
                success_count=existing.success_count if existing else 0,
                last_used=existing.last_used if existing else 0.0,
            )
            new_tools[name] = schema

        self._tools = new_tools
        self._tool_cache_ts = now
        logger.debug("Discovered %d tools from %s", len(new_tools), self.config.name)
        return list(self._tools.values())

    # ── tool invocation ───────────────────────────────────────────────────────

    def call_tool(self, tool_name: str, arguments: Dict) -> Any:
        schema = self._tools.get(tool_name)
        if schema is None:
            # Try without cache in case tool was added recently
            self.list_tools(force_refresh=True)
            schema = self._tools.get(tool_name)
            if schema is None:
                raise KeyError(f"Tool '{tool_name}' not found on server '{self.config.name}'")

        schema.call_count += 1
        schema.last_used = time.time()

        try:
            result = self._rpc("tools/call", {"name": tool_name, "arguments": arguments})
            schema.success_count += 1
            # Flatten content array → string
            return self._extract_content(result)
        except Exception:
            raise   # caller handles stats

    @staticmethod
    def _extract_content(result: Dict) -> str:
        """Convert MCP content array to a plain string."""
        content = result.get("content", [])
        if not content:
            return json.dumps(result)
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    parts.append(f"[image: {item.get('url', '')}]")
                else:
                    parts.append(json.dumps(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    # ── resources (bonus) ─────────────────────────────────────────────────────

    def list_resources(self) -> List[Dict]:
        with contextlib.suppress(Exception):
            return self._rpc("resources/list").get("resources", [])
        return []

    def read_resource(self, uri: str) -> str:
        result = self._rpc("resources/read", {"uri": uri})
        return self._extract_content(result)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tool Search Engine — BM25 + trie prefix search, no embeddings
# ─────────────────────────────────────────────────────────────────────────────

class ToolSearchEngine:
    """
    Fast offline tool search using:
      1. BM25 term-frequency scoring over tool names + descriptions
      2. Prefix trie for instant name autocomplete
      3. Recency + success-rate boosting for ranking

    Zero external dependencies — uses Python stdlib only.
    """

    # BM25 parameters
    K1 = 1.5
    B  = 0.75

    def __init__(self):
        self._tools: Dict[str, MCPToolSchema] = {}     # qualified_name → schema
        self._index: Dict[str, Dict[str, int]] = {}    # term → {qname: freq}
        self._doc_lengths: Dict[str, int] = {}         # qname → token count
        self._avg_doc_len: float = 0.0
        self._trie: Dict = {}                          # prefix trie for names

    # ── index management ──────────────────────────────────────────────────────

    def index_tool(self, schema: MCPToolSchema) -> None:
        qname = schema.qualified_name
        self._tools[qname] = schema

        # Tokenise name + description (simple word split)
        tokens = self._tokenize(f"{schema.name} {schema.description}")
        freq: Dict[str, int] = defaultdict(int)
        for tok in tokens:
            freq[tok] += 1

        # Update inverted index
        for term, count in freq.items():
            if term not in self._index:
                self._index[term] = {}
            self._index[term][qname] = count

        self._doc_lengths[qname] = len(tokens)
        self._avg_doc_len = sum(self._doc_lengths.values()) / len(self._doc_lengths)

        # Update trie with qualified name and bare name
        self._trie_insert(schema.qualified_name)
        self._trie_insert(schema.name)

    def remove_server(self, server_name: str) -> None:
        """Remove all tools from a disconnected server."""
        gone = [q for q in self._tools if q.startswith(f"{server_name}::")]
        for qname in gone:
            schema = self._tools.pop(qname, None)
            self._doc_lengths.pop(qname, None)
            for term_dict in self._index.values():
                term_dict.pop(qname, None)
        if self._doc_lengths:
            self._avg_doc_len = sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def reindex_server(self, server_name: str, tools: List[MCPToolSchema]) -> None:
        self.remove_server(server_name)
        for t in tools:
            self.index_tool(t)

    # ── search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        server_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Returns top_k tools ranked by BM25 + recency + success-rate.
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scores: Dict[str, float] = defaultdict(float)
        N = len(self._tools)
        if N == 0:
            return []

        for term in query_terms:
            if term not in self._index:
                continue
            postings = self._index[term]
            # IDF (smoothed)
            df = len(postings)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            for qname, tf in postings.items():
                dl = self._doc_lengths.get(qname, 1)
                norm_tf = (tf * (self.K1 + 1)) / (
                    tf + self.K1 * (1 - self.B + self.B * dl / max(self._avg_doc_len, 1))
                )
                scores[qname] += idf * norm_tf

        # Filter by server
        if server_filter:
            scores = {q: s for q, s in scores.items() if q.startswith(f"{server_filter}::")}

        # Boost by recency and success rate
        now = time.time()
        for qname in list(scores):
            schema = self._tools[qname]
            # Recency boost: 0–0.3 extra (decays over 24 h)
            age_h = (now - schema.last_used) / 3600 if schema.last_used else 24
            recency_boost = 0.3 * math.exp(-age_h / 24)
            # Trust boost: 0–0.2 extra based on success rate
            trust_boost = 0.2 * schema.success_rate
            scores[qname] += recency_boost + trust_boost

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for qname, score in ranked:
            schema = self._tools[qname]
            reason = self._explain_match(query_terms, schema)
            results.append(SearchResult(tool=schema, score=score, match_reason=reason))
        return results

    def prefix_search(self, prefix: str, limit: int = 10) -> List[MCPToolSchema]:
        """Instant prefix autocomplete from trie."""
        prefix = prefix.lower()
        names = self._trie_collect(prefix, limit)
        out = []
        for name in names:
            # Could be bare name or qualified name
            if "::" in name:
                schema = self._tools.get(name)
            else:
                # Find first matching qualified name
                schema = next(
                    (s for q, s in self._tools.items() if s.name == name), None
                )
            if schema:
                out.append(schema)
        return out[:limit]

    def all_tools(self, server: Optional[str] = None) -> List[MCPToolSchema]:
        if server:
            return [s for s in self._tools.values() if s.server_name == server]
        return list(self._tools.values())

    # ── trie helpers ──────────────────────────────────────────────────────────

    def _trie_insert(self, word: str) -> None:
        node = self._trie
        for ch in word.lower():
            node = node.setdefault(ch, {})
        node["$"] = word   # terminal stores original string

    def _trie_collect(self, prefix: str, limit: int) -> List[str]:
        node = self._trie
        for ch in prefix:
            if ch not in node:
                return []
            node = node[ch]
        results: List[str] = []
        self._trie_dfs(node, results, limit)
        return results

    def _trie_dfs(self, node: Dict, results: List[str], limit: int) -> None:
        if len(results) >= limit:
            return
        if "$" in node:
            results.append(node["$"])
        for ch, child in node.items():
            if ch == "$":
                continue
            self._trie_dfs(child, results, limit)
            if len(results) >= limit:
                return

    # ── text utils ────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase, split on non-alphanumeric, remove stopwords."""
        STOPWORDS = {
            "a", "an", "the", "is", "it", "in", "on", "at", "to",
            "for", "of", "and", "or", "with", "this", "that", "can",
            "be", "are", "was", "has", "have", "do", "does", "will",
        }
        tokens = re.split(r"[^a-z0-9]+", text.lower())
        return [t for t in tokens if t and t not in STOPWORDS and len(t) > 1]

    @staticmethod
    def _explain_match(query_terms: List[str], schema: MCPToolSchema) -> str:
        text = f"{schema.name} {schema.description}".lower()
        matched = [t for t in query_terms if t in text]
        if not matched:
            return "boosted by recency/trust"
        return f"matched: {', '.join(matched[:4])}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. MCPToolBridge — wraps MCPToolSchema as a RunnableTool
# ─────────────────────────────────────────────────────────────────────────────

class MCPToolBridge(RunnableTool):
    """
    Adapts a single MCP tool into the harness RunnableTool interface.
    Routes calls through the owning MCPClient with retry logic.
    """

    def __init__(self, schema: MCPToolSchema, client: MCPClient, max_retries: int = 2):
        self._schema = schema
        self._client = client
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        # Use qualified name to avoid conflicts
        return self._schema.qualified_name.replace("::", "__")

    @property
    def description(self) -> str:
        return (
            f"[MCP:{self._schema.server_name}] {self._schema.description}"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return self._schema.input_schema

    def run(self, **kwargs) -> Any:
        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(1, self._max_retries + 2):
            try:
                return self._client.call_tool(self._schema.name, kwargs)
            except (TimeoutError, ConnectionError) as exc:
                last_exc = exc
                logger.warning(
                    "MCPToolBridge retry %d/%d for %s: %s",
                    attempt, self._max_retries + 1, self.name, exc,
                )
                if attempt <= self._max_retries:
                    reconnected = self._client.reconnect()
                    if not reconnected:
                        break
            except Exception as exc:
                raise  # Non-transient errors bubble up immediately
        raise last_exc


# ─────────────────────────────────────────────────────────────────────────────
# 7. Lifecycle Manager — health checks + reconnection
# ─────────────────────────────────────────────────────────────────────────────

class ServerLifecycleManager:
    """
    Runs background health-check loop.
    Attempts reconnection on failure with exponential back-off.
    """

    def __init__(self, check_interval: float = 30.0):
        self._clients: Dict[str, MCPClient] = {}
        self._check_interval = check_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._on_reconnect: Optional[Callable[[str, MCPClient], None]] = None
        self._backoff: Dict[str, float] = {}   # server_name → next retry ts

    def register(self, name: str, client: MCPClient) -> None:
        self._clients[name] = client

    def unregister(self, name: str) -> None:
        self._clients.pop(name, None)

    def on_reconnect(self, callback: Callable[[str, MCPClient], None]) -> None:
        """Register callback invoked after a successful reconnection."""
        self._on_reconnect = callback

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="mcp-lifecycle"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop_event.wait(timeout=self._check_interval):
            now = time.time()
            for name, client in list(self._clients.items()):
                # Respect back-off window
                if now < self._backoff.get(name, 0):
                    continue
                if not client.is_alive():
                    logger.warning("Server %s appears dead — reconnecting…", name)
                    success = client.reconnect()
                    if success:
                        logger.info("Server %s reconnected.", name)
                        self._backoff.pop(name, None)
                        if self._on_reconnect:
                            self._on_reconnect(name, client)
                    else:
                        # Exponential back-off: 30s → 60s → 120s → 300s max
                        prev = self._backoff.get(name, now)
                        delay = min(300, max(30, (prev - now) * 2 + 30))
                        self._backoff[name] = now + delay
                        logger.error(
                            "Reconnect failed for %s; next try in %.0fs", name, delay
                        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. MCPClientManager — top-level facade
# ─────────────────────────────────────────────────────────────────────────────

class MCPClientManager:
    """
    Central coordinator for all MCP server connections.

    Usage:
        mgr = MCPClientManager.from_config(config_dict)
        mgr.start()

        # Get all tools as RunnableTool for the orchestrator
        tools = mgr.get_all_runnable_tools()

        # Search
        results = mgr.search_tools("create a jira ticket", top_k=3)

        # Smart context injection for LLM
        snippet = mgr.build_tool_context_snippet(user_prompt)

        mgr.stop()
    """

    def __init__(self, health_check_interval: float = 30.0):
        self._clients: Dict[str, MCPClient] = {}
        self._search_engine = ToolSearchEngine()
        self._lifecycle = ServerLifecycleManager(check_interval=health_check_interval)
        self._lifecycle.on_reconnect(self._on_server_reconnect)
        self._lock = threading.RLock()

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: Dict, **kwargs) -> "MCPClientManager":
        mgr = cls(**kwargs)
        for name, cfg in config.get("servers", {}).items():
            sc = ServerConfig(
                name=name,
                transport=cfg["transport"],
                command=cfg.get("command"),
                env=cfg.get("env"),
                url=cfg.get("url"),
                timeout=cfg.get("timeout", 30.0),
                max_retries=cfg.get("max_retries", 3),
            )
            mgr.add_server(sc)
        return mgr

    @classmethod
    def from_config_file(cls, path: str, **kwargs) -> "MCPClientManager":
        with open(path) as f:
            return cls.from_config(json.load(f), **kwargs)

    # ── server management ─────────────────────────────────────────────────────

    def add_server(self, config: ServerConfig) -> bool:
        """Connect to a new MCP server and index its tools. Returns True on success."""
        try:
            client = MCPClient(config)
            tools = client.list_tools()
            with self._lock:
                self._clients[config.name] = client
                self._search_engine.reindex_server(config.name, tools)
                self._lifecycle.register(config.name, client)
            logger.info("Added server '%s' with %d tools", config.name, len(tools))
            return True
        except Exception as exc:
            logger.error("Failed to add server '%s': %s", config.name, exc)
            return False

    def remove_server(self, name: str) -> None:
        with self._lock:
            client = self._clients.pop(name, None)
            self._search_engine.remove_server(name)
            self._lifecycle.unregister(name)
        if client:
            client.close()

    def _on_server_reconnect(self, name: str, client: MCPClient) -> None:
        """Called by lifecycle manager after reconnect — re-discover tools."""
        try:
            tools = client.list_tools(force_refresh=True)
            with self._lock:
                self._search_engine.reindex_server(name, tools)
            logger.info("Re-indexed %d tools from %s after reconnect", len(tools), name)
        except Exception as exc:
            logger.warning("Could not re-index %s after reconnect: %s", name, exc)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._lifecycle.start()

    def stop(self) -> None:
        self._lifecycle.stop()
        with self._lock:
            for client in self._clients.values():
                client.close()

    def __enter__(self) -> "MCPClientManager":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── tool access ───────────────────────────────────────────────────────────

    def get_all_runnable_tools(self) -> List[RunnableTool]:
        """Return all MCP tools as RunnableTool instances for the orchestrator."""
        tools = []
        with self._lock:
            for name, client in self._clients.items():
                for schema in client.list_tools():
                    tools.append(MCPToolBridge(schema, client))
        return tools

    def get_tool(self, qualified_name: str) -> Optional[RunnableTool]:
        """Resolve server::tool_name → RunnableTool."""
        if "::" not in qualified_name:
            return None
        server_name, tool_name = qualified_name.split("::", 1)
        client = self._clients.get(server_name)
        if not client:
            return None
        schemas = {s.name: s for s in client.list_tools()}
        schema = schemas.get(tool_name)
        if not schema:
            return None
        return MCPToolBridge(schema, client)

    # ── search API ────────────────────────────────────────────────────────────

    def search_tools(
        self,
        query: str,
        top_k: int = 5,
        server_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search across all connected MCP servers."""
        return self._search_engine.search(query, top_k=top_k, server_filter=server_filter)

    def autocomplete_tool(self, prefix: str, limit: int = 10) -> List[MCPToolSchema]:
        return self._search_engine.prefix_search(prefix, limit=limit)

    def list_all_tool_schemas(self, server: Optional[str] = None) -> List[MCPToolSchema]:
        return self._search_engine.all_tools(server=server)

    # ── context injection ─────────────────────────────────────────────────────

    def build_tool_context_snippet(
        self,
        user_prompt: str,
        max_tools: int = 8,
        max_chars: int = 2000,
    ) -> str:
        """
        Given a user prompt, return a compact text snippet describing the most
        relevant MCP tools. Inject this into the LLM system prompt BEFORE the
        LLM decides which tool to call.

        Strategy:
          1. BM25 search the prompt against all tools
          2. Rank by score; trim to max_chars
          3. Format as a concise bulleted list
        """
        results = self.search_tools(user_prompt, top_k=max_tools)
        if not results:
            return ""

        lines = ["[Available MCP Tools — relevant to your request]"]
        total = len(lines[0])

        for r in results:
            t = r.tool
            # Compact description: qualified_name — first sentence of description
            first_sentence = t.description.split(".")[0].strip()
            line = f"• {t.qualified_name}({self._param_summary(t.input_schema)}): {first_sentence}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)

        lines.append(
            "\nTo call: use tool name as 'server__toolname' (double underscore)."
        )
        return "\n".join(lines)

    @staticmethod
    def _param_summary(schema: Dict) -> str:
        """Compact parameter summary: name:type, name:type"""
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        parts = []
        for pname, pdef in list(props.items())[:4]:   # max 4 params shown
            ptype = pdef.get("type", "any")
            marker = "" if pname in required else "?"
            parts.append(f"{pname}{marker}:{ptype}")
        if len(props) > 4:
            parts.append("…")
        return ", ".join(parts)

    # ── server status ─────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return health summary for all servers."""
        result = {}
        with self._lock:
            for name, client in self._clients.items():
                tools = client.list_tools()
                result[name] = {
                    "alive": client.is_alive(),
                    "transport": client.config.transport,
                    "tool_count": len(tools),
                    "tools": [
                        {
                            "name": t.name,
                            "calls": t.call_count,
                            "success_rate": round(t.success_rate, 2),
                            "last_used": t.last_used,
                        }
                        for t in tools
                    ],
                }
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 9. Configuration Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_mcp_config(source: str | Dict) -> Dict:
    """
    Load MCP server config from:
      - A dict (already parsed)
      - A file path (JSON)
      - A JSON string

    Returns validated config dict.
    """
    if isinstance(source, dict):
        config = source
    elif isinstance(source, str):
        try:
            # Try as JSON string first
            config = json.loads(source)
        except json.JSONDecodeError:
            # Treat as file path
            with open(source) as f:
                config = json.load(f)
    else:
        raise TypeError(f"Expected dict or str, got {type(source)}")

    _validate_config(config)
    return config


def _validate_config(config: Dict) -> None:
    if "servers" not in config:
        raise ValueError("Config must have a 'servers' key")
    for name, srv in config["servers"].items():
        t = srv.get("transport")
        if t not in ("stdio", "http"):
            raise ValueError(f"Server '{name}': transport must be 'stdio' or 'http'")
        if t == "stdio" and not srv.get("command"):
            raise ValueError(f"Server '{name}': stdio transport requires 'command'")
        if t == "http" and not srv.get("url"):
            raise ValueError(f"Server '{name}': http transport requires 'url'")


# ─────────────────────────────────────────────────────────────────────────────
# 10. AgentOrchestrator integration helpers
# ─────────────────────────────────────────────────────────────────────────────

class MCPAwareOrchestrator:
    """
    Thin wrapper that enriches an existing AgentOrchestrator with MCP tools.

    Usage (pseudocode):
        base_orchestrator = AgentOrchestrator(llm=..., tools=[calc, shell])
        mcp_mgr = MCPClientManager.from_config(cfg)
        orchestrator = MCPAwareOrchestrator(base_orchestrator, mcp_mgr)
        orchestrator.run("create a github issue for the failing tests")
    """

    def __init__(self, base_orchestrator, mcp_manager: MCPClientManager):
        self._base = base_orchestrator
        self._mcp = mcp_manager

    def _enrich_context(self, prompt: str) -> str:
        snippet = self._mcp.build_tool_context_snippet(prompt)
        if snippet:
            return f"{snippet}\n\n---\n{prompt}"
        return prompt

    def _all_tools(self) -> List[RunnableTool]:
        base_tools: List[RunnableTool] = getattr(self._base, "tools", [])
        mcp_tools = self._mcp.get_all_runnable_tools()
        return base_tools + mcp_tools

    def run(self, prompt: str) -> str:
        enriched = self._enrich_context(prompt)
        all_tools = self._all_tools()
        # Inject into base orchestrator — adapt to your actual API
        return self._base.run(enriched, tools=all_tools)

    def search_and_run(self, query: str, tool_query: str | None = None) -> str:
        """
        Search for the most relevant tool, then run the agent with it pre-selected.
        """
        results = self._mcp.search_tools(tool_query or query, top_k=3)
        if results:
            best = results[0]
            logger.info(
                "Pre-selected tool: %s (score=%.2f, reason=%s)",
                best.tool.qualified_name, best.score, best.match_reason,
            )
        return self.run(query)
