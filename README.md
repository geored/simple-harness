<p align="center">
  <h1 align="center">AI Harness</h1>
  <p align="center">
    A single-file agentic runtime with tool use, living memory, multi-agent planning, MCP integration, and messaging channels.
  </p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#features">Features</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#tools">Tools</a> &middot;
  <a href="#providers">Providers</a> &middot;
  <a href="#skills">Skills</a> &middot;
  <a href="#multi-agent">Multi-Agent</a> &middot;
  <a href="#mcp">MCP</a> &middot;
  <a href="#channels">Channels</a>
</p>

---

## What is this?

AI Harness is a **~4,600-line Python runtime** that turns any OpenAI-compatible LLM into an autonomous agent. It provides a complete agent loop with tool execution, persistent memory, conversation checkpointing, multi-agent orchestration, MCP server connectivity, and external messaging channels -- all from a single `python harness.py` command.

```
                     ┌─────────────────────────────────────────────┐
                     │              AI Harness Runtime              │
                     │                                             │
   Terminal REPL ──▶ │  Agent Loop (ReAct)                        │
   Telegram     ──▶ │    ├── LLM Client (OpenAI / Vertex Claude) │
   Slack (soon) ──▶ │    ├── Tool Orchestrator (14 built-in)      │
                     │    ├── Output Observer (self-correction)    │
                     │    ├── Living Memory (cross-session)        │
                     │    ├── Session Context (within-session)     │
                     │    ├── Skill Registry (agentskills.io)      │
                     │    ├── MCP Client (STDIO + HTTP)            │
                     │    └── Multi-Agent Planner (plan + delegate)│
                     │                                             │
                     │  History Backend (JSONL checkpointing)      │
                     └─────────────────────────────────────────────┘
```

## Quick Start

```bash
# Clone and set up
git clone <repo-url> && cd ai-harness
python3 -m venv .venv
.venv/bin/pip install requests pydantic rich ddgs pyyaml anthropic

# Run with local Ollama (default)
.venv/bin/python harness.py

# Run with Claude via Vertex AI
.venv/bin/python harness.py --provider vertex-claude

# Single-shot mode (no REPL)
.venv/bin/python harness.py "What is the mass of Jupiter in kilograms?"
```

Create a `.env` file for secrets (auto-loaded on startup):

```env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
```

## Features

### Autonomous Agent Loop

A ReAct-style loop where the LLM reasons, selects tools, executes them, observes results, and repeats until it has a final answer. Includes loop-detection to prevent runaway tool calls (3 identical calls trigger a warning, 5 force an exit).

### Self-Observation

An `OutputObserver` classifies every response the agent produces -- factual answer, uncertain claim, partial completion, generated code -- and decides whether to accept it or trigger a refinement loop. Heuristic classification runs locally; uncertain cases escalate to the LLM itself.

```
  ● web_search(quantum computing 2026)
  ● read_file(results.md)
  ● [observer] uncertain_answer → refining
  ● web_search(quantum computing latest papers)
  ● [observer] factual_answer → terminal (90%)
```

### Living Memory

Two self-rewriting documents maintained by the LLM after every interaction:

| Document | Scope | Persists |
|---|---|---|
| `memory.md` | Cross-session | User identity, preferences, long-term facts |
| `session.md` | Current session | Conversation flow, topic context |
| `memory_archive.md` | Backup | Every version before rewrite (90-day retention) |

The LLM rewrites these documents itself -- compressing old information, resolving contradictions, preserving identity. No embeddings, no vector DB. Just a living document that grows smarter over time.

### Conversation Checkpointing

Every turn is append-only checkpointed to `.checkpoints/` as JSONL. Resume any previous session:

```bash
python harness.py --list-checkpoints
python harness.py --checkpoint-id 20260503T150640
```

### Workspace Sandboxing

All file operations (`read_file`, `write_file`, `shell`) are sandboxed to the working directory. Path traversal is blocked at the tool level. Shell commands are restricted to an allowlist (`ls`, `cat`, `grep`, `git`, `find`, etc.) -- no `rm`, `sudo`, or `python3`.

## Architecture

```
harness.py      2,350 lines   Config, tools, LLM clients, agent loop, REPL, CLI
mcp_client.py   1,063 lines   MCP protocol client (STDIO + HTTP transports)
a2a.py            473 lines   Agent-to-Agent communication (registry, circuit breaker)
memory.py         292 lines   Living document memory engine
channels.py       262 lines   Messaging channels (Telegram)
skills.py         108 lines   agentskills.io skill registry
```

### Core Subsystems

```
AgentConfig          Pydantic-validated configuration
HistoryBackend       Thread-safe conversation history + JSONL checkpointing
RunnableTool         Abstract base for all tools (auto-generates OpenAI schemas)
AgentOrchestrator    Tool execution engine (sync/async, timeout, failure recording)
LLMClient            OpenAI-compatible HTTP client with retry + backoff
VertexClaudeClient   Claude via Vertex AI (Anthropic SDK)
OutputObserver       Response classification + refinement triggers
AgentLoop            The main ReAct loop tying everything together
```

## Tools

14 built-in tools, all auto-generating their OpenAI function-calling schema from Python type hints:

| Tool | Description |
|---|---|
| `calculator` | AST-safe arithmetic (supports `+`, `-`, `*`, `/`, `**`, `abs()`, `round()`) |
| `web_search` | DuckDuckGo search (no API key required) |
| `shell` | Sandboxed shell commands (allowlisted executables only) |
| `read_file` | Read workspace files (path-traversal protected, 1MB limit) |
| `write_file` | Write files with auto-created parent directories |
| `list_files` | Directory listing with file sizes |
| `python_exec` | Sandboxed Python execution (blocked: `os`, `subprocess`, `socket`, etc.) |
| `http_fetch` | Fetch URLs (500KB response limit) |
| `run_skill` | Activate a skill from the registry |
| `list_skills` | List available skills |
| `create_skill` | Create new skills at runtime |
| `plan_agents` | Analyze a task and create specialist sub-agents |
| `delegate` | Send a task to a specialist agent |
| `list_agents` | Show available agents |

## Providers

The harness talks to any OpenAI-compatible endpoint. Switch providers with CLI flags:

| Provider | Command |
|---|---|
| **Ollama** (local, free) | `python harness.py` |
| **OpenAI** | `python harness.py --provider openai --base-url https://api.openai.com/v1 --api-key sk-...` |
| **Claude via Vertex AI** | `python harness.py --provider vertex-claude --project my-gcp-project` |
| **vLLM / LiteLLM / any** | `python harness.py --base-url http://your-server:8000/v1` |

## Skills

Skills follow the [agentskills.io](https://agentskills.io) open standard -- portable across Claude Code, Cursor, Gemini CLI, VS Code Copilot, and 30+ other agent tools. Each skill is a directory with a `SKILL.md` file containing YAML frontmatter and markdown instructions.

```
skills/
  research/       Web search + summarize
  code-review/    Code quality review
  file-stats/     Directory analysis
  multi-review/   Review all files in a directory
  telegram-bot/   Telegram Bot API guide
```

**Use skills three ways:**

```
❯ /research                              # slash command
❯ Use the research skill on quantum computing   # natural language
❯ Create a skill called "lint" that checks style  # create at runtime
```

Skills can require specific tools (`allowed-tools: web_search http_fetch`) and MCP servers -- the harness validates availability before activation.

## Multi-Agent

For complex tasks, the harness creates specialist agents on the fly. A planner agent (Claude Opus) decomposes the task, assigns models and tools, then the main agent delegates execution step by step.

```
❯ Build a REST API with authentication and tests

  ● plan_agents(Build a REST API...)
  Created 3 specialist agents:
    - architect (Designs system architecture)
    - coder (Writes Python code and tests)
    - tester (Write and run tests)

  ● delegate → architect
  ● delegate → coder
  ● delegate → tester
```

Pre-define agents in `agents.json`:

```json
{
  "agents": {
    "researcher": {
      "description": "Searches the web and summarizes findings",
      "model": "qwen3:8b",
      "tools": ["web_search", "http_fetch"]
    },
    "coder": {
      "description": "Writes and tests Python code",
      "model": "claude-sonnet-4-6",
      "tools": ["read_file", "write_file", "python_exec", "shell"]
    }
  }
}
```

```bash
python harness.py --agents agents.json
```

## MCP

Connect to external tool servers using the [Model Context Protocol](https://modelcontextprotocol.io). The harness implements both STDIO and HTTP/SSE transports, with BM25-based tool search so the LLM sees only the most relevant tools for each prompt.

```bash
python harness.py --mcp-config mcp_servers.json
```

```json
{
  "servers": {
    "my-server": {
      "transport": "stdio",
      "command": ["python3", "my_mcp_server.py"]
    }
  }
}
```

**Under the hood:**
- Tool schemas are cached for 5 minutes; auto-refreshed on reconnection
- Background health checks with exponential-backoff reconnection (30s to 300s)
- BM25 ranking with IDF weighting, recency boost, and success-rate trust scoring
- Trie-based prefix autocomplete for tool names

## Channels

Run the REPL and external messaging simultaneously. Messages from Telegram (or other channels) are processed through independent agent instances with their own history.

```bash
python harness.py --provider vertex-claude --channels channels.json
```

```json
{
  "channels": {
    "telegram": {
      "token": "$TELEGRAM_BOT_TOKEN",
      "allowed_chats": []
    }
  }
}
```

Terminal shows channel activity alongside your REPL:

```
  ┌ telegram · Georgy
  │ What's the capital of France?
  │ → Paris is the capital of France...
  └ 3s · 142 chars
```

## Agent-to-Agent Communication

The `a2a.py` library provides production-grade primitives for multi-agent mesh networking:

- **Registry** -- TTL-based agent discovery with automatic expiry
- **Circuit Breaker** -- per-agent fault isolation (Closed / Open / Half-Open)
- **Bounded Task Queue** -- backpressure-aware work queue with explicit `QueueFullError`
- **A2A Client** -- HTTP messaging with retry, exponential backoff, and circuit-breaker integration
- **Agent Worker Pool** -- thread pool draining the task queue
- **Agent Node** -- facade wiring everything together for a single agent in the mesh

## REPL

```
─────────────────────────────────────
AI Harness · qwen3:8b
Tools: 14 · Memory: 127 words
─────────────────────────────────────

❯ your prompt here
❯ /code-review                     # activate a skill
❯ """                              # multi-line input
  ... paste text here
  ... """
❯ exit
```

## CLI Reference

```
python harness.py [prompt] [options]

Positional:
  prompt                    Single-shot prompt (skip REPL)

Provider:
  --provider                openai | vertex-claude (default: openai)
  --base-url URL            OpenAI-compatible endpoint
  --api-key KEY             API key (default: ollama)
  --model MODEL             Model name (default: qwen3:8b)
  --project ID              GCP project (vertex-claude)
  --region REGION           Vertex AI region (default: global)

Model:
  --temperature FLOAT       Sampling temperature 0.0-2.0 (default: 0.7)
  --max-tokens INT          Max response tokens (default: 16384)
  --max-retries INT         HTTP retries (default: 3)
  --timeout FLOAT           Per-request timeout in seconds (default: 60)

Session:
  --workspace PATH          Working directory for file operations
  --checkpoint-id ID        Resume a previous conversation
  --list-checkpoints        Print saved checkpoints and exit

Extensions:
  --mcp-config PATH         MCP server configuration file
  --agents PATH             Pre-defined agent configuration file
  --channels PATH           Messaging channel configuration file

Debug:
  --demo                    Run built-in demo (no LLM needed)
  --verbose                 Enable debug logging
```

## Dependencies

```
requests        HTTP client for LLM + MCP + channels
pydantic        Configuration validation
rich            Markdown rendering in terminal
ddgs            DuckDuckGo search (web_search tool)
pyyaml          Skill frontmatter parsing
anthropic       Claude via Vertex AI (optional)
```

## License

MIT
