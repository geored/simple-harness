# Simple Harness — Running Guide

## Quick Start

```bash
cd ai-harness

# Install dependencies
python3 -m venv .venv
.venv/bin/pip install requests pydantic rich ddgs pyyaml anthropic

# Run with local Ollama (default)
.venv/bin/python harness.py

# Run with Claude Sonnet via Vertex AI
.venv/bin/python harness.py --provider vertex-claude

# Run with Claude Opus
.venv/bin/python harness.py --provider vertex-claude --model claude-opus-4-6

# Single-shot (no REPL)
.venv/bin/python harness.py "What is 42 * 17?"
```

## Environment Setup

Create a `.env` file in the project root (auto-loaded on startup):

```
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
```

No need to export manually — the harness reads `.env` automatically.

## Providers

| Provider | Flag | Requirements |
|----------|------|-------------|
| Ollama (local) | `--provider openai` (default) | Ollama running on localhost:11434 |
| OpenAI | `--provider openai --base-url https://api.openai.com/v1 --api-key sk-...` | OpenAI API key |
| Claude via Vertex | `--provider vertex-claude` | `GOOGLE_CLOUD_PROJECT` in .env + gcloud auth |
| Any OpenAI-compatible | `--provider openai --base-url http://your-server/v1` | Server running |

## Tools (14 built-in)

| Tool | What it does |
|------|-------------|
| `calculator` | AST-safe math evaluation |
| `web_search` | DuckDuckGo search (no API key) |
| `shell` | Run allow-listed commands (ls, git, grep, cat, find...) |
| `read_file` | Read files (workspace-sandboxed) |
| `write_file` | Write files (workspace-sandboxed) |
| `list_files` | List directory contents |
| `python_exec` | Execute Python code (sandboxed, blocked modules) |
| `http_fetch` | Fetch URLs |
| `list_skills` | Show available skills |
| `run_skill` | Activate an agentskills.io skill |
| `create_skill` | Create a new skill dynamically |
| `plan_agents` | Plan and create specialist agents for complex tasks |
| `delegate` | Delegate a task to a specialist agent |
| `list_agents` | Show available agents |

## REPL Commands

```
❯ your prompt here              # normal prompt
❯ /code-review                  # activate a skill by name
❯ /research                     # activate research skill
❯ """                           # start multi-line input
  ... paste text here
  ... """                       # end multi-line input
❯ exit                          # quit
```

## Memory System

Two living documents maintained by the LLM:

| File | Scope | Persists |
|------|-------|----------|
| `memory.md` | Cross-session | Yes — user identity, preferences, long-term facts |
| `session.md` | Current session | No — deleted on exit |
| `memory_archive.md` | Backup | Yes — every version of memory.md before rewrite |

Memory is automatic — no commands needed. The LLM rewrites these documents after each interaction.

## Checkpointing

Every conversation is checkpointed to `.checkpoints/` as append-only JSONL.

```bash
# List saved sessions
python harness.py --list-checkpoints

# Resume a previous session
python harness.py --checkpoint-id 20260503T150640
```

## Workspace

Control where the agent creates files:

```bash
# Files go to ~/projects/my-app/ instead of current directory
python harness.py --workspace ~/projects/my-app
```

## Skills (agentskills.io format)

Skills are in `skills/` as directories with `SKILL.md` files:

```
skills/
  research/SKILL.md         # web search + summarize
  code-review/SKILL.md      # code quality review
  file-stats/SKILL.md       # directory analysis
  multi-review/SKILL.md     # review all files in a directory
  telegram-bot/SKILL.md     # Telegram Bot API guide
```

Use in REPL:
```
❯ /code-review                           # slash command
❯ Use the research skill on AI trends    # natural language
❯ Create a skill called "deploy" that checks git status and pushes to main
```

## Multi-Agent Planning

For complex tasks, the harness plans and creates specialist agents automatically:

```bash
# With Vertex Claude (Opus plans, Sonnet executes)
python harness.py --provider vertex-claude

❯ Build a REST API with authentication and tests
# → Opus creates architect + coder + tester agents
# → Each agent works on its part
# → Results combined
```

Pre-defined agents:
```bash
python harness.py --agents agents.json
```

Example `agents.json`:
```json
{
  "agents": {
    "researcher": {
      "description": "Searches the web and summarizes findings",
      "provider": "openai",
      "model": "qwen3:8b",
      "tools": ["web_search", "http_fetch"]
    },
    "coder": {
      "description": "Writes and tests Python code",
      "provider": "vertex-claude",
      "model": "claude-sonnet-4-6",
      "tools": ["read_file", "write_file", "python_exec", "shell"]
    }
  }
}
```

## MCP Servers

Connect to external tool servers via MCP:

```bash
python harness.py --mcp-config mcp_servers.json
```

Example `mcp_servers.json`:
```json
{
  "servers": {
    "test": {
      "transport": "stdio",
      "command": ["python3", "test_mcp_server.py"]
    }
  }
}
```

MCP tools are discovered automatically via BM25 search — the LLM sees relevant MCP tools based on your prompt without manual selection.

## Messaging Channels (Telegram)

Run the REPL + Telegram bot simultaneously:

```bash
python harness.py --provider vertex-claude --channels channels.json
```

Example `channels.json`:
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

Terminal shows channel activity:
```
  ┌ telegram · Georgy
  │ What is the weather today?
  │ → I don't have real-time weather access...
  └ 8s · 245 chars
```

## Self-Observation

The agent loop includes an OutputObserver that classifies responses:

```
  ● web_search(AI news 2026)
  ● read_file(harness.py)
  ● [observer] factual_answer → terminal (90%)
```

If the output is uncertain or incomplete, the observer triggers a refinement loop (max 2 retries).

## Demo Mode

Test without any LLM endpoint:

```bash
python harness.py --demo
```

## Full Example Session

```bash
# Start with all features
python harness.py --provider vertex-claude --channels channels.json --workspace ~/projects/demo

# In the REPL:
❯ My name is Georgy and I prefer concise answers
❯ What is 42 * 17?
❯ Search for latest Rust news
❯ /code-review
❯ Review harness.py for security issues
❯ Build a simple todo app with SQLite
❯ List available skills
❯ Create a skill called "lint" that reads a file and checks for style issues
❯ What do you remember about me?
❯ exit

# Next time — memory persists:
python harness.py --provider vertex-claude
❯ Who am I?
# → "Your name is Georgy, you prefer concise answers"
```

## Architecture

```
harness.py    — Main: config, tools, LLM clients, agent loop, REPL, CLI
memory.py     — Living document memory (memory.md + session.md)
skills.py     — agentskills.io skill registry
mcp_client.py — MCP protocol client (STDIO + HTTP)
channels.py   — Messaging channels (Telegram)
a2a.py        — Agent-to-Agent communication library (future use)
```
