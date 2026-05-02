# Agent Planning Spec — Dynamic Agent Creation from User Intent

## v2 — Updated after code review

## Problem

Currently agents are pre-defined in `agents.json`. The user must manually decide what specialists are needed before starting. This has three failures:
1. Wrong agents for the task (research agent when you need a coder)
2. Missing agents (no tester agent when the task needs testing)
3. Generic agents that aren't optimized for the specific task

## Solution

A `plan_agents` tool that the main LLM calls as its first action on complex prompts. The tool:
1. Analyzes the user's prompt
2. Calls **Claude Opus 4.6** to design a team of specialist agents (planning always uses the most capable model)
3. Creates all agents in one step
4. Returns the plan so the main agent knows how to delegate

## Design

### The Tool

```
Name: plan_agents
Description: Analyze a task and create specialist agents to handle it.
             Call this FIRST on complex tasks that need multiple steps
             or expertise areas. Returns a plan with created agents.
Input: task_description (str) — the user's full prompt/request
Output: JSON with created agents and execution plan
```

### Constructor Dependencies

`PlanAgentsTool.__init__` receives:
- `planner_llm`: LLM client for the planning call (always Opus via Vertex, falls back to main LLM)
- `provider_config`: dict with `base_url`, `api_key`, `vertex_project`, `vertex_region` — needed to construct sub-agent LLM clients for any provider
- `tool_pool`: dict of available tool instances by name
- `agent_registry`: mutable dict where created agents are registered (shared with DelegateTool)

### How It Works

```
User prompt arrives
    ↓
Main LLM decides: "This is complex, I need a team"
    ↓
Calls plan_agents(task_description="Build a price monitoring scraper...")
    ↓
Inside plan_agents:
    1. Send planning prompt to OPUS 4.6 via Vertex AI:
       "Given this task, what specialist agents are needed?
        For each agent specify: name, role, tools (from pool), model.
        Also provide an execution plan: steps in order.
        Respond with ONLY valid JSON."
    
    2. Opus returns structured JSON (validated, retried once on failure)
    
    3. Validate plan:
       - Max 5 agents
       - Max 10 plan steps
       - All tools exist in the pool
       - No circular dependencies (topological sort, reject if cycle found)
       - All model names are valid
    
    4. Create agents via factory function
    
    5. Register agents in shared registry
       (DelegateTool and ListAgentsTool auto-see them)
    
    6. Return plan to main agent as text summary
    ↓
Main agent follows the plan, delegating sequentially
```

### Planning Model

The planning call ALWAYS uses **Claude Opus 4.6** via Vertex AI when `GOOGLE_CLOUD_PROJECT` is set. This ensures reliable structured JSON output regardless of what model the main agent or sub-agents use.

Fallback chain:
1. Opus 4.6 via Vertex (primary — most reliable for structured output)
2. Main agent's LLM (if Vertex unavailable)
3. Return "no agents needed" (if both fail)

Small local models (qwen3:8b) are NOT used for planning — they cannot reliably produce the required JSON schema.

### When NOT to Create Agents

The LLM decides whether agents are needed — no hard-coded thresholds. The planning prompt includes guidance:

```
If this task is simple (single question, one tool call, quick answer),
respond with: {"agents": [], "plan": [], "reason": "Task is simple enough for direct handling"}
```

The tool enforces no word-count or step-count gates. Opus is capable enough to make this judgment correctly.

### Resource Limits

Hard caps enforced by the tool BEFORE creating anything:
- **Max 5 agents** per plan
- **Max 10 plan steps** per plan
- **Max 30 iterations** per sub-agent (existing `_HARD_ITERATION_CEILING`)
- Plans exceeding limits are truncated with a warning

### Agent Template

Each auto-created agent gets:
- Its own `AgentLoop` with its own `HistoryBackend`
- A system prompt: "You are a specialist agent. Complete the task using your tools. When done, respond with a TEXT SUMMARY of what you did and what you produced. Do NOT keep calling tools indefinitely — finish and report back."
- Tools selected from the available pool
- The model specified by the planner
- An `agent_name` for spinner visualization

### Available Tool Pool

The planner can assign from:
```
calculator, web_search, shell, read_file, write_file,
list_files, python_exec, http_fetch
```

Skills and MCP tools are NOT available to sub-agents (they use the main agent's context).

### Provider Resolution

When the planner assigns a model to an agent, the factory resolves the provider:

| Model pattern | Provider | Client | Requires |
|--------------|----------|--------|----------|
| `claude-*` | vertex-claude | `VertexClaudeClient` | `GOOGLE_CLOUD_PROJECT` |
| `gpt-*` | openai | `LLMClient` | `--api-key` with OpenAI key |
| anything else | openai (Ollama) | `LLMClient` | Ollama running locally |

If the requested model is unreachable:
1. Try the main agent's model as fallback
2. If that fails, skip the agent and log a warning
3. The plan continues with remaining agents

Credential requirements:
- Vertex Claude agents need `GOOGLE_CLOUD_PROJECT` env var (same as main agent)
- Ollama agents need Ollama running on `--base-url` (default localhost:11434)
- No agent can require credentials the user hasn't already provided

### Model Selection Guidelines

The planning prompt includes these guidelines for Opus:

```
Choose models based on subtask complexity:
- Research, search, simple formatting → "qwen3:8b" (local, fast, free)
- Code generation, file creation → "claude-sonnet-4-6" (capable, moderate cost)
- Architecture, design, complex reasoning → "claude-sonnet-4-6" (good enough for sub-tasks)
- Only use opus for the planning call itself, never for sub-agents
```

### Execution Plan Format

```json
{
  "agents": [
    {
      "name": "researcher",
      "role": "Find best practices and examples",
      "tools": ["web_search", "http_fetch"],
      "model": "qwen3:8b"
    },
    {
      "name": "architect",
      "role": "Design project structure and file layout",
      "tools": ["write_file", "list_files"],
      "model": "claude-sonnet-4-6"
    },
    {
      "name": "coder",
      "role": "Implement the code based on architecture",
      "tools": ["read_file", "write_file", "python_exec", "shell"],
      "model": "claude-sonnet-4-6"
    }
  ],
  "plan": [
    {"step": 1, "agent": "researcher", "task": "Research scraping patterns and anti-bot strategies"},
    {"step": 2, "agent": "architect", "task": "Design project: files, modules, data flow"},
    {"step": 3, "agent": "coder", "task": "Implement scraper, database, and CLI"},
    {"step": 4, "agent": "coder", "task": "Write tests"}
  ]
}
```

**Note:** All steps execute sequentially. The `depends_on` field from v1 is removed — the current runtime cannot execute parallel tool calls (single tool call per LLM turn). Sequential execution is correct and sufficient for v1.

### Data Flow Between Steps

Each step's output is automatically appended to the next step's task prompt:

```
Step 1 runs → researcher returns: "Found: use requests + BeautifulSoup..."
Step 2 prompt becomes:
  "Design project: files, modules, data flow

   Context from previous steps:
   [researcher] Found: use requests + BeautifulSoup..."

Step 2 runs → architect returns: "Created: scraper.py, db.py, alerts.py..."
Step 3 prompt becomes:
  "Implement scraper, database, and CLI

   Context from previous steps:
   [researcher] Found: use requests + BeautifulSoup...
   [architect] Created: scraper.py, db.py, alerts.py..."
```

Context accumulates but is truncated at 2000 chars per step to prevent overflow.

### Integration with Existing Code

- `plan_agents` is ALWAYS registered (not just with `--agents`)
- `--agents` still works for pre-defined agents alongside dynamic ones
- `delegate` and `list_agents` are ALWAYS registered (empty initially, populated after planning)
- After `plan_agents` creates agents, the main agent loop refreshes `tool_schemas()` on the next iteration so the LLM sees updated delegate descriptions
- Pre-defined agents from `--agents` and dynamically created agents coexist in the same registry

### Agent Lifecycle and Cleanup

- After a sub-agent completes its delegated task, its `AgentOrchestrator` is shut down (`executor.shutdown()`)
- The agent entry remains in the registry (for potential reuse within the same prompt)
- When the REPL moves to the next prompt (`history.clear()`), all dynamic agents are cleaned up
- On harness exit, all agent orchestrators are shut down

### What Changes in harness.py

1. New `PlanAgentsTool` class (~200 lines)
2. New `_create_agent_from_spec()` factory function (~80 lines)
3. Planning prompt template with few-shot example (~30 lines)
4. `plan_agents`, `delegate`, `list_agents` always registered in `_build_agent()`
5. Agent cleanup in REPL loop and shutdown
6. Tool schema refresh mechanism after agent creation

Estimated total: ~350 lines of new code.

### Error Handling

- Planner returns invalid JSON → retry once with "You must respond with ONLY valid JSON", then fall back to "no agents needed"
- Agent can't be created (bad model, unreachable provider) → skip it, create the rest, warn in output
- Circular dependencies in plan → reject plan, report error, let main agent handle directly
- Sub-agent hits iteration limit → return partial results, mark step as failed
- All agents fail → main agent handles the task directly with its own tools
- Total plan execution exceeds 10 minutes → abort remaining steps, return what's done

### Workspace

All agents share the same `--workspace` directory. Files created by the architect are readable by the coder. No agent isolation — this is intentional (they're collaborating on the same project).

### Memory

Sub-agents do NOT get their own memory. They're ephemeral — created for one task, discarded after. The main agent's `memory.md` captures the overall session outcome. The main agent's `session.md` captures which agents were used and what they produced.

### Visualization

The spinner shows agent creation and execution:
```
● Thinking · Planning agents (Opus) · 5s
● Thinking · Created: researcher, architect, coder · 6s
● Thinking · Step 1/4 · Delegating to researcher · 12s
  ● researcher  · web_search
● Thinking · Step 2/4 · Delegating to architect · 30s
  ● architect   · write_file
● Thinking · Step 3/4 · Delegating to coder · 65s
  ● coder       · python_exec
  ● coder       · write_file
● Thinking · Step 4/4 · Delegating to coder · 90s
  ● coder       · python_exec
● Thinking · 95s
```

### Security

- Dynamically created agents inherit the same tool restrictions as static agents:
  - Shell: allow-list only, no rm/sudo/python3
  - File I/O: workspace-sandboxed
  - Python exec: blocked modules (os, subprocess, etc.)
- The planner cannot create tools that don't exist in the pool
- The planner cannot grant an agent more permissions than the harness has
- Model selection is validated against known provider patterns
