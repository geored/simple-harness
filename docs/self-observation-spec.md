# Self-Observable Agent Loop Spec

## v2 — Updated after code review (7 findings addressed)

## Problem

The agent loop has a binary exit: LLM calls a tool (continue) or returns text (done). There's no quality gate. The agent never evaluates whether its output is actually complete, correct, or sufficient. It blindly trusts that when the LLM stops calling tools, the job is done.

This leads to:
- Incomplete answers returned without the agent noticing
- Uncertain answers ("I think...", "possibly...") treated as final
- Code generated but never tested
- Multi-part tasks where only part 1 is completed
- No post-delivery verification (dead links, stale data)

## Solution

An `OutputObserver` layer between "LLM produced text" and "return to user" that:
1. **Classifies** the output type (factual answer, code, research, uncertain)
2. **Evaluates** if it's complete and sufficient
3. **Decides** whether to return (terminal) or re-enter the loop (continue with refinement)
4. **Triggers** optional parallel follow-up actions after returning

## Architecture

```
Agent Loop Iteration:
  LLM returns text response
      ↓
  OutputObserver.observe(output, context)
      ↓
  ┌─────────────────────────────────┐
  │ 1. Classify: what kind?         │
  │ 2. Evaluate: is it sufficient?  │
  │ 3. Decide: terminal or continue │
  └──────────┬──────────────────────┘
             │
     ┌───────┴───────┐
     ▼               ▼
  TERMINAL        CONTINUE
  (return)        (remove rejected msg,
     │             inject system refinement,
     │             re-enter loop)
     │
     ├── optional parallel follow-ups
     │   (verify links, cache, notify)
     ↓
  User sees answer
```

---

## Output Classification

```python
class OutputClass(Enum):
    # TERMINAL — return immediately
    FACTUAL_ANSWER    = "factual_answer"
    OPINION_ANALYSIS  = "opinion_analysis"
    CREATIVE_CONTENT  = "creative_content"
    ACKNOWLEDGMENT    = "acknowledgment"
    CLARIFICATION_REQ = "clarification_req"

    # CONTINUABLE — may need refinement
    RESEARCH_RESULT   = "research_result"
    CODE_GENERATED    = "code_generated"
    UNCERTAIN_ANSWER  = "uncertain_answer"
    MULTI_PART_TASK   = "multi_part_task"
```

---

## Observation Result

```python
@dataclass
class ObservationResult:
    output_class: OutputClass
    confidence: float             # 0.0-1.0
    is_terminal: bool
    follow_up_action: Optional[str]
    follow_up_parallel: bool      # can follow-up run while user sees result?
    reasoning: str
```

---

## Base Cases (Prevent Infinite Loops)

Three mechanisms ensure the observer never creates an infinite refinement loop:

### 1. Max Refinements (hard cap)
```python
MAX_REFINEMENTS = 2
```
After 2 re-entries, force terminal regardless of classification.

### 2. Low Confidence Default
```python
MIN_CONFIDENCE = 0.6
```
If the classifier isn't confident, default to terminal.

### 3. Repeated Classification
If the same output class appears twice in a row, force terminal.

### Counter Location (Finding #5)

The `_refinement_count` lives on the `OutputObserver` instance, NOT derived from the agent loop iteration. The observer tracks its own state:

```python
class OutputObserver:
    def __init__(self):
        self._refinement_count = 0
        self._previous_class = None

    def reset(self):
        """Called at start of each AgentLoop.run()."""
        self._refinement_count = 0
        self._previous_class = None

    def observe(self, output, context):
        # Base case 1: hard cap
        if self._refinement_count >= self.MAX_REFINEMENTS:
            return terminal("Max refinements reached")

        result = self._classify(output, context)

        # Base case 2: low confidence
        if result.confidence < self.MIN_CONFIDENCE:
            result.is_terminal = True
            return result

        # Base case 3: repeated class
        if result.output_class == self._previous_class:
            result.is_terminal = True
            return result

        self._previous_class = result.output_class
        if not result.is_terminal:
            self._refinement_count += 1
        return result
```

---

## Classification Engine (Hybrid)

### Layer 1: Heuristic (free, instant)

```python
def _classify_heuristic(self, output, context):
    words = output.split()

    # Short direct answer, no tools used → factual (terminal)
    if len(words) < 50 and self._refinement_count == 0:
        return ObservationResult(FACTUAL_ANSWER, 0.85, True, None, False,
                                "Short direct answer")

    # Empty or near-empty → acknowledgment (terminal)
    if len(words) < 5:
        return ObservationResult(ACKNOWLEDGMENT, 0.9, True, None, False,
                                "Minimal response")

    # Uncertainty markers (Finding #1: confidence 0.65, below LLM threshold)
    # Require 2+ markers to avoid false positives on careful phrasing
    uncertainty = ["i'm not sure", "i think", "possibly", "might be",
                   "couldn't find", "no results found"]
    matches = sum(1 for m in uncertainty if m in output.lower())
    if matches >= 2:
        return ObservationResult(UNCERTAIN_ANSWER, 0.65, False,
                                "Multiple uncertainty markers — verify with additional search",
                                False, "Output contains hedging language")

    # Code generated → terminal + parallel test suggestion
    if "```" in output and ("def " in output or "class " in output):
        return ObservationResult(CODE_GENERATED, 0.75, True,
                                "Consider running tests on generated code",
                                True, "Code generated")

    # Multi-part without conclusion (Finding #6: expanded completion markers)
    partial_markers = ["step 1", "first,", "part 1", "phase 1"]
    completion_markers = ["finally", "conclusion", "complete", "all done",
                         "in summary", "here's how", "here are the steps",
                         "that covers", "to summarize"]
    has_partial = any(w in output.lower() for w in partial_markers)
    has_completion = any(w in output.lower() for w in completion_markers)
    if has_partial and not has_completion:
        return ObservationResult(MULTI_PART_TASK, 0.65, False,
                                "Continue with remaining steps",
                                False, "Appears to be partial completion")

    # Default: trust the agent
    return ObservationResult(FACTUAL_ANSWER, 0.9, True, None, False,
                            "Output appears complete")
```

### Layer 2: LLM (accurate, escalated when heuristic unsure)

Only called when heuristic confidence < 0.7 (Finding #1: uncertainty at 0.65 now properly escalates):

```
Prompt: "Classify this agent output. Is it complete and sufficient?
         Original request: {prompt}
         Output (first 500 chars): {output[:500]}
         Respond with ONLY JSON: {class, is_complete, confidence, reasoning}"
```

### Escalation Logic

```python
def _classify(self, output, context):
    heuristic = self._classify_heuristic(output, context)
    if heuristic.confidence > 0.7:  # strict greater-than (Finding #4)
        return heuristic
    if self._llm:
        try:
            return self._classify_llm(output, context)
        except Exception:
            heuristic.is_terminal = True  # LLM failed, default to terminal
            return heuristic
    return heuristic
```

---

## Refinement Prompt Design (Finding #2)

When the observer decides to continue, it injects a **system message** (not user — prevents model confusion) and **removes the rejected assistant message** from history (Finding #3 — prevents memory pollution):

```python
if not observation.is_terminal:
    # Remove the rejected output from history (Finding #3)
    snap = self._history.snapshot()
    if snap and snap[-1].get("role") == "assistant":
        self._history._history.pop()  # remove last entry

    # Inject refinement as system message (Finding #2)
    self._history.append({
        "role": "system",
        "content": (
            f"[Observer] Your previous response was insufficient. "
            f"Classification: {observation.output_class.value}. "
            f"Issue: {observation.reasoning}. "
            f"Action: {observation.follow_up_action}. "
            f"Provide a more complete response."
        ),
    })
    continue  # re-enter agent loop
```

---

## Observer Activation Scope (Finding #4)

| Context | Observer Mode | Why |
|---------|-------------|-----|
| REPL (main agent) | `"hybrid"` | User-facing, quality matters |
| Channel agents (Telegram) | `"heuristic"` | Fast, no LLM cost per message |
| Sub-agents (delegate) | `"off"` | Sub-agents have system prompts controlling their behavior |
| Plan agents | `"off"` | Planner output is structured JSON, not user-facing |

Implementation: `AgentLoop.__init__` accepts `observer_mode` parameter. Default is `"hybrid"`. `_create_agent_from_spec` passes `"off"`. Channel factory passes `"heuristic"`.

---

## Parallel Follow-Up Actions

Actions that run in background threads after the user already sees the answer:

| Action | When | What it does |
|--------|------|-------------|
| verify_links | RESEARCH_RESULT | Checks URLs in output are alive, logs dead ones |
| suggest_tests | CODE_GENERATED | Logs suggestion to terminal |
| cache_result | Any terminal | Saves to `.cache/results/{hash}.json` |

Follow-ups are fire-and-forget daemon threads. If they fail, they log a warning.

---

## Integration with AgentLoop

### Changes to `__init__`:
```python
def __init__(self, ..., observer_mode="hybrid"):
    ...
    self._observer = OutputObserver(
        llm_client=llm_client if observer_mode == "hybrid" else None,
        mode=observer_mode,
    )
```

### Changes to `run()`:
```python
self._observer.reset()  # at start

# Branch B (text response):
if response.has_text:
    self._history.append({"role": "assistant", "content": response.text})

    if self._observer.mode != "off":
        observation = self._observer.observe(response.text, {
            "original_prompt": user_prompt,
        })

        if not observation.is_terminal:
            # Remove rejected output, inject system refinement
            self._history._history.pop()
            self._history.append({
                "role": "system",
                "content": f"[Observer] {observation.reasoning}. {observation.follow_up_action}."
            })
            continue

        if observation.follow_up_action and observation.follow_up_parallel:
            threading.Thread(target=self._run_follow_up,
                           args=(observation,), daemon=True).start()

    self._consolidate_async(snapshot)
    return response.text
```

---

## What Changes

| File | Change | Lines |
|------|--------|-------|
| harness.py | OutputObserver class + heuristic classifier | ~100 |
| harness.py | ObservationResult dataclass + OutputClass enum | ~25 |
| harness.py | Branch B modification + refinement logic | ~30 |
| harness.py | Follow-up action stubs | ~30 |
| harness.py | AgentLoop.__init__ + factory changes | ~15 |
| harness.py | LLM classifier (optional) | ~20 |
| **Total** | | **~220 lines** |

---

## What This Does NOT Do

- Does not change how tools are called
- Does not add new tools (observer is internal)
- Does not change the REPL or channel interface
- Does not add latency for simple answers (heuristic is instant)
- Does not break existing behavior (`mode="off"` returns immediately)

---

## Visualization

Terminal shows observer decisions inline:

```
● Thinking · 15s
  ● web_search
  ● read_file
● Thinking · 25s
  ● [observer] uncertain_answer → refining
  ● web_search
● Thinking · 40s
  ● [observer] research_result → terminal
[answer shown]
  [follow-up] 3 links verified (2 alive, 1 dead)
```

---

## Validation

1. Simple "What is 2+2?" → FACTUAL_ANSWER, no refinement, instant
2. "I think it might be..." with 2+ hedges → UNCERTAIN_ANSWER → refinement → better answer
3. Code generation → CODE_GENERATED → terminal + "suggest tests" log
4. "Step 1: ..." without conclusion → MULTI_PART_TASK → re-enters loop
5. Max refinements (2) → forces terminal
6. `observer_mode="off"` → existing behavior unchanged
7. Sub-agents → no observer
8. Channel agents → heuristic only (no LLM cost)
