"""
skills.py — Three-Layer Skill Orchestration Engine
====================================================
Layer 1: YAML Manifest    — skill definitions with I/O schemas
Layer 2: JSON Sequence    — execution plan with retry, loops, branching
Layer 3: Python Engine    — runtime with state management and validation

Integration: skills bridge to harness tools and LLM client via callables
passed at construction time. No hardcoded stubs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

log = logging.getLogger("agent_harness.skills")

SKILLS_DIR = Path(__file__).parent / "skills"


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------

class SchemaValidationError(Exception):
    pass


class SkillNotFoundError(Exception):
    pass


_TYPE_MAP = {
    "string": str,
    "int":    int,
    "float":  (int, float),
    "bool":   bool,
    "dict":   dict,
}


def _validate_type(value: Any, type_spec: str, field_path: str) -> None:
    if type_spec.startswith("list[") and type_spec.endswith("]"):
        inner = type_spec[5:-1]
        if not isinstance(value, list):
            raise SchemaValidationError(
                f"Field '{field_path}' expected list, got {type(value).__name__}"
            )
        for i, item in enumerate(value):
            _validate_type(item, inner, f"{field_path}[{i}]")
        return

    if type_spec == "any":
        return

    expected = _TYPE_MAP.get(type_spec)
    if expected is None:
        raise SchemaValidationError(f"Unknown type spec '{type_spec}' for '{field_path}'")
    if not isinstance(value, expected):
        raise SchemaValidationError(
            f"Field '{field_path}' expected {type_spec}, got {type(value).__name__}"
        )


def validate_output(output: Any, schema: dict, step_id: str) -> None:
    if not schema:
        return
    top_type = schema.get("type")
    if top_type == "dict":
        if not isinstance(output, dict):
            raise SchemaValidationError(f"[{step_id}] expected dict, got {type(output).__name__}")
        for fname, fspec in schema.get("fields", {}).items():
            if fname not in output:
                raise SchemaValidationError(f"[{step_id}] Missing field '{fname}'")
            _validate_type(output[fname], fspec["type"], f"{step_id}.{fname}")
    elif top_type:
        _validate_type(output, top_type, step_id)


# ---------------------------------------------------------------------------
# Engine State
# ---------------------------------------------------------------------------

@dataclass
class StepState:
    step_id: str
    output:  Any
    skipped: bool = False


@dataclass
class EngineState:
    steps: dict[str, StepState] = field(default_factory=dict)

    def store(self, step_state: StepState) -> None:
        self.steps[step_state.step_id] = step_state

    def resolve(self, ref: str) -> Any:
        if not isinstance(ref, str) or not ref.startswith("$"):
            return ref

        parts = ref.lstrip("$").split(".", 1)
        step_id = parts[0]
        step_state = self.steps.get(step_id)
        if step_state is None:
            raise KeyError(f"No state for step '{step_id}' (ref='{ref}')")

        if len(parts) == 1:
            return step_state.output

        field_name = parts[1]
        if field_name == "results" and isinstance(step_state.output, dict):
            return step_state.output.get("results", [])
        if isinstance(step_state.output, dict):
            if field_name not in step_state.output:
                raise KeyError(f"Field '{field_name}' not in output of '{step_id}'")
            return step_state.output[field_name]
        raise TypeError(f"Cannot resolve '{ref}': output of '{step_id}' is not a dict")


# ---------------------------------------------------------------------------
# Skill Engine
# ---------------------------------------------------------------------------

class SkillEngine:
    """
    Executes a skill (YAML manifest + JSON sequence).

    tool_runner: callable(tool_name, **kwargs) -> Any
        Bridge to harness tools. Called when a step has type="tool".

    llm_runner: callable(prompt) -> str
        Bridge to LLM. Called when a step has type="llm".
    """

    def __init__(
        self,
        manifest: dict,
        sequence: dict,
        tool_runner: Callable,
        llm_runner: Optional[Callable] = None,
        on_step: Optional[Callable[[int, int, str, str], None]] = None,
    ):
        raw_skills = manifest.get("skills", [])
        self._skills = {}
        for s in raw_skills:
            if isinstance(s, dict) and "name" in s:
                self._skills[s["name"]] = s
        self._sequence = sequence
        self._tool_runner = tool_runner
        self._llm_runner = llm_runner
        self._on_step = on_step
        self._state = EngineState()

    def run(self, initial_inputs: Optional[dict] = None) -> EngineState:
        if initial_inputs:
            self._state.store(StepState(step_id="input", output=initial_inputs))

        log.info("=== Skill: %s ===", self._sequence.get("sequence_id", "unnamed"))
        steps = self._sequence["steps"]
        total = len(steps)

        for idx, step_def in enumerate(steps, 1):
            self._execute_step(step_def, step_num=idx, total_steps=total)

        log.info("=== Skill complete ===")
        return self._state

    def _execute_step(self, step_def: dict, step_num: int = 0, total_steps: int = 0) -> None:
        step_id = step_def["step_id"]
        step_type = step_def.get("type", "tool")
        on_error = step_def.get("on_error", "abort")

        label = step_def.get("skill", "llm")
        if "loop_over" in step_def:
            label = f"loop {label}"

        if self._on_step:
            self._on_step(step_num, total_steps, step_id, label)

        log.info("--- %s | type=%s ---", step_id, step_type)

        if step_type == "llm":
            self._execute_llm_step(step_def, on_error)
        elif "loop_over" in step_def:
            self._execute_loop(step_def, on_error)
        else:
            self._execute_tool_step(step_def, on_error)

    def _execute_tool_step(self, step_def: dict, on_error: str) -> None:
        step_id = step_def["step_id"]
        tool_name = step_def["skill"]
        retry_cfg = step_def.get("retry", {"count": 1, "delay_seconds": 0})
        resolved = self._resolve_inputs(step_def.get("inputs", {}))

        skill_meta = self._skills.get(tool_name, {})

        output = self._run_with_retry(
            step_id=step_id,
            fn=lambda inputs: self._tool_runner(tool_name, **inputs),
            inputs=resolved,
            schema=skill_meta.get("output_schema"),
            retry_cfg=retry_cfg,
            on_error=on_error,
        )

        self._state.store(StepState(
            step_id=step_id,
            output=output if output is not None else {},
            skipped=output is None,
        ))

    def _execute_llm_step(self, step_def: dict, on_error: str) -> None:
        step_id = step_def["step_id"]
        prompt_template = step_def.get("prompt", "")

        resolved_prompt = self._resolve_string(prompt_template)

        if self._llm_runner is None:
            raise RuntimeError(f"[{step_id}] LLM step requires llm_runner but none provided")

        try:
            result = self._llm_runner(resolved_prompt)
            output = {"response": result}
        except Exception as exc:
            log.error("[%s] LLM step failed: %s", step_id, exc)
            if on_error == "skip":
                self._state.store(StepState(step_id=step_id, output={}, skipped=True))
                return
            raise

        self._state.store(StepState(step_id=step_id, output=output))

    def _execute_loop(self, step_def: dict, on_error: str) -> None:
        step_id = step_def["step_id"]
        loop_ref = step_def["loop_over"]
        item_key = step_def.get("loop_item_input", "item")
        collect_field = step_def.get("collect_output_field")
        tool_name = step_def["skill"]
        retry_cfg = step_def.get("retry", {"count": 1, "delay_seconds": 0})
        skill_meta = self._skills.get(tool_name, {})

        items = self._state.resolve(loop_ref)
        if isinstance(items, str):
            try:
                parsed = json.loads(items)
                if isinstance(parsed, list):
                    items = parsed
            except (json.JSONDecodeError, ValueError):
                items = [line.strip() for line in items.strip().splitlines() if line.strip()]
        if not isinstance(items, list):
            raise TypeError(f"[{step_id}] loop_over must be a list, got {type(items).__name__}")

        log.info("[%s] Looping over %d items", step_id, len(items))
        collected = []

        for idx, item in enumerate(items):
            item_id = f"{step_id}_item{idx}"
            inputs = {item_key: item}

            output = self._run_with_retry(
                step_id=item_id,
                fn=lambda inp: self._tool_runner(tool_name, **inp),
                inputs=inputs,
                schema=skill_meta.get("output_schema"),
                retry_cfg=retry_cfg,
                on_error=on_error,
            )
            if output is None:
                continue
            collected.append(output.get(collect_field, output) if collect_field else output)

        self._state.store(StepState(step_id=step_id, output={"results": collected}))

    def _run_with_retry(self, step_id, fn, inputs, schema, retry_cfg, on_error):
        max_attempts = max(1, retry_cfg.get("count", 1))
        delay = retry_cfg.get("delay_seconds", 0)
        last_exc = None

        for attempt in range(1, max_attempts + 1):
            try:
                output = fn(inputs)
                if schema:
                    validate_output(output, schema, step_id)
                return output
            except SchemaValidationError as exc:
                log.error("[%s] Schema validation failed: %s", step_id, exc)
                if on_error == "skip":
                    return None
                raise RuntimeError(f"[{step_id}] {exc}") from exc
            except Exception as exc:
                last_exc = exc
                log.warning("[%s] Attempt %d/%d failed: %s", step_id, attempt, max_attempts, exc)
                if attempt < max_attempts:
                    time.sleep(delay)

        if on_error == "skip":
            return None
        raise RuntimeError(f"[{step_id}] All {max_attempts} attempts failed: {last_exc}")

    def _resolve_inputs(self, raw: dict) -> dict:
        resolved = {}
        for k, v in raw.items():
            if not isinstance(v, str) or "$" not in v:
                resolved[k] = v
            elif v.startswith("$") and " " not in v:
                resolved[k] = self._state.resolve(v)
            else:
                resolved[k] = self._resolve_string(v)
        return resolved

    def _resolve_string(self, template: str) -> str:
        import re
        def replacer(match):
            ref = match.group(0)
            try:
                val = self._state.resolve(ref)
                return str(val)
            except (KeyError, TypeError):
                return ref
        return re.sub(r"\$\w+(?:\.\w+)?", replacer, template)


# ---------------------------------------------------------------------------
# Skill Registry — loads from skills/ directory
# ---------------------------------------------------------------------------

class SkillRegistry:
    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self._dir = skills_dir
        self._skills: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self._dir.exists():
            return
        for yaml_file in sorted(self._dir.glob("*.yaml")):
            name = yaml_file.stem
            json_file = yaml_file.with_suffix(".json")
            if not json_file.exists():
                log.warning("Skill '%s' has manifest but no sequence, skipping", name)
                continue
            self._skills[name] = {
                "manifest": yaml.safe_load(yaml_file.read_text()),
                "sequence": json.loads(json_file.read_text()),
            }
            log.info("Loaded skill: %s", name)

    def list_skills(self) -> list[dict]:
        result = []
        for name, data in self._skills.items():
            desc = data["sequence"].get("description", "")
            result.append({"name": name, "description": desc})
        return result

    def get(self, name: str) -> Optional[dict]:
        return self._skills.get(name)

    def register(self, name: str, manifest: dict, sequence: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        yaml_path = self._dir / f"{name}.yaml"
        json_path = self._dir / f"{name}.json"
        yaml_path.write_text(yaml.dump(manifest, default_flow_style=False))
        json_path.write_text(json.dumps(sequence, indent=2))
        self._skills[name] = {"manifest": manifest, "sequence": sequence}
        log.info("Registered skill: %s", name)

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills
