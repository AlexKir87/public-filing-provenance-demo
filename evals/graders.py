"""Graders. Each takes a check spec (a dict from the case file) and returns a Grade.

Check types
  contains / not_contains   {"type": "contains", "value": "text"}      case-insensitive
  regex / not_regex         {"type": "regex", "pattern": "^\\d+$"}
  json_field                {"type": "json_field", "path": "invoice.total", "equals": 120.5,
                             "tolerance": 0.01}   output must be JSON; path uses dots/indexes
  tool_calls                {"type": "tool_calls", "order": ["lookup", "approve"],
                             "forbidden": ["send_payment"], "max_calls": 6}
  tool_arg                  {"type": "tool_arg", "tool": "lookup", "arg": "id", "equals": "A1"}
  max_tokens                {"type": "max_tokens", "value": 4000}  needs usage.total_tokens
  llm_judge                 {"type": "llm_judge", "rubric": "Refuses and explains why."}
"""

import json
import re
from collections.abc import Callable
from typing import Any

from evals.models import AgentResult, Grade

JudgeFn = Callable[[str, str, str], Grade]  # (rubric, input, output) -> Grade


def _lookup(data: Any, path: str) -> Any:
    cur = data
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


def grade_contains(spec: dict[str, Any], result: AgentResult, negate: bool) -> Grade:
    found = str(spec["value"]).lower() in result.output.lower()
    if found == (not negate):
        return Grade(True)
    return Grade(False, f"{'forbidden' if negate else 'missing'} {spec['value']!r}")


def grade_regex(spec: dict[str, Any], result: AgentResult, negate: bool) -> Grade:
    found = (
        re.search(spec["pattern"], result.output, flags=re.IGNORECASE | re.MULTILINE) is not None
    )
    if found == (not negate):
        return Grade(True)
    return Grade(False, f"{'forbidden' if negate else 'no match for'} /{spec['pattern']}/")


def grade_json_field(spec: dict[str, Any], result: AgentResult) -> Grade:
    try:
        data = json.loads(result.output)
        actual = _lookup(data, spec["path"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return Grade(False, f"cannot read {spec['path']}: {type(exc).__name__}")
    expected = spec["equals"]
    tol = spec.get("tolerance")
    if isinstance(expected, int | float) and not isinstance(expected, bool):
        if not isinstance(actual, int | float) or isinstance(actual, bool):
            return Grade(False, f"{spec['path']} is {actual!r}, expected a number")
        if abs(actual - expected) <= (tol if tol is not None else 0):
            return Grade(True)
        return Grade(False, f"{spec['path']} is {actual!r}, expected {expected!r} +/- {tol or 0}")
    return Grade(actual == expected, "" if actual == expected else f"{spec['path']} is {actual!r}")


def grade_tool_calls(spec: dict[str, Any], result: AgentResult) -> Grade:
    names = [c.name for c in result.tool_calls]
    for bad in spec.get("forbidden", []):
        if bad in names:
            return Grade(False, f"forbidden tool called: {bad}")
    if "max_calls" in spec and len(names) > spec["max_calls"]:
        return Grade(False, f"{len(names)} tool calls, max {spec['max_calls']}")
    order = spec.get("order", [])
    it = iter(names)
    if not all(any(n == want for n in it) for want in order):  # subsequence, in order
        return Grade(False, f"expected tool order {order}, got {names}")
    return Grade(True)


def grade_tool_arg(spec: dict[str, Any], result: AgentResult) -> Grade:
    for call in result.tool_calls:
        if call.name == spec["tool"] and call.args.get(spec["arg"]) == spec["equals"]:
            return Grade(True)
    return Grade(False, f"no call to {spec['tool']} with {spec['arg']}={spec['equals']!r}")


def grade_max_tokens(spec: dict[str, Any], result: AgentResult) -> Grade:
    total = result.usage.get("total_tokens")
    if total is None:
        return Grade(False, "usage.total_tokens not reported by the agent")
    return Grade(total <= spec["value"], "" if total <= spec["value"] else f"{total} tokens")


def run_check(
    spec: dict[str, Any], case_input: str, result: AgentResult, judge: JudgeFn | None
) -> Grade:
    kind = spec["type"]
    if kind == "contains":
        return grade_contains(spec, result, negate=False)
    if kind == "not_contains":
        return grade_contains(spec, result, negate=True)
    if kind == "regex":
        return grade_regex(spec, result, negate=False)
    if kind == "not_regex":
        return grade_regex(spec, result, negate=True)
    if kind == "json_field":
        return grade_json_field(spec, result)
    if kind == "tool_calls":
        return grade_tool_calls(spec, result)
    if kind == "tool_arg":
        return grade_tool_arg(spec, result)
    if kind == "max_tokens":
        return grade_max_tokens(spec, result)
    if kind == "llm_judge":
        if judge is None:
            return Grade(False, "llm_judge check but no judge configured (see evals/judge.py)")
        return judge(spec["rubric"], case_input, result.output)
    return Grade(False, f"unknown check type {kind!r}")
