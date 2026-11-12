from evals.graders import run_check
from evals.models import AgentResult, Grade, ToolCall


def res(output: str = "", calls: list[str] | None = None, **usage: int) -> AgentResult:
    return AgentResult(output, [ToolCall(n, {"order_id": "A1"}) for n in calls or []], usage)


def ok(spec: dict[str, object], result: AgentResult) -> bool:
    return run_check(spec, "q", result, None).passed


def test_contains_and_not_contains() -> None:
    assert ok({"type": "contains", "value": "HELLO"}, res("say hello"))
    assert not ok({"type": "contains", "value": "bye"}, res("hello"))
    assert ok({"type": "not_contains", "value": "bye"}, res("hello"))
    assert not ok({"type": "not_contains", "value": "hello"}, res("Hello"))


def test_regex() -> None:
    assert ok({"type": "regex", "pattern": r"\d{3}"}, res("code 123"))
    assert ok({"type": "not_regex", "pattern": r"\d{3}"}, res("none"))


def test_json_field_with_tolerance() -> None:
    out = '{"total": 120.5, "items": [{"sku": "x"}], "ccy": "EUR"}'
    assert ok({"type": "json_field", "path": "total", "equals": 120.5}, res(out))
    assert ok(
        {"type": "json_field", "path": "total", "equals": 120.49, "tolerance": 0.02}, res(out)
    )
    assert not ok({"type": "json_field", "path": "total", "equals": 120.0}, res(out))
    assert ok({"type": "json_field", "path": "items.0.sku", "equals": "x"}, res(out))
    assert not ok({"type": "json_field", "path": "ccy", "equals": "USD"}, res(out))
    assert not ok({"type": "json_field", "path": "missing", "equals": 1}, res(out))
    assert not ok({"type": "json_field", "path": "total", "equals": 1}, res("not json"))


def test_tool_calls_order_forbidden_and_max() -> None:
    r = res("", ["lookup", "check", "approve"])
    assert ok({"type": "tool_calls", "order": ["lookup", "approve"]}, r)
    assert not ok({"type": "tool_calls", "order": ["approve", "lookup"]}, r)
    assert not ok({"type": "tool_calls", "forbidden": ["check"]}, r)
    assert not ok({"type": "tool_calls", "max_calls": 2}, r)
    assert ok({"type": "tool_arg", "tool": "lookup", "arg": "order_id", "equals": "A1"}, r)
    assert not ok({"type": "tool_arg", "tool": "lookup", "arg": "order_id", "equals": "B"}, r)


def test_max_tokens() -> None:
    assert ok({"type": "max_tokens", "value": 100}, res("", total_tokens=50))
    assert not ok({"type": "max_tokens", "value": 100}, res("", total_tokens=500))
    assert not ok({"type": "max_tokens", "value": 100}, res(""))


def test_llm_judge_uses_injected_judge_and_fails_without_one() -> None:
    spec = {"type": "llm_judge", "rubric": "declines"}
    assert not run_check(spec, "q", res("x"), None).passed
    assert run_check(spec, "q", res("x"), lambda r, i, o: Grade(True)).passed
    assert not run_check(spec, "q", res("x"), lambda r, i, o: Grade(False, "no")).passed


def test_unknown_check_type_fails() -> None:
    assert not ok({"type": "nope"}, res("x"))
