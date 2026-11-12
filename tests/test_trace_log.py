import json
from pathlib import Path

import pytest

from app.trace_log import Trace, record_result, steps_of
from evals.models import AgentResult, ToolCall
from evals.run_evals import run_once


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(ln) for ln in path.read_text().splitlines()]


def test_span_records_status_time_and_parent(tmp_path: Path) -> None:
    log = tmp_path / "ops" / "trace.jsonl"
    trace = Trace(request_id="r-1", path=log)
    with trace.span("invoke_agent", "advisor") as root:
        with trace.span("execute_tool", "get_quote", parent=root) as tool:
            tool.set(rows=3)
    out = rows(log)
    assert [r["name"] for r in out] == ["get_quote", "advisor"]  # children end first
    assert out[0]["parent_id"] == root.span_id
    assert out[0]["attrs"] == {"rows": 3}
    assert {r["trace_id"] for r in out} == {trace.trace_id}
    assert all(r["request_id"] == "r-1" and r["status"] == "ok" for r in out)


def test_error_keeps_class_name_never_the_message(tmp_path: Path) -> None:
    log = tmp_path / "t.jsonl"
    trace = Trace(path=log)
    with pytest.raises(ValueError):
        with trace.span("execute_tool", "get_quote"):
            raise ValueError("account 12345 of Jane Doe")
    row = rows(log)[0]
    assert row["status"] == "error" and row["error_type"] == "ValueError"
    assert "Jane" not in log.read_text() and trace.failed


def test_content_only_when_enabled(tmp_path: Path) -> None:
    off, on = tmp_path / "off.jsonl", tmp_path / "on.jsonl"
    with Trace(path=off).span("chat", "m", prompt="secret") as _:
        pass
    with Trace(path=on, content=True).span("chat", "m", prompt="secret"):
        pass
    assert "secret" not in off.read_text() and "content" not in rows(off)[0]
    assert rows(on)[0]["content"] == {"prompt": "secret"}


def test_logging_never_raises(tmp_path: Path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("x")
    with Trace(path=blocked / "sub" / "t.jsonl").span("chat", "m"):  # parent is a file
        pass


def test_record_result_makes_tool_and_model_spans(tmp_path: Path) -> None:
    log = tmp_path / "t.jsonl"
    result = AgentResult(
        output="done",
        tool_calls=[ToolCall("search", {"q": "x"}), ToolCall("send", {})],
        usage={"calls": [{"model": "m1", "input_tokens": 10, "output_tokens": 4}]},
    )
    record_result(Trace(path=log, content=True), "agent", "hi", result, 12)
    out = {r["name"]: r for r in rows(log)}
    assert out["search"]["content"] == {"args": {"q": "x"}}
    assert (out["m1"]["in"], out["m1"]["out"]) == (10, 4)
    assert out["agent"]["ms"] == 12 and steps_of(result) == ["search", "send"]


def test_run_once_traces_failed_case_and_crash(tmp_path: Path) -> None:
    case = {"id": "c1", "input": "q", "checks": [{"type": "contains", "value": "yes"}]}
    log = tmp_path / "c1.jsonl"
    steps: list[str] = []

    def agent(text: str) -> AgentResult:
        return AgentResult(output="no", tool_calls=[ToolCall("lookup")])

    ok, problems, _ = run_once(case, agent, None, True, Trace(path=log), steps)
    assert not ok and problems and steps == ["lookup"]
    root = [r for r in rows(log) if r["op"] == "invoke_agent"][0]
    assert root["status"] == "error"

    def crash(text: str) -> str:
        raise RuntimeError("boom")

    crash_log = tmp_path / "c2.jsonl"
    ok, _, _ = run_once(case, crash, None, True, Trace(path=crash_log))
    assert not ok and rows(crash_log)[0]["error_type"] == "RuntimeError"
