"""Step trace of one agent run: one JSONL line per span, no content by default.

A trace answers "what did the agent do and where did it fail" without a tracing vendor. It is
the always-on layer: the same agent can also export OpenTelemetry to a backend
(docs/observability.md). The two are linked by `trace_id`: pass the OTel trace id of the active
span to `Trace(trace_id=...)`.

  trace = Trace(request_id="r-123")                       # production: metadata only
  with trace.span("invoke_agent", "advisor") as root:
      with trace.span("execute_tool", "get_quote", parent=root):
          ...

A line: {"ts", "trace_id", "span_id", "parent_id", "request_id", "op", "name", "status",
"ms", "error_type"?, "in"?, "out"?, "attrs"?, "content"?}. `op` is `invoke_agent`, `execute_tool`
or `chat` (names of the OpenTelemetry GenAI conventions, status Development: they may change).
An exception is recorded by its class name only, never its message (messages can hold client data).
`content` (inputs, outputs, tool arguments) is written only when `Trace(content=True)`: use it for
synthetic eval cases, never for client requests unless the owner approved it for that client at
kickoff.
Where it goes: `COGNITIA_TRACE_LOG` (default `ops/trace.jsonl`). Logging never raises.
"""

import json
import os
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.usage import calls_of

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "ops" / "trace.jsonl"


def new_id(nbytes: int = 8) -> str:
    return secrets.token_hex(nbytes)


def _count(value: Any) -> int:
    ok = isinstance(value, int) and not isinstance(value, bool) and value >= 0
    return value if ok else 0


class Span:
    """Mutable record of one span; `set` adds attributes, `fail` marks it as failed."""

    def __init__(self, span_id: str, parent_id: str) -> None:
        self.span_id = span_id
        self.parent_id = parent_id
        self.status = "ok"
        self.error_type = ""
        self.attrs: dict[str, Any] = {}
        self.content: dict[str, Any] = {}
        self.tokens_in = 0
        self.tokens_out = 0

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    def fail(self, error_type: str) -> None:
        self.status = "error"
        self.error_type = error_type


class Trace:
    def __init__(
        self,
        request_id: str = "",
        trace_id: str = "",
        path: Path | None = None,
        content: bool = False,
    ) -> None:
        self.request_id = request_id
        self.trace_id = trace_id or new_id(16)
        self.path = path or Path(os.environ.get("COGNITIA_TRACE_LOG", str(DEFAULT_LOG)))
        self.content = content
        self.failed = False

    @contextmanager
    def span(
        self, op: str, name: str, parent: Span | None = None, **content: Any
    ) -> Iterator[Span]:
        """Time a block, record it when it ends (also when it raises). `content` is dropped
        unless this trace was created with `content=True`."""
        span = Span(new_id(), parent.span_id if parent else "")
        if self.content:
            span.content.update(content)
        start = time.perf_counter()
        try:
            yield span
        except BaseException as exc:
            span.fail(type(exc).__name__)
            raise
        finally:
            self._write(op, name, span, round((time.perf_counter() - start) * 1000))

    def record(
        self,
        op: str,
        name: str,
        parent: Span | None = None,
        ms: int = 0,
        error_type: str = "",
        tokens: tuple[int, int] = (0, 0),
        **content: Any,
    ) -> Span:
        """Record an already finished step (for example from an AgentResult)."""
        span = Span(new_id(), parent.span_id if parent else "")
        if error_type:
            span.fail(error_type)
        span.tokens_in, span.tokens_out = tokens
        if self.content:
            span.content.update(content)
        self._write(op, name, span, ms)
        return span

    def _write(self, op: str, name: str, span: Span, ms: int) -> None:
        if span.status == "error":
            self.failed = True
        row: dict[str, Any] = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "trace_id": self.trace_id,
            "span_id": span.span_id,
            "parent_id": span.parent_id,
            "request_id": self.request_id,
            "op": op,
            "name": name,
            "status": span.status,
            "ms": ms,
        }
        if span.error_type:
            row["error_type"] = span.error_type
        if span.tokens_in or span.tokens_out:
            row["in"], row["out"] = span.tokens_in, span.tokens_out
        if span.attrs:
            row["attrs"] = span.attrs
        if span.content:
            row["content"] = span.content
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except (OSError, TypeError, ValueError):
            pass


def record_result(
    trace: Trace, agent_name: str, input_text: str, result: Any, ms: int = 0, error: str = ""
) -> Span:
    """Spans for a finished run from an `AgentResult` (evals): the agent, its tool calls, its model
    calls (token counts from `usage`). `error` marks the agent span as failed."""
    root = trace.record(
        "invoke_agent", agent_name, ms=ms, error_type=error, input=input_text,
        output=str(getattr(result, "output", "")),
    )  # fmt: skip
    for call in getattr(result, "tool_calls", []) or []:
        trace.record("execute_tool", str(call.name), parent=root, args=dict(call.args))
    for usage in calls_of(getattr(result, "usage", {}) or {}):
        tokens = (_count(usage.get("input_tokens")), _count(usage.get("output_tokens")))
        trace.record("chat", str(usage.get("model") or ""), parent=root, tokens=tokens)
    return root


def steps_of(result: Any) -> list[str]:
    """Names of the tools the agent called, in order: the one-line trajectory for reports."""
    return [str(c.name) for c in getattr(result, "tool_calls", []) or []]
