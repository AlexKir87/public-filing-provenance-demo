"""Shared types for the eval framework."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """What the agent returns for one input.

    `run_agent` may return a plain string (output only) or an AgentResult. Return an
    AgentResult when cases check tool calls, token usage or latency. Map the framework's
    message history to `tool_calls` (LangGraph, Pydantic AI and the SDKs all expose it).
    """

    output: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class Grade:
    passed: bool
    detail: str = ""
