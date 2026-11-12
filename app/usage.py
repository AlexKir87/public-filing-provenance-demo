"""Normalise a model call's token usage into the dict the cost tools read (app code and evals).

The standard shape (also what `AgentResult.usage` should hold):
  {"model": "<exact id>", "input_tokens": N, "output_tokens": N, "cached_input_tokens": N,
   "cost_usd": x}          # cost_usd optional; without it the price comes from ops/prices.json
A run that makes several calls (or uses several models) returns {"calls": [<standard shape>, ...]}.
`input_tokens` includes the cached ones.

`normalize` accepts the usage object or dict of a provider reply and maps the usual alias names.
The alias lists below are a convenience, not a statement about any SDK: check the field names of
the SDK version in `docs/reference/` (the pinned wiki snapshot) and fix `ALIASES` if they differ.
Tokens that cannot be found are 0 and `complete` stays False so that the report can flag them.
"""

from collections.abc import Iterable, Mapping
from typing import Any

ALIASES = {
    "input_tokens": ("input_tokens", "prompt_tokens", "request_tokens"),
    "output_tokens": ("output_tokens", "completion_tokens", "response_tokens"),
    "cached_input_tokens": ("cached_input_tokens", "cache_read_input_tokens", "cached_tokens"),
}
NESTED_CACHED = (
    ("prompt_tokens_details", "cached_tokens"),
    ("input_token_details", "cache_read"),
)


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)


def _count(value: Any) -> int | None:
    ok = isinstance(value, int) and not isinstance(value, bool) and value >= 0
    return value if ok else None


def normalize(raw: Any, model: str = "") -> dict[str, Any]:
    """The standard usage dict from a provider usage object or dict. Never raises."""
    out: dict[str, Any] = {"model": model or str(_get(raw, "model") or "")}
    complete = True
    for field, names in ALIASES.items():
        value = next((c for n in names if (c := _count(_get(raw, n))) is not None), None)
        if value is None and field == "cached_input_tokens":
            for outer, inner in NESTED_CACHED:
                value = _count(_get(_get(raw, outer), inner))
                if value is not None:
                    break
            value = value or 0  # no cache figure means no cache hit was reported
        elif value is None:
            complete = False
        out[field] = value or 0
    cost = _get(raw, "cost_usd")
    if isinstance(cost, int | float) and not isinstance(cost, bool) and cost >= 0:
        out["cost_usd"] = float(cost)
    out["complete"] = complete
    return out


def merge(calls: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Several standard usage dicts as one `{"calls": [...]}` result for AgentResult.usage."""
    return {"calls": [c for c in calls if c]}


def calls_of(usage: dict[str, Any]) -> list[dict[str, Any]]:
    """The per-call usage dicts of an AgentResult.usage (flat or `calls`). [] = no model call."""
    if not usage:
        return []
    inner = usage.get("calls")
    if isinstance(inner, list):
        return [c for c in inner if isinstance(c, dict) and c]
    return [usage]
