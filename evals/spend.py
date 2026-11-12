"""Model spend and token use of one eval run, appended to the build ledger ops/spend.jsonl.

The ledger is what `scripts/build_report.py spend` and `validate_stage.py step --spend-log` read
({"step": N, "usd": x}, one line per run). Since tokenomics each line also says what was used, so
that `scripts/build_report.py tokenomics` can show tokens x price per model:
  {"step": 4, "usd": 0.62, "ts": "...", "kind": "eval", "source": "agent|judge", "model": "...",
   "calls": 12, "requests": 12, "in": 18000, "out": 2400, "cached": 0}
(`requests` = agent runs, only on agent lines). Nothing here guesses a price: the agent reports
`usage["cost_usd"]` itself, or the call is priced from ops/prices.json (model, source_url and the
date checked), and the judge is priced by `usd_per_mtok` in evals/judge_config.json (copy the
figures from the provider's price page). A call that cannot be priced is counted as unpriced, and
then no ledger line is written, so the spend check stays n/a and fails closed.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.usage import calls_of
from evals import pricing


@dataclass
class Group:
    calls: int = 0
    requests: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cached: int = 0
    usd: float = 0.0


@dataclass
class Meter:
    usd: float = 0.0
    unpriced: int = 0
    groups: dict[tuple[str, str], Group] = field(default_factory=dict)
    prices: dict[str, pricing.Price] | None = None  # None: read ops/prices.json when needed

    def reset(self) -> None:
        self.usd, self.unpriced, self.groups = 0.0, 0, {}


METER = Meter()


def _count(value: Any) -> int:
    ok = isinstance(value, int) and not isinstance(value, bool) and value >= 0
    return value if ok else 0


def _price_table() -> dict[str, pricing.Price]:
    if METER.prices is None:
        METER.prices = pricing.load_prices()
    return METER.prices


def _record(
    source: str, model: str, tokens: tuple[int, int, int], usd: float | None, request: bool = False
) -> None:
    group = METER.groups.setdefault((source, model), Group())
    group.calls += 1
    group.requests += int(request)
    group.tokens_in += tokens[0]
    group.tokens_out += tokens[1]
    group.cached += tokens[2]
    if usd is None:
        METER.unpriced += 1
    else:
        METER.usd += usd
        group.usd += usd


def add_agent(usage: dict[str, Any]) -> None:
    """Count one agent run. An empty usage means no model call (nothing to count).

    `usage` is flat or `{"calls": [...]}` (see app/usage.py)."""
    for i, call in enumerate(calls_of(usage)):
        model = str(call.get("model") or "")
        tokens = (
            _count(call.get("input_tokens")),
            _count(call.get("output_tokens")),
            _count(call.get("cached_input_tokens")),
        )
        cost = call.get("cost_usd")
        if isinstance(cost, int | float) and not isinstance(cost, bool) and cost >= 0:
            usd: float | None = float(cost)
        elif model in _price_table():
            usd = _price_table()[model].cost(*tokens)
        else:
            usd = None
        _record("agent", model, tokens, usd, request=(i == 0))


def add_judge(input_tokens: int, output_tokens: int, cfg: dict[str, Any]) -> None:
    """Count one judge call, priced from `usd_per_mtok: {"input": x, "output": y}`."""
    model = str(cfg.get("model") or "")
    prices: Any = cfg.get("usd_per_mtok")
    try:
        rate_in, rate_out = float(prices["input"]), float(prices["output"])
    except (TypeError, KeyError, ValueError):
        _record("judge", model, (input_tokens, output_tokens, 0), None)
        return
    usd = (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000
    _record("judge", model, (input_tokens, output_tokens, 0), usd)


def append_ledger(path: Path, step: str, kind: str = "eval", now: datetime | None = None) -> str:
    """Append this run's spend for `step`; return a one-line note for the run output."""
    if not step.isdigit():
        return "spend not recorded: no plan step (set BUILD_STEP or pass --step)"
    if METER.unpriced:
        return (
            f"spend not recorded: {METER.unpriced} model calls without a price "
            "(usage['cost_usd'] from the agent, a model entry in ops/prices.json, "
            "usd_per_mtok in evals/judge_config.json)"
        )
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict[str, Any]] = []
    for (source, model), g in sorted(METER.groups.items()):
        row: dict[str, Any] = {"step": int(step), "usd": round(g.usd, 6), "ts": stamp}
        row.update(kind=kind, source=source, model=model, calls=g.calls)
        if source == "agent":
            row["requests"] = g.requests
        row.update({"in": g.tokens_in, "out": g.tokens_out, "cached": g.cached})
        rows.append(row)
    rows = rows or [{"step": int(step), "usd": 0.0, "ts": stamp, "kind": kind}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return f"spend recorded: step {step}, ${METER.usd:.4f} -> {path.name}"
