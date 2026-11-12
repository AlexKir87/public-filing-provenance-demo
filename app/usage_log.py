"""Token log of production requests: one JSONL line per model per request, counts only.

Call `log_request(usage, request_id)` once per user request with the usage the agent collected
(flat or `{"calls": [...]}`, see app/usage.py). A line holds no prompt, answer or client data:

  {"ts": "...", "kind": "prod", "request_id": "r-123", "model": "...", "calls": 3,
   "in": 5200, "out": 610, "cached": 0, "usd": 0.0041}      # usd only when the agent reported it

Prices are applied when the report is built (`scripts/build_report.py tokenomics`, from
ops/prices.json), so a price change never rewrites the log. Where the log lives is the
operator's choice: `COGNITIA_USAGE_LOG` (default `ops/usage.jsonl`); ship it to the client's log
store in production. Logging never raises: a full disk must not fail a user request.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.usage import calls_of

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "ops" / "usage.jsonl"


def _count(value: Any) -> int:
    ok = isinstance(value, int) and not isinstance(value, bool) and value >= 0
    return value if ok else 0


def usage_rows(usage: dict[str, Any], request_id: str, now: datetime | None = None) -> list[str]:
    """JSONL lines (one per model) for one request. [] when the request made no model call."""
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    per_model: dict[str, dict[str, Any]] = {}
    for call in calls_of(usage):
        model = str(call.get("model") or "")
        row = per_model.setdefault(
            model,
            {"ts": stamp, "kind": "prod", "request_id": request_id, "model": model, "calls": 0,
             "in": 0, "out": 0, "cached": 0},
        )  # fmt: skip
        row["calls"] += 1
        row["in"] += _count(call.get("input_tokens"))
        row["out"] += _count(call.get("output_tokens"))
        row["cached"] += _count(call.get("cached_input_tokens"))
        cost = call.get("cost_usd")
        if isinstance(cost, int | float) and not isinstance(cost, bool) and cost >= 0:
            row["_usd"] = float(row.get("_usd", 0.0)) + float(cost)
            row["_priced"] = row.get("_priced", 0) + 1
    out = []
    for _, row in sorted(per_model.items()):
        usd, priced = row.pop("_usd", None), row.pop("_priced", 0)
        if usd is not None and priced == row["calls"]:  # a partly priced model gets no usd
            row["usd"] = round(usd, 6)
        out.append(json.dumps(row))
    return out


def log_request(usage: dict[str, Any], request_id: str, path: Path | None = None) -> None:
    target = path or Path(os.environ.get("COGNITIA_USAGE_LOG", str(DEFAULT_LOG)))
    try:
        lines = usage_rows(usage, request_id)
        if lines:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
    except OSError:
        pass
