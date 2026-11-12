"""Model price table: ops/prices.json, copied by hand from the providers' price pages.

Nothing here knows a price. The table is empty at kickoff; the owner or the delivery thread adds
a model only with its `source_url` and `checked_on` (the date the page was read). An entry without
both, or with a negative or non-numeric rate, is ignored, so the model counts as unpriced and the
tokenomics report says so instead of guessing.

{
  "currency": "USD",
  "models": {
    "<exact model id>": {
      "input": 0.30, "output": 1.20, "cached_input": 0.03,   # USD per 1M tokens; cached optional
      "source_url": "https://...", "checked_on": "2026-11-06"
    }
  }
}
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRICES = Path(__file__).resolve().parent.parent / "ops" / "prices.json"
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Price:
    model: str
    input: float  # USD per 1M tokens
    output: float
    cached_input: float | None
    source_url: str
    checked_on: str

    def cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        """USD for one call. `input_tokens` includes the cached ones; cached are billed at the
        cached rate, or at the full input rate when the table has no cached rate (overstates)."""
        cached = min(max(cached_tokens, 0), max(input_tokens, 0))
        cached_rate = self.input if self.cached_input is None else self.cached_input
        fresh = max(input_tokens, 0) - cached
        return (fresh * self.input + cached * cached_rate + output_tokens * self.output) / 1e6


def _rate(value: Any) -> float | None:
    ok = isinstance(value, int | float) and not isinstance(value, bool) and value >= 0
    return float(value) if ok else None


def parse_prices(data: Any) -> dict[str, Price]:
    """Valid entries only. Anything malformed is dropped (fail closed: unpriced)."""
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, dict):
        return {}
    found: dict[str, Price] = {}
    for name, entry in models.items():
        if not isinstance(entry, dict):
            continue
        rate_in, rate_out = _rate(entry.get("input")), _rate(entry.get("output"))
        cached = entry.get("cached_input")
        url, day = entry.get("source_url"), entry.get("checked_on")
        if rate_in is None or rate_out is None or (cached is not None and _rate(cached) is None):
            continue
        if not (isinstance(url, str) and url.startswith("https://")):
            continue
        if not (isinstance(day, str) and DATE.match(day)):
            continue
        found[str(name)] = Price(
            str(name), rate_in, rate_out, None if cached is None else float(cached), url, day
        )
    return found


def load_prices(path: Path = PRICES) -> dict[str, Price]:
    try:
        return parse_prices(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}
