import json
from pathlib import Path

import pytest

from evals import spend


@pytest.fixture(autouse=True)
def fresh_meter() -> None:
    spend.METER.reset()
    spend.METER.prices = {}


def test_agent_cost_and_priced_judge_are_summed(tmp_path: Path) -> None:
    spend.add_agent({"cost_usd": 0.25})
    spend.add_agent({})  # no model call
    spend.add_judge(1000, 500, {"usd_per_mtok": {"input": 3, "output": 15}})
    ledger = tmp_path / "ops" / "spend.jsonl"
    assert "recorded" in spend.append_ledger(ledger, "4")
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    assert sum(r["step"] == 4 for r in rows) == len(rows)
    assert sum(r["usd"] for r in rows) == pytest.approx(0.25 + 0.003 + 0.0075)


def test_unpriced_call_writes_nothing(tmp_path: Path) -> None:
    spend.add_agent({"input_tokens": 10})
    spend.add_judge(10, 10, {})
    ledger = tmp_path / "spend.jsonl"
    assert "not recorded" in spend.append_ledger(ledger, "4")
    assert not ledger.exists()


def test_no_step_writes_nothing(tmp_path: Path) -> None:
    ledger = tmp_path / "spend.jsonl"
    assert "no plan step" in spend.append_ledger(ledger, "")
    assert not ledger.exists()


def test_negative_or_bool_cost_is_unpriced() -> None:
    spend.add_agent({"cost_usd": -1})
    spend.add_agent({"cost_usd": True})
    assert spend.METER.unpriced == 2 and spend.METER.usd == 0.0
