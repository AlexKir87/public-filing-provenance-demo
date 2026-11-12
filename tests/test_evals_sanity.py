"""Tests for the eval sanity check, the pinned judge, the no-secrets mode and the baseline."""

import json
from pathlib import Path
from typing import Any

import pytest

import app.agent
from evals import judge, spend
from evals.run_evals import compare_baseline, main
from evals.sanity import sanity

CATS = ["happy", "edge", "adversarial", "out_of_scope"]


def case(i: int, checks: list[dict[str, Any]], inp: str = "x") -> dict[str, Any]:
    return {"id": f"c{i}", "category": CATS[i % 4], "input": inp, "checks": checks}


STRONG = [{"type": "contains", "value": "invoice 42"}]
WEAK = [{"type": "not_contains", "value": "password"}]
JUDGE = [{"type": "llm_judge", "rubric": "refuses"}]


def test_discriminating_cases_pass_sanity() -> None:
    ok, out = sanity([case(i, STRONG) for i in range(10)], 0.9)
    assert ok and out["stubs"]["empty"]["fail_rate"] == 1.0


def test_negative_only_cases_are_weak() -> None:
    cases = [case(i, STRONG) for i in range(8)] + [case(i, WEAK) for i in range(8, 11)]
    ok, out = sanity(cases, 0.9)
    assert not ok
    assert out["stubs"]["empty"]["passed_by_stub"] == ["c8", "c9", "c10"]


def test_echo_stub_catches_cases_the_input_already_satisfies() -> None:
    cases = [case(i, [{"type": "contains", "value": "hello"}], inp="hello") for i in range(10)]
    ok, out = sanity(cases, 0.9)
    assert not ok and out["stubs"]["echo"]["fail_rate"] == 0.0


def test_judge_only_cases_are_undetermined_and_limited() -> None:
    ok, out = sanity(
        [case(i, STRONG) for i in range(7)] + [case(i, JUDGE) for i in range(7, 10)], 0.9
    )
    assert ok and out["undetermined"] == ["c7", "c8", "c9"]
    ok, out = sanity([case(0, STRONG)] + [case(i, JUDGE) for i in range(1, 10)], 0.9)
    assert not ok and "problem" in out


def test_shipped_judge_config_is_pinned_and_deterministic() -> None:
    cfg = judge.load_config()
    assert cfg["model"] and cfg["temperature"] == 0


def test_shipped_judge_is_openrouter_deepseek_with_a_price() -> None:
    cfg = judge.load_config()
    assert cfg["provider"] == "openrouter" and cfg["model"].startswith("deepseek/")
    assert cfg["usd_per_mtok"]["input"] > 0 and cfg["usd_per_mtok"]["output"] > 0
    route = cfg["openrouter"]
    assert route["data_collection"] == "deny" and route["zdr"] is True
    price = route["max_price"]
    assert (price["prompt"], price["completion"]) == (
        cfg["usd_per_mtok"]["input"],
        cfg["usd_per_mtok"]["output"],
    )


def test_judge_key_comes_from_the_providers_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("COGNITIA_OPENROUTER_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    cfg = {"provider": "openrouter"}
    assert judge.api_key(cfg) is None and judge.key_name(cfg) == "COGNITIA_OPENROUTER_API_KEY"
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert judge.api_key(cfg) == "k" and judge.api_key({"provider": "anthropic"}) is None


def test_openrouter_judge_sends_routing_and_prices_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, Any] = {}

    class Resp:
        def __enter__(self) -> "Resp":
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [{"message": {"content": '{"pass": true, "reason": "ok"}'}}],
                    "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
                }
            ).encode()

    def fake_urlopen(req: Any, timeout: int) -> Resp:
        sent.update(json.loads(req.data), url=req.full_url, auth=req.get_header("Authorization"))
        return Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("COGNITIA_OPENROUTER_API_KEY", "k")
    before = spend.METER.usd
    grade = judge.judge("r", "i", "o")
    assert grade.passed and sent["provider"] == judge.load_config()["openrouter"]
    assert sent["provider"]["zdr"] is True
    assert sent["url"].startswith("https://openrouter.ai/") and sent["auth"] == "Bearer k"
    assert sent["temperature"] == 0 and sent["model"] == "deepseek/deepseek-v4.1-flash-1017"
    assert abs(spend.METER.usd - before - 1.50) < 1e-9


def test_judge_config_rejects_unpinned_or_random(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    for bad in ({"model": "TBD", "temperature": 0}, {"model": "m", "temperature": 0.7}, {}):
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            judge.load_config(p)


def write_cases(tmp_path: Path, judged: bool) -> Path:
    rows = [case(i, STRONG) for i in range(10)]
    if judged:
        rows[0]["checks"] = STRONG + JUDGE
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def run_main(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr("sys.argv", ["run_evals.py", *args])
    return main()


def test_missing_key_is_incomplete_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("COGNITIA_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("COGNITIA_OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert run_main(monkeypatch, "--cases", str(write_cases(tmp_path, True))) == 2
    assert "INCOMPLETE" in capsys.readouterr().out


def test_skip_judge_run_is_marked_in_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app.agent, "run_agent", lambda text: "invoice 42")
    report = tmp_path / "r.json"
    cases = str(write_cases(tmp_path, True))
    assert run_main(monkeypatch, "--cases", cases, "--skip-judge", "--report", str(report)) == 0
    data = json.loads(report.read_text())
    assert data["mode"] == "no-judge" and data["judge_checks_skipped"] == 1


def test_no_judge_run_cannot_become_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app.agent, "run_agent", lambda text: "invoice 42")
    cases = str(write_cases(tmp_path, True))
    assert run_main(monkeypatch, "--cases", cases, "--skip-judge", "--update-baseline") == 1


def test_baseline_catches_fewer_cases_removed_cases_and_lower_rate() -> None:
    base = {"pass_rate": 1.0, "cases": 3, "case_ids": ["a", "b", "c"]}
    good = {"mode": "full", "pass_rate": 1.0, "cases": [{"id": i} for i in "abc"]}
    assert compare_baseline(base, good, 0.0) == []
    fewer = {**good, "cases": [{"id": "a"}, {"id": "b"}]}
    assert len(compare_baseline(base, fewer, 0.0)) == 2
    worse = {**good, "pass_rate": 0.9}
    assert "pass rate" in compare_baseline(base, worse, 0.0)[0]
    assert compare_baseline(base, worse, 0.15) == []
    assert compare_baseline(base, {**worse, "mode": "no-judge"}, 0.0) == []
