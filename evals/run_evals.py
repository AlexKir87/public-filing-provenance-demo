#!/usr/bin/env python3
"""Run eval cases against app.agent.run_agent and fail below a pass-rate threshold.

Case format, one JSON object per line in evals/cases.jsonl:
  {"id": "c1", "category": "happy|edge|adversarial|out_of_scope", "input": "...",
   "checks": [ {"type": "contains", "value": "..."}, ... ]}
Check types are documented in evals/graders.py. The old format
  {"expected_contains": [...], "forbidden_contains": [...]} is still accepted.

Usage:
  python evals/run_evals.py --threshold 0.9 --report evals/report.json
  python evals/run_evals.py --repeats 3      # run each case 3 times, majority decides
  python evals/run_evals.py --skip-judge     # skip llm_judge checks (offline; marked in the report)
  python evals/run_evals.py --baseline evals/report.baseline.json   # fail below the golden report
  python evals/run_evals.py --update-baseline                       # write the golden report
  python evals/run_evals.py --ledger ops/spend.jsonl --step 4       # append this run's model spend
  python evals/run_evals.py --traces-dir evals/traces               # one step trace per case

Exit codes: 0 passed; 1 failed; 2 incomplete (llm_judge checks need a key that is not available).
The judge is pinned in evals/judge_config.json (exact model id, temperature 0).
"""

import argparse
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.trace_log import Trace, record_result, steps_of  # noqa: E402
from evals import spend  # noqa: E402
from evals.graders import JudgeFn, run_check  # noqa: E402
from evals.models import AgentResult, Grade  # noqa: E402

BASELINE = ROOT / "evals" / "report.baseline.json"
REQUIRED_CATEGORIES = ("happy", "edge", "adversarial", "out_of_scope")


def load_cases(path: Path) -> list[dict[str, Any]]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def checks_for(case: dict[str, Any]) -> list[dict[str, Any]]:
    checks = list(case.get("checks", []))
    checks += [{"type": "contains", "value": s} for s in case.get("expected_contains", [])]
    checks += [{"type": "not_contains", "value": s} for s in case.get("forbidden_contains", [])]
    return checks


def to_result(raw: Any) -> AgentResult:
    return raw if isinstance(raw, AgentResult) else AgentResult(output=str(raw))


def run_once(
    case: dict[str, Any],
    run_agent: Any,
    judge: JudgeFn | None,
    skip_judge: bool,
    trace: Trace | None = None,
    steps: list[str] | None = None,
) -> tuple[bool, list[str], float]:
    """One run of one case. `trace` gets the step spans; `steps` is filled with the tool names."""
    start = time.perf_counter()
    try:
        result = to_result(run_agent(case["input"]))
    except Exception as exc:  # a crash is a failed case, not a crashed run
        elapsed = time.perf_counter() - start
        if trace:
            record_result(
                trace, "agent", case["input"], AgentResult(output=""),
                round(elapsed * 1000), type(exc).__name__,
            )  # fmt: skip
        return False, [f"exception {type(exc).__name__}: {exc}"], elapsed
    elapsed = time.perf_counter() - start
    if steps is not None:
        steps[:] = steps_of(result)
    spend.add_agent(result.usage)
    problems: list[str] = []
    for spec in checks_for(case):
        if skip_judge and spec["type"] == "llm_judge":
            continue
        try:
            grade: Grade = run_check(spec, case["input"], result, judge)
        except Exception as exc:  # a broken judge or malformed check fails the case, not the run
            grade = Grade(False, f"{type(exc).__name__}: {exc}")
        if not grade.passed:
            problems.append(f"{spec['type']}: {grade.detail}")
    if not checks_for(case):
        problems.append("case has no checks")
    if trace:
        record_result(
            trace, "agent", case["input"], result, round(elapsed * 1000),
            "eval-check-failed" if problems else "",
        )  # fmt: skip
    return not problems, problems, elapsed


def judge_check_count(cases: list[dict[str, Any]]) -> int:
    return sum(s["type"] == "llm_judge" for c in cases for s in checks_for(c))


def compare_baseline(
    baseline: dict[str, Any], report: dict[str, Any], tolerance: float
) -> list[str]:
    """Reasons the run is worse than the golden report. Empty = no regression."""
    problems: list[str] = []
    ids = {c["id"] for c in report["cases"]}
    if len(ids) < baseline["cases"]:
        problems.append(f"{len(ids)} cases, baseline has {baseline['cases']}")
    gone = sorted(set(baseline["case_ids"]) - ids)
    if gone:
        problems.append(f"cases removed since the baseline: {gone}")
    if report["mode"] == "no-judge":
        return problems  # pass rate over fewer checks is not comparable
    if report["pass_rate"] < baseline["pass_rate"] - tolerance:
        problems.append(
            f"pass rate {report['pass_rate']:.2%} < baseline {baseline['pass_rate']:.2%}"
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(ROOT / "evals" / "cases.jsonl"))
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--min-cases", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--report", default="")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--allow-missing-categories", action="store_true")
    parser.add_argument("--baseline", default="", help="golden report; fail if the run is worse")
    parser.add_argument("--baseline-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--update-baseline", action="store_true", help="write evals/report.baseline.json"
    )
    parser.add_argument("--ledger", default="", help="append model spend, e.g. ops/spend.jsonl")
    parser.add_argument("--step", default=os.environ.get("BUILD_STEP", ""), help="plan step")
    parser.add_argument(
        "--traces-dir", default="", help="write one step trace per case, e.g. evals/traces"
    )
    parser.add_argument(
        "--no-trace-content",
        action="store_true",
        help="traces keep tool names and token counts only (use when cases hold real data)",
    )
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    if len(cases) < args.min_cases:
        print(f"FAIL: {len(cases)} cases, need at least {args.min_cases} (spec.md requires 10)")
        return 1
    present = {c.get("category", "") for c in cases}
    missing = [c for c in REQUIRED_CATEGORIES if c not in present]
    if missing and not args.allow_missing_categories:
        print(f"FAIL: no cases in categories {missing}; spec.md requires all of them")
        return 1

    from evals.judge import api_key, key_name, load_config
    from evals.judge import judge as pinned_judge

    judge_checks = judge_check_count(cases)
    judge: JudgeFn | None = None
    judge_cfg: dict[str, Any] = {}
    if judge_checks and not args.skip_judge:
        try:
            judge_cfg = load_config()
        except (OSError, ValueError) as exc:
            print(f"FAIL: judge config: {exc}")
            return 1
        if not api_key(judge_cfg):
            print(
                f"INCOMPLETE: {judge_checks} llm_judge checks need {key_name(judge_cfg)}, "
                "which this run does not have (a fork or a PR without secrets). Nothing was "
                "graded. Re-run where the secret exists, or use --skip-judge for a marked "
                "partial run."
            )
            return 2
        judge = pinned_judge
    skipped = judge_checks if args.skip_judge else 0
    run_agent = importlib.import_module("app.agent").run_agent

    by_category: dict[str, list[bool]] = defaultdict(list)
    report: list[dict[str, Any]] = []
    traces_dir = Path(args.traces_dir) if args.traces_dir else None
    if traces_dir:
        traces_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        trace: Trace | None = None
        if traces_dir:
            path = traces_dir / f"{case['id']}.jsonl"
            path.unlink(missing_ok=True)  # a re-run replaces the trace of the case
            trace = Trace(request_id=case["id"], path=path, content=not args.no_trace_content)
        steps: list[str] = []
        runs = [
            run_once(case, run_agent, judge, args.skip_judge, trace, steps)
            for _ in range(args.repeats)
        ]
        passes = sum(ok for ok, _, _ in runs)
        passed = passes * 2 > len(runs)  # strict majority
        problems = next((p for ok, p, _ in runs if not ok), [])
        by_category[case.get("category", "uncategorized")].append(passed)
        mean_s = sum(t for _, _, t in runs) / len(runs)
        report.append(
            {
                "id": case["id"],
                "category": case.get("category", ""),
                "passed": passed,
                "passes": f"{passes}/{len(runs)}",
                "seconds": round(mean_s, 3),
                "problems": problems,
                **({"trace_id": trace.trace_id, "steps": steps} if trace else {}),
            }
        )
        status = "PASS" if passed else "FAIL"
        print(f"{status} {case['id']} ({passes}/{len(runs)}) {'; '.join(problems)}")

    total = sum(r["passed"] for r in report)
    rate = total / len(report)
    for cat, results in sorted(by_category.items()):
        print(f"  {cat}: {sum(results)}/{len(results)}")
    print(f"pass rate {rate:.2%} ({total}/{len(report)}), threshold {args.threshold:.0%}")
    summary: dict[str, Any] = {
        "mode": "no-judge" if skipped else "full",
        "judge_checks_skipped": skipped,
        "judge": {"model": judge_cfg.get("model"), "temperature": judge_cfg.get("temperature")}
        if judge
        else None,
        "repeats": args.repeats,
        "pass_rate": rate,
        "cases": report,
    }
    if skipped:
        print(f"NOTE: {skipped} llm_judge checks skipped (--skip-judge); report mode is no-judge")
    if args.report:
        Path(args.report).write_text(json.dumps(summary, indent=2))
    ok = rate >= args.threshold
    if args.ledger:
        print(spend.append_ledger(Path(args.ledger), args.step))
    if args.update_baseline:
        if skipped:
            print("FAIL: refusing to write a baseline from a run that skipped judge checks")
            return 1
        BASELINE.write_text(
            json.dumps(
                {
                    "pass_rate": rate,
                    "cases": len(report),
                    "case_ids": sorted(r["id"] for r in report),
                    "judge": summary["judge"],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"baseline written: {BASELINE.relative_to(ROOT)}")
    if args.baseline:
        base_path = Path(args.baseline)
        if not base_path.is_file():
            print(f"FAIL: baseline {args.baseline} not found")
            return 1
        regress = compare_baseline(
            json.loads(base_path.read_text()), summary, args.baseline_tolerance
        )
        for p in regress:
            print(f"REGRESSION: {p}")
        ok = ok and not regress
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
