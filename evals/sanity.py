#!/usr/bin/env python3
"""Do the eval cases discriminate? Run them against stub agents that know nothing.

A stub that returns nothing, and one that echoes the input, must fail at least --min-fail-rate of
the cases. A case a stub passes asserts too little (usually only `not_contains` checks, or a
`contains` the input already holds): add a positive check. No model calls and no keys: llm_judge
checks are skipped, and a case made only of judge checks cannot be judged here, so it is reported
as undetermined (at least half of the cases must be determined).

Usage: python evals/sanity.py [--min-fail-rate 0.9] [--report evals/sanity.json]
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.run_evals import checks_for, load_cases, run_once  # noqa: E402

STUBS: dict[str, Callable[[str], str]] = {
    "empty": lambda text: "",
    "echo": lambda text: text,
}


def undetermined(case: dict[str, Any]) -> bool:
    checks = checks_for(case)
    return bool(checks) and all(c["type"] == "llm_judge" for c in checks)


def sanity(cases: list[dict[str, Any]], min_fail_rate: float) -> tuple[bool, dict[str, Any]]:
    determined = [c for c in cases if not undetermined(c)]
    out: dict[str, Any] = {
        "cases": len(cases),
        "undetermined": [c["id"] for c in cases if undetermined(c)],
        "stubs": {},
    }
    ok = len(determined) * 2 >= len(cases) and bool(determined)
    if not ok:
        out["problem"] = "fewer than half of the cases can be judged without a model"
    for name, stub in STUBS.items():
        passed = [c["id"] for c in determined if run_once(c, stub, None, True)[0]]
        fail_rate = 1 - len(passed) / len(determined) if determined else 0.0
        out["stubs"][name] = {"fail_rate": round(fail_rate, 3), "passed_by_stub": passed}
        ok = ok and fail_rate >= min_fail_rate
    return ok, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "evals" / "cases.jsonl"))
    ap.add_argument("--min-fail-rate", type=float, default=0.9)
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    ok, out = sanity(load_cases(Path(a.cases)), a.min_fail_rate)
    for name, res in out["stubs"].items():
        print(f"stub {name}: fails {res['fail_rate']:.0%} of cases (need {a.min_fail_rate:.0%})")
        for cid in res["passed_by_stub"]:
            print(f"  weak case, passed by the {name} stub: {cid}")
    if out["undetermined"]:
        print(f"undetermined (judge-only): {out['undetermined']}")
    if "problem" in out:
        print(f"FAIL: {out['problem']}")
    if a.report:
        Path(a.report).write_text(json.dumps(out, indent=2))
    print("SANITY PASSED" if ok else "SANITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
