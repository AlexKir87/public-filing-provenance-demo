"""Score an agent run on the 20 labelled figures (spec "Accuracy target and measurement").

Usage: python tests/grounding/score.py [--labels PATH] [--answers PATH] [--min-correct N]

`--answers` is a JSONL file with one line per label id: {"id": "g-stm-01", "bundle": {...}}
where `bundle` is the agent's `AnswerBundle` JSON (`app/models.py`). The default is
`tests/fixtures/replay/grounding_run.jsonl`, written by step 10.

Rules, per label:
- Expected `verified` (statement, dimensional, narrative): correct when the answer holds a figure
  with the expected value, status `verified`, and provenance equal to the label's (xbrl: tag,
  period start and end, dimensions, accession; text: accession and the expected quote inside the
  figure's quote, whitespace-normalised).
- Expected `unverified` (planted wrong claim): correct when the claimed number appears as a figure
  with status `unverified` and no figure with that number is `verified`.
- False verified: any figure with status `verified` that is not the filing's value with the
  label's provenance (for a planted claim, the filing's true value). Conservative on purpose:
  an extra verified figure the question did not ask for also counts, because the scorer has no
  filing facts of its own. A label with no figure at all is incorrect, not false verified.
The exit code is 1 when correct < `--min-correct` or false verified > 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.models import AnswerBundle  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_LABELS = HERE / "labels.jsonl"
DEFAULT_ANSWERS = HERE.parent / "fixtures" / "replay" / "grounding_run.jsonl"
_TOL = Decimal("0.0005")


@dataclass
class Score:
    total: int = 0
    correct: int = 0
    false_verified: int = 0
    correct_ids: list[str] = field(default_factory=list)
    wrong_ids: list[str] = field(default_factory=list)
    false_verified_ids: list[str] = field(default_factory=list)


def load_labels(path: Path = DEFAULT_LABELS) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def load_answers(path: Path = DEFAULT_ANSWERS) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = AnswerBundle.model_validate(row["bundle"]).model_dump(mode="json")
    return out


def _same_value(a: Any, b: Any) -> bool:
    return abs(Decimal(str(a)) - Decimal(str(b))) <= _TOL


def _norm(text: str) -> str:
    return " ".join(text.split())


def _prov_matches(label: dict[str, Any], prov: dict[str, Any] | None) -> bool:
    if not prov:
        return False
    want = label["provenance"]
    if prov.get("kind") != want["kind"] or prov.get("accession") != want["accession"]:
        return False
    if want["kind"] == "text":
        return _norm(want["quote"]) in _norm(prov.get("quote") or "")
    return (
        prov.get("tag") == want["tag"]
        and prov.get("period_start") == want["period_start"]
        and prov.get("period_end") == want["period_end"]
        and (prov.get("dimensions") or {}) == want["dimensions"]
    )


def _exact(label: dict[str, Any], fig: dict[str, Any]) -> bool:
    """Verified figure with the filing's value and the label's provenance."""
    return (
        fig["status"] == "verified"
        and _same_value(fig["value"], label["true_value" if "true_value" in label else "value"])
        and _prov_matches(label, fig.get("provenance"))
    )


def grade_label(label: dict[str, Any], bundle: dict[str, Any] | None) -> tuple[bool, bool]:
    """Return (correct, false_verified) for one label.

    False verified is judged per figure, not per label: every `verified` figure in the answer
    must be the filing's value with the label's provenance, otherwise it counts. A wrong value
    cannot escape by also getting its tag, period, dimensions or quote wrong.
    """
    figures = (bundle or {}).get("figures", [])
    false_verified = any(f["status"] == "verified" and not _exact(label, f) for f in figures)
    if label["expected_status"] == "unverified":
        shown = any(
            f["status"] == "unverified" and _same_value(f["value"], label["value"]) for f in figures
        )
        return (shown and not false_verified), false_verified
    correct = any(_exact(label, f) for f in figures)
    return (correct and not false_verified), false_verified


def score(labels: list[dict[str, Any]], answers: dict[str, dict[str, Any]]) -> Score:
    result = Score(total=len(labels))
    for label in labels:
        ok, fv = grade_label(label, answers.get(label["id"]))
        (result.correct_ids if ok else result.wrong_ids).append(label["id"])
        if fv:
            result.false_verified_ids.append(label["id"])
    result.correct = len(result.correct_ids)
    result.false_verified = len(result.false_verified_ids)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    ap.add_argument(
        "--min-correct", type=int, default=None, help="default: 90%% of labels, rounded up"
    )
    args = ap.parse_args(argv)
    if not args.answers.exists():
        print(f"no answers file at {args.answers}; run the live pass first", file=sys.stderr)
        return 2
    labels = load_labels(args.labels)
    min_correct = args.min_correct if args.min_correct is not None else -(-len(labels) * 9 // 10)
    res = score(labels, load_answers(args.answers))
    print(f"correct: {res.correct} of {res.total}")
    print(f"false verified: {res.false_verified}")
    if res.wrong_ids:
        print("incorrect: " + ", ".join(res.wrong_ids))
    if res.false_verified_ids:
        print("FALSE VERIFIED: " + ", ".join(res.false_verified_ids))
    passed = res.correct >= min_correct and res.false_verified == 0
    print("PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
