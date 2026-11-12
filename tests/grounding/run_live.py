"""One live run of the 20 labelled questions (plan step 10); writes the answers file for score.py.

Usage (needs COGNITIA_LIVE=1, COGNITIA_MODEL, COGNITIA_OPENROUTER_API_KEY, EDGAR_IDENTITY):
  COGNITIA_LIVE=1 COGNITIA_RECORD=1 BUILD_STEP=10 python tests/grounding/run_live.py [--out PATH]

Each question goes through `app.agent.run_agent`; with COGNITIA_RECORD=1 each run is also saved as a
replay fixture. Spend goes to ops/spend.jsonl through `evals.spend`. Then run score.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.agent import run_agent  # noqa: E402
from evals import spend  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "fixtures" / "replay" / "grounding_run.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(HERE / "labels.jsonl"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--ledger", default=str(ROOT / "ops" / "spend.jsonl"))
    args = parser.parse_args()
    if os.environ.get("COGNITIA_LIVE") != "1":
        print("refusing to run: set COGNITIA_LIVE=1 (this is the fresh live run)")
        return 2
    labels = [json.loads(x) for x in Path(args.labels).read_text().splitlines() if x.strip()]
    spend.METER.reset()
    lines = []
    for row in labels:
        t0 = time.time()
        result = run_agent(row["question"])
        spend.add_agent(result.usage)
        lines.append(json.dumps({"id": row["id"], "bundle": json.loads(result.output)}))
        print(f"{row['id']} {time.time() - t0:.1f}s")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(spend.append_ledger(Path(args.ledger), os.environ.get("BUILD_STEP", "")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
