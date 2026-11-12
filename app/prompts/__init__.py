"""Prompts of the agent: system instructions and tool descriptions, kept out of the code.

`PROMPT_VERSION` is a digest of both files. It goes into the audit record and into the
replay key, so a changed prompt never replays a transcript recorded for the old one
([[pattern-replay-evals-vs-live-runs]] rule 6).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.models import FilingRef

_DIR = Path(__file__).resolve().parent
_INSTRUCTIONS = (_DIR / "instructions.md").read_text(encoding="utf-8")
_TOOLS_RAW = (_DIR / "tools.json").read_text(encoding="utf-8")

TOOL_DESCRIPTIONS: dict[str, str] = json.loads(_TOOLS_RAW)
_DIGEST = hashlib.sha256((_INSTRUCTIONS + "\n" + _TOOLS_RAW).encode()).hexdigest()
PROMPT_VERSION = "p-" + _DIGEST[:12]

_FIELDS = ("form", "company", "cik", "accession", "period_end")


def instructions(ref: FilingRef) -> str:
    """The system instructions for the one filing, from `instructions.md`."""
    text = _INSTRUCTIONS
    values = {
        "form": ref.form,
        "company": ref.company,
        "cik": ref.cik,
        "accession": ref.accession,
        "period_end": ref.period_end.isoformat(),
    }
    for name in _FIELDS:
        text = text.replace(f"<<{name}>>", values[name])
    if "<<" in text:
        raise ValueError("instructions.md has an unknown <<placeholder>>")
    return text


__all__ = ["PROMPT_VERSION", "TOOL_DESCRIPTIONS", "instructions"]
