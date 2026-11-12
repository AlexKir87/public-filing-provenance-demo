"""Typed inputs and outputs of the four read-only tools (spec.md "Tools").

The model sends the inputs; code builds every output from the filing. Limits are
enforced here so no tool can return a page of data larger than the spec allows.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.edgar_source import Statement, StatementLine
from app.models import Fact, Figure, Provenance, TextSpan

MAX_FACTS = 10
MAX_SPANS = 5

_IN = ConfigDict(extra="forbid")
_OUT = ConfigDict(extra="forbid", frozen=True)


class LookupFactInput(BaseModel):
    model_config = _IN

    concept: str = Field(min_length=1, description="XBRL tag, e.g. 'us-gaap:GrossProfit'")
    period_end: date | None = Field(default=None, description="Period end date, ISO")
    dimensions: dict[str, str] | None = Field(
        default=None,
        description="Exact dimension members, e.g. {'srt:ProductOrServiceAxis': 'iPhone'}; "
        "omit for the filing-wide (undimensioned) value",
    )


class LookupFactOutput(BaseModel):
    model_config = _OUT

    facts: list[Fact] = Field(max_length=MAX_FACTS)
    total: int = Field(ge=0, description="matches before the cap")
    truncated: bool
    note: str = ""
    provenance: list[Provenance] = Field(
        default_factory=list,
        max_length=MAX_FACTS,
        description="provenance[i] belongs to facts[i]: copy it unchanged into a figure",
    )


class ListStatementLinesInput(BaseModel):
    model_config = _IN

    statement: Statement
    period_end: date | None = None
    offset: int = Field(default=0, ge=0)


class ListStatementLinesOutput(BaseModel):
    model_config = _OUT

    lines: list[StatementLine] = Field(max_length=MAX_FACTS)
    total: int = Field(ge=0)
    next_offset: int | None = Field(default=None, description="pass as offset for more lines")


class SearchFilingTextInput(BaseModel):
    model_config = _IN

    query: str = Field(min_length=1)
    section: str | None = Field(default=None, description="e.g. 'Item 1'; omit to search all")


class SearchFilingTextOutput(BaseModel):
    model_config = _OUT

    spans: list[TextSpan] = Field(max_length=MAX_SPANS)
    note: str = ""
    provenance: list[Provenance] = Field(
        default_factory=list,
        max_length=MAX_SPANS,
        description="provenance[i] belongs to spans[i]: copy it unchanged into a figure",
    )


class ComputeRatioInput(BaseModel):
    model_config = _IN

    numerator: Fact
    denominator: Fact
    kind: Literal["ratio", "change_pct"] = Field(
        description="ratio = numerator / denominator; change_pct = numerator is the current "
        "value and denominator the prior value"
    )


# compute_ratio returns a `Figure` (derived provenance, or unverified with a reason).
__all__ = [
    "MAX_FACTS",
    "MAX_SPANS",
    "ComputeRatioInput",
    "Figure",
    "ListStatementLinesInput",
    "ListStatementLinesOutput",
    "LookupFactInput",
    "LookupFactOutput",
    "SearchFilingTextInput",
    "SearchFilingTextOutput",
]
