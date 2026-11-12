"""Typed state for the grounded 10-K agent (spec.md "State schema").

Every type here is built by code or validated by code. The model never constructs
a `Fact`, `TextSpan` or `Provenance`; it only receives them from tools.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_STRICT = ConfigDict(extra="forbid", frozen=True)


class FilingRef(BaseModel):
    """The one filing the deployment answers about."""

    model_config = _STRICT

    accession: str = Field(min_length=1)
    form: str = Field(min_length=1)
    period_end: date
    company: str = Field(min_length=1)
    cik: str = Field(min_length=1)


class Fact(BaseModel):
    """One XBRL fact from the filing, as EdgarTools returns it."""

    model_config = _STRICT

    tag: str = Field(min_length=1)
    taxonomy: str = Field(min_length=1)
    value: Decimal
    unit: str = Field(min_length=1)
    period_start: date | None = None
    period_end: date
    dimensions: dict[str, str] = Field(default_factory=dict)
    accession: str = Field(min_length=1)

    @model_validator(mode="after")
    def _period_order(self) -> Fact:
        if self.period_start is not None and self.period_start > self.period_end:
            raise ValueError("period_start is after period_end")
        return self


class TextSpan(BaseModel):
    """A quoted passage from the filing text, with its offsets in that text."""

    model_config = _STRICT

    section: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _offsets(self) -> TextSpan:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class Provenance(BaseModel):
    """Where a figure comes from. `derived` carries the formula and every input."""

    model_config = _STRICT

    kind: Literal["xbrl", "text", "derived"]
    accession: str = Field(min_length=1)
    statement_or_section: str = Field(min_length=1)
    tag: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    dimensions: dict[str, str] = Field(default_factory=dict)
    quote: str | None = None
    inputs: list[Provenance] = Field(default_factory=list)
    formula: str | None = None

    @model_validator(mode="after")
    def _kind_requirements(self) -> Provenance:
        if self.kind == "xbrl":
            if not self.tag or self.period_end is None:
                raise ValueError("xbrl provenance needs tag and period_end")
        elif self.kind == "text":
            if not self.quote:
                raise ValueError("text provenance needs quote")
        else:  # derived
            if not self.formula or not self.inputs:
                raise ValueError("derived provenance needs formula and inputs")
        return self


def _figure_schema(schema: dict[str, Any]) -> None:
    """Show `provenance` as a required key in the schema the model sees.

    Parsing still defaults it to None; a weak model that is told a key is optional tends to drop
    it from a verified figure, which the validator below then rejects (step 9 live runs).
    """
    required = schema.setdefault("required", [])
    if "provenance" not in required:
        required.append("provenance")


class Figure(BaseModel):
    """One number the agent states. `unverified` has a reason and no provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_extra=_figure_schema)

    label: str = Field(min_length=1)
    value: Decimal
    unit: str = Field(min_length=1)
    status: Literal["verified", "unverified"]
    provenance: Provenance | None = None
    reason: str = ""
    claimed_by: Literal["agent", "user"]

    @model_validator(mode="after")
    def _status_requirements(self) -> Figure:
        if self.status == "unverified":
            if self.provenance is not None:
                raise ValueError("unverified figure must not carry provenance")
            if not self.reason.strip():
                raise ValueError("unverified figure needs a reason")
        else:
            if self.provenance is None:
                raise ValueError("verified figure needs provenance")
        return self


class AnswerBundle(BaseModel):
    """Final output of the agent (the Pydantic AI output type)."""

    model_config = _STRICT

    answer: str
    figures: list[Figure] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    """Name and digest of one tool call. Arguments themselves are not logged."""

    model_config = _STRICT

    name: str = Field(min_length=1)
    arg_digest: str = Field(min_length=1)


class AuditRecord(BaseModel):
    """Metadata only, written to ops/trace.jsonl. Never holds question text."""

    model_config = _STRICT

    request_id: str = Field(min_length=1)
    filing_accession: str = Field(min_length=1)
    model: str
    prompt_version: str = Field(min_length=1)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    figure_statuses: list[Literal["verified", "unverified"]] = Field(default_factory=list)


Provenance.model_rebuild()
