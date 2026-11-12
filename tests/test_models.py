from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.models import (
    AnswerBundle,
    AuditRecord,
    Fact,
    Figure,
    FilingRef,
    Provenance,
    TextSpan,
    ToolCallRecord,
)

ACCESSION = "0000000000-00-000001"


def _xbrl_prov() -> Provenance:
    return Provenance(
        kind="xbrl",
        accession=ACCESSION,
        statement_or_section="Income statement",
        tag="us-gaap:Revenues",
        period_start=date(2024, 9, 29),
        period_end=date(2025, 9, 27),
    )


def _fact() -> Fact:
    return Fact(
        tag="us-gaap:Revenues",
        taxonomy="us-gaap",
        value=Decimal("416161000000"),
        unit="USD",
        period_start=date(2024, 9, 29),
        period_end=date(2025, 9, 27),
        dimensions={},
        accession=ACCESSION,
    )


def test_filing_ref_valid() -> None:
    ref = FilingRef(
        accession=ACCESSION,
        form="10-K",
        period_end=date(2025, 9, 27),
        company="Example Co",
        cik="0000000001",
    )
    assert ref.form == "10-K"


def test_fact_keeps_decimal_value_exact() -> None:
    fact = _fact()
    assert fact.value == Decimal("416161000000")
    assert isinstance(fact.value, Decimal)


def test_fact_rejects_reversed_period() -> None:
    with pytest.raises(ValidationError):
        Fact(
            tag="us-gaap:Revenues",
            taxonomy="us-gaap",
            value=Decimal(1),
            unit="USD",
            period_start=date(2025, 9, 27),
            period_end=date(2024, 9, 29),
            accession=ACCESSION,
        )


def test_text_span_requires_end_after_start() -> None:
    TextSpan(section="Item 1", quote="Net sales were up.", start=10, end=28)
    with pytest.raises(ValidationError):
        TextSpan(section="Item 1", quote="Net sales were up.", start=28, end=10)


def test_text_provenance_needs_quote() -> None:
    with pytest.raises(ValidationError):
        Provenance(kind="text", accession=ACCESSION, statement_or_section="Item 1")


def test_derived_provenance_needs_formula_and_inputs() -> None:
    with pytest.raises(ValidationError):
        Provenance(kind="derived", accession=ACCESSION, statement_or_section="ratio")
    derived = Provenance(
        kind="derived",
        accession=ACCESSION,
        statement_or_section="ratio",
        formula="a / b",
        inputs=[_xbrl_prov(), _xbrl_prov()],
    )
    assert len(derived.inputs) == 2


def test_verified_figure_with_provenance() -> None:
    figure = Figure(
        label="Total net sales",
        value=Decimal("416161000000"),
        unit="USD",
        status="verified",
        provenance=_xbrl_prov(),
        claimed_by="agent",
    )
    assert figure.status == "verified"


def test_verified_figure_without_provenance_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Figure(
            label="Total net sales",
            value=Decimal("416161000000"),
            unit="USD",
            status="verified",
            claimed_by="agent",
        )


def test_unverified_figure_needs_reason_and_no_provenance() -> None:
    figure = Figure(
        label="Employees",
        value=Decimal("999"),
        unit="count",
        status="unverified",
        reason="no matching fact",
        claimed_by="agent",
    )
    assert figure.provenance is None
    with pytest.raises(ValidationError):
        Figure(
            label="Employees",
            value=Decimal("999"),
            unit="count",
            status="unverified",
            claimed_by="agent",
        )
    with pytest.raises(ValidationError):
        Figure(
            label="Employees",
            value=Decimal("999"),
            unit="count",
            status="unverified",
            reason="no matching fact",
            provenance=_xbrl_prov(),
            claimed_by="agent",
        )


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AnswerBundle.model_validate({"answer": "x", "figures": [], "notes": [], "status": "ok"})


def test_answer_bundle_round_trips_through_json() -> None:
    bundle = AnswerBundle(
        answer="Total net sales were $416.2 billion.",
        figures=[
            Figure(
                label="Total net sales",
                value=Decimal("416161000000"),
                unit="USD",
                status="verified",
                provenance=_xbrl_prov(),
                claimed_by="agent",
            ),
            Figure(
                label="Employees",
                value=Decimal("1"),
                unit="count",
                status="unverified",
                reason="quote not found",
                claimed_by="user",
            ),
        ],
        notes=["Figures from the FY2025 10-K."],
    )
    restored = AnswerBundle.model_validate_json(bundle.model_dump_json())
    assert restored == bundle
    assert restored.figures[0].value == Decimal("416161000000")


def test_audit_record_holds_metadata_only() -> None:
    record = AuditRecord(
        request_id="req-fixed-0001",
        filing_accession=ACCESSION,
        model="model-id-unknown",
        prompt_version="v1",
        tool_calls=[ToolCallRecord(name="lookup_fact", arg_digest="sha256:abc")],
        figure_statuses=["verified", "unverified"],
    )
    dumped = record.model_dump_json()
    assert "question" not in dumped
    assert AuditRecord.model_validate_json(dumped) == record
