"""Step 3 with step 4: what the real tools return is grounded by the validator.

Offline against the committed snapshot (tests/fixtures/), no network.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.edgar_source import EdgarSource, get_source
from app.grounding import REASON_VALUE, RunEvidence, ground_figure
from app.models import Figure, Provenance
from app.tools.compute import compute_ratio
from app.tools.edgar_tools import lookup_fact, search_filing_text
from app.tools.schemas import ComputeRatioInput, LookupFactInput, SearchFilingTextInput

FY = date(2025, 9, 27)
PY = date(2024, 9, 28)
REVENUE = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"


@pytest.fixture(scope="module")
def src() -> EdgarSource:
    return get_source(offline=True)


def _evidence(src: EdgarSource) -> RunEvidence:
    return RunEvidence(
        accession=src.ref.accession,
        section_texts={s: src.section_text(s) for s in src.sections()},
    )


def _lookup(src: EdgarSource, ev: RunEvidence, concept: str, end: date = FY):
    out = lookup_fact(src, LookupFactInput(concept=concept, period_end=end))
    ev.add_facts(out.facts)
    assert len(out.facts) == 1, out
    return out.facts[0]


@pytest.mark.parametrize("kind", ["ratio", "change_pct"])
def test_compute_ratio_output_is_verified_by_the_validator(src: EdgarSource, kind: str) -> None:
    ev = _evidence(src)
    num = _lookup(src, ev, REVENUE if kind == "change_pct" else "us-gaap:GrossProfit")
    den = _lookup(src, ev, REVENUE, PY if kind == "change_pct" else FY)
    fig = compute_ratio(src, ComputeRatioInput(numerator=num, denominator=den, kind=kind))
    assert fig.status == "verified"
    assert ground_figure(fig, ev).status == "verified"
    # The same figure with a nudged value is caught.
    wrong = Figure.model_validate({**fig.model_dump(), "value": fig.value + Decimal("0.01")})
    out = ground_figure(wrong, ev)
    assert out.status == "unverified" and out.reason == REASON_VALUE


def test_lookup_fact_output_grounds_an_xbrl_figure(src: EdgarSource) -> None:
    ev = _evidence(src)
    fact = _lookup(src, ev, REVENUE)
    prov = Provenance(
        kind="xbrl",
        accession=fact.accession,
        statement_or_section="Income statement",
        tag=f"{fact.taxonomy}:{fact.tag}",
        period_start=fact.period_start,
        period_end=fact.period_end,
        dimensions=fact.dimensions,
    )
    fig = Figure(
        label="Total net sales",
        value=fact.value,
        unit="USD",
        status="verified",
        provenance=prov,
        claimed_by="agent",
    )
    assert ground_figure(fig, ev).status == "verified"
    assert fact.value == Decimal("416161000000")


def test_search_filing_text_spans_sit_at_their_offsets(src: EdgarSource) -> None:
    ev = _evidence(src)
    out = search_filing_text(src, SearchFilingTextInput(query="employees"))
    assert out.spans, out
    ev.add_spans(out.spans)
    for span in out.spans:
        prov = Provenance(
            kind="text",
            accession=src.ref.accession,
            statement_or_section=span.section,
            quote=span.quote,
        )
        # A made-up number in a real span is never verified.
        fig = Figure(
            label="x",
            value=Decimal("123456789"),
            unit="employees",
            status="verified",
            provenance=prov,
            claimed_by="agent",
        )
        assert ground_figure(fig, ev).status == "unverified"
        text = ev.section_texts[span.section][span.start : span.end]
        assert " ".join(text.split()) == " ".join(span.quote.split())


@pytest.mark.parametrize(
    ("query", "value", "unit"),
    [
        ("full-time equivalent employees", "166000", "employees"),
        ("shares of common stock were issued and outstanding", "14776353000", "shares"),
    ],
)
def test_narrative_figures_from_the_snapshot_verify(
    src: EdgarSource, query: str, value: str, unit: str
) -> None:
    ev = _evidence(src)
    spans = search_filing_text(src, SearchFilingTextInput(query=query)).spans
    ev.add_spans(spans)
    span = next(s for s in spans if value[:3] in s.quote.replace(",", ""))
    prov = Provenance(
        kind="text",
        accession=src.ref.accession,
        statement_or_section=span.section,
        quote=span.quote,
    )
    fig = Figure(
        label=query,
        value=Decimal(value),
        unit=unit,
        status="verified",
        provenance=prov,
        claimed_by="agent",
    )
    assert ground_figure(fig, ev).status == "verified"
    # The date in the same quote is not a figure.
    for bad in ("27", "2025", "17"):
        wrong = Figure.model_validate({**fig.model_dump(), "value": Decimal(bad)})
        assert ground_figure(wrong, ev).status == "unverified"
