"""Step 4: the four read-only tools, offline against the committed snapshot."""

from __future__ import annotations

import inspect
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

import app.tools as tools
from app.edgar_source import EdgarSource, get_source
from app.models import Fact, TextSpan
from app.tools import (
    ComputeRatioInput,
    ListStatementLinesInput,
    LookupFactInput,
    SearchFilingTextInput,
    compute_ratio,
    list_statement_lines,
    lookup_fact,
    search_filing_text,
)
from app.tools.schemas import MAX_FACTS, MAX_SPANS

FY = date(2025, 9, 27)
REVENUE = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
IPHONE = {"srt:ProductOrServiceAxis": "aapl:IPhoneMember"}


@pytest.fixture(scope="module")
def src() -> EdgarSource:
    return get_source(offline=True)


def _fact(src: EdgarSource, concept: str, dims: dict[str, str] | None = None, end: date = FY):
    out = lookup_fact(src, LookupFactInput(concept=concept, period_end=end, dimensions=dims))
    assert len(out.facts) == 1, out
    return out.facts[0]


def test_lookup_fact_total_net_sales(src):
    out = lookup_fact(src, LookupFactInput(concept=REVENUE, period_end=FY))
    assert [f.value for f in out.facts] == [Decimal(416161000000)]
    assert out.facts[0].dimensions == {}
    assert out.facts[0].accession == src.ref.accession
    assert not out.truncated


def test_lookup_fact_dimensional_does_not_leak_into_total(src):
    total = _fact(src, REVENUE)
    iphone = _fact(src, REVENUE, IPHONE)
    assert iphone.value == Decimal(209586000000)
    assert iphone.dimensions == IPHONE
    assert total.value != iphone.value


def test_lookup_fact_unknown_tag_says_so(src):
    out = lookup_fact(src, LookupFactInput(concept="us-gaap:NoSuchTag"))
    assert out.facts == [] and out.total == 0 and "no matching fact" in out.note


def test_lookup_fact_capped_at_ten_latest_first(src):
    # No period filter: every period of a tag. Use the tag with the most periods.
    best = max(
        {f.tag for f in src.facts(any_dimensions=True)},
        key=lambda t: len(src.facts(t, any_dimensions=True, dimensions=None)),
    )
    out = lookup_fact(src, LookupFactInput(concept=best))
    assert len(out.facts) <= MAX_FACTS
    ends = [f.period_end for f in out.facts]
    assert ends == sorted(ends, reverse=True)


def test_list_statement_lines_pages_ten_at_a_time(src):
    first = list_statement_lines(src, ListStatementLinesInput(statement="income"))
    assert len(first.lines) == MAX_FACTS and first.total > MAX_FACTS
    assert first.next_offset == MAX_FACTS
    seen = list(first.lines)
    nxt = first.next_offset
    while nxt is not None:
        page = list_statement_lines(src, ListStatementLinesInput(statement="income", offset=nxt))
        assert len(page.lines) <= MAX_FACTS
        seen += page.lines
        nxt = page.next_offset
    assert len(seen) == first.total
    assert all(x.statement == "income" for x in seen)


def test_list_statement_lines_period_filter(src):
    out = list_statement_lines(src, ListStatementLinesInput(statement="balance", period_end=FY))
    assert out.lines and all(x.period_end == FY for x in out.lines)
    assert out.total >= len(out.lines)


def test_list_statement_lines_offset_past_end(src):
    out = list_statement_lines(src, ListStatementLinesInput(statement="cashflow", offset=10_000))
    assert out.lines == [] and out.next_offset is None


def test_search_filing_text_returns_verifiable_spans(src):
    out = search_filing_text(
        src, SearchFilingTextInput(query="full-time equivalent employees", section="Item 1")
    )
    assert 1 <= len(out.spans) <= MAX_SPANS
    assert "166,000" in out.spans[0].quote
    assert all(src.verify_span(s) for s in out.spans)


def test_search_filing_text_unknown_section_and_no_hit(src):
    bad = search_filing_text(src, SearchFilingTextInput(query="employees", section="Item 99"))
    assert bad.spans == [] and "unknown section" in bad.note
    none = search_filing_text(src, SearchFilingTextInput(query="zzzqqq nonexistentword"))
    assert none.spans == [] and none.note


def test_search_filing_text_capped_at_five(src):
    out = search_filing_text(src, SearchFilingTextInput(query="the"))
    assert len(out.spans) <= MAX_SPANS


def test_compute_ratio_gross_margin_with_both_inputs(src):
    gross = _fact(src, "us-gaap:GrossProfit")
    revenue = _fact(src, REVENUE)
    fig = compute_ratio(src, ComputeRatioInput(numerator=gross, denominator=revenue, kind="ratio"))
    assert fig.status == "verified" and fig.claimed_by == "agent"
    assert abs(fig.value - Decimal("0.46905")) < Decimal("0.0005")
    assert fig.value == (gross.value / revenue.value).quantize(Decimal("0.000001"))
    prov = fig.provenance
    assert prov is not None and prov.kind == "derived" and prov.formula
    # bare tags, like every other provenance (the eval cases expect `GrossProfit`)
    assert [i.tag for i in prov.inputs] == ["GrossProfit", REVENUE.split(":")[-1]]
    assert all(i.accession == src.ref.accession and i.period_end == FY for i in prov.inputs)


def test_compute_ratio_change_pct_uses_prior_as_base(src):
    cur = _fact(src, REVENUE)
    prior = _fact(src, REVENUE, end=date(2024, 9, 28))
    fig = compute_ratio(src, ComputeRatioInput(numerator=cur, denominator=prior, kind="change_pct"))
    expected = ((cur.value - prior.value) / prior.value * 100).quantize(Decimal("0.000001"))
    assert fig.status == "verified" and fig.unit == "percent" and fig.value == expected


def test_compute_ratio_dimensional_input(src):
    fig = compute_ratio(
        src,
        ComputeRatioInput(
            numerator=_fact(src, REVENUE, IPHONE),
            denominator=_fact(src, REVENUE),
            kind="ratio",
        ),
    )
    assert fig.status == "verified"
    assert fig.provenance is not None
    assert fig.provenance.inputs[0].dimensions == IPHONE
    assert Decimal("0.5") < fig.value < Decimal("0.52")


def test_compute_ratio_fabricated_input_is_unverified(src):
    real = _fact(src, REVENUE)
    forged = real.model_copy(update={"value": Decimal(450_000_000_000)})
    fig = compute_ratio(src, ComputeRatioInput(numerator=forged, denominator=real, kind="ratio"))
    assert fig.status == "unverified" and fig.provenance is None
    assert "not found" in fig.reason


def test_compute_ratio_unit_mismatch_and_zero_denominator(src):
    revenue = _fact(src, REVENUE)
    eps = _fact(src, "us-gaap:EarningsPerShareBasic")
    mixed = compute_ratio(src, ComputeRatioInput(numerator=revenue, denominator=eps, kind="ratio"))
    assert mixed.status == "unverified" and "units" in mixed.reason

    zero = Fact(**{**revenue.model_dump(), "value": Decimal(0)})
    # a zero fact that is not in the filing is rejected before arithmetic
    out = compute_ratio(src, ComputeRatioInput(numerator=revenue, denominator=zero, kind="ratio"))
    assert out.status == "unverified"


def test_compute_ratio_zero_denominator_from_filing():
    zero = Fact(
        tag="Z", taxonomy="us-gaap", value=Decimal(0), unit="USD", period_end=FY, accession="a-1"
    )
    one = zero.model_copy(update={"tag": "O", "value": Decimal(1)})

    class _Stub:
        class ref:  # noqa: N801
            accession = "a-1"

        @staticmethod
        def facts(concept, period_end, dimensions, any_dimensions=False):
            tag = concept.split(":")[1]
            return [f for f in (zero, one) if f.tag == tag]

    out = compute_ratio(_Stub(), ComputeRatioInput(numerator=one, denominator=zero, kind="ratio"))
    assert out.status == "unverified" and out.reason == "denominator is zero"


def test_inputs_reject_unknown_fields_and_bad_values():
    with pytest.raises(ValidationError):
        LookupFactInput(concept="us-gaap:GrossProfit", period="2025")
    with pytest.raises(ValidationError):
        ListStatementLinesInput(statement="equity")
    with pytest.raises(ValidationError):
        ListStatementLinesInput(statement="income", offset=-1)
    with pytest.raises(ValidationError):
        SearchFilingTextInput(query="")


def test_tools_are_read_only():
    """No tool takes anything but a source and a typed input, and none can write or send."""
    for name in tools.TOOL_NAMES:
        fn = getattr(tools, name)
        assert list(inspect.signature(fn).parameters) == ["source", "args"]
    import app.tools.compute as c
    import app.tools.edgar_tools as e

    for mod in (c, e):
        text = inspect.getsource(mod)
        for banned in (
            "open(",
            "write_text",
            "requests",
            "httpx",
            "urllib",
            "subprocess",
            "os.environ",
        ):
            assert banned not in text, (mod.__name__, banned)


def _stub_source(facts):
    class _Stub:
        class ref:  # noqa: N801
            accession = "a-1"

        @staticmethod
        def facts(concept, period_end, dimensions, any_dimensions=False):
            return list(facts)

        @staticmethod
        def sections():
            return ["Item 1"]

        @staticmethod
        def search_text(query, section=None, limit=5):
            return [TextSpan(section="Item 1", quote="x", start=i, end=i + 1) for i in range(20)][
                :limit
            ]

    return _Stub()


def test_lookup_fact_truncates_over_ten():
    facts = [
        Fact(
            tag="T",
            taxonomy="us-gaap",
            value=Decimal(i),
            unit="USD",
            period_end=date(2000 + i, 12, 31),
            accession="a-1",
        )  # fmt: skip
        for i in range(12)
    ]
    out = lookup_fact(_stub_source(facts), LookupFactInput(concept="us-gaap:T"))
    assert len(out.facts) == MAX_FACTS and out.total == 12 and out.truncated
    assert out.facts[0].period_end == date(2011, 12, 31)


def test_search_filing_text_passes_cap_of_five_to_source():
    out = search_filing_text(_stub_source([]), SearchFilingTextInput(query="x"))
    assert len(out.spans) == MAX_SPANS


def test_compute_ratio_negative_prior_is_unverified(src):
    cur = _fact(src, REVENUE)
    neg = cur.model_copy(update={"value": Decimal(-1)})

    class _S:
        ref = src.ref

        @staticmethod
        def facts(*a, **k):
            return [cur, neg]

    out = compute_ratio(_S(), ComputeRatioInput(numerator=cur, denominator=neg, kind="change_pct"))
    assert out.status == "unverified" and "negative" in out.reason


def test_lookup_and_search_return_aligned_provenance(src: EdgarSource) -> None:
    out = lookup_fact(src, LookupFactInput(concept="us-gaap:GrossProfit", period_end=FY))
    assert out.facts and len(out.provenance) == len(out.facts)
    for f, p in zip(out.facts, out.provenance, strict=True):
        assert p.kind == "xbrl" and p.tag == f.tag and p.period_end == f.period_end
    spans = search_filing_text(src, SearchFilingTextInput(query="full-time equivalent employees"))
    assert spans.spans and len(spans.provenance) == len(spans.spans)
    assert all(
        p.kind == "text" and p.quote == s.quote
        for p, s in zip(spans.provenance, spans.spans, strict=True)
    )


def test_figure_schema_requires_provenance_but_parsing_still_defaults() -> None:
    from app.models import Figure

    assert "provenance" in Figure.model_json_schema()["required"]
    fig = Figure(
        label="x", value=1, unit="USD", status="unverified", reason="r", claimed_by="agent"
    )
    assert fig.provenance is None
