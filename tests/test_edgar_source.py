"""Step 2: EDGAR source, offline against the committed snapshot.

The snapshot is one recorded fetch of the Apple FY2025 10-K (public SEC data). Network
tests use an injected fetcher, so nothing here touches SEC.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from app import edgar_source as es
from app.edgar_source import (
    DEFAULT_ACCESSION,
    DEFAULT_SNAPSHOT_DIR,
    EdgarSource,
    MissingIdentityError,
    SourceUnavailable,
    get_source,
    load_filing_data,
    save_filing_data,
)
from app.models import TextSpan

FY_END = date(2025, 9, 27)
NET_SALES = "RevenueFromContractWithCustomerExcludingAssessedTax"


@pytest.fixture(scope="module")
def src() -> EdgarSource:
    return get_source(offline=True)


def test_snapshot_identifies_the_fixed_filing(src: EdgarSource) -> None:
    ref = src.ref
    assert ref.accession == DEFAULT_ACCESSION == "0000320193-25-000079"
    assert (ref.form, ref.period_end, ref.company) == ("10-K", FY_END, "Apple Inc.")
    assert src.origin == "snapshot"


def test_total_net_sales_matches_the_recorded_fetch(src: EdgarSource) -> None:
    (fact,) = src.facts(NET_SALES, FY_END)
    assert fact.value == Decimal("416161000000")
    assert (fact.unit, fact.period_start, fact.dimensions) == ("USD", date(2024, 9, 29), {})
    assert fact.accession == DEFAULT_ACCESSION and fact.taxonomy == "us-gaap"


@pytest.mark.parametrize("concept", [f"us-gaap:{NET_SALES}", f"us-gaap_{NET_SALES}", NET_SALES])
def test_concept_spellings_are_equivalent(src: EdgarSource, concept: str) -> None:
    assert [f.value for f in src.facts(concept, FY_END)] == [Decimal("416161000000")]


def test_wrong_taxonomy_matches_nothing(src: EdgarSource) -> None:
    assert src.facts(f"dei:{NET_SALES}", FY_END) == []


def test_dimensioned_fact_is_not_returned_for_a_total(src: EdgarSource) -> None:
    totals = src.facts(NET_SALES)
    assert totals and all(f.dimensions == {} for f in totals)


def test_dimensional_fact_by_exact_dimensions(src: EdgarSource) -> None:
    dims = {"srt:ProductOrServiceAxis": "aapl:IPhoneMember"}
    (fact,) = src.facts(NET_SALES, FY_END, dims)
    assert fact.value == Decimal("209586000000") and fact.dimensions == dims


def test_any_dimensions_returns_the_whole_set(src: EdgarSource) -> None:
    every = src.facts(NET_SALES, FY_END, any_dimensions=True)
    assert len(every) > 1 and any(f.dimensions for f in every)


def test_cover_fact_has_no_period_start(src: EdgarSource) -> None:
    (fact,) = src.facts("dei:EntityCommonStockSharesOutstanding", any_dimensions=True)
    assert fact.value == Decimal("14776353000") and fact.period_start is None


def test_facts_are_numeric_with_units_only(src: EdgarSource) -> None:
    allf = src.facts(any_dimensions=True)
    assert allf and all(f.unit and f.unit != "nan" for f in allf)
    assert src.facts("dei:DocumentFiscalYearFocus", any_dimensions=True) == []


def test_statement_lines(src: EdgarSource) -> None:
    income = src.statement_lines("income")
    assert any(
        ln.tag == f"us-gaap:{NET_SALES}" and ln.value == Decimal("416161000000") for ln in income
    )
    assert all(ln.statement == "income" for ln in income)
    for name in ("balance", "cashflow"):
        assert src.statement_lines(name)  # type: ignore[arg-type]
    assert len(src.statement_lines("income", include_dimensioned=True)) > len(income)


def test_text_search_finds_employees_with_verified_offsets(src: EdgarSource) -> None:
    spans = src.search_text("full-time equivalent employees")
    assert spans and all(src.verify_span(s) for s in spans)
    top = spans[0]
    assert top.section == "Item 1" and "166,000" in top.quote
    assert src.section_text(top.section)[top.start : top.end] == top.quote


def test_text_search_cover_shares_and_section_filter(src: EdgarSource) -> None:
    spans = src.search_text("shares of common stock were issued and outstanding", "Cover")
    assert spans and "14,776,353,000" in spans[0].quote
    assert src.search_text("shares of common stock were issued and outstanding", "Item 2") == []


def test_text_search_limit_and_empty(src: EdgarSource) -> None:
    assert len(src.search_text("Apple", limit=2)) == 2
    assert src.search_text("") == [] and src.search_text("the of") == []


def test_unknown_section_raises(src: EdgarSource) -> None:
    with pytest.raises(KeyError):
        src.section_text("Item 99")
    with pytest.raises(KeyError):
        src.search_text("Apple", "Item 99")


def test_verify_span_rejects_altered_quote_and_offsets(src: EdgarSource) -> None:
    good = src.search_text("full-time equivalent employees")[0]
    assert not src.verify_span(good.model_copy(update={"quote": good.quote.replace("166", "167")}))
    assert not src.verify_span(
        good.model_copy(update={"start": good.start + 1, "end": good.end + 1})
    )
    assert not src.verify_span(TextSpan(section="Nope", quote="x", start=0, end=1))


def test_disk_round_trip_is_lossless(src: EdgarSource, tmp_path: Path) -> None:
    save_filing_data(src._data, tmp_path)  # noqa: SLF001 - format test
    assert load_filing_data(tmp_path) == src._data  # noqa: SLF001


# ----------------------------------------------------------------------- modes and failures


def _snapshot_fetch(accession: str):  # type: ignore[no-untyped-def]
    return load_filing_data(DEFAULT_SNAPSHOT_DIR)


def test_offline_ignores_cache_and_network(tmp_path: Path) -> None:
    def boom(_: str):  # type: ignore[no-untyped-def]
        raise AssertionError("network used")

    s = get_source(offline=True, fetch=boom, cache_dir=tmp_path)
    assert s.origin == "snapshot"


def test_offline_env_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(es.ENV_OFFLINE, "1")
    assert get_source().origin == "snapshot"


def test_offline_without_matching_snapshot_is_unavailable() -> None:
    with pytest.raises(SourceUnavailable):
        get_source("0000000000-00-000000", offline=True)


def test_live_fetch_then_cache(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(acc: str):  # type: ignore[no-untyped-def]
        calls.append(acc)
        return _snapshot_fetch(acc)

    first = get_source(offline=False, fetch=fetch, cache_dir=tmp_path)
    again = get_source(offline=False, fetch=fetch, cache_dir=tmp_path)
    assert (first.origin, again.origin) == ("network", "cache")
    assert calls == [DEFAULT_ACCESSION]


def test_retry_with_backoff_then_success(tmp_path: Path) -> None:
    attempts: list[int] = []
    waits: list[float] = []

    def flaky(acc: str):  # type: ignore[no-untyped-def]
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("reset")
        return _snapshot_fetch(acc)

    s = get_source(offline=False, fetch=flaky, cache_dir=tmp_path, sleep=waits.append)
    assert s.origin == "network" and len(attempts) == 3 and waits == [1.0, 2.0]


def test_failed_fetch_falls_back_to_snapshot(tmp_path: Path) -> None:
    def down(_: str):  # type: ignore[no-untyped-def]
        raise OSError("down")

    s = get_source(offline=False, fetch=down, cache_dir=tmp_path, sleep=lambda _: None)
    assert s.origin == "snapshot"


def test_no_cache_no_snapshot_is_source_unavailable_without_exception_text(tmp_path: Path) -> None:
    def down(_: str):  # type: ignore[no-untyped-def]
        raise OSError("User-Agent: secret-identity")

    with pytest.raises(SourceUnavailable) as err:
        get_source(
            "0000000000-00-000000",
            offline=False,
            fetch=down,
            cache_dir=tmp_path,
            sleep=lambda _: None,
        )
    assert "secret-identity" not in str(err.value) and "OSError" in str(err.value)


def test_missing_identity_fails_fast_and_never_calls_sec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(es.ENV_IDENTITY, raising=False)
    with pytest.raises(MissingIdentityError) as err:
        get_source(offline=False, cache_dir=tmp_path)
    assert es.ENV_IDENTITY in str(err.value)


def test_identity_value_never_in_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(es.ENV_IDENTITY, "Test Person test@example.invalid")
    assert es._identity() == "Test Person test@example.invalid"  # noqa: SLF001
    monkeypatch.setenv(es.ENV_IDENTITY, "  ")
    with pytest.raises(MissingIdentityError) as err:
        es._identity()  # noqa: SLF001
    assert "example.invalid" not in str(err.value)


def test_accession_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(es.ENV_ACCESSION, "0000000000-00-000000")
    with pytest.raises(SourceUnavailable):
        get_source(offline=True)


def test_snapshot_holds_no_email_address() -> None:
    blob = "".join(p.read_text(errors="ignore") for p in DEFAULT_SNAPSHOT_DIR.iterdir())
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", blob)
