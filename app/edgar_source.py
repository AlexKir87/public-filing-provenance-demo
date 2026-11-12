"""EDGAR source: the one fixed filing, as XBRL facts and filing text sections.

Read-only. The rest of the app (tools, grounding validator) never talks to SEC or to
EdgarTools; it asks an `EdgarSource`, which answers from one `FilingData`:

* offline  - reads only the committed snapshot (`tests/fixtures/filing_snapshot/`);
* live     - disk cache first, then one fetch through EdgarTools (retry with backoff),
             then the snapshot if the fetch fails and the accession matches;
             otherwise `SourceUnavailable`.

`EDGAR_IDENTITY` is read by name from the environment, passed to EdgarTools and never
logged, stored or put into an exception message (spec "Observability").
API facts: docs/reference/edgartools-api-selection.md, sec-edgar-data-apis.md.
The filing's own XBRL (`filing.xbrl()`) is used, not company facts, because dimensional
facts are outside company facts.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from app.models import Fact, FilingRef, TextSpan

DEFAULT_ACCESSION = "0000320193-25-000079"
ENV_IDENTITY = "EDGAR_IDENTITY"
ENV_ACCESSION = "FILING_ACCESSION"
ENV_OFFLINE = "EDGAR_OFFLINE"
ENV_CACHE_DIR = "EDGAR_CACHE_DIR"
ENV_SNAPSHOT_DIR = "EDGAR_SNAPSHOT_DIR"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "tests" / "fixtures" / "filing_snapshot"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "edgar"

Statement = Literal["income", "balance", "cashflow"]
_STATEMENT_TYPES: dict[str, str] = {
    "income": "IncomeStatement",
    "balance": "BalanceSheet",
    "cashflow": "CashFlowStatement",
}
_UNITS = {"usd": "USD", "usdPerShare": "USD/shares"}
_COVER = "Cover"

_FILE_FILING = "filing.json"
_FILE_FACTS = "facts.jsonl"
_FILE_SECTIONS = "sections.json"


class EdgarSourceError(Exception):
    """Base class. Messages never contain the SEC identity."""


class MissingIdentityError(EdgarSourceError):
    """`EDGAR_IDENTITY` is not set (SEC requires it for live requests)."""


class SourceUnavailable(EdgarSourceError):
    """Neither SEC, the cache nor the snapshot can serve the filing."""


_STRICT = ConfigDict(extra="forbid", frozen=True)


class FactRecord(BaseModel):
    """A `Fact` plus the presentation context EdgarTools gives it."""

    model_config = _STRICT

    fact: Fact
    label: str = ""
    statement: str | None = None


class StatementLine(BaseModel):
    """One undimensioned line of a primary statement (input of `list_statement_lines`)."""

    model_config = _STRICT

    statement: Statement
    label: str
    tag: str
    unit: str
    period_start: date | None = None
    period_end: date
    value: Decimal


class FilingData(BaseModel):
    """Everything the app knows about the filing; also the on-disk format."""

    model_config = _STRICT

    ref: FilingRef
    facts: list[FactRecord]
    sections: dict[str, str]


Fetcher = Callable[[str], FilingData]


# --------------------------------------------------------------------------- disk format


def save_filing_data(data: FilingData, directory: Path) -> None:
    """Write `data` in the snapshot/cache layout (deterministic bytes)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / _FILE_FILING).write_text(
        data.ref.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    with (directory / _FILE_FACTS).open("w", encoding="utf-8") as fh:
        for rec in data.facts:
            fh.write(rec.model_dump_json() + "\n")
    (directory / _FILE_SECTIONS).write_text(
        json.dumps(data.sections, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def load_filing_data(directory: Path) -> FilingData:
    """Read the snapshot/cache layout. Raises `FileNotFoundError` when incomplete."""
    ref = FilingRef.model_validate_json((directory / _FILE_FILING).read_text(encoding="utf-8"))
    with (directory / _FILE_FACTS).open(encoding="utf-8") as fh:
        facts = [FactRecord.model_validate_json(line) for line in fh if line.strip()]
    sections = json.loads((directory / _FILE_SECTIONS).read_text(encoding="utf-8"))
    return FilingData(ref=ref, facts=facts, sections=sections)


# --------------------------------------------------------------------------- queries


def _norm_tag(concept: str) -> tuple[str | None, str]:
    """'us-gaap:Foo', 'us-gaap_Foo' or 'Foo' -> (taxonomy or None, 'Foo')."""
    concept = concept.strip()
    if ":" in concept:
        tax, _, name = concept.partition(":")
        return tax, name
    if "_" in concept:
        tax, _, name = concept.partition("_")
        if tax in {"us-gaap", "dei", "srt", "ifrs-full"} or tax.isalpha() and tax.islower():
            return tax, name
    return None, concept


_SENTENCE = re.compile(r"\S.*?(?:(?<=[.!?])\s+(?=[A-Z“\"(])|\n\s*\n|\Z)", re.S)
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset({"the", "a", "an", "of", "in", "and", "or", "to", "is", "was", "for", "on"})


def _terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class EdgarSource:
    """Query interface over one `FilingData`. Pure code, no network."""

    def __init__(self, data: FilingData, origin: str = "memory") -> None:
        self._data = data
        self.origin = origin  # "snapshot", "cache", "network" or "memory"; never a secret

    @property
    def ref(self) -> FilingRef:
        return self._data.ref

    def facts(
        self,
        concept: str | None = None,
        period_end: date | None = None,
        dimensions: dict[str, str] | None = None,
        *,
        any_dimensions: bool = False,
    ) -> list[Fact]:
        """Facts matching every given filter.

        `dimensions=None` means *undimensioned* (the filing-wide number) unless
        `any_dimensions=True`, so a segment value is never returned for a total.
        A dict matches exactly those dimensions.
        """
        tax, name = _norm_tag(concept) if concept else (None, "")
        out: list[Fact] = []
        for rec in self._data.facts:
            f = rec.fact
            if concept and (f.tag != name or (tax is not None and f.taxonomy != tax)):
                continue
            if period_end is not None and f.period_end != period_end:
                continue
            if not any_dimensions and f.dimensions != (dimensions or {}):
                continue
            if any_dimensions and dimensions is not None and f.dimensions != dimensions:
                continue
            out.append(f)
        return out

    def statement_lines(
        self, statement: Statement, *, include_dimensioned: bool = False
    ) -> list[StatementLine]:
        """Lines of a primary statement in filing order, all periods shown in the filing."""
        wanted = _STATEMENT_TYPES[statement]
        lines: list[StatementLine] = []
        for rec in self._data.facts:
            f = rec.fact
            if rec.statement != wanted or (f.dimensions and not include_dimensioned):
                continue
            lines.append(
                StatementLine(
                    statement=statement,
                    label=rec.label or f.tag,
                    tag=f"{f.taxonomy}:{f.tag}",
                    unit=f.unit,
                    period_start=f.period_start,
                    period_end=f.period_end,
                    value=f.value,
                )
            )
        return lines

    def sections(self) -> list[str]:
        return list(self._data.sections)

    def section_text(self, section: str) -> str:
        try:
            return self._data.sections[section]
        except KeyError:
            raise KeyError(f"unknown section {section!r}") from None

    def search_text(self, query: str, section: str | None = None, limit: int = 5) -> list[TextSpan]:
        """Sentences containing every query word, as verbatim spans with offsets.

        Offsets are positions in `section_text(span.section)`; `verify_span` re-checks them.
        Ranked: whole query as a phrase first, then shorter sentences, then document order.
        """
        terms = _terms(query)
        if not terms or limit < 1:
            return []
        names = [section] if section is not None else self.sections()
        phrase = " ".join(_WORD.findall(query.lower()))
        hits: list[tuple[int, int, int, int, TextSpan]] = []
        for order, name in enumerate(names):
            text = self.section_text(name)
            for m in _SENTENCE.finditer(text):
                start, end = m.start(), m.start() + len(m.group().rstrip())
                if end <= start:
                    continue
                quote = text[start:end]
                words = set(_WORD.findall(quote.lower()))
                if not all(t in words for t in terms):
                    continue
                joined = " ".join(_WORD.findall(quote.lower()))
                span = TextSpan(section=name, quote=quote, start=start, end=end)
                hits.append((0 if phrase in joined else 1, len(quote), order, start, span))
        hits.sort(key=lambda h: h[:4])
        return [h[4] for h in hits[:limit]]

    def verify_span(self, span: TextSpan) -> bool:
        """True when the span's quote sits verbatim at its offsets in its section."""
        text = self._data.sections.get(span.section)
        return text is not None and text[span.start : span.end] == span.quote


# --------------------------------------------------------------------------- live fetch


def _identity() -> str:
    value = os.environ.get(ENV_IDENTITY, "").strip()
    if not value:
        raise MissingIdentityError(f"{ENV_IDENTITY} is not set")
    return value


def _to_decimal(raw: Any) -> Decimal | None:
    try:
        value = Decimal(str(raw).strip())
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() else None


def _to_date(raw: Any) -> date | None:
    if raw is None:
        return None
    text = str(raw)[:10]
    if text in {"", "NaT", "nan", "None"}:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _present(raw: Any) -> bool:
    return raw is not None and str(raw) not in {"", "nan", "NaN", "None", "<NA>"}


def _records_from_dataframe(df: Any, accession: str) -> list[FactRecord]:
    """Numeric facts of the filing's own XBRL as typed records; non-numeric ones are skipped."""
    dim_cols = [c for c in df.columns if str(c).startswith("dim_")]
    records: list[FactRecord] = []
    seen: set[tuple[Any, ...]] = set()
    for row in df.to_dict("records"):
        value = _to_decimal(row.get("value"))
        period_end = _to_date(row.get("period_end")) or _to_date(row.get("period_instant"))
        concept = str(row.get("concept", ""))
        if value is None or period_end is None or ":" not in concept:
            continue
        taxonomy, _, tag = concept.partition(":")
        dims = {
            str(c)[4:].replace("_", ":", 1): str(row[c]) for c in dim_cols if _present(row.get(c))
        }
        if not dims and _present(row.get("dimension")) and _present(row.get("member")):
            dims = {str(row["dimension"]): str(row["member"])}
        if not _present(row.get("unit_ref")):
            continue  # no unit: a text or flag fact that merely looks numeric (e.g. fiscal year)
        unit_ref = str(row["unit_ref"]).strip()
        unit = _UNITS.get(unit_ref, unit_ref)
        period_start = _to_date(row.get("period_start"))
        key = (taxonomy, tag, period_start, period_end, tuple(sorted(dims.items())), unit)
        if key in seen:
            continue
        seen.add(key)
        stmt = row.get("statement_type")
        records.append(
            FactRecord(
                fact=Fact(
                    tag=tag,
                    taxonomy=taxonomy,
                    value=value,
                    unit=unit,
                    period_start=period_start,
                    period_end=period_end,
                    dimensions=dims,
                    accession=accession,
                ),
                label=str(row.get("label") or ""),
                statement=str(stmt) if _present(stmt) else None,
            )
        )
    return records


def _sections_from_filing(filing: Any) -> dict[str, str]:
    """Cover text plus every non-empty 'Item N' of the 10-K, as plain text."""
    report = filing.obj()
    full = filing.text()
    sections: dict[str, str] = {}
    first = re.search(r"\n\s*Item\s+1\.\s+Business", full)
    if first and first.start() > 0:
        sections[_COVER] = full[: first.start()].strip()
    for item in report.items:
        body = report[item]
        if body and str(body).strip():
            sections[str(item)] = str(body).strip()
    return sections


def fetch_filing_from_edgar(accession: str) -> FilingData:
    """One real fetch through EdgarTools (network). Needs `EDGAR_IDENTITY`."""
    identity = _identity()
    from edgar import get_by_accession_number, set_identity  # lazy: offline needs no network stack

    set_identity(identity)
    filing = get_by_accession_number(accession)
    if filing is None:
        raise SourceUnavailable(f"filing {accession} not found at SEC")
    xbrl = filing.xbrl()
    if xbrl is None:
        raise SourceUnavailable(f"filing {accession} has no XBRL")
    ref = FilingRef(
        accession=accession,
        form=str(filing.form),
        period_end=_to_date(filing.period_of_report) or date.min,
        company=str(filing.company),
        cik=str(filing.cik).zfill(10),
    )
    return FilingData(
        ref=ref,
        facts=_records_from_dataframe(xbrl.facts.to_dataframe(), accession),
        sections=_sections_from_filing(filing),
    )


# --------------------------------------------------------------------------- entry point


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def get_source(
    accession: str | None = None,
    *,
    offline: bool | None = None,
    snapshot_dir: Path | None = None,
    cache_dir: Path | None = None,
    fetch: Fetcher | None = None,
    retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> EdgarSource:
    """Build the source for the fixed filing.

    `offline` defaults to env `EDGAR_OFFLINE`; offline never touches the network or the
    cache, only the snapshot. `fetch` is injectable for tests (default: EdgarTools).
    """
    accession = accession or os.environ.get(ENV_ACCESSION) or DEFAULT_ACCESSION
    if offline is None:
        offline = _truthy(os.environ.get(ENV_OFFLINE))
    snap = snapshot_dir or Path(os.environ.get(ENV_SNAPSHOT_DIR) or DEFAULT_SNAPSHOT_DIR)

    def from_snapshot() -> EdgarSource | None:
        try:
            data = load_filing_data(snap)
        except (FileNotFoundError, ValueError):
            return None
        return EdgarSource(data, "snapshot") if data.ref.accession == accession else None

    if offline:
        src = from_snapshot()
        if src is None:
            raise SourceUnavailable(f"no snapshot for {accession} in offline mode")
        return src

    cache = (cache_dir or Path(os.environ.get(ENV_CACHE_DIR) or DEFAULT_CACHE_DIR)) / accession
    try:
        cached = load_filing_data(cache)
        if cached.ref.accession == accession:
            return EdgarSource(cached, "cache")
    except (FileNotFoundError, ValueError):
        pass

    do_fetch = fetch or fetch_filing_from_edgar
    if fetch is None:
        _identity()  # fail fast and clearly; a missing identity is not a transient error
    last: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            data = do_fetch(accession)
        except MissingIdentityError:
            raise
        except Exception as exc:  # transport and parse errors: retry, then fall back
            last = exc
            if attempt + 1 < retries:
                sleep(2.0**attempt)
            continue
        try:
            save_filing_data(data, cache)
        except OSError:
            pass  # a cache that cannot be written must not fail the request
        return EdgarSource(data, "network")

    src = from_snapshot()
    if src is not None:
        return src
    # class name only: exception text from the HTTP layer could echo request headers
    raise SourceUnavailable(f"source unavailable ({type(last).__name__ if last else 'unknown'})")


def build_snapshot(out_dir: Path, accession: str | None = None) -> FilingData:
    """Fetch once from SEC and write the committed snapshot layout. Network, needs identity."""
    data = fetch_filing_from_edgar(accession or os.environ.get(ENV_ACCESSION) or DEFAULT_ACCESSION)
    save_filing_data(data, out_dir)
    return data


if __name__ == "__main__":  # python -m app.edgar_source <out_dir>
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SNAPSHOT_DIR
    built = build_snapshot(target)
    print(f"wrote {len(built.facts)} facts, {len(built.sections)} sections to {target}")
