"""Read-only tools over the fixed filing: `lookup_fact`, `list_statement_lines`,
`search_filing_text`. Pure functions of an `EdgarSource`; no side effects."""

from __future__ import annotations

from app.edgar_source import EdgarSource
from app.models import Provenance
from app.tools.compute import xbrl_provenance
from app.tools.schemas import (
    MAX_FACTS,
    MAX_SPANS,
    ListStatementLinesInput,
    ListStatementLinesOutput,
    LookupFactInput,
    LookupFactOutput,
    SearchFilingTextInput,
    SearchFilingTextOutput,
)


def lookup_fact(source: EdgarSource, args: LookupFactInput) -> LookupFactOutput:
    """Facts for a tag, optionally one period and exact dimensions (undimensioned by default)."""
    found = source.facts(args.concept, args.period_end, args.dimensions)
    found.sort(key=lambda f: f.period_end, reverse=True)
    kept = found[:MAX_FACTS]
    note = ""
    if not found:
        note = "no matching fact; check the tag, the period and the dimensions"
    elif len(found) > MAX_FACTS:
        note = f"{len(found)} matches, showing the {MAX_FACTS} latest; narrow by period_end"
    return LookupFactOutput(
        facts=kept,
        total=len(found),
        truncated=len(found) > len(kept),
        note=note,
        provenance=[xbrl_provenance(f) for f in kept],
    )


def list_statement_lines(
    source: EdgarSource, args: ListStatementLinesInput
) -> ListStatementLinesOutput:
    """Undimensioned lines of a primary statement in filing order, `MAX_FACTS` per call."""
    lines = source.statement_lines(args.statement)
    if args.period_end is not None:
        lines = [x for x in lines if x.period_end == args.period_end]
    page = lines[args.offset : args.offset + MAX_FACTS]
    end = args.offset + len(page)
    return ListStatementLinesOutput(
        lines=page, total=len(lines), next_offset=end if end < len(lines) else None
    )


def search_filing_text(source: EdgarSource, args: SearchFilingTextInput) -> SearchFilingTextOutput:
    """Verbatim sentences of the filing containing every query word, with offsets."""
    try:
        spans = source.search_text(args.query, args.section, limit=MAX_SPANS)
    except KeyError:
        return SearchFilingTextOutput(
            spans=[], note=f"unknown section; choose one of {', '.join(source.sections())}"
        )
    return SearchFilingTextOutput(
        spans=spans,
        note="" if spans else "no sentence contains every query word",
        provenance=[
            Provenance(
                kind="text",
                accession=source.ref.accession,
                statement_or_section=sp.section,
                quote=sp.quote,
            )
            for sp in spans
        ],
    )
