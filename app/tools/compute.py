"""`compute_ratio`: pure arithmetic over two facts, returned as a derived `Figure`.

The model passes the `Fact` objects a lookup returned; code checks both against the
filing before computing, so a fabricated or altered fact never yields a verified figure.
"""

from __future__ import annotations

from decimal import Decimal

from app.edgar_source import EdgarSource
from app.models import Fact, Figure, Provenance
from app.tools.schemas import ComputeRatioInput

_PLACES = Decimal("0.000001")
FORMULAS = {
    "ratio": "numerator / denominator",
    "change_pct": "(numerator - denominator) / denominator * 100",
}
_UNITS = {"ratio": "pure", "change_pct": "percent"}


def _in_filing(source: EdgarSource, fact: Fact) -> bool:
    return fact in source.facts(
        f"{fact.taxonomy}:{fact.tag}", fact.period_end, fact.dimensions, any_dimensions=True
    )


def xbrl_provenance(fact: Fact) -> Provenance:
    return Provenance(
        kind="xbrl",
        accession=fact.accession,
        statement_or_section="XBRL",
        tag=fact.tag,
        period_start=fact.period_start,
        period_end=fact.period_end,
        dimensions=fact.dimensions,
    )


def compute_ratio(source: EdgarSource, args: ComputeRatioInput) -> Figure:
    num, den, kind = args.numerator, args.denominator, args.kind
    label = f"{num.tag} / {den.tag}" if kind == "ratio" else f"change in {num.tag}"

    def unverified(reason: str) -> Figure:
        return Figure(
            label=label, value=Decimal(0), unit=_UNITS[kind], status="unverified",
            reason=reason, claimed_by="agent",
        )  # fmt: skip

    if not (_in_filing(source, num) and _in_filing(source, den)):
        return unverified("input fact not found in the filing")
    if num.unit != den.unit:
        return unverified("inputs have different units")
    if den.value == 0:
        return unverified("denominator is zero")
    if kind == "change_pct" and den.value < 0:
        return unverified("prior value is negative; percent change is not meaningful")
    if kind == "ratio":
        value = num.value / den.value
    else:
        value = (num.value - den.value) / den.value * 100
    return Figure(
        label=label,
        value=value.quantize(_PLACES),
        unit=_UNITS[kind],
        status="verified",
        provenance=Provenance(
            kind="derived",
            accession=source.ref.accession,
            statement_or_section="derived",
            formula=FORMULAS[kind],
            inputs=[xbrl_provenance(num), xbrl_provenance(den)],
        ),
        claimed_by="agent",
    )
