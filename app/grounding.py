"""Grounding validator and answer renderer (spec.md "Output schemas", rules 1 to 6).

Pure code over `app/models.py`. The model proposes figures; this module decides
every status. A figure is `verified` only when code finds the evidence for it in
what the tools returned during this run (`RunEvidence`). Nothing here reads text
as an instruction: quotes, labels and notes are data and cannot set a status.

How the agent (step 5) uses it:
- tools append what they return to one `RunEvidence` per run;
- the Pydantic AI output validator calls `validate_bundle(bundle, evidence,
  final_attempt=...)` and turns `StrayNumbersError` into `ModelRetry`;
- the model writes the answer with placeholders `{fig:N}` (N is the index in
  `figures`); code renders every number and the origin lines.

Reason codes for `unverified` figures (spec rule 4, plus two for derived figures):
`no matching fact`, `value differs from filing`, `period differs`, `quote not found`,
`input unverified`, `cannot recompute`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from datetime import date
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext

from pydantic import BaseModel, ConfigDict, Field

from app.models import AnswerBundle, Fact, Figure, Provenance, TextSpan

REASON_NO_FACT = "no matching fact"
REASON_VALUE = "value differs from filing"
REASON_PERIOD = "period differs"
REASON_QUOTE = "quote not found"
REASON_INPUT = "input unverified"
REASON_RECOMPUTE = "cannot recompute"
REASONS = (
    REASON_NO_FACT,
    REASON_VALUE,
    REASON_PERIOD,
    REASON_QUOTE,
    REASON_INPUT,
    REASON_RECOMPUTE,
)

MASK = "[unverified figure]"
MASK_NOTE = (
    "Numbers in the answer text that were not rendered from a checked figure "
    "are shown as [unverified figure]."
)
STATUS_WORDS_NOTE = (
    "Status words in the model's answer text were removed; each figure's status is set by code."
)
MISSING_PLACEHOLDER_NOTE = (
    "The answer referred to a figure that does not exist; it is shown as [unverified figure]."
)

DERIVED_FORMULAS = ("ratio", "change_pct", "sum", "difference")
# The expressions `compute_ratio` (app/tools/compute.py, step 4) writes into
# `Provenance.formula`. Spaces are ignored when matching; tests pin both spellings.
FORMULA_ALIASES = {
    "numerator/denominator": "ratio",
    "(numerator-denominator)/denominator*100": "change_pct",
}


def canonical_formula(formula: str | None) -> str:
    """`ratio`, `change_pct`, `sum` or `difference`, or "" when the formula is unknown."""
    f = (formula or "").strip().lower()
    f = FORMULA_ALIASES.get(re.sub(r"\s+", "", f), f)
    return f if f in DERIVED_FORMULAS else ""


_MAX_DERIVED_DEPTH = 5
_PLACEHOLDER = re.compile(r"\{fig:(\d+)\}")


class RunEvidence(BaseModel):
    """Everything the tools returned in one run, plus the filing text to check quotes.

    `section_texts` maps a section name to that section's full text, as the EDGAR
    source holds it; a `TextSpan` is checked against it at `start:end`.
    """

    model_config = ConfigDict(extra="forbid")

    accession: str = Field(min_length=1)
    facts: list[Fact] = Field(default_factory=list)
    spans: list[TextSpan] = Field(default_factory=list)
    section_texts: dict[str, str] = Field(default_factory=dict)

    def add_facts(self, facts: Iterable[Fact]) -> None:
        self.facts.extend(facts)

    def add_spans(self, spans: Iterable[TextSpan]) -> None:
        self.spans.extend(spans)


class StrayNumbersError(ValueError):
    """The answer text holds numbers that were not rendered from a figure.

    The agent turns this into `ModelRetry` (at most 2 output retries); on the
    last attempt `validate_bundle(..., final_attempt=True)` masks them instead.
    """

    def __init__(self, numbers: Sequence[str]) -> None:
        self.numbers = list(numbers)
        super().__init__(
            "The answer text contains numbers that are not placeholders: "
            + ", ".join(repr(n) for n in self.numbers)
            + ". Put every number in `figures` and refer to it as {fig:N} in `answer`."
        )


# --- units and numbers -------------------------------------------------------

_SCALE_WORDS = {
    "thousand": Decimal(1_000),
    "thousands": Decimal(1_000),
    "million": Decimal(1_000_000),
    "millions": Decimal(1_000_000),
    "billion": Decimal(1_000_000_000),
    "billions": Decimal(1_000_000_000),
}


def _base_unit(unit: str) -> str:
    """Canonical base unit: lower case, no taxonomy prefix, `per share` as `/shares`."""
    u = unit.strip().lower()
    for prefix in ("iso4217:", "xbrli:"):
        u = u.replace(prefix, "")
    u = re.sub(r"\s+per\s+shares?$", "/shares", u)
    u = re.sub(r"\s*/\s*", "/", u)
    u = re.sub(r"/share$", "/shares", u)
    return u


def split_unit(unit: str) -> tuple[str, Decimal]:
    """`'USD millions'` -> (`'usd'`, 1000000). A unit without a scale word has scale 1."""
    parts = unit.strip().split()
    if len(parts) >= 2 and parts[-1].lower() in _SCALE_WORDS:
        return _base_unit(" ".join(parts[:-1])), _SCALE_WORDS[parts[-1].lower()]
    return _base_unit(unit), Decimal(1)


def raw_value(figure: Figure) -> Decimal:
    """The figure's value in the base unit (raw dollars, raw shares)."""
    _, scale = split_unit(figure.unit)
    return figure.value * scale


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE = re.compile(
    rf"\b(?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{1,2}}/\d{{1,2}}/\d{{2,4}}\b"
)
_QUOTE_NUMBER = re.compile(
    r"(?<![A-Za-z0-9.,])"
    r"(?P<open>\()?\s*(?P<sign>[-−])?\s*(?P<cur>[$€£])?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![A-Za-z0-9])"
    r"\s*(?P<close>\))?"
    r"(?:\s*(?P<pct>%|percent\b)|\s*(?P<scale>thousand|million|billion)s?\b)?",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z]+")


class QuoteNumber(BaseModel):
    """One number printed in a quote, with the marks around it that say what it counts."""

    model_config = ConfigDict(frozen=True)

    printed: Decimal
    scaled: Decimal | None
    currency: str | None
    percent: bool
    following_words: tuple[str, ...]

    @property
    def value(self) -> Decimal:
        """ "$90.7 billion" stands for 90700000000 only, never for 90.7."""
        return self.scaled if self.scaled is not None else self.printed


_UNIT_WINDOW = 4


def quote_numbers(text: str) -> list[QuoteNumber]:
    """Numbers printed in a quote. Dates and numbers glued to letters (FY2025, A320)
    are not figures. Brackets around a number mean negative. Digits are never altered."""
    cleaned = _DATE.sub(" ", text)
    found: list[QuoteNumber] = []
    for m in _QUOTE_NUMBER.finditer(cleaned):
        try:
            value = Decimal(m.group("num").replace(",", ""))
        except InvalidOperation:  # pragma: no cover - the regex only admits digits
            continue
        sign_at = m.start("sign")
        # "5 - 6 employees" is a range, not minus six.
        real_sign = sign_at >= 0 and not cleaned[:sign_at].rstrip()[-1:].isdigit()
        if (m.group("open") and m.group("close")) or real_sign:
            value = -value
        scale = m.group("scale")
        # Unit words belong to this number only up to the next number or punctuation:
        # in "166 stores and 12 employees", "employees" is not the unit of 166.
        tail = re.split(r"[\d.,;:()$€£%]", cleaned[m.end() :], maxsplit=1)[0]
        words = tuple(w.lower() for w in _WORD.findall(tail)[:_UNIT_WINDOW])
        found.append(
            QuoteNumber(
                printed=value,
                scaled=value * _SCALE_WORDS[scale.lower()] if scale else None,
                currency=m.group("cur"),
                percent=bool(m.group("pct")),
                following_words=words,
            )
        )
    return found


def numbers_in_text(text: str) -> list[tuple[Decimal, Decimal | None]]:
    """(printed value, value times its scale word or None) for each number in a quote."""
    return [(n.printed, n.scaled) for n in quote_numbers(text)]


def _stem(word: str) -> str:
    w = word.lower()
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def unit_printed_with(number: QuoteNumber, base_unit: str) -> bool:
    """Does the quote say what the figure's unit says? Fails closed for unknown units.

    - percent: the number is followed by `%` or "percent";
    - usd: `$` before it and a scale word after it ("$90.7 billion"), or "dollars"/"USD"
      right after it; "$391,035" alone may come from a table in millions, so it fails;
    - any other unit (shares, employees, stores): that word within the next four words,
      before any other number or punctuation.
    A bare number in a quote never verifies a figure: the table it came from may be in
    millions, or it may count something else.
    """
    if number.percent:
        return base_unit in ("%", "percent")
    if base_unit in ("%", "percent"):
        return False
    if base_unit == "usd":
        if number.currency == "$":
            return number.scaled is not None
        nxt = [w for w in number.following_words if w not in _SCALE_WORDS][:1]
        return nxt in (["dollars"], ["usd"])
    if number.currency is not None or not base_unit or not base_unit.isalpha():
        return False
    target = _stem(base_unit)
    return any(_stem(w) == target for w in number.following_words)


def _normalise_ws(text: str) -> str:
    """Collapse whitespace only. Digits, brackets and signs are never altered."""
    return re.sub(r"\s+", " ", text).strip()


# --- matching one provenance -------------------------------------------------


def _tag_matches(prov_tag: str, fact: Fact) -> bool:
    return prov_tag == fact.tag or prov_tag == f"{fact.taxonomy}:{fact.tag}"


def _same_period(prov: Provenance, fact: Fact) -> bool:
    return prov.period_start == fact.period_start and prov.period_end == fact.period_end


def _xbrl_fact(prov: Provenance, evidence: RunEvidence) -> tuple[Fact | None, str]:
    """The one fact this xbrl provenance points to, or the reason there is none."""
    if prov.tag is None or prov.accession != evidence.accession:
        return None, REASON_NO_FACT
    same_tag = [
        f
        for f in evidence.facts
        if f.accession == evidence.accession
        and _tag_matches(prov.tag, f)
        and f.dimensions == prov.dimensions
    ]
    if not same_tag:
        return None, REASON_NO_FACT
    exact = [f for f in same_tag if _same_period(prov, f)]
    if not exact:
        return None, REASON_PERIOD
    values = {f.value for f in exact}
    if len(values) != 1:
        # Two facts with the same tag, period and dimensions but different values:
        # the evidence cannot say which one the figure means.
        return None, REASON_VALUE
    return exact[0], ""


def _quote_span(prov: Provenance, evidence: RunEvidence) -> TextSpan | None:
    """The returned span whose quote is this provenance's quote and sits verbatim in the filing."""
    if prov.quote is None or prov.accession != evidence.accession:
        return None
    wanted = _normalise_ws(prov.quote)
    if not wanted:
        return None
    for span in evidence.spans:
        if span.section != prov.statement_or_section:
            continue
        if _normalise_ws(span.quote) != wanted:
            continue
        text = evidence.section_texts.get(span.section)
        if text is None or span.end > len(text):
            continue
        if _normalise_ws(text[span.start : span.end]) == wanted:
            return span
    return None


class _Resolved(BaseModel):
    """A value computed by code from evidence, with its base unit (None when unknown)."""

    model_config = ConfigDict(frozen=True)

    value: Decimal | None = None
    unit: str | None = None
    reason: str = ""


def _resolve_value(prov: Provenance, evidence: RunEvidence, depth: int = 0) -> _Resolved:
    """The value a provenance stands for, computed by code from evidence only."""
    if prov.kind == "xbrl":
        fact, reason = _xbrl_fact(prov, evidence)
        if fact is None:
            return _Resolved(reason=reason)
        return _Resolved(value=fact.value, unit=_base_unit(fact.unit))
    if prov.kind == "text":
        span = _quote_span(prov, evidence)
        if span is None:
            return _Resolved(reason=REASON_QUOTE)
        numbers = quote_numbers(span.quote)
        if len(numbers) != 1:
            # A quote with several numbers does not say which one is the input.
            return _Resolved(reason=REASON_RECOMPUTE)
        # The unit of a narrative number is not known to code, so a text input can
        # only feed a ratio or a percentage change, never a sum or a difference.
        return _Resolved(value=numbers[0].value, unit=None)
    return _recompute(prov, evidence, depth + 1)


def _recompute(prov: Provenance, evidence: RunEvidence, depth: int) -> _Resolved:
    if depth > _MAX_DERIVED_DEPTH or prov.accession != evidence.accession:
        return _Resolved(reason=REASON_RECOMPUTE)
    formula = canonical_formula(prov.formula)
    if not formula:
        return _Resolved(reason=REASON_RECOMPUTE)
    inputs: list[_Resolved] = []
    for inp in prov.inputs:
        resolved = _resolve_value(inp, evidence, depth)
        if resolved.value is None:
            return _Resolved(reason=REASON_INPUT)
        inputs.append(resolved)
    values = [i.value for i in inputs if i.value is not None]
    units = {i.unit for i in inputs}
    with localcontext() as ctx:
        ctx.prec = 34
        ctx.traps[DivisionByZero] = True
        ctx.traps[InvalidOperation] = True
        try:
            if formula in ("sum", "difference"):
                # Only like with like: USD with USD, shares with shares.
                if len(units) != 1 or None in units:
                    return _Resolved(reason=REASON_RECOMPUTE)
                unit = units.pop()
                if formula == "sum":
                    if len(values) < 2:
                        return _Resolved(reason=REASON_RECOMPUTE)
                    return _Resolved(value=sum(values, Decimal(0)), unit=unit)
                if len(values) != 2:
                    return _Resolved(reason=REASON_RECOMPUTE)
                return _Resolved(value=values[0] - values[1], unit=unit)
            if len(values) != 2:
                return _Resolved(reason=REASON_RECOMPUTE)
            a, b = values
            if formula == "ratio":
                same = inputs[0].unit is not None and inputs[0].unit == inputs[1].unit
                return _Resolved(value=a / b, unit="pure" if same else None)
            return _Resolved(value=(a - b) / b * 100, unit="%")  # change_pct
        except (DivisionByZero, InvalidOperation):
            return _Resolved(reason=REASON_RECOMPUTE)


def _significant_digits(value: Decimal) -> int:
    digits = value.normalize().as_tuple().digits
    return len(digits) if value != 0 else 1


_PURE_UNITS = ("pure", "ratio", "x", "times")


def _derived_unit_matches(figure: Figure, computed: _Resolved) -> bool:
    """The figure's unit must say what the formula yields (sum of USD is USD, and so on)."""
    base, scale = split_unit(figure.unit)
    if computed.unit is None:  # ratio of different or unknown units, e.g. USD per employee
        return scale == 1 and base not in ("%", "percent", *_PURE_UNITS)
    if computed.unit == "%":
        return base in ("%", "percent") and scale == 1
    if computed.unit == "pure":
        return base in _PURE_UNITS and scale == 1
    return base == computed.unit


def _derived_value_matches(figure: Figure, computed: _Resolved, formula: str) -> bool:
    """Exact for sums and differences; for ratios, the figure's own stated precision.

    A ratio or percentage may be rounded to the decimals the figure states, but it
    must state at least two significant digits, so that `0` or `6` cannot stand for
    any small ratio.
    """
    if formula in ("sum", "difference"):
        return raw_value(figure) == computed.value
    stated = figure.value
    if _significant_digits(stated) < 2:
        return False
    exponent = stated.as_tuple().exponent
    if not isinstance(exponent, int):  # pragma: no cover - non-finite handled earlier
        return False
    with localcontext() as ctx:
        ctx.prec = 34
        try:
            if computed.value is None:
                return False
            rounded = computed.value.quantize(Decimal(1).scaleb(exponent))
        except InvalidOperation:
            return False
    return rounded == stated


# --- one figure --------------------------------------------------------------


def _unverified(figure: Figure, reason: str) -> Figure:
    data = figure.model_dump()
    data.update(status="unverified", provenance=None, reason=reason)
    return Figure.model_validate(data)


def _verified(figure: Figure) -> Figure:
    data = figure.model_dump()
    data.update(status="verified", reason="")
    return Figure.model_validate(data)


def ground_figure(figure: Figure, evidence: RunEvidence) -> Figure:
    """Rules 1 to 4 for one figure. Never promotes: an `unverified` input stays as it is."""
    if figure.status == "unverified":
        # Keep the model's own "unverified", but only with a reason code from the list:
        # free text from the model never reaches the rendered answer as a reason.
        return figure if figure.reason in REASONS else _unverified(figure, REASON_NO_FACT)
    prov = figure.provenance
    if prov is None:  # models.py forbids it; kept so the rule does not depend on it
        return _unverified(figure, REASON_NO_FACT)
    if not figure.value.is_finite():  # models.py rejects NaN and infinity as well
        return _unverified(figure, REASON_VALUE)

    if prov.kind == "xbrl":
        fact, reason = _xbrl_fact(prov, evidence)
        if fact is None:
            return _unverified(figure, reason)
        base, _ = split_unit(figure.unit)
        if base != _base_unit(fact.unit) or raw_value(figure) != fact.value:
            return _unverified(figure, REASON_VALUE)
        return _verified(figure)

    if prov.kind == "text":
        span = _quote_span(prov, evidence)
        if span is None:
            return _unverified(figure, REASON_QUOTE)
        target = raw_value(figure)
        base, _ = split_unit(figure.unit)
        for number in quote_numbers(span.quote):
            if target == number.value and unit_printed_with(number, base):
                return _verified(figure)
        return _unverified(figure, REASON_VALUE)

    # derived
    computed = _recompute(prov, evidence, depth=1)
    if computed.value is None:
        return _unverified(figure, computed.reason)
    formula = canonical_formula(prov.formula)
    if not _derived_unit_matches(figure, computed) or not _derived_value_matches(
        figure, computed, formula
    ):
        return _unverified(figure, REASON_VALUE)
    return _verified(figure)


def _clean_label(figure: Figure) -> Figure:
    label = clean_model_line(figure.label) or "figure"
    if label == figure.label:
        return figure
    data = figure.model_dump()
    data.update(label=label)
    return Figure.model_validate(data)


def ground_figures(
    figures: Sequence[Figure], evidence: RunEvidence
) -> tuple[list[Figure], list[int]]:
    """Ground every figure; put user-claimed figures first (rule 6), keep every figure.

    Returns the grounded figures and `order`, where `order[new_index] = old_index`,
    so that `{fig:N}` placeholders written against the model's order still resolve.
    """
    grounded = [_clean_label(ground_figure(f, evidence)) for f in figures]
    order = [i for i, f in enumerate(grounded) if f.claimed_by == "user"]
    order += [i for i, f in enumerate(grounded) if f.claimed_by != "user"]
    return [grounded[i] for i in order], order


# --- answer text -------------------------------------------------------------

_FORM_OR_REFERENCE = re.compile(
    r"\b(?:10-K|10-Q|8-K|20-F|S-1)\b|\b(?:Item|Note|Part)\s+\d{1,2}[A-Z]?\b(?![.,]?\d)",
    re.IGNORECASE,
)
_NOT_A_FIGURE = r"(?!\s*(?:%|percent|thousand|million|billion|trillion)\b|\s*%|[.,]\d)"
_YEAR_PATTERNS = (
    re.compile(
        r"\b(?:fiscal|FY|year|ended|ending)\s+(?:year\s+)?"
        r"(?P<y>(?:19|20)\d{2})\b" + _NOT_A_FIGURE,
        re.IGNORECASE,
    ),
    re.compile(r"\b(?P<y>FY(?:19|20)?\d{2})\b", re.IGNORECASE),
    re.compile(
        rf"\b(?:{_MONTHS})\s+(?P<y>(?:\d{{1,2}},?\s+)?(?:19|20)\d{{2}})\b" + _NOT_A_FIGURE,
        re.IGNORECASE,
    ),
)
_STRAY = re.compile(
    r"[$€£]?\s?\(?[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?"
    r"(?:\s?(?:%|percent\b|thousand\b|million\b|billion\b|trillion\b))?",
    re.IGNORECASE,
)
_STATUS_WORDS = re.compile(r"\b(?:un)?verified\b", re.IGNORECASE)
# Cyrillic and Greek letters that look like Latin ones, so "vеrified" with a Cyrillic
# "е" is still a status word. One character for one, so positions do not move.
_CONFUSABLES = str.maketrans(
    "аеіорсухАВЕНІКМОРСТХѵνεοιρ",
    "aeiopcyxABEHIKMOPCTXvveoip",
)
_INVISIBLE = re.compile(r"[\u00ad\u200b-\u200f\u2060-\u2064\ufeff]")


def normalise_model_text(text: str) -> str:
    """NFKC and no invisible characters, so that "ver\u200bified" or full-width digits
    cannot slip past the status-word and stray-number checks."""
    return _INVISIBLE.sub("", unicodedata.normalize("NFKC", text)).translate(_CONFUSABLES)


def clean_model_line(text: str) -> str:
    """Model-written label or note: one line, no status words, no badges, no numbers
    (a number in a label would bypass the checks, so it is masked like a stray one)."""
    text, _ = strip_status_words(normalise_model_text(text))
    text = re.sub(r"\[\s*\d+[^\]]*\]", "", text)  # no forged "[n, ...]" badges
    text = re.sub(r"\borigin\s*:", "", text, flags=re.IGNORECASE)
    return mask_stray_numbers(_normalise_ws(text))


def _allowed_spans(answer: str) -> list[tuple[int, int]]:
    spans = [m.span() for m in _PLACEHOLDER.finditer(answer)]
    spans += [m.span() for m in _FORM_OR_REFERENCE.finditer(answer)]
    for pattern in _YEAR_PATTERNS:
        spans += [m.span("y") for m in pattern.finditer(answer)]
    return spans


def _stray_matches(answer: str) -> list[re.Match[str]]:
    allowed = _allowed_spans(answer)
    out = []
    for m in _STRAY.finditer(answer):
        # Skip a number only when all of it lies inside a placeholder, a form name,
        # an item reference or a year ("Item 5,391,000" is still a stray number).
        lead = len(m.group()) - len(m.group().lstrip())
        start, end = m.start() + lead, m.end()
        if any(a <= start and end <= b for a, b in allowed):
            continue
        out.append(m)
    return out


def find_stray_numbers(answer: str) -> list[str]:
    """Numbers in the model's answer text that are not `{fig:N}` placeholders (rule 5)."""
    return [m.group().strip() for m in _stray_matches(answer)]


def mask_stray_numbers(answer: str) -> str:
    """Replace every stray number with `[unverified figure]`."""
    pieces: list[str] = []
    last = 0
    for m in _stray_matches(answer):
        start, end = m.span()
        lead = len(m.group()) - len(m.group().lstrip())
        pieces.append(answer[last : start + lead])
        pieces.append(MASK)
        last = end
    pieces.append(answer[last:])
    return "".join(pieces)


def _plain(value: Decimal, grouping: bool = True) -> str:
    text = format(value.normalize(), ",f" if grouping else "f")
    return text


def format_value(figure: Figure) -> str:
    """Display text for a figure's value, exact (never rounded beyond what it states)."""
    base, scale = split_unit(figure.unit)
    value = figure.value * scale
    if not value.is_finite():
        return f"{figure.value} {figure.unit}"
    if base == "usd/shares":
        return f"${_plain(value)} per share"
    if base in ("%", "percent", "pure%"):
        return f"{_plain(value)}%"
    number = _plain(value)
    if abs(value) >= 1_000_000 and base in ("usd", "shares"):
        for word, size in (("billion", Decimal(1_000_000_000)), ("million", Decimal(1_000_000))):
            scaled = value / size
            if abs(scaled) >= 1 and scaled == scaled.quantize(Decimal("0.001")):
                number = f"{_plain(scaled)} {word}"
                break
    if base == "usd":
        return f"${number}" if not number.startswith("-") else f"-${number[1:]}"
    if base == "shares":
        return f"{number} shares"
    return f"{number} {figure.unit.strip()}"


def _period_text(prov: Provenance) -> str:
    def d(x: date) -> str:
        return x.isoformat()

    if prov.period_start and prov.period_end:
        return f"period {d(prov.period_start)} to {d(prov.period_end)}"
    if prov.period_end:
        return f"as of {d(prov.period_end)}"
    return "period unknown"


def origin_text(prov: Provenance) -> str:
    """One origin line: accession, statement or section, tag and period, or the quote."""
    if prov.kind == "xbrl":
        dims = ""
        if prov.dimensions:
            dims = ", dimensions " + "; ".join(
                f"{k}={v}" for k, v in sorted(prov.dimensions.items())
            )
        return (
            f"accession {prov.accession}, {prov.statement_or_section}, tag {prov.tag}, "
            f"{_period_text(prov)}{dims}"
        )
    if prov.kind == "text":
        quote = _normalise_ws(prov.quote or "")
        return f'accession {prov.accession}, {prov.statement_or_section}, quote "{quote}"'
    inputs = " | ".join(origin_text(i) for i in prov.inputs)
    return f"computed by code: {prov.formula} of [{inputs}]"


def figure_line(index: int, figure: Figure) -> str:
    head = f"[{index + 1}] {clean_model_line(figure.label) or 'figure'}: {format_value(figure)}"
    if figure.claimed_by == "user":
        head += " (stated by the user)"
    if figure.status == "verified" and figure.provenance is not None:
        return f"{head}, verified. Origin: {origin_text(figure.provenance)}."
    return f"{head}, unverified ({figure.reason}). Not matched to the filing."


def strip_status_words(answer: str) -> tuple[str, bool]:
    """Remove "verified" / "unverified" from the model's own text: only code states a status."""
    if not _STATUS_WORDS.search(answer):
        return answer, False
    answer = _STATUS_WORDS.sub("", answer)
    answer = re.sub(r"\(\s*\)|\[\s*\]", "", answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r"[ \t]+([.,;:!?])", r"\1", answer)
    return answer, True


def render_answer(
    answer: str, figures: Sequence[Figure], order: Sequence[int]
) -> tuple[str, list[str]]:
    """Fill `{fig:N}` from the figures and append one status and origin line per figure.

    `answer` must already be free of stray numbers and status words (see
    `validate_bundle`). Returns the text and any notes.
    """
    notes: list[str] = []
    new_index = {old: new for new, old in enumerate(order)}

    def fill(m: re.Match[str]) -> str:
        old = int(m.group(1))
        if old not in new_index:
            if MISSING_PLACEHOLDER_NOTE not in notes:
                notes.append(MISSING_PLACEHOLDER_NOTE)
            return MASK
        i = new_index[old]
        fig = figures[i]
        return f"{format_value(fig)} [{i + 1}, {fig.status}]"

    body = _PLACEHOLDER.sub(fill, answer).strip()
    if not figures:
        return body, notes
    lines = [figure_line(i, f) for i, f in enumerate(figures)]
    return body + "\n\nFigures:\n" + "\n".join(lines), notes


# --- the output validator ----------------------------------------------------


def validate_bundle(
    bundle: AnswerBundle, evidence: RunEvidence, *, final_attempt: bool = False
) -> AnswerBundle:
    """The output validator (rules 1 to 6).

    Raises `StrayNumbersError` when the answer text holds numbers outside `{fig:N}`
    placeholders and this is not the final attempt; on the final attempt those
    numbers are masked and a note is added. Figures are never dropped and never
    promoted to `verified` without evidence.
    """
    notes = [n for n in (clean_model_line(x) for x in bundle.notes) if n]
    answer, stripped = strip_status_words(normalise_model_text(bundle.answer))
    if stripped:
        notes.append(STATUS_WORDS_NOTE)
    stray = find_stray_numbers(answer)
    if stray:
        if not final_attempt:
            raise StrayNumbersError(stray)
        answer = mask_stray_numbers(answer)
        notes.append(MASK_NOTE)

    figures, order = ground_figures(bundle.figures, evidence)
    text, render_notes = render_answer(answer, figures, order)
    notes.extend(n for n in render_notes if n not in notes)
    return AnswerBundle(answer=text, figures=figures, notes=notes)
