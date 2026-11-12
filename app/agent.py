"""The agent: one Pydantic AI `Agent[Deps, AnswerBundle]` over the fixed filing (spec.md).

The model chooses read-only tools and proposes an `AnswerBundle`; the output validator
(`app/grounding.py`) decides every status, renders every number and asks the model to
retry (at most twice) when the answer text holds a number of its own.

Run modes:
- default: **replay**. A recorded live transcript from `tests/fixtures/replay/` plays the
  model; the tools, the snapshot and the validator run for real. No network, no spend.
  A question without a recording is a hard failure (`ReplayMissing`), never a guess
  ([[pattern-replay-evals-vs-live-runs]]).
- `COGNITIA_LIVE=1`: the model named in `COGNITIA_MODEL` (and `COGNITIA_FALLBACK_MODEL`
  through `FallbackModel`), priced from `ops/prices.json`; an unpriced model does not run.
  `COGNITIA_RECORD=1` saves the model's responses as a replay fixture.

Limits per run: `UsageLimits` (requests, tool calls), a USD spend cap checked after every
model response, a wall-clock `asyncio.timeout`, output retries at most 2. A run that hits
any of them returns an answer with no figures and a note.

The trace (`app/trace_log.py`) holds metadata only: no question, answer or tool arguments.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, TypeVar

import pydantic_ai
from pydantic import BaseModel, ValidationError
from pydantic_ai import (
    Agent,
    FallbackExceptionGroup,
    ModelAPIError,
    ModelMessagesTypeAdapter,
    ModelResponse,
    ModelRetry,
    RunContext,
    Tool,
    ToolOutput,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UsageLimits,
)
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app import prompts
from app import tools as tool_impl
from app.edgar_source import EdgarSource, SourceUnavailable, Statement, get_source
from app.grounding import RunEvidence, StrayNumbersError, validate_bundle
from app.models import AnswerBundle, AuditRecord, Fact, Figure, ToolCallRecord
from app.tools.schemas import (
    ComputeRatioInput,
    ListStatementLinesInput,
    ListStatementLinesOutput,
    LookupFactInput,
    LookupFactOutput,
    SearchFilingTextInput,
    SearchFilingTextOutput,
)
from app.trace_log import Span, Trace
from app.usage import merge, normalize
from evals.models import AgentResult, ToolCall
from evals.pricing import Price, load_prices

AGENT_NAME = "provenance-demo-agent"

# No start-up banner on stdout: evals and the demo print only the answer.
pydantic_ai.BANNER_ENABLED = False

ENV_LIVE = "COGNITIA_LIVE"
ENV_RECORD = "COGNITIA_RECORD"
ENV_MODEL = "COGNITIA_MODEL"
ENV_FALLBACK_MODEL = "COGNITIA_FALLBACK_MODEL"
ENV_REPLAY_DIR = "COGNITIA_REPLAY_DIR"
ENV_SPEND_CAP = "COGNITIA_RUN_SPEND_CAP_USD"
ENV_TIMEOUT = "COGNITIA_RUN_TIMEOUT_S"
ENV_REQUEST_LIMIT = "COGNITIA_RUN_REQUEST_LIMIT"
ENV_TOOL_CALLS_LIMIT = "COGNITIA_RUN_TOOL_CALLS_LIMIT"
# OpenRouter key under the project's own name (the eval judge uses the same variable).
ENV_OPENROUTER_KEY_NAME = "COGNITIA_OPENROUTER_API_KEY"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPLAY_DIR = REPO_ROOT / "tests" / "fixtures" / "replay"

# Spec "Output schemas" rule 5 and "Failure modes": at most 2 output retries.
OUTPUT_RETRIES = 2
TOOL_RETRIES = 1

# Defaults are provisional ("unknown until measured", spec); step 7 measures them.
DEFAULT_REQUEST_LIMIT = 10
DEFAULT_TOOL_CALLS_LIMIT = 8
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_SPEND_CAP_USD = 0.10

NOTE_LIMIT = "Stopped at limit: the run reached its request, tool-call or spend limit."
NOTE_UNPRICED = "Stopped at limit: the answering model has no price entry, so spend is unknown."
NOTE_TIMEOUT = "Stopped at limit: the run took longer than its time limit."
NOTE_MALFORMED = "The model did not return a valid answer after its retries."
NOTE_PROVIDER = "The model provider is unavailable (the fallback model failed as well)."
NOTE_SOURCE = "Source unavailable: the filing could not be loaded."
ANSWER_STOPPED = "No answer: {note} No figures are shown."
ANSWER_EMPTY = (
    "Please provide a question about the filing, for example which figure or "
    "statement line you need."
)


class AgentConfigError(RuntimeError):
    """The live run is not configured (model id missing, or a model has no price)."""


class ReplayMissing(RuntimeError):
    """No recorded transcript for this question and prompt version."""


class ReplayMismatch(RuntimeError):
    """The run asked the recorded transcript for more model responses than it holds."""


class SpendCapExceeded(RuntimeError):
    """The run's priced spend went over its cap."""


class UnpricedModel(RuntimeError):
    """A model answered that `ops/prices.json` has no entry for: spend cannot be capped."""


@dataclass(frozen=True)
class RunConfig:
    """Per-run limits. `prices=None` turns off spend accounting (tests and replay)."""

    request_limit: int = DEFAULT_REQUEST_LIMIT
    tool_calls_limit: int = DEFAULT_TOOL_CALLS_LIMIT
    timeout_s: float = DEFAULT_TIMEOUT_S
    spend_cap_usd: float = DEFAULT_SPEND_CAP_USD
    prices: dict[str, Price] | None = None

    @classmethod
    def from_env(cls, prices: dict[str, Price] | None) -> RunConfig:
        return cls(
            request_limit=_env_int(ENV_REQUEST_LIMIT, DEFAULT_REQUEST_LIMIT),
            tool_calls_limit=_env_int(ENV_TOOL_CALLS_LIMIT, DEFAULT_TOOL_CALLS_LIMIT),
            timeout_s=_env_float(ENV_TIMEOUT, DEFAULT_TIMEOUT_S),
            spend_cap_usd=_env_float(ENV_SPEND_CAP, DEFAULT_SPEND_CAP_USD),
            prices=prices,
        )


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise AgentConfigError(f"{name} is not a number") from None
    if not value > 0:
        raise AgentConfigError(f"{name} must be greater than 0")
    return value


def _env_int(name: str, default: int) -> int:
    value = _env_float(name, float(default))
    if value != int(value):
        raise AgentConfigError(f"{name} must be a whole number")
    return int(value)


# --- dependencies and tools ----------------------------------------------------


@dataclass
class Deps:
    """What one run needs. `evidence` collects every fact and span the tools returned."""

    source: EdgarSource
    evidence: RunEvidence
    trace: Trace
    root: Span
    tool_calls: list[ToolCall] = field(default_factory=list)
    audit_calls: list[ToolCallRecord] = field(default_factory=list)


_In = TypeVar("_In", bound=BaseModel)


def _inputs(model: type[_In], **values: Any) -> _In:
    """Validate tool arguments; a bad argument asks the model to retry the call."""
    try:
        return model(**values)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise ModelRetry(f"invalid arguments ({problems})") from None


def _note_call(deps: Deps, name: str, args: BaseModel) -> None:
    payload = args.model_dump(mode="json")
    deps.tool_calls.append(ToolCall(name=name, args=payload))
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    deps.audit_calls.append(ToolCallRecord(name=name, arg_digest=digest))


def lookup_fact(
    ctx: RunContext[Deps],
    concept: str,
    period_end: date | None = None,
    dimensions: dict[str, str] | None = None,
) -> LookupFactOutput:
    args = _inputs(LookupFactInput, concept=concept, period_end=period_end, dimensions=dimensions)
    deps = ctx.deps
    _note_call(deps, "lookup_fact", args)
    with deps.trace.span("execute_tool", "lookup_fact", parent=deps.root) as span:
        out = tool_impl.lookup_fact(deps.source, args)
        span.set(results=len(out.facts))
    deps.evidence.add_facts(out.facts)
    return out


def list_statement_lines(
    ctx: RunContext[Deps],
    statement: Statement,
    period_end: date | None = None,
    offset: int = 0,
) -> ListStatementLinesOutput:
    args = _inputs(
        ListStatementLinesInput, statement=statement, period_end=period_end, offset=offset
    )
    deps = ctx.deps
    _note_call(deps, "list_statement_lines", args)
    with deps.trace.span("execute_tool", "list_statement_lines", parent=deps.root) as span:
        out = tool_impl.list_statement_lines(deps.source, args)
        span.set(results=len(out.lines))
    # A line is a fact of the filing shown in another shape: the validator needs the fact.
    for line in out.lines:
        deps.evidence.add_facts(
            f
            for f in deps.source.facts(line.tag, line.period_end)
            if f.value == line.value and f.period_start == line.period_start
        )
    return out


def search_filing_text(
    ctx: RunContext[Deps], query: str, section: str | None = None
) -> SearchFilingTextOutput:
    args = _inputs(SearchFilingTextInput, query=query, section=section)
    deps = ctx.deps
    _note_call(deps, "search_filing_text", args)
    with deps.trace.span("execute_tool", "search_filing_text", parent=deps.root) as span:
        out = tool_impl.search_filing_text(deps.source, args)
        span.set(results=len(out.spans))
    deps.evidence.add_spans(out.spans)
    return out


def compute_ratio(
    ctx: RunContext[Deps],
    numerator: Fact,
    denominator: Fact,
    kind: Literal["ratio", "change_pct"],
) -> Figure:
    args = _inputs(ComputeRatioInput, numerator=numerator, denominator=denominator, kind=kind)
    deps = ctx.deps
    _note_call(deps, "compute_ratio", args)
    with deps.trace.span("execute_tool", "compute_ratio", parent=deps.root) as span:
        figure = tool_impl.compute_ratio(deps.source, args)
        span.set(status=figure.status)
    # The inputs are not added to the evidence: a derived figure verifies only when both
    # inputs were returned by a lookup in this run (spec rule 1), not merely exist in the filing.
    return figure


def _tools() -> list[Tool[Deps]]:
    def tool(fn: Callable[..., Any]) -> Tool[Deps]:
        name = fn.__name__
        return Tool(fn, takes_ctx=True, name=name, description=prompts.TOOL_DESCRIPTIONS[name])

    tools = [tool(lookup_fact), tool(list_statement_lines), tool(search_filing_text)]
    tools.append(tool(compute_ratio))
    if tuple(t.name for t in tools) != tool_impl.TOOL_NAMES:
        raise RuntimeError("the agent must register exactly the four tools of the spec")
    return tools


# --- the agent -----------------------------------------------------------------


def _instructions(ctx: RunContext[Deps]) -> str:
    return prompts.instructions(ctx.deps.source.ref)


async def _ground(ctx: RunContext[Deps], bundle: AnswerBundle) -> AnswerBundle:
    """Output validator: rules 1 to 6 of the spec, in code (`app/grounding.py`)."""
    final_attempt = ctx.retry >= ctx.max_retries
    try:
        return validate_bundle(bundle, ctx.deps.evidence, final_attempt=final_attempt)
    except StrayNumbersError as exc:
        raise ModelRetry(str(exc)) from None


def build_agent() -> Agent[Deps, AnswerBundle]:
    agent: Agent[Deps, AnswerBundle] = Agent(
        None,
        output_type=ToolOutput(AnswerBundle),
        deps_type=Deps,
        instructions=_instructions,
        retries={"tools": TOOL_RETRIES, "output": OUTPUT_RETRIES},
        name=AGENT_NAME,
        tools=_tools(),
        defer_model_check=True,
    )
    agent.output_validator(_ground)
    return agent


AGENT = build_agent()


# --- models and prices -----------------------------------------------------------


def _build_model(model_id: str) -> Model:
    """A model from a `provider:model` id. OpenRouter also takes the project's key name."""
    if model_id.startswith("openrouter:") and not os.environ.get("OPENROUTER_API_KEY"):
        key = os.environ.get(ENV_OPENROUTER_KEY_NAME)
        if key:
            from pydantic_ai.models.openrouter import OpenRouterModel
            from pydantic_ai.providers.openrouter import OpenRouterProvider

            return OpenRouterModel(
                model_id.split(":", 1)[1], provider=OpenRouterProvider(api_key=key)
            )
    return infer_model(model_id)


def model_ids_from_env() -> list[str]:
    """Primary model id, then the fallback id if set. No id is hard-coded (plan step 5)."""
    primary = os.environ.get(ENV_MODEL, "").strip()
    if not primary:
        raise AgentConfigError(f"{ENV_MODEL} is not set; name the primary model as provider:model")
    fallback = os.environ.get(ENV_FALLBACK_MODEL, "").strip()
    return [primary, fallback] if fallback else [primary]


def build_model(model_ids: Sequence[str]) -> Model:
    models = [_build_model(i) for i in model_ids]
    return models[0] if len(models) == 1 else FallbackModel(models[0], *models[1:])


def price_for(prices: dict[str, Price], model_name: str) -> Price | None:
    """The price entry for a model id, with or without its `provider:` prefix."""
    for key in (model_name, model_name.split(":", 1)[-1]):
        if key and key in prices:
            return prices[key]
    return None


def check_priced(model_ids: Sequence[str], prices: dict[str, Price]) -> None:
    missing = [m for m in model_ids if price_for(prices, m) is None]
    if missing:
        raise AgentConfigError(
            "no price entry in ops/prices.json for: "
            + ", ".join(missing)
            + " (add it from the provider's price page with source_url and checked_on)"
        )


def _response_cost(response: ModelResponse, prices: dict[str, Price]) -> float | None:
    price = price_for(prices, response.model_name or "")
    if price is None:
        return None
    u = response.usage
    return price.cost(u.input_tokens, u.output_tokens, u.cache_read_tokens)


# --- replay ----------------------------------------------------------------------


def replay_dir() -> Path:
    return Path(os.environ.get(ENV_REPLAY_DIR) or DEFAULT_REPLAY_DIR)


def replay_key(question: str, accession: str) -> str:
    """Key of a recording: prompt version, filing and question. The question is not stored."""
    raw = f"{prompts.PROMPT_VERSION}\n{accession}\n{question.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def replay_path(question: str, accession: str, directory: Path | None = None) -> Path:
    return (directory or replay_dir()) / f"{replay_key(question, accession)}.json"


def save_replay(path: Path, responses: Sequence[ModelResponse]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "prompt_version": prompts.PROMPT_VERSION,
        "models": sorted({r.model_name or "" for r in responses}),
        "responses": json.loads(ModelMessagesTypeAdapter.dump_json(list(responses))),
    }
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def load_replay(path: Path) -> list[ModelResponse]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("prompt_version") != prompts.PROMPT_VERSION:
        raise ReplayMissing(f"{path.name} was recorded for another prompt version")
    messages = ModelMessagesTypeAdapter.validate_python(data["responses"])
    responses = [m for m in messages if isinstance(m, ModelResponse)]
    if len(responses) != len(messages) or not responses:
        raise ReplayMissing(f"{path.name} holds no model responses")
    return responses


def replay_model(responses: Sequence[ModelResponse]) -> FunctionModel:
    """A model that answers with the recorded responses in order, then fails."""
    queue = list(responses)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not queue:
            raise ReplayMismatch("the recorded transcript has no more model responses")
        return deepcopy(queue.pop(0))

    return FunctionModel(respond, model_name="replay")


# --- one run ---------------------------------------------------------------------


@dataclass
class RunOutcome:
    bundle: AnswerBundle
    tool_calls: list[ToolCall]
    responses: list[ModelResponse]
    usage: dict[str, Any]
    cost_usd: float | None
    error_type: str = ""
    audit: AuditRecord | None = None


def _stopped(note: str) -> AnswerBundle:
    return AnswerBundle(answer=ANSWER_STOPPED.format(note=note), figures=[], notes=[note])


def _usage(responses: Sequence[ModelResponse], costs: Sequence[float | None]) -> dict[str, Any]:
    calls = []
    for resp, cost in zip(responses, costs, strict=True):
        raw: dict[str, Any] = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cached_input_tokens": resp.usage.cache_read_tokens,
        }
        if cost is not None:
            raw["cost_usd"] = cost
        calls.append(normalize(raw, model=resp.model_name or ""))
    usage = merge(calls)
    usage["total_tokens"] = sum(c["input_tokens"] + c["output_tokens"] for c in calls)
    return usage


async def answer(
    question: str,
    *,
    model: Model,
    source: EdgarSource,
    config: RunConfig | None = None,
    trace: Trace | None = None,
    replay: bool = False,
) -> RunOutcome:
    """Run the agent once. Never raises for a stop the spec handles; returns a bundle."""
    config = config or RunConfig()
    trace = trace or Trace(request_id=uuid.uuid4().hex)
    evidence = RunEvidence(
        accession=source.ref.accession,
        section_texts={s: source.section_text(s) for s in source.sections()},
    )
    responses: list[ModelResponse] = []
    costs: list[float | None] = []
    error = ""
    with trace.span("invoke_agent", AGENT_NAME) as root:
        deps = Deps(source=source, evidence=evidence, trace=trace, root=root)
        limits = UsageLimits(
            request_limit=config.request_limit, tool_calls_limit=config.tool_calls_limit
        )
        try:
            async with asyncio.timeout(config.timeout_s):
                async with AGENT.iter(question, deps=deps, model=model, usage_limits=limits) as run:
                    started = time.perf_counter()
                    async for node in run:
                        if Agent.is_model_request_node(node):
                            started = time.perf_counter()
                        elif Agent.is_call_tools_node(node):
                            _on_response(node.model_response, started, deps, config, responses,
                                         costs, replay)  # fmt: skip
                    result = run.result
            assert result is not None  # noqa: S101 - iteration ended at an End node
            bundle = result.output
        except (UsageLimitExceeded, SpendCapExceeded) as exc:
            bundle, error = _stopped(NOTE_LIMIT), type(exc).__name__
        except UnpricedModel as exc:
            bundle, error = _stopped(NOTE_UNPRICED), type(exc).__name__
        except TimeoutError as exc:
            bundle, error = _stopped(NOTE_TIMEOUT), type(exc).__name__
        except UnexpectedModelBehavior as exc:
            bundle, error = _stopped(NOTE_MALFORMED), type(exc).__name__
        except (ModelAPIError, FallbackExceptionGroup) as exc:
            bundle, error = _stopped(NOTE_PROVIDER), type(exc).__name__
        if error:
            root.fail(error)
        audit = AuditRecord(
            request_id=trace.request_id or "none",
            filing_accession=source.ref.accession,
            model=model.model_name,
            prompt_version=prompts.PROMPT_VERSION,
            tool_calls=deps.audit_calls,
            figure_statuses=[f.status for f in bundle.figures],
        )
        root.set(
            mode="replay" if replay else ("live" if config.prices is not None else "test"),
            model=audit.model,
            prompt_version=audit.prompt_version,
            filing_accession=audit.filing_accession,
            tool_calls=[c.model_dump() for c in audit.tool_calls],
            figures_verified=audit.figure_statuses.count("verified"),
            figures_unverified=audit.figure_statuses.count("unverified"),
            requests=len(responses),
        )
    priced = [c for c in costs if c is not None]
    return RunOutcome(
        bundle=bundle,
        tool_calls=deps.tool_calls,
        responses=responses,
        usage=_usage(responses, costs),
        cost_usd=sum(priced) if priced or replay else None,
        error_type=error,
        audit=audit,
    )


def _on_response(
    response: ModelResponse,
    started: float,
    deps: Deps,
    config: RunConfig,
    responses: list[ModelResponse],
    costs: list[float | None],
    replay: bool,
) -> None:
    """Record one model response; stop the run when its spend goes over the cap."""
    responses.append(response)
    ms = round((time.perf_counter() - started) * 1000)
    tokens = (response.usage.input_tokens, response.usage.output_tokens)
    deps.trace.record("chat", response.model_name or "", parent=deps.root, ms=ms, tokens=tokens)
    if replay:
        costs.append(0.0)  # a replayed response costs nothing
        return
    if config.prices is None:
        costs.append(None)
        return
    cost = _response_cost(response, config.prices)
    costs.append(cost)
    if cost is None:
        raise UnpricedModel(response.model_name or "")
    if sum(c or 0.0 for c in costs) > config.spend_cap_usd:
        raise SpendCapExceeded(f"spend over {config.spend_cap_usd} USD")


# --- output for evals and the UI ---------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def bundle_json(bundle: AnswerBundle) -> str:
    """The bundle as JSON with numbers as numbers (evals compare `figures.N.value`)."""
    return json.dumps(bundle.model_dump(mode="python"), default=_json_default, ensure_ascii=False)


def empty_question_bundle() -> AnswerBundle:
    return AnswerBundle(answer=ANSWER_EMPTY, figures=[], notes=[])


async def arun_agent(text: str) -> AgentResult:
    """`run_agent` for callers that already run an event loop."""
    if not text.strip():
        bundle = empty_question_bundle()  # nothing to ask the model; no tokens spent
        return AgentResult(output=bundle_json(bundle), usage=_usage([], []))
    live = _truthy(os.environ.get(ENV_LIVE))
    try:
        source = get_source() if live else get_source(offline=True)
    except SourceUnavailable:
        return AgentResult(output=bundle_json(_stopped(NOTE_SOURCE)), usage=_usage([], []))
    if live:
        ids = model_ids_from_env()
        prices = load_prices()
        check_priced(ids, prices)
        outcome = await answer(
            text, model=build_model(ids), source=source, config=RunConfig.from_env(prices)
        )
        if _truthy(os.environ.get(ENV_RECORD)) and not outcome.error_type:
            save_replay(replay_path(text, source.ref.accession), outcome.responses)
    else:
        path = replay_path(text, source.ref.accession)
        if not path.is_file():
            raise ReplayMissing(
                f"no recorded run for this question (prompt {prompts.PROMPT_VERSION}) in "
                f"{path.parent}; record one with {ENV_LIVE}=1 {ENV_RECORD}=1"
            )
        outcome = await answer(
            text,
            model=replay_model(load_replay(path)),
            source=source,
            config=RunConfig.from_env(None),
            replay=True,
        )
    return AgentResult(
        output=bundle_json(outcome.bundle), tool_calls=outcome.tool_calls, usage=outcome.usage
    )


def run_agent(text: str) -> AgentResult:
    """Entry point for evals (`evals/run_evals.py`): the `AnswerBundle` JSON, tool calls, usage."""
    return asyncio.run(arun_agent(text))
