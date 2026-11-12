"""Step 8: the demo UI shows status and origin per figure and the notice, and nothing else.

The app is the real `Agent.to_web()` Starlette app; its chat endpoint is called through
Starlette's test client. The pipeline runs for real on the offline filing snapshot with a
scripted model (no network, no spend).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from starlette.testclient import TestClient

from app import agent as ag
from app import ui
from app.edgar_source import EdgarSource, get_source
from app.models import AnswerBundle
from evals.pricing import Price

ACCESSION = "0000320193-25-000079"
FY_START, FY_END = "2024-09-29", "2025-09-27"
REVENUE = "RevenueFromContractWithCustomerExcludingAssessedTax"
REVENUE_VALUE = 416161000000
HTML = "<html><body>chat ui</body></html>"


def _figure(label: str, value: int, claimed_tag: str = REVENUE) -> dict[str, Any]:
    return {
        "label": label,
        "value": value,
        "unit": "USD",
        "status": "verified",  # the model's claim; code decides
        "claimed_by": "agent",
        "provenance": {
            "kind": "xbrl",
            "accession": ACCESSION,
            "statement_or_section": "Income statement",
            "tag": claimed_tag,
            "period_start": FY_START,
            "period_end": FY_END,
        },
    }


class _Script:
    def __init__(self, *steps: Any) -> None:
        self.steps, self.calls = list(steps), 0
        self.model = FunctionModel(self._respond, model_name="scripted")

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        response: ModelResponse = step(info)
        return response


def _call_lookup(info: AgentInfo) -> ModelResponse:
    args = {"concept": f"us-gaap:{REVENUE}", "period_end": FY_END}
    return ModelResponse(parts=[ToolCallPart("lookup_fact", args)])


def _final(answer: str, figures: list[dict[str, Any]]) -> Callable[[AgentInfo], ModelResponse]:
    def step(info: AgentInfo) -> ModelResponse:
        args = {"answer": answer, "figures": figures, "notes": []}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return step


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("COGNITIA_TRACE_LOG", str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv(ag.ENV_REPLAY_DIR, str(tmp_path / "replay"))
    monkeypatch.setenv("EDGAR_OFFLINE", "1")
    for name in (ag.ENV_LIVE, ag.ENV_RECORD, ag.ENV_MODEL, ag.ENV_FALLBACK_MODEL,
                 ui.ENV_ALLOWED_HOSTS, ui.ENV_HTML_SOURCE):  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture(scope="module")
def src() -> EdgarSource:
    return get_source(offline=True)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    page = tmp_path / "ui.html"
    page.write_text(HTML)
    return TestClient(ui.build_app(html_source=page), base_url="http://localhost")


def _live(monkeypatch: pytest.MonkeyPatch, script: _Script, src: EdgarSource) -> None:
    monkeypatch.setenv(ag.ENV_LIVE, "1")
    monkeypatch.setenv(ag.ENV_MODEL, "test:scripted")
    monkeypatch.setattr(ag, "get_source", lambda *a, **k: src)
    monkeypatch.setattr(ag, "build_model", lambda ids: script.model)
    prices = {
        m: Price(m, 1.0, 1.0, None, "https://example.com/prices", "2026-11-11")
        for m in ("test:scripted", "scripted")
    }
    monkeypatch.setattr(ag, "load_prices", lambda: prices)


def _chat(client: TestClient, text: str, headers: dict[str, str] | None = None) -> Any:
    body = {
        "trigger": "submit-message",
        "id": "chat-1",
        "messages": [
            {"id": "m1", "role": "user", "parts": [{"type": "text", "text": text}]},
        ],
    }
    return client.post("/api/chat", json=body, headers=headers or {})


def _shown_text(response: Any) -> str:
    """All text deltas of the Vercel AI data stream joined."""
    out: list[str] = []
    for line in response.text.splitlines():
        if line.startswith("data: {"):
            event = json.loads(line[len("data: ") :])
            if event.get("type") == "text-delta":
                out.append(event["delta"])
    return "".join(out)


# --- the page and the host allow-list ---------------------------------------------


def test_app_serves_the_chat_page_and_health(client: TestClient) -> None:
    assert client.get("/").text == HTML
    assert client.get("/api/health").json() == {"ok": True}


def test_unlisted_host_name_is_refused_and_listed_one_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = tmp_path / "ui.html"
    page.write_text(HTML)
    refused = TestClient(ui.build_app(html_source=page), base_url="http://evil.example")
    assert refused.get("/api/health").status_code == 421
    monkeypatch.setenv(ui.ENV_ALLOWED_HOSTS, "demo.example.com, ")
    assert ui.allowed_hosts_from_env() == ["demo.example.com"]
    allowed = TestClient(ui.build_app(html_source=page), base_url="http://demo.example.com")
    assert allowed.get("/api/health").status_code == 200


# --- what a figure looks like ------------------------------------------------------


def test_chat_shows_status_and_origin_per_figure_and_the_notice(
    client: TestClient, src: EdgarSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = _figure("Total net sales", REVENUE_VALUE)
    wrong = _figure("Planted wrong claim", REVENUE_VALUE + 7)
    script = _Script(
        _call_lookup,
        _final("Net sales were {fig:0}; a planted claim says {fig:1}.", [good, wrong]),
    )
    _live(monkeypatch, script, src)
    text = _shown_text(_chat(client, "What were total net sales for fiscal 2025?"))

    assert "[1, verified]" in text and "[2, unverified]" in text
    figure_lines = [x for x in text.split("Figures:")[1].splitlines() if x.startswith("[")]
    assert len(figure_lines) == 2
    first, second = figure_lines
    assert ", verified. Origin: accession " + ACCESSION in first
    assert f"tag {REVENUE}" in first and f"period {FY_START} to {FY_END}" in first
    assert ", unverified (" in second and "Not matched to the filing." in second
    assert "Origin:" not in second
    assert ui.NOTICE in text


def test_ui_never_upgrades_a_status_the_pipeline_did_not_give(src: EdgarSource) -> None:
    bundle = AnswerBundle(answer="Nothing grounded. [1, unverified]", figures=[], notes=["n"])
    text = ui.render_reply(bundle)
    assert "unverified" in text and "[1, verified]" not in text
    assert text.count(ui.NOTICE) == 1 and f"{ui.NOTE_PREFIX}n" in text


# --- stops are shown, not guessed ----------------------------------------------------


def test_question_without_recording_shows_message_and_no_figures(client: TestClient) -> None:
    text = _shown_text(_chat(client, "A question nobody recorded"))
    assert ui.MSG_NO_RECORDING in text and ui.NOTICE in text
    assert "Figures:" not in text


def test_empty_question_gets_a_prompt_to_ask(client: TestClient) -> None:
    text = _shown_text(_chat(client, "   "))
    assert ui.MSG_NO_QUESTION in text and ui.NOTICE in text


def test_live_config_error_names_only_the_error_class(
    client: TestClient, src: EdgarSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    _live(monkeypatch, _Script(_call_lookup), src)
    monkeypatch.setattr(ag, "load_prices", lambda: {})
    text = _shown_text(_chat(client, "q"))
    assert "AgentConfigError" in text and ui.NOTICE in text


def test_only_the_last_user_message_is_answered() -> None:
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="first")]),
        ModelResponse(parts=[TextPart("earlier answer")]),
        ModelRequest(parts=[UserPromptPart(content="second")]),
    ]
    assert ui.last_user_text(history) == "second"
    assert ui.last_user_text([]) == ""


def test_ui_has_no_side_effect_tools_or_approval_actions() -> None:
    agent = ui.build_ui_agent()
    assert agent._function_toolset.tools == {}


def test_any_pipeline_failure_still_shows_the_notice_and_only_the_class_name(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(text: str) -> Any:
        raise ag.SpendCapExceeded("secret detail")

    monkeypatch.setattr(ag, "arun_agent", boom)
    text = _shown_text(_chat(client, "q"))
    assert "SpendCapExceeded" in text and "secret detail" not in text and ui.NOTICE in text
