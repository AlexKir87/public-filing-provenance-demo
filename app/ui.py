"""Demo web UI: `Agent.to_web()` over a thin text agent (spec.md, plan.md step 8).

The framework's chat UI shows plain text, so the demo shows what the validator already
rendered: the answer with `[N, verified|unverified]` marks and one origin line per figure
(accession, statement or section, tag and period, or the quote). The UI agent's "model" is
a function that runs the real pipeline (`app.agent.arun_agent`: replay by default, live
with `COGNITIA_LIVE=1`) and returns that text plus the notice. It never changes a status,
never adds a figure and has no "mark as verified" action. The UI keeps no conversation
state: only the last user message is answered.

Run: `python -m app.ui` (host `COGNITIA_UI_HOST`, default 127.0.0.1, port `COGNITIA_UI_PORT`,
default 7932). Behind a host name, list it in `COGNITIA_UI_ALLOWED_HOSTS` (comma separated).
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from starlette.applications import Starlette

from app import agent as pipeline
from app.models import AnswerBundle

UI_AGENT_NAME = "provenance-demo-ui"
ENV_HOST = "COGNITIA_UI_HOST"
ENV_PORT = "COGNITIA_UI_PORT"
ENV_ALLOWED_HOSTS = "COGNITIA_UI_ALLOWED_HOSTS"
ENV_HTML_SOURCE = "COGNITIA_UI_HTML_SOURCE"

NOTICE = (
    "Not investment advice. Figures are taken from the SEC filing named in each origin line. "
    "`verified` means the code matched the figure to that filing; `unverified` means it did "
    "not, so do not rely on it."
)
NOTE_PREFIX = "Note: "
MSG_NO_RECORDING = (
    "No recorded demo run exists for this question, so no answer is shown (the demo never "
    "guesses). Ask one of the prepared demo questions, or run the demo live."
)
MSG_NOT_CONFIGURED = "The demo is not configured to answer this question: {why}"
MSG_NO_QUESTION = "Ask a question about the filing."


def last_user_text(messages: list[ModelMessage]) -> str:
    """The text of the latest user message (earlier turns are ignored: no state)."""
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in reversed(message.parts):
                if isinstance(part, UserPromptPart):
                    content = part.content
                    if isinstance(content, str):
                        return content
                    return " ".join(c for c in content if isinstance(c, str))
    return ""


def render_reply(bundle: AnswerBundle) -> str:
    """The bundle as chat text: answer (already with marks and origin lines), notes, notice."""
    lines = [bundle.answer.strip()]
    lines.extend(f"{NOTE_PREFIX}{n}" for n in bundle.notes)
    lines.append("---\n" + NOTICE)
    return "\n\n".join(lines)


def notice_only(message: str) -> str:
    return f"{message}\n\n---\n{NOTICE}"


async def reply_for(question: str) -> str:
    """Run the pipeline for one question and return the text to show."""
    if not question.strip():
        return notice_only(MSG_NO_QUESTION)
    try:
        result = await pipeline.arun_agent(question)
    except pipeline.ReplayMissing:
        return notice_only(MSG_NO_RECORDING)
    except Exception as exc:  # noqa: BLE001 - last boundary: show the class name, never the text
        return notice_only(MSG_NOT_CONFIGURED.format(why=type(exc).__name__))
    return render_reply(AnswerBundle.model_validate(json.loads(result.output)))


async def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(await reply_for(last_user_text(messages)))])


async def _stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield await reply_for(last_user_text(messages))


def build_ui_agent() -> Agent[None, str]:
    model = FunctionModel(_respond, stream_function=_stream, model_name="demo-pipeline")
    return Agent(model, output_type=str, name=UI_AGENT_NAME)


def allowed_hosts_from_env() -> list[str]:
    raw = os.environ.get(ENV_ALLOWED_HOSTS, "")
    return [h.strip() for h in raw.split(",") if h.strip()]


def build_app(*, html_source: str | Path | None = None) -> Starlette:
    """The Starlette app. IP addresses and localhost are always allowed; other host names
    must be listed in `COGNITIA_UI_ALLOWED_HOSTS` (never `*` unless something in front of
    the app authenticates)."""
    source = html_source if html_source is not None else os.environ.get(ENV_HTML_SOURCE)
    return build_ui_agent().to_web(allowed_hosts=allowed_hosts_from_env(), html_source=source)


def main() -> None:  # pragma: no cover - starts a server
    import uvicorn

    host = os.environ.get(ENV_HOST, "127.0.0.1")
    port = int(os.environ.get(ENV_PORT, "7932"))
    uvicorn.run(build_app(), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
