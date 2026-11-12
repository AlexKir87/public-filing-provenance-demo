# How it works

The demo answers questions about one public filing: Apple Inc. 10-K, accession `0000320193-25-000079`, fiscal year ended 2025-09-27 (SEC EDGAR).

**Principle: the model proposes, the code verifies.** The model chooses read-only tools; each tool returns typed facts (XBRL) or text spans built by code. The model returns an answer with figures; an output validator then checks every figure against what the tools returned in that same run. A figure it cannot match is shown as `unverified` with a reason, never as fact. A stray number in the answer text is rejected and retried, then masked.

| Part | What it does | Code |
|---|---|---|
| Agent | One Pydantic AI agent with typed output, run limits, optional fallback model, replay of recorded runs | `app/agent.py` |
| Models | Typed `Fact`, `TextSpan`, `Provenance`, `Figure`, `AnswerBundle` | `app/models.py` |
| EDGAR source | The fixed filing: XBRL facts and filing text (EdgarTools), local cache, committed snapshot as a fallback | `app/edgar_source.py` |
| Tools | `lookup_fact`, `list_statement_lines`, `search_filing_text`, `compute_ratio` (all read-only) | `app/tools/` |
| Grounding | Matches figures to facts and text spans, sets status and reason, renders one origin line per figure | `app/grounding.py` |
| Prompts | Instructions and tool descriptions | `app/prompts/` |
| Demo UI | Chat page; every reply carries the origin line and a "not investment advice" notice | `app/ui.py` |

There are no side-effect tools: the agent cannot write, send or store anything. No conversation memory (last message only).

## Two modes
- **Replay (default offline):** `EDGAR_OFFLINE=1 python -m app.ui`. Model replies come from recorded runs in `tests/fixtures/replay/`, so it runs without keys or network, but only recorded questions are answered. Any other question shows a message; the demo never guesses.
- **Live:** set `COGNITIA_LIVE=1`, a model id in `COGNITIA_MODEL`, the provider key in an environment variable, and `EDGAR_IDENTITY` (name and email, required by SEC). Any question about the filing is answered by the real model and checked by the code. Keys are read from environment variables only; never put them in a file.

## Limits
One filing, English only. A quote with several numbers verifies any one of them whose unit is printed with it; numbers written as words are not detected. A derived ratio is shown with the unit `pure`. No click-through to the position in the filing, no authentication on the UI: run it locally. Not investment advice.
