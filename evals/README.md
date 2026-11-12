# Evals

Run: `python evals/run_evals.py --threshold 0.9 --report evals/report.json`

Cases live in `cases.jsonl` (see `cases.example.jsonl` for every check type). Each case has a `category` (`happy`, `edge`, `adversarial`, `out_of_scope`; all four are required) and a list of `checks`.

| Check | Use it for |
| :--- | :--- |
| `contains`, `not_contains`, `regex`, `not_regex` | Simple text expectations and forbidden content |
| `json_field` | Extracted fields and numbers, with an optional numeric `tolerance`. The agent's output must be JSON |
| `tool_calls`, `tool_arg` | Which tools were called, in what order, with what arguments, and which were never allowed |
| `max_tokens` | Cost ceiling per case (needs `usage.total_tokens`) |
| `llm_judge` | Qualities text matching cannot see, such as refusing correctly. Judge pinned in `judge_config.json`; default: DeepSeek V4.1 Flash via OpenRouter, key `COGNITIA_OPENROUTER_API_KEY` (provider `anthropic` needs `COGNITIA_ANTHROPIC_API_KEY`) |

Tokens and cost: the agent's `AgentResult.usage` is `{"model", "input_tokens", "output_tokens", "cached_input_tokens", "cost_usd"?}` or `{"calls": [...]}` for several calls (helpers in `app/usage.py`). With `--ledger`, each run appends one line per source (agent, judge) and model to `ops/spend.jsonl`: tokens, calls, requests and USD. USD comes from the agent's `cost_usd`, else from `ops/prices.json` (model, `source_url`, `checked_on`; empty at kickoff, filled only from the provider's page), the judge from `usd_per_mtok`. No price means no ledger line and an n/a spend check. `python scripts/build_report.py tokenomics` turns the ledger and the production log `ops/usage.jsonl` (`app/usage_log.py`) into the client's table and forecast.

Traces: `--traces-dir evals/traces` writes one step trace per case (`evals/traces/<case>.jsonl`: the agent run, each tool call, each model call, with inputs, outputs and tool arguments; `--no-trace-content` drops them) and adds `trace_id` and `steps` to every case in the report. CI keeps the folder as artifact `eval-traces` for 14 days and prints the failed cases with the tools the agent called into the job summary (`python scripts/build_report.py failures`). Production errors: `python scripts/build_report.py errors`. See `docs/observability.md`.

To check tool calls or usage, `run_agent` returns an `AgentResult` (see `evals/models.py`) instead of a string. Map the framework's message history to `tool_calls`.

Quality gates (CI)
- `python evals/sanity.py`: a stub agent that returns nothing and one that echoes the input must each fail at least 90% of the cases. Cases a stub passes are listed: add a positive check (`contains`, `json_field`, `tool_calls`). Judge-only cases cannot be judged offline and count as undetermined (at least half the cases must be determined). No keys needed.
- `evals/report.baseline.json` is the golden report: `python evals/run_evals.py --update-baseline` after a green full run, committed by the owner. CI compares with the copy on the base branch and fails when the pass rate drops, the number of cases falls, or a baseline case disappears.
- The judge is pinned in `judge_config.json` (exact model id, `temperature` 0; CI runs `--repeats 3`). Without the judge key for the configured provider a run with `llm_judge` checks ends `INCOMPLETE` (exit 2) with a clear message instead of a stack trace, and the `gate` fails: a skipped job is not a pass. `--skip-judge` gives a marked partial run (`mode: no-judge` in the report) that cannot become the baseline.

Practical rules
- Prefer exact checks (`json_field`, `tool_calls`) over `llm_judge` wherever the answer can be checked exactly. Judge only what you must.
- Use `--repeats 3` for agents whose output varies; a case passes on a strict majority.
- Every production incident becomes a new case. Never edit a case to make a failing run pass; the fix-task hook blocks that.
- Data sent to the judge: every `llm_judge` call carries the case input and the agent output to the judge provider (OpenRouter, then a model host). `judge_config.json` asks OpenRouter to route only to zero-retention endpoints (`"zdr": true`) that also do not train on prompts (`data_collection: deny`), and caps the endpoint price at the ledger rate (`max_price` = `usd_per_mtok`). If no endpoint matches, the call fails and the run is `INCOMPLETE`; it never falls back to a retaining endpoint. Eval cases should be synthetic or anonymised. If they must contain real client data, the owner decides at kickoff to use `--judge anthropic` or a contract-approved provider.
- `usd_per_mtok` in `judge_config.json` prices judge calls for the spend ledger. It is the highest provider rate listed on the model page, so the ledger overstates rather than understates. Re-check it on the page when you change the model.
- Keep a hand-checked sample of judge verdicts to make sure the judge agrees with you.
