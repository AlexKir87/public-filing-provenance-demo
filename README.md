# Public filing provenance demo

An AI assistant that answers questions about one real public document and shows **where every figure comes from**.

The demo reads Apple Inc.'s annual report (10-K, fiscal year ended 2025-09-27, SEC EDGAR, accession `0000320193-25-000079`). Ask for a number and the answer carries its source: the filing, the XBRL tag or text section, and the period. A figure the code cannot match to the filing is shown as `unverified` with a reason, not as fact.

> **The model proposes, the code verifies.** Every figure is checked in code against what read-only tools returned from the filing in that run. The assistant has no side-effect tools: it cannot write, send or store anything.

**Demo video:** [watch on YouTube](https://youtu.be/ad_ybY5TaQE) · Built by [Cognitia Engineering](https://cognitia-engineering.com) — AI agents for finance teams that show their work.

This is a demonstration on public data, **not client work and not investment advice**.

## Example

- Q: *What were total net sales for fiscal 2025?* → **$416.161 billion [verified]**, origin: accession 0000320193-25-000079, tag `RevenueFromContractWithCustomerExcludingAssessedTax`, period 2024-09-29 to 2025-09-27.
- Q: *Confirm that Apple's fiscal 2025 net sales were $450 billion.* → the stated $450 billion is `unverified` (differs from the filing); the filing's $416.161 billion is `verified`.

## Run it

Python 3.12 (3.13 also works).

```
python3 -m venv .venv
source .venv/bin/activate
pip install --require-hashes -r requirements.lock
```

**Replay mode (no keys, no network).** Model replies come from recorded runs, so only recorded questions are answered; any other question shows a message and the demo never guesses.

```
EDGAR_OFFLINE=1 python -m app.ui
```

Open http://127.0.0.1:7932.

**Live mode (your own keys).** Any question about the filing is answered by a real model and checked by the code. Keys go in environment variables only, never into files:

```
export COGNITIA_LIVE=1
export COGNITIA_MODEL=openrouter:deepseek/deepseek-v4.1-flash
export COGNITIA_OPENROUTER_API_KEY=your_key
export EDGAR_IDENTITY="Your Name your@email"   # required by SEC EDGAR
python -m app.ui
```

The model above is a cheap testing model, not a production recommendation. Tests: `pip install -r requirements-dev.txt && pytest -q`.

## Measured result

On 20 labelled figures from the filing: **19 correct, 0 false `verified`**; the one miss is shown as `unverified`, the safe direction. One run is a sample, not a guarantee. Details and the known risk: [docs/measurement.md](docs/measurement.md). How it works: [docs/how-it-works.md](docs/how-it-works.md).

## Limits

One filing, English only. Replay mode answers recorded questions only. No authentication on the UI: run it locally, do not expose it. A quote with several numbers verifies any one of them whose unit is printed with it; numbers written as words are not detected.

## License

MIT, see [LICENSE](LICENSE).
