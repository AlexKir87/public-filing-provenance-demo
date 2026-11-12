# Replay fixtures

Recorded live model responses, one JSON file per question, written by `app/agent.py` when a live
run has `COGNITIA_LIVE=1 COGNITIA_RECORD=1` (step 9 and step 10). Without `COGNITIA_LIVE=1` the
agent replays them: the recorded responses play the model, while the tools, the filing snapshot
and the grounding validator run for real. No network, no spend.

- File name: the first 24 hex digits of sha256(prompt version, accession, question). The question
  text itself is not stored.
- A changed prompt (`app/prompts/`) changes the key, so old recordings are not used; re-record.
- A question without a recording, or a transcript that ends early, is a hard failure.
- Passing replay proves plumbing only. Accuracy comes from a fresh live run.
