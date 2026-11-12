# Measurement on 20 labelled figures

Filing: Apple Inc. 10-K, accession `0000320193-25-000079`. Labelled set: `tests/grounding/labels.jsonl` (20 figures: 12 from statements, 3 dimensional, 3 narrative, 2 planted wrong claims). Values were cross-checked against the SEC filing.

Live run with the model `deepseek/deepseek-v4.1-flash` through OpenRouter (a cheap testing model, not a production recommendation):

```
$ python tests/grounding/score.py
correct: 19 of 20
false verified: 0
incorrect: g-txt-03
```

The scorer is strict: any extra `verified` figure that is not the filing's value with the expected provenance counts as a false `verified`.

**The one miss (g-txt-03):** the question asks about *voting* stock; the filing's cover page says "voting and non-voting stock". The agent found the sentence but the figure was shown as `unverified` with a note on the wording difference. That is the safe direction: a correct figure shown as unverified.

**Risk to know:** the code checks a figure against the filing's text span, not against the meaning of the question. In another run a figure with a small wording difference could be shown `verified` with only a note flagging it.

Earlier runs on the way to 19 of 20 (17, then 18) are in the git history of the working repository, not here. One 20-figure run is a sample, not a guarantee; model output varies between runs. Cost of one grounding run: about $0.006 on the testing model.

Reproduce: `COGNITIA_LIVE=1 COGNITIA_RECORD=1 python tests/grounding/run_live.py`, then `python tests/grounding/score.py` (the scorer also runs on the recorded run offline).
