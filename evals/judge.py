"""LLM judge. Grades one output against a plain-language rubric.

Two providers, chosen by `provider` in evals/judge_config.json:
  openrouter (default)  OpenAI-compatible API at openrouter.ai; key in COGNITIA_OPENROUTER_API_KEY
                        (falls back to OPENROUTER_API_KEY). Sends the `openrouter` block as
                        `provider`: zdr (zero-retention endpoints only), data_collection deny
                        and max_price (no endpoint dearer than the ledger rate `usd_per_mtok`).
  anthropic             needs the `anthropic` package and COGNITIA_ANTHROPIC_API_KEY (falls back to
                        ANTHROPIC_API_KEY, which cloud sessions do not pass through).
Judge prompts contain the eval case input and the agent output. Use the `anthropic` provider
(kickoff --judge anthropic) when the cases hold real client data; see evals/README.md.
The judge is pinned in evals/judge_config.json (exact model id, temperature 0): the same output
gets the same verdict from run to run, and changing the judge is a reviewed change to that file, not
a repository variable. Use a different model from the agent's where possible so the judge does not
grade its own habits. Check the id and price against the provider's model page when you change it.
Keys are read from the environment only. Keep the signature (rubric, input, output) -> Grade.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from evals import spend
from evals.models import Grade

CONFIG = Path(__file__).resolve().parent / "judge_config.json"

PROMPT = """You grade an AI agent's answer against a rubric. Be strict.
<rubric>{rubric}</rubric>
<user_input>{case_input}</user_input>
<agent_output>{output}</agent_output>
Reply with only JSON: {{"pass": true or false, "reason": "one sentence"}}"""


KEY_VARS = {
    "anthropic": ("COGNITIA_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    "openrouter": ("COGNITIA_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"),
}
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    """The pinned judge settings. Raises ValueError when they are missing or not deterministic."""
    cfg: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("provider", "anthropic") not in KEY_VARS:
        raise ValueError(f"{path.name}: `provider` must be one of {sorted(KEY_VARS)}")
    model = cfg.get("model")
    if not isinstance(model, str) or not model.strip() or "TBD" in model:
        raise ValueError(f"{path.name}: set `model` to an exact model id")
    if cfg.get("temperature") != 0:
        raise ValueError(f"{path.name}: `temperature` must be 0 for a repeatable judge")
    return cfg


def key_name(cfg: dict[str, Any]) -> str:
    """The environment variable the configured provider's key is expected in."""
    return KEY_VARS[cfg.get("provider", "anthropic")][0]


def api_key(cfg: dict[str, Any] | None = None) -> str | None:
    provider = (cfg or {}).get("provider", "anthropic")
    return next((v for name in KEY_VARS[provider] if (v := os.environ.get(name))), None)


def parse_verdict(text: str) -> Grade:
    try:
        verdict = json.loads(text[text.index("{") : text.rindex("}") + 1])
        return Grade(bool(verdict["pass"]), str(verdict.get("reason", "")))
    except (ValueError, KeyError):
        return Grade(False, f"judge returned unparseable reply: {text[:80]!r}")


def judge(rubric: str, case_input: str, output: str) -> Grade:
    """Grade with the provider named in judge_config.json."""
    try:
        cfg = load_config()
    except (OSError, ValueError) as exc:
        return Grade(False, f"judge config: {exc}")
    key = api_key(cfg)
    if not key:
        return Grade(False, f"{key_name(cfg)} is not set")
    prompt = PROMPT.format(rubric=rubric, case_input=case_input, output=output)
    if cfg.get("provider") == "openrouter":
        return openrouter_judge(cfg, key, prompt)
    return anthropic_judge(cfg, key, prompt)


def openrouter_judge(cfg: dict[str, Any], key: str, prompt: str) -> Grade:
    body: dict[str, Any] = {
        "model": cfg["model"],
        "max_tokens": int(cfg.get("max_tokens", 300)),
        "temperature": cfg["temperature"],
        "messages": [{"role": "user", "content": prompt}],
    }
    routing = cfg.get("openrouter")
    if isinstance(routing, dict) and routing:
        body["provider"] = routing  # e.g. {"data_collection": "deny", "zdr": true}
    req = urllib.request.Request(  # noqa: S310  (fixed https URL)
        OPENROUTER_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return Grade(False, f"openrouter HTTP {exc.code}: {exc.read()[:120]!r}")
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return Grade(False, f"openrouter request failed: {exc}")
    usage = data.get("usage") or {}
    spend.add_judge(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), cfg)
    try:
        text = str(data["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        return Grade(False, f"openrouter reply without choices: {str(data)[:80]!r}")
    return parse_verdict(text)


def anthropic_judge(cfg: dict[str, Any], key: str, prompt: str) -> Grade:
    import anthropic  # imported here so the rest of the framework works without it

    # Explicit base_url: a cloud session's own ANTHROPIC_BASE_URL must not redirect the judge.
    client = anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com")
    params: dict[str, Any] = {
        "model": cfg["model"],
        "max_tokens": int(cfg.get("max_tokens", 300)),
        "temperature": cfg["temperature"],
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        msg = client.messages.create(**params)
    except anthropic.BadRequestError as exc:
        if "temperature" not in str(exc).lower():
            raise
        # This model takes no temperature: retry without it (the pinned id still fixes the judge).
        del params["temperature"]
        msg = client.messages.create(**params)
    spend.add_judge(msg.usage.input_tokens, msg.usage.output_tokens, cfg)
    return parse_verdict("".join(getattr(block, "text", "") for block in msg.content))
