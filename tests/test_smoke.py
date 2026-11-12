import importlib


def test_agent_entrypoint_exists() -> None:
    assert callable(importlib.import_module("app.agent").run_agent)
