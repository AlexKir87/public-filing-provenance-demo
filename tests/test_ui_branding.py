"""The default page is the Cognitia-branded shell; its brand files are served by the app itself."""

from __future__ import annotations

import re

import pytest
from starlette.testclient import TestClient

from app import ui


@pytest.fixture
def client() -> TestClient:
    return TestClient(ui.build_app(), base_url="http://localhost")


def test_default_page_is_the_branded_shell(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "<title>Cognitia Engineering" in page.text
    assert "Pydantic AI</title>" not in page.text
    assert "/brand/favicon.svg" in page.text and "/brand/brand.css" in page.text


def test_brand_script_uses_the_mark_and_the_demo_notice(client: TestClient) -> None:
    script = client.get("/brand/brand.js")
    assert script.status_code == 200
    assert "/brand/cognitia-mark_v1.svg" in script.text
    assert "Cognitia Engineering" in script.text
    assert "not investment advice" in script.text
    for path in ("/brand/cognitia-mark_v1.svg", "/brand/cognitia-mark-on-dark_v1.svg"):
        assert client.get(path).status_code == 200


def test_fonts_and_logos_are_local(client: TestClient) -> None:
    for path in ("/brand/fonts/Geist-Variable.ttf", "/brand/fonts/GeistMono-Variable.ttf"):
        assert client.get(path).status_code == 200
    css = client.get("/brand/brand.css").text
    assert not re.search(r"https?://", css)
    page = client.get("/").text
    hosts = set(re.findall(r"https?://([^/\"']+)", page))
    assert hosts <= {"cdn.jsdelivr.net"}  # only the stock chat bundle stays on the CDN


def test_html_source_env_still_overrides_the_default(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = tmp_path / "x.html"
    page.write_text("<html><title>custom</title></html>")
    monkeypatch.setenv(ui.ENV_HTML_SOURCE, str(page))
    custom = TestClient(ui.build_app(), base_url="http://localhost")
    assert "custom" in custom.get("/").text
