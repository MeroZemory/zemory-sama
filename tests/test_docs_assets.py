"""Documentation asset integrity tests."""

from __future__ import annotations

import re
import tomllib
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL_REPORT = ROOT / "docs" / "reports" / "2026-06-28-final-comparison"
REPO_NAME = "realtime-voice-runtime"


class _ImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "img":
            return
        values = dict(attrs)
        src = values.get("src")
        if src:
            self.sources.append(src)


def _markdown_image_targets(markdown: str) -> list[str]:
    return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown)


def test_root_readme_does_not_embed_dense_benchmark_svgs() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    embedded_benchmark_svgs = [
        target
        for target in _markdown_image_targets(readme)
        if target.startswith("docs/benchmarks/") and target.endswith(".svg")
    ]

    assert embedded_benchmark_svgs == []
    assert "docs/reports/2026-06-28-final-comparison" in readme


def test_public_repo_name_is_consistent() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert readme.startswith(f"# {REPO_NAME}\n")
    assert pyproject["project"]["name"] == REPO_NAME
    old_url = "github.com/MeroZemory/" + "zemory" + "-sama"
    assert old_url not in readme


def test_final_report_image_sources_exist_and_are_valid_svg() -> None:
    html = (FINAL_REPORT / "index.html").read_text(encoding="utf-8")
    parser = _ImageParser()
    parser.feed(html)

    assert parser.sources
    for src in parser.sources:
        assert not src.startswith(("http://", "https://"))
        target = FINAL_REPORT / src
        assert target.exists(), src
        root = ET.parse(target).getroot()
        assert root.tag.endswith("svg"), src
        assert root.attrib.get("viewBox") or (
            root.attrib.get("width") and root.attrib.get("height")
        ), src


def test_final_report_keeps_external_latency_claims_bounded() -> None:
    html = (FINAL_REPORT / "index.html").read_text(encoding="utf-8")

    for repo in [
        "AIRI",
        "Open-LLM-VTuber",
        "LiveKit Agents",
        "RealtimeVoiceChat",
        "Neuro",
        "AI-Waifu-Vtuber",
        "AIRIS-VtuberAI",
    ]:
        assert repo in html

    assert "외부 repo 대비 “우리가 더 빠르다”는 직접 결론은 내리지 않는다" in html
    assert "warm-cache" in html
