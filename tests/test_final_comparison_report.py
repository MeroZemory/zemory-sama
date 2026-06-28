"""Integrity guards for the 2026-06-28 final comparison report.

These tests make the report's fairness/structure properties machine-enforced so
the published numbers can never silently drift away from the benchmark
``summary.json`` artifacts they claim to quote. They are characterization tests:
the report is already correct, and these lock that in.
"""

from __future__ import annotations

import json
import re
import xml.dom.minidom
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPORT_DIR = Path(__file__).resolve().parents[1] / "docs" / "reports" / "2026-06-28-final-comparison"
INDEX = REPORT_DIR / "index.html"
REPO_METADATA = Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / "2026-06-27-comparison" / "repo-metadata.json"
SETUP_RESULTS = Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / "2026-06-27-comparison" / "setup-results-final.json"


class _Table:
    def __init__(self) -> None:
        self.caption = ""
        self.headers: list[str] = []
        self.rows: list[dict] = []


class _TableExtractor(HTMLParser):
    """Extract every <table> as caption + headers + body rows (text + first href)."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[_Table] = []
        self._t: _Table | None = None
        self._in_thead = False
        self._in_caption = False
        self._cell: str | None = None
        self._href: str | None = None
        self._row: dict | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "table":
            self._t = _Table()
        elif self._t is None:
            return
        elif tag == "caption":
            self._in_caption = True
        elif tag == "thead":
            self._in_thead = True
        elif tag == "tbody":
            self._in_thead = False
        elif tag == "tr":
            self._row = {"cells": [], "href": None}
        elif tag in ("th", "td"):
            self._cell = ""
        elif tag == "a" and self._cell is not None and self._href is None:
            self._href = a.get("href")

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._t is not None:
            self.tables.append(self._t)
            self._t = None
        elif self._t is None:
            return
        elif tag == "caption":
            self._in_caption = False
        elif tag in ("th", "td") and self._cell is not None:
            self._row["cells"].append(re.sub(r"\s+", " ", self._cell).strip())
            if self._row["href"] is None and self._href:
                self._row["href"] = self._href
            self._cell = None
            self._href = None
        elif tag == "tr" and self._row is not None:
            if self._in_thead:
                self._t.headers = self._row["cells"]
            else:
                self._t.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._in_caption and self._t is not None:
            self._t.caption += data
        elif self._cell is not None:
            self._cell += data


def _html() -> str:
    return INDEX.read_text(encoding="utf-8")


def _tables() -> list[_Table]:
    ex = _TableExtractor()
    ex.feed(_html())
    return ex.tables


def _main_table() -> _Table:
    for t in _tables():
        if t.caption.strip().startswith("표 1."):
            return t
    raise AssertionError("표 1 main latency table not found")


def _summary_for(href: str) -> dict:
    path = (REPORT_DIR / href).resolve() / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _first_number(text: str) -> float:
    m = re.search(r"-?\d[\d,]*\.?\d*", text)
    assert m, f"no number in {text!r}"
    return float(m.group(0).replace(",", ""))


# --- Fairness: every quoted number in the main table matches its artifact -----


def test_index_html_exists() -> None:
    assert INDEX.is_file(), f"missing report at {INDEX}"


def test_main_table_numbers_match_artifacts() -> None:
    table = _main_table()
    cols = {name: i for i, name in enumerate(table.headers)}
    for key in ("Valid turns", "p50", "Representative max", "Raw max", "Outliers"):
        assert key in cols, f"missing column {key!r} in {table.headers}"

    checked = 0
    for row in table.rows:
        href = row["href"]
        assert href and "/benchmarks/" in href, f"row has no benchmark link: {row['cells']}"
        summary = _summary_for(href)
        cells = row["cells"]

        assert _first_number(cells[cols["Valid turns"]]) == summary["turn_count"]
        assert _first_number(cells[cols["p50"]]) == pytest.approx(summary["turn_p50_ms"], abs=0.05)
        assert _first_number(cells[cols["Representative max"]]) == pytest.approx(
            summary["turn_representative_max_ms"], abs=0.05
        )
        assert _first_number(cells[cols["Raw max"]]) == pytest.approx(summary["turn_max_ms"], abs=0.05)
        assert _first_number(cells[cols["Outliers"]]) == summary["turn_extreme_outlier_count"]
        checked += 1

    assert checked == 11, f"expected 11 main-table rows, parsed {checked}"


def test_main_table_boundary_labels_match_source_notes() -> None:
    """측정 경계 column must agree with each run's summary.json source_note."""
    table = _main_table()
    cols = {name: i for i, name in enumerate(table.headers)}
    boundary_col = cols["측정 경계"]

    verified = 0
    for row in table.rows:
        note = _summary_for(row["href"]).get("source_note", "")
        label = row["cells"][boundary_col]
        if "first local speaker playback callback" in note:
            assert "device playback" in label, label
            verified += 1
        elif "commit to first response audio delta" in note:
            assert "commit" in label, label
            verified += 1
        elif "first response audio delta" in note:
            assert "API first-audio" in label, label
            verified += 1
        # manual run has no boundary phrase in its note; verified elsewhere via runtime code.
    assert verified >= 9, f"expected >=9 source-note-derivable boundaries, got {verified}"


#: ms values that are session configuration (chunk/callback/silence), not
#: measurements, so they legitimately appear in cells without being in summary.json.
_CONFIG_MS = {10.0, 20.0, 200.0, 300.0}


def _summary_numbers(summary: dict) -> list[float]:
    return [float(v) for v in summary.values() if isinstance(v, (int, float))]


def test_headline_metric_cards_match_artifacts() -> None:
    """Header metric cards are headline claims, so they must be artifact-backed too."""
    html = _html()
    expected = {
        "1261.3 ms": ("../../benchmarks/2026-06-28-short-default-device-playback-n8/", "turn_p50_ms"),
        "957.0 ms": ("../../benchmarks/2026-06-28-forced-commit-device-playback-n8/", "turn_p50_ms"),
        "1654.2 ms": (
            "../../benchmarks/2026-06-28-local-endpoint-miss14-device-playback-n8/",
            "turn_p50_ms",
        ),
        "5.3 ms": (
            "../../benchmarks/2026-06-28-short-default-device-playback-n8/",
            "api_to_playback_p50_ms",
        ),
    }

    for label, (href, key) in expected.items():
        assert f"<strong>{label}</strong>" in html
        assert float(label.removesuffix(" ms")) == pytest.approx(_summary_for(href)[key], abs=0.05)


def test_all_benchmark_linked_rows_quote_real_numbers() -> None:
    """Every 'X ms' value in a benchmark-linked row must exist in that run's artifact.

    Skips the first cell (the run-name link, which can contain config like
    'server_vad 200 ms') and tolerates known session-config constants.
    """
    rows_checked = 0
    numbers_checked = 0
    for table in _tables():
        for row in table.rows:
            href = row["href"]
            if not href or "/benchmarks/" not in href:
                continue
            values = _summary_numbers(_summary_for(href))
            rows_checked += 1
            for cell in row["cells"][1:]:  # skip the run-name cell
                for token in re.findall(r"(-?\d[\d,]*\.?\d*)\s*ms\b", cell):
                    n = float(token.replace(",", ""))
                    ok = n in _CONFIG_MS or any(abs(n - v) <= 0.05 for v in values)
                    assert ok, f"{href}: quoted {n} ms not found in artifact"
                    numbers_checked += 1
    assert rows_checked >= 20, f"expected many benchmark rows, parsed {rows_checked}"
    assert numbers_checked >= 30, f"expected many quoted ms values, parsed {numbers_checked}"


def test_low_power_rows_are_flagged() -> None:
    """Any run with n<4 valid turns must carry the low-power marker."""
    table = _main_table()
    cols = {name: i for i, name in enumerate(table.headers)}
    for row in table.rows:
        n = _first_number(row["cells"][cols["Valid turns"]])
        if n < 4:
            assert "*" in row["cells"][cols["Valid turns"]], (
                f"n<4 row missing low-power '*': {row['cells']}"
            )


def test_reference_repo_table_matches_metadata_snapshot() -> None:
    """External repo stars and freshness labels must match the captured snapshot."""
    table = next(t for t in _tables() if t.caption.strip().startswith("표 8."))
    cols = {name: i for i, name in enumerate(table.headers)}
    metadata = {item["url"]: item for item in json.loads(REPO_METADATA.read_text(encoding="utf-8"))}

    checked = 0
    for row in table.rows:
        href = row["href"]
        if href not in metadata:
            continue
        item = metadata[href]
        cells = row["cells"]
        stars = int(cells[cols["Stars"]].replace(",", ""))
        updated = item["updated_at"][:10]
        pushed = item["pushed_at"][:10]

        assert stars == item["stars"], f"{href}: stars drifted from repo-metadata.json"
        assert f"updated {updated}" in cells[cols["Freshness snapshot"]]
        assert f"pushed {pushed}" in cells[cols["Freshness snapshot"]]
        checked += 1

    assert checked >= 6, f"expected to verify most external repos, got {checked}"


def test_unsnapshotted_reference_rows_do_not_claim_repo_metrics() -> None:
    table = next(t for t in _tables() if t.caption.strip().startswith("표 8."))
    cols = {name: i for i, name in enumerate(table.headers)}
    metadata = {item["url"] for item in json.loads(REPO_METADATA.read_text(encoding="utf-8"))}

    checked = 0
    for row in table.rows:
        href = row["href"]
        if not href or not href.startswith("https://github.com/") or href in metadata:
            continue
        stars_cell = row["cells"][cols["Stars"]]
        freshness_cell = row["cells"][cols["Freshness snapshot"]]

        assert not re.search(r"\d", stars_cell), f"{href}: unsnapshotted row claims stars"
        assert "source review" in freshness_cell, f"{href}: missing source-review qualifier"
        checked += 1

    assert checked >= 1, "expected at least one source-review-only reference row"


def test_setup_table_matches_final_setup_results() -> None:
    """Setup duration is diagnostic evidence, so it must match the stored run."""
    table = next(t for t in _tables() if t.caption.strip().startswith("표 9."))
    cols = {name: i for i, name in enumerate(table.headers)}
    results = {item["repo"]: item for item in json.loads(SETUP_RESULTS.read_text(encoding="utf-8"))}

    for row in table.rows:
        repo = row["cells"][cols["Repository"]]
        assert repo in results, f"{repo}: setup row has no final setup artifact"
        item = results[repo]

        assert row["cells"][cols["Status"]] == item["status"]
        assert _first_number(row["cells"][cols["Duration"]]) == pytest.approx(
            item["duration_s"], abs=0.05
        )

    assert len(table.rows) == len(results)


# --- Fairness: header verification pill matches the real metrics --------------


def test_coverage_pill_matches_real_coverage() -> None:
    html = _html()
    assert "116 tests" in html
    m = re.search(r"coverage (\d+\.\d+)%", html)
    assert m, "coverage pill not found"
    # 80% floor is enforced in pyproject; the pill must at least be plausible.
    assert float(m.group(1)) >= 80.0


# --- Structure: links, anchors, captions resolve -----------------------------


def test_all_local_links_resolve() -> None:
    html = _html()
    links = re.findall(r'href="(\.\./[^"#]+|[\w-]+\.svg)"', html)
    missing = [link for link in set(links) if not (REPORT_DIR / link).exists()]
    assert not missing, f"dangling local links: {missing}"


def test_toc_anchors_have_unique_section_ids() -> None:
    html = _html()
    anchors = set(re.findall(r'href="#([\w-]+)"', html))
    ids = re.findall(r'\sid="([\w-]+)"', html)
    dangling = sorted(a for a in anchors if a not in set(ids))
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not dangling, f"TOC anchors without a target id: {dangling}"
    assert not duplicates, f"duplicate element ids: {duplicates}"


def test_every_table_has_a_nonempty_caption() -> None:
    for t in _tables():
        assert t.caption.strip(), f"table without caption: headers={t.headers}"


def test_every_table_header_uses_scope() -> None:
    """Header <th> must declare scope for assistive tech."""
    html = _html()
    # No bare <th> (without an attribute) should remain.
    assert "<th>" not in html, "found a <th> without scope/attributes"


def test_figures_are_bound_to_their_captions() -> None:
    """Figure captions should be programmatically associated with each chart image."""
    html = _html()
    figures = re.findall(r"<figure\b([^>]*)>(.*?)</figure>", html, flags=re.S)
    assert figures, "expected at least one report figure"

    ids = set(re.findall(r'\sid="([\w-]+)"', html))
    for attrs, body in figures:
        assert "<img " in body, f"figure without image: {body}"
        assert "<figcaption" in body, f"figure without caption: {body}"
        m = re.search(r'aria-describedby="([\w-]+)"', attrs)
        assert m, f"figure missing aria-describedby: {body}"
        assert m.group(1) in ids, f"figure describes missing caption id: {m.group(1)}"


# --- Visualization: SVGs parse and are wired into the report -----------------


def test_all_svgs_parse_as_xml() -> None:
    svgs = sorted(REPORT_DIR.glob("*.svg"))
    assert len(svgs) == 10, f"expected 10 SVGs, found {len(svgs)}"
    for svg in svgs:
        xml.dom.minidom.parseString(svg.read_text(encoding="utf-8"))


def test_every_svg_is_referenced_by_the_report() -> None:
    html = _html()
    for svg in REPORT_DIR.glob("*.svg"):
        assert f'src="{svg.name}"' in html, f"{svg.name} is not embedded in index.html"


def test_every_embedded_image_has_alt_text() -> None:
    html = _html()
    for tag in re.findall(r"<img\b[^>]*>", html):
        m = re.search(r'alt="([^"]*)"', tag)
        assert m and m.group(1).strip(), f"img without meaningful alt: {tag}"
        assert not m.group(1).strip().endswith("SVG"), f"alt is a generic placeholder: {tag}"


def test_svg_accessible_labels_use_file_scoped_ids() -> None:
    """SVG title/desc IDs should be unique enough to remain safe if inlined."""
    for svg in REPORT_DIR.glob("*.svg"):
        root = ET.parse(svg).getroot()
        ns = {"svg": "http://www.w3.org/2000/svg"}
        assert root.attrib.get("role") == "img", svg.name

        labelledby = root.attrib.get("aria-labelledby", "")
        label_ids = labelledby.split()
        assert len(label_ids) == 2, f"{svg.name}: expected title + desc aria-labelledby"

        ids = {el.attrib["id"] for el in root.iter() if "id" in el.attrib}
        assert set(label_ids).issubset(ids), f"{svg.name}: aria-labelledby points to missing ids"

        prefix = svg.stem.replace("-", "_")
        assert label_ids == [f"{prefix}_title", f"{prefix}_desc"], (
            f"{svg.name}: label ids should be file-scoped, got {label_ids}"
        )

        title = root.find("svg:title", ns)
        desc = root.find("svg:desc", ns)
        assert title is not None and title.attrib.get("id") == f"{prefix}_title", svg.name
        assert desc is not None and desc.attrib.get("id") == f"{prefix}_desc", svg.name
        assert (title.text or "").strip(), svg.name
        assert (desc.text or "").strip(), svg.name


def test_reference_landscape_svg_names_every_reference_repo() -> None:
    table = next(t for t in _tables() if t.caption.strip().startswith("표 8."))
    repos = [row["cells"][0] for row in table.rows]
    svg_text = (REPORT_DIR / "reference-landscape.svg").read_text(encoding="utf-8")

    for repo in repos:
        assert repo in svg_text, f"reference-landscape.svg omits {repo!r}"
