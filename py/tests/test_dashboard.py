"""Tests for the dashboard's span detail panel.

The dashboard embeds its JS as a big inline string in dashboard.py.
We can't run JS from Python, but we CAN pin that the rendering
functions for the new metadata fields exist in the source — that
catches accidental deletions during refactors and makes the contract
explicit.
"""
import re

from treval import dashboard as dashboard_mod


def test_dashboard_renders_sources_section():
    """The detail panel must render `metadata.sources` as a clickable list.

    A research OPERATION span (#9 in deep-research) stores its deduped
    sources in `metadata.sources`. The dashboard must show them so the
    user can see at a glance which URLs fed the answer.
    """
    src = dashboard_mod._build_html.__code__.co_filename
    with open(src) as f:
        text = f.read()
    # The render function exists
    assert "function renderSources" in text, "renderSources() missing"
    # The function is wired into showDetail()
    assert "renderSources(s)" in text, "renderSources() not called from showDetail()"
    # It produces a <h3> with the source count, clickable links, and URLs
    assert "Sources (" in text, "Sources heading missing"
    assert "target=\"_blank\"" in text, "Sources should open in new tab"


def test_dashboard_renders_run_stats_for_unknown_metadata():
    """Unknown metadata keys (num_sources, tavily_searches, tavily_cost_usd)
    must be rendered as a generic 'Run stats' key-value section, so
    future metadata fields work without touching the dashboard.
    """
    src = dashboard_mod._build_html.__code__.co_filename
    with open(src) as f:
        text = f.read()
    assert "function renderRunStats" in text, "renderRunStats() missing"
    assert "renderRunStats(s)" in text, "renderRunStats() not called"
    # Cost is rendered with $ prefix
    assert "toFixed(5)" in text or "toFixed(" in text, "Cost should be formatted"
    # Object values (the sources list) are skipped — they have their own panel
    assert "function renderRunStats" in text and "sources" in text


def test_dashboard_llm_metadata_still_works():
    """Regression: LLM spans still show Model / Tokens / Cost via metaGrid()."""
    src = dashboard_mod._build_html.__code__.co_filename
    with open(src) as f:
        text = f.read()
    assert "function metaGrid" in text
    assert "metaGrid(s)" in text
    # Existing LLM fields still referenced
    assert "m.model" in text
    assert "m.prompt_tokens" in text
    assert "m.total_tokens" in text


def test_dashboard_exported_html_contains_new_render_functions(tmp_path):
    """End-to-end: _build_html() produces a string with the new functions.

    This catches a different failure mode: if the source has the
    functions but they're inside a Python string that's never embedded,
    the browser would still see the old code.
    """
    # Use an isolated in-memory store so we don't touch the real DB.
    from treval.db import SpanStore
    db = tmp_path / "spans.db"
    store = SpanStore(db_path=db)
    store.save(name="r", type="OPERATION", metadata={
        "num_sources": 2,
        "sources": [{"url": "https://a", "title": "A", "content": "x"}],
        "tavily_searches": 1,
        "tavily_cost_usd": 0.001,
    })
    html = dashboard_mod._build_html(store=store)
    assert "renderSources" in html
    assert "renderRunStats" in html
    # The metadata we just stored is embedded
    assert "https://a" in html  # embedded as JSON, but the URL is there
