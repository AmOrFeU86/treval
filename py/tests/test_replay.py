"""
Test of the interactive replay module.
"""
from treval.db import SpanStore
from treval.replay import ReplaySession


def test_replay_session_creates():
    """Create a ReplaySession from an existing span."""
    store = SpanStore()
    store.clear()
    sid = store.save(name="test_tool", type="TOOL", input="hello", output="world")

    session = ReplaySession(sid)
    assert session.span_id == sid
    assert session.span_type == "TOOL"
    assert session.span_name == "test_tool"
    assert session.original["input"] == "hello"
    assert session.is_replayable


def test_replay_not_replayable_without_input():
    """Span without input is not replayable."""
    store = SpanStore()
    store.clear()
    sid = store.save(name="no_input", type="AGENT")

    session = ReplaySession(sid)
    assert not session.is_replayable


def test_replay_set_input():
    """Modify input before replay."""
    store = SpanStore()
    store.clear()
    sid = store.save(name="t", type="TOOL", input="original")

    session = ReplaySession(sid)
    assert session.modified_input == "original"

    session.set_input("nuevo input")
    assert session.modified_input == "nuevo input"


def test_replay_set_model():
    """Cambiar modelo."""
    store = SpanStore()
    store.clear()
    sid = store.save(name="t", type="TOOL", input="test")

    session = ReplaySession(sid)
    assert session.modified_model is None

    session.set_model("gpt-4")
    assert session.modified_model == "gpt-4"


def test_replay_no_api_key_uses_env():
    """Without API key, replay uses the env var and works if configured."""
    import os
    store = SpanStore()
    store.clear()
    sid = store.save(name="t", type="TOOL", input="test",
                     output="result")

    session = ReplaySession(sid)
    # With empty api_key, it will try to read from env
    # This is an integration test, we just verify it doesn't crash
    result = session.replay(api_key="")
    assert isinstance(result, dict)
    # If the env var is configured, it should contain output
    # If not, it should have an error
    if not os.environ.get("OPENROUTER_API_KEY"):
        assert "error" in result