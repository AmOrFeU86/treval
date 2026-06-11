"""
Tests para @treval.tool
"""
import pytest
from treval import tool
from treval.db import SpanStore


@tool
def add(a: int, b: int) -> int:
    return a + b


@tool(name="multiplicar")
def multiply(a: int, b: int) -> int:
    return a * b


def test_tool_preserves_function():
    assert add(2, 3) == 5


def test_tool_preserves_with_strings():
    assert add("hello ", "world") == "hello world"


def test_tool_records_span():
    store = SpanStore()
    store.clear()
    add(2, 3)
    spans = store.list_spans(type="TOOL")
    assert len(spans) == 1
    s = spans[0]
    assert s["name"] == "add"
    assert s["type"] == "TOOL"
    assert s["status"] == "ok"


def test_multiple_calls_multiple_spans():
    store = SpanStore()
    store.clear()
    add(1, 1)
    add(2, 2)
    add(3, 3)
    assert store.count() == 3


def test_tool_custom_name():
    store = SpanStore()
    store.clear()
    multiply(4, 5)
    spans = store.list_spans(type="TOOL")
    assert spans[0]["name"] == "multiplicar"


def test_tool_records_error():
    store = SpanStore()
    store.clear()

    @tool
    def failing():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        failing()

    spans = store.list_spans(type="TOOL")
    assert len(spans) == 1
    assert spans[0]["status"] == "error"
    assert "ValueError" in spans[0]["output"]


def test_tool_metadata_fn_recorded_as_json():
    """metadata_fn gets (args, kwargs, result) and stores its return as JSON."""
    import json
    store = SpanStore()
    store.clear()

    @tool(metadata_fn=lambda args, kwargs, result: {
        "query": kwargs["q"],
        "num_results": len(result),
    })
    def search(q: str):
        return ["a", "b", "c"]

    search(q="hello")
    spans = store.list_spans(type="TOOL")
    assert len(spans) == 1
    meta = json.loads(spans[0]["metadata"])
    assert meta == {"query": "hello", "num_results": 3}


def test_tool_metadata_fn_not_called_on_error():
    """On exception, metadata_fn is not called and metadata stays None."""
    import json
    store = SpanStore()
    store.clear()

    def fail_meta(args, kwargs, result):
        raise AssertionError("should not be called on error")

    @tool(metadata_fn=fail_meta)
    def boom():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        boom()
    spans = store.list_spans(type="TOOL")
    assert len(spans) == 1
    assert spans[0]["status"] == "error"
    assert spans[0]["metadata"] is None


def test_tool_without_metadata_fn_has_none_metadata():
    """Backward compat: no metadata_fn → metadata column is NULL."""
    store = SpanStore()
    store.clear()

    @tool
    def plain(x: int) -> int:
        return x * 2

    plain(7)
    spans = store.list_spans(type="TOOL")
    assert spans[0]["metadata"] is None


def test_tool_metadata_fn_receives_positional_args():
    """metadata_fn's first arg is a tuple of positional args."""
    import json
    store = SpanStore()
    store.clear()

    @tool(metadata_fn=lambda args, kwargs, result: {
        "args": list(args),
        "kwargs": dict(kwargs),
    })
    def fn(a, b, c=0):
        return a + b + c

    fn(1, 2, c=3)
    spans = store.list_spans(type="TOOL")
    meta = json.loads(spans[0]["metadata"])
    assert meta["args"] == [1, 2]
    assert meta["kwargs"] == {"c": 3}