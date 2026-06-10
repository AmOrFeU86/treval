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