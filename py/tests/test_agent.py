"""
Tests para @treval.agent y @treval.operation
"""
import pytest
from treval import agent, operation, tool
from treval.db import SpanStore


@agent
class WeatherAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key

    @operation
    def get_forecast(self, location: str) -> str:
        return f"sunny in {location}"

    @operation(name="predecir_temp")
    def get_temperature(self, location: str) -> int:
        return 25


@agent(name="MySearcher")
class SearchAgent:
    def __init__(self):
        pass

    @operation
    def search(self, query: str) -> str:
        return f"results for {query}"


# --- Agent tests ---

def test_agent_records_span():
    store = SpanStore()
    store.clear()
    WeatherAgent("test-key")
    spans = store.list_spans(type="AGENT")
    assert len(spans) == 1
    assert spans[0]["name"] == "WeatherAgent"
    assert spans[0]["type"] == "AGENT"


def test_agent_custom_name():
    store = SpanStore()
    store.clear()
    SearchAgent()
    spans = store.list_spans(type="AGENT")
    assert spans[0]["name"] == "MySearcher"


def test_agent_records_error_in_init():
    store = SpanStore()
    store.clear()

    @agent
    class BadAgent:
        def __init__(self):
            raise RuntimeError("init failed")

    with pytest.raises(RuntimeError):
        BadAgent()

    spans = store.list_spans(type="AGENT")
    assert len(spans) >= 1
    assert spans[0]["status"] == "error"
    assert "RuntimeError" in spans[0]["output"]


# --- Operation tests ---

def test_operation_records_span():
    store = SpanStore()
    store.clear()
    w = WeatherAgent("key")
    w.get_forecast("Madrid")
    spans = store.list_spans(type="OPERATION")
    assert len(spans) == 1
    assert spans[0]["name"] == "get_forecast"
    assert spans[0]["status"] == "ok"


def test_operation_custom_name():
    store = SpanStore()
    store.clear()
    w = WeatherAgent("key")
    w.get_temperature("Barcelona")
    spans = store.list_spans(type="OPERATION")
    assert spans[0]["name"] == "predecir_temp"


def test_operation_linked_to_agent():
    """El operation debe tener parent_id apuntando al span del agente."""
    store = SpanStore()
    store.clear()
    w = WeatherAgent("key")
    w.get_forecast("Madrid")

    agent_spans = store.list_spans(type="AGENT")
    op_spans = store.list_spans(type="OPERATION")
    assert len(agent_spans) == 1
    assert len(op_spans) == 1
    assert op_spans[0]["parent_id"] == agent_spans[0]["id"]


def test_multiple_operations():
    store = SpanStore()
    store.clear()
    w = WeatherAgent("key")
    w.get_forecast("Madrid")
    w.get_temperature("BCN")
    ops = store.list_spans(type="OPERATION")
    assert len(ops) == 2


def test_operation_records_error():
    store = SpanStore()
    store.clear()

    @agent
    class FragileAgent:
        def __init__(self):
            pass

        @operation
        def crash(self):
            raise ValueError("exploded")

    f = FragileAgent()
    with pytest.raises(ValueError):
        f.crash()

    ops = store.list_spans(type="OPERATION")
    assert ops[0]["status"] == "error"


# --- Mixed: tool within agent ---

def test_tool_within_agent():
    store = SpanStore()
    store.clear()

    @agent
    class CalcAgent:
        def __init__(self):
            pass

        @operation
        def compute(self, a, b):
            return add_wrapped(a, b)

    @tool
    def add_wrapped(a, b):
        return a + b

    c = CalcAgent()
    result = c.compute(2, 3)
    assert result == 5

    tools = store.list_spans(type="TOOL")
    ops = store.list_spans(type="OPERATION")
    assert len(tools) == 1
    assert len(ops) == 1
    assert tools[0]["parent_id"] == ops[0]["id"]