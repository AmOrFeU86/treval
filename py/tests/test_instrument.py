"""
Tests para treval.instrument().
Verificamos idempotencia y que las funciones auxiliares funcionan.
The real integration test is done with demo_agent.
"""
import json

from treval import instrument
from treval.instrument import _summarize_messages, _extract_response, _extract_usage
from treval.db import SpanStore


def test_instrument_is_idempotent():
    """Llamar instrument() dos veces no rompe nada."""
    instrument()
    instrument()
    assert True


def test_summarize_messages_truncates():
    """Mensajes largos se truncan correctamente."""
    messages = [
        {"role": "user", "content": "Hola" * 1000},  # 4000 chars
    ]
    result = _summarize_messages(messages)
    assert len(result) < 2500  # truncado
    assert "user:" in result


def test_summarize_messages_multiple():
    """Multiple messages are included."""
    messages = [
        {"role": "system", "content": "Eres un asistente"},
        {"role": "user", "content": "Hola"},
        {"role": "assistant", "content": "Hello, how can I help you?"},
    ]
    result = _summarize_messages(messages)
    assert "system:" in result
    assert "user:" in result
    assert "assistant:" in result


class FakeResponse:
    def __init__(self):
        self.choices = [
            type("", (), {"message": type("", (), {"content": "Hello!"})})()
        ]
        self.usage = type("", (), {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})()


def test_extract_response():
    """Extrae texto de respuesta."""
    result = _extract_response(FakeResponse())
    assert result == "Hello!"


def test_extract_usage():
    """Extrae tokens de respuesta."""
    usage = _extract_usage(FakeResponse())
    assert usage["prompt_tokens"] == 10
    assert usage["total_tokens"] == 15