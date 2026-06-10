"""
Tests para wrap, callbacks y gateway.
"""
from treval.db import SpanStore
from treval.wrap import _summarize, _extract_response


# ─── Wrap ───

def test_summarize_empty():
    assert _summarize([]) == ""


def test_summarize_single():
    result = _summarize([{"role": "user", "content": "hola"}])
    assert "user: hola" in result


def test_summarize_multiple():
    msgs = [
        {"role": "system", "content": "Eres un asistente"},
        {"role": "user", "content": "Hola"},
    ]
    result = _summarize(msgs)
    assert "system: Eres un asistente" in result
    assert "user: Hola" in result


def test_extract_response_standard():
    class FakeChoice:
        def __init__(self):
            self.message = type("", (), {"content": "Hello!"})()

    class FakeResponse:
        def __init__(self):
            self.choices = [FakeChoice()]

    result = _extract_response(FakeResponse())
    assert result == "Hello!"


def test_extract_response_empty():
    class FakeResponse:
        def __init__(self):
            self.choices = []

    result = _extract_response(FakeResponse())
    assert "FakeResponse" in result or result == ""


# ─── Callbacks ───

def test_trace_context_manager():
    """The trace() context manager should create and register spans."""
    store = SpanStore()
    store.clear()

    from treval.callbacks import trace

    with trace("test_block", type="OPERATION", metadata={"key": "val"}):
        span_id = None
        from treval.context import current_span_id
        # El contexto debe tener el span activo dentro del with
        ctx = current_span_id()
        assert ctx is not None

    # After the with block, there should be a span
    spans = store.list_spans(type="OPERATION")
    assert len(spans) == 1
    assert spans[0]["name"] == "test_block"
    assert spans[0]["status"] == "ok"


def test_trace_captures_error():
    """If code inside trace() throws an error, the span should reflect it."""
    store = SpanStore()
    store.clear()

    from treval.callbacks import trace

    try:
        with trace("failing_block"):
            raise ValueError("something went wrong")
    except ValueError:
        pass

    spans = store.list_spans(type="OPERATION")
    assert len(spans) == 1
    assert spans[0]["status"] == "error"
    assert "ValueError" in spans[0]["output"]


# ─── Gateway ───

def test_gateway_handler_has_upstream():
    """Verify the gateway has basic configurations."""
    from treval.gateway import GatewayHandler
    assert hasattr(GatewayHandler, "upstream_url")
    assert hasattr(GatewayHandler, "store")


def test_gateway_model_list():
    """GET / debe devolver info del gateway."""
    from treval.gateway import GatewayHandler
    import json

    # Configurar upstream_url
    GatewayHandler.upstream_url = "https://api.openai.com/v1"

    # Simular un handler
    handler = GatewayHandler.__new__(GatewayHandler)
    handler.path = "/v1/chat/completions"
    handler.headers = {}
    handler.command = "POST"
    handler.wfile = None

    # Verificar que _build_upstream_request funciona
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    req = handler._build_upstream_request(data)
    assert req.method == "POST"
    assert req.data is not None
    assert "gpt-4" in req.data.decode()