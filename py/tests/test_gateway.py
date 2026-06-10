"""Tests for the treval gateway/proxy.

Covers:
- Basic proxy (sync POST forwarding)
- Streaming SSE
- Upstream errors (timeout, 4xx)
- CORS headers
- Health endpoint (GET /)
- Spans registered in DB
- SSE parsing
"""

import json
import threading
import time
from unittest.mock import Mock, patch
from io import BytesIO
from http.client import HTTPConnection

import pytest
from treval.db import SpanStore


# ─── Helpers ────────────────────────────────────────────────────────────

def _make_fake_upstream_response(body_bytes, _status=200, headers=None):
    """Crea un objeto respuesta simulado para urlopen.

    Simula correctamente read(chunk_size): devuelve el contenido una vez,
    luego b'' en llamadas sucesivas (como un archivo real).
    """
    hdrs = headers or {"Content-Type": "application/json"}
    data = [body_bytes]  # Lista mutable para state

    class FakeStream:
        """Simula un HTTPResponse con read() que se agota."""

        def read(self, size=-1):
            if not data:
                return b""
            result = data[0]
            data.clear()
            return result

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    FakeStream.status = _status
    FakeStream.headers = hdrs
    return FakeStream()


def _start_test_server(monkeypatch, urlopen_mock, port=9199):
    """Arranca el gateway en un thread y devuelve (server, thread)."""
    from treval.gateway import GatewayServer, GatewayHandler, OPENROUTER_BASE

    monkeypatch.setattr("treval.gateway.urlopen", urlopen_mock)

    store = SpanStore()
    store.clear()

    GatewayHandler.upstream_url = "https://fake.test/v1"
    GatewayServer.allow_reuse_address = True
    server = GatewayServer(("127.0.0.1", port), GatewayHandler)

    t = threading.Thread(target=server.serve_forever)
    t.daemon = True
    t.start()
    time.sleep(0.3)
    return server, t


# ─── Parseo SSE ─────────────────────────────────────────────────────────

def test_sse_parse():
    """Parsear chunks SSE."""
    from treval.gateway import _parse_sse_chunks
    raw = 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: {"choices":[{"delta":{"content":" world"}}]}\n\ndata: [DONE]\n'
    chunks = _parse_sse_chunks(raw)
    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[1]["choices"][0]["delta"]["content"] == " world"


def test_sse_empty():
    from treval.gateway import _parse_sse_chunks
    assert _parse_sse_chunks("") == []
    assert _parse_sse_chunks("data: [DONE]\n") == []


def test_sse_reassemble():
    from treval.gateway import _reassemble_content
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " "}}]},
        {"choices": [{"delta": {"content": "world"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    assert _reassemble_content(chunks) == "Hello world"


def test_sse_no_content_chunks():
    from treval.gateway import _reassemble_content
    assert _reassemble_content([{"choices": [{"delta": {}}]}, {"choices": [{"delta": {"content": "A"}}]}]) == "A"


# ─── Threading server ───────────────────────────────────────────────────

def test_threading_server_used():
    from treval.gateway import GatewayServer
    import socketserver
    assert issubclass(GatewayServer, socketserver.ThreadingMixIn)


# ─── Health endpoint (GET /) ────────────────────────────────────────────

def test_health_endpoint(monkeypatch):
    """GET / debe devolver estado del gateway."""
    from treval.gateway import urlopen as real_urlopen
    from urllib.request import Request

    def fake_urlopen(req, timeout=120):
        if hasattr(req, "get_full_url"):
            url = req.get_full_url()
        else:
            url = str(req)
        if "health" in url or url.endswith("/"):
            # Simular health check
            return _make_fake_upstream_response(b'{"models": []}')
        return _make_fake_upstream_response(
            json.dumps({"id": "test", "model": "test/model",
                        "choices": [{"message": {"content": "ok"}}]}).encode()
        )

    server, t = _start_test_server(monkeypatch, fake_urlopen, port=9190)

    try:
        conn = HTTPConnection("127.0.0.1", 9190, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        data = json.loads(resp.read())
        assert data["service"] == "treval-gateway"
        assert data["status"] == "running"
        assert "streaming" in data
        assert data["streaming"] is True
        conn.close()
    finally:
        server.shutdown()
        server.server_close()


# ─── Sync proxy (stream=false) ──────────────────────────────────────────

def test_sync_proxy(monkeypatch):
    """POST /v1/chat/completions debe reenviar y devolver respuesta."""
    upstream_body = json.dumps({
        "id": "cmpl-test",
        "model": "test/model",
        "choices": [{"message": {"content": "respuesta del proxy"}}],
    })

    def fake_urlopen(req, timeout=120):
        return _make_fake_upstream_response(upstream_body.encode())

    server, t = _start_test_server(monkeypatch, fake_urlopen, port=9191)

    try:
        store = SpanStore()
        before = store.count()
        body = json.dumps({
            "model": "test/model",
            "messages": [{"role": "user", "content": "hola"}],
            "stream": False,
        })
        conn = HTTPConnection("127.0.0.1", 9191, timeout=5)
        conn.request("POST", "/v1/chat/completions", body,
                     {"Content-Type": "application/json",
                      "Authorization": "Bearer sk-test"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        assert data["choices"][0]["message"]["content"] == "respuesta del proxy"
        conn.close()

        # Verificar span guardado
        assert store.count() > before
    finally:
        server.shutdown()
        server.server_close()


# ─── Streaming SSE ──────────────────────────────────────────────────────

def test_streaming_proxy(monkeypatch):
    """POST con stream=true debe reenviar SSE y trazar contenido."""
    sse_body = 'data: {"choices":[{"delta":{"content":"Hello"}}],"model":"test/model"}\n\ndata: {"choices":[{"delta":{"content":" world"}}]}\n\ndata: [DONE]\n'

    def fake_urlopen(req, timeout=120):
        return _make_fake_upstream_response(sse_body.encode(),
                                            headers={"Content-Type": "text/event-stream"})

    server, t = _start_test_server(monkeypatch, fake_urlopen, port=9192)

    try:
        store = SpanStore()
        before = store.count()

        body = json.dumps({
            "model": "test/model",
            "messages": [{"role": "user", "content": "hola"}],
            "stream": True,
        })
        conn = HTTPConnection("127.0.0.1", 9192, timeout=5)
        conn.request("POST", "/v1/chat/completions", body,
                     {"Content-Type": "application/json",
                      "Authorization": "Bearer sk-test"})
        resp = conn.getresponse()
        raw = resp.read().decode()
        assert "Hello" in raw
        assert "world" in raw
        conn.close()

        # Verificar span guardado con contenido reensamblado
        assert store.count() > before
        spans = store.list_spans()
        if spans:
            last = spans[0]
            assert "Hello world" in last.get("output", "")
    finally:
        server.shutdown()
        server.server_close()


# ─── CORS ───────────────────────────────────────────────────────────────

def test_cors_preflight(monkeypatch):
    """OPTIONS debe devolver CORS headers."""
    def fake_urlopen(req, timeout=120):
        return _make_fake_upstream_response(b'{}')

    server, t = _start_test_server(monkeypatch, fake_urlopen, port=9193)

    try:
        conn = HTTPConnection("127.0.0.1", 9193, timeout=5)
        conn.request("OPTIONS", "/v1/chat/completions")
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 204
        assert resp.getheader("Access-Control-Allow-Origin") == "*"
        conn.close()
    finally:
        server.shutdown()
        server.server_close()


def test_cors_on_post(monkeypatch):
    """POST debe incluir CORS headers."""
    def fake_urlopen(req, timeout=120):
        return _make_fake_upstream_response(
            json.dumps({"id": "t", "model": "m",
                        "choices": [{"message": {"content": "ok"}}]}).encode())

    server, t = _start_test_server(monkeypatch, fake_urlopen, port=9194)

    try:
        body = json.dumps({"model": "test", "messages": []})
        conn = HTTPConnection("127.0.0.1", 9194, timeout=5)
        conn.request("POST", "/v1/chat/completions", body,
                     {"Content-Type": "application/json",
                      "Authorization": "Bearer sk-test"})
        resp = conn.getresponse()
        resp.read()
        assert resp.getheader("Access-Control-Allow-Origin") == "*"
        conn.close()
    finally:
        server.shutdown()
        server.server_close()


# ─── Errores ────────────────────────────────────────────────────────────

def test_upstream_http_error(monkeypatch):
    """Upstream 4xx error must propagate with the same code."""
    import urllib.error

    def fake_urlopen(req, timeout=120):
        err_body = json.dumps({"error": "rate limit"}).encode()
        raise urllib.error.HTTPError(
            url="http://fake/v1",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=BytesIO(err_body),
        )

    server, t = _start_test_server(monkeypatch, fake_urlopen, port=9195)

    try:
        body = json.dumps({"model": "test", "messages": []})
        conn = HTTPConnection("127.0.0.1", 9195, timeout=5)
        conn.request("POST", "/v1/chat/completions", body,
                     {"Content-Type": "application/json",
                      "Authorization": "Bearer sk-test"})
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 429  # Propagate real code, not 502
        conn.close()
    finally:
        server.shutdown()
        server.server_close()


def test_upstream_connection_error(monkeypatch):
    """Si el upstream no responde, devuelve 502."""
    def fake_urlopen(req, timeout=120):
        raise OSError("Connection refused")

    server, t = _start_test_server(monkeypatch, fake_urlopen, port=9196)

    try:
        body = json.dumps({"model": "test", "messages": []})
        conn = HTTPConnection("127.0.0.1", 9196, timeout=5)
        conn.request("POST", "/v1/chat/completions", body,
                     {"Content-Type": "application/json",
                      "Authorization": "Bearer sk-test"})
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 502
        conn.close()
    finally:
        server.shutdown()
        server.server_close()


# ─── Span registration ──────────────────────────────────────────────────

def test_span_registered_on_sync(monkeypatch):
    """Cada request debe guardar un span en la BD."""
    def fake_urlopen(req, timeout=120):
        return _make_fake_upstream_response(
            json.dumps({"id": "t", "model": "m",
                        "choices": [{"message": {"content": "ok"}}]}).encode())

    store = SpanStore()
    store.clear()
    assert store.count() == 0

    server, t = _start_test_server(monkeypatch, fake_urlopen, port=9197)

    try:
        body = json.dumps({"model": "test/m", "messages": [{"role": "user", "content": "hi"}]})
        conn = HTTPConnection("127.0.0.1", 9197, timeout=5)
        conn.request("POST", "/v1/chat/completions", body,
                     {"Content-Type": "application/json", "Authorization": "Bearer sk"})
        resp = conn.getresponse()
        resp.read()
        conn.close()
        time.sleep(0.2)

        assert store.count() >= 1
        spans = store.list_spans()
        last = spans[0]
        assert last["type"] == "LLM"
    finally:
        server.shutdown()
        server.server_close()