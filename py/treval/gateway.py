"""
Local gateway/proxy: intercepts HTTP traffic to LLM APIs and traces it.

No need to modify agent code — just point the base URL
to the gateway.

Usage:
    treval gateway --port 9090
    # In agent code:
    client = OpenAI(base_url="http://localhost:9090/v1", api_key="...")
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from socketserver import ThreadingMixIn
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from treval.db import SpanStore

OPENROUTER_BASE = "https://openrouter.ai/api"
OPENAI_BASE = "https://api.openai.com"


class GatewayServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server for the gateway."""
    allow_reuse_address = True
    daemon_threads = True


def _parse_sse_chunks(raw: str) -> list[dict]:
    """Parses an SSE (Server-Sent Events) body and returns a list of JSON chunks.

    SSE format:
        data: {...json...}

        data: [DONE]
    """
    chunks = []
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                chunks.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return chunks


def _reassemble_content(chunks: list[dict]) -> str:
    """Reassembles full content from SSE chunks."""
    content = ""
    for chunk in chunks:
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content += delta.get("content", "")
    return content


def _extract_model_from_chunks(chunks: list[dict]) -> str:
    """Extracts the model name from the last SSE chunk."""
    for chunk in reversed(chunks):
        model = chunk.get("model", "")
        if model:
            return model
    return "unknown"


class GatewayHandler(BaseHTTPRequestHandler):
    """HTTP proxy that forwards requests to the real API and traces spans.

    Supports:
    - Synchronous requests (stream=false)
    - SSE streaming (stream=true) → forwards chunks in real time
    - CORS headers for browser usage
    - Upstream HTTP error code propagation
    """

    store = SpanStore()
    upstream_url = ""

    def _cors_headers(self):
        """Adds CORS headers to the response."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, HTTP-Referer, X-Title")

    def do_OPTIONS(self):
        """Handles CORS preflight."""
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # 1. Registrar span de entrada
        start = time.perf_counter()
        span_id = self.store.save(
            name="gateway.request",
            type="LLM",
            status="pending",
            input=body[:2000].decode("utf-8", errors="replace"),
        )

        try:
            data = json.loads(body)
            is_stream = data.get("stream", False)
            upstream = self._build_upstream_request(data)
            upstream_response = urlopen(upstream, timeout=120)
            duration_ms = (time.perf_counter() - start) * 1000

            if is_stream:
                self._handle_stream_response(upstream_response, span_id, data, start)
            else:
                self._handle_sync_response(upstream_response, span_id, data, start)

        except HTTPError as e:
            duration_ms = (time.perf_counter() - start) * 1000
            err_body = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
            self.store.update(
                span_id,
                status="error",
                output=f"HTTP {e.code}: {err_body[:500]}",
                duration_ms=duration_ms,
            )
            self.send_response(e.code)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            self.store.update(
                span_id,
                status="error",
                output=f"{type(e).__name__}: {e}",
                duration_ms=duration_ms,
            )
            self.send_response(502)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _handle_sync_response(self, upstream_response, span_id, data, start):
        """Processes a non-streaming response: reads all, traces, returns."""
        response_body = upstream_response.read()
        status_code = upstream_response.status
        response_data = json.loads(response_body)
        model = response_data.get("model", data.get("model", "unknown"))
        content = ""
        if response_data.get("choices"):
            content = response_data["choices"][0].get("message", {}).get("content", "")

        duration_ms = (time.perf_counter() - start) * 1000
        self.store.update(
            span_id,
            name=f"llm.{model}",
            status="ok",
            output=content[:2000],
            duration_ms=duration_ms,
        )

        self.send_response(status_code)
        self._cors_headers()
        for key, value in upstream_response.headers.items():
            if key.lower() not in ("transfer-encoding", "content-encoding", "content-length"):
                self.send_header(key, value)
        self.send_header("Content-Length", len(response_body))
        self.end_headers()
        self.wfile.write(response_body)

    def _handle_stream_response(self, upstream_response, span_id, data, start):
        """Processes streaming response (SSE): forwards chunks in real time and
        accumulates content for the final span."""
        buffer = b""
        full_content = ""
        chunk_count = 0

        # Headers de respuesta streaming
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        # Leer y reenviar chunks
        while True:
            chunk = upstream_response.read(4096)
            if not chunk:
                break
            buffer += chunk
            self.wfile.write(chunk)
            self.wfile.flush()
            chunk_count += 1

        # Procesar buffer completo para extraer spans
        raw_text = buffer.decode("utf-8", errors="replace")
        sse_chunks = _parse_sse_chunks(raw_text)
        full_content = _reassemble_content(sse_chunks)
        model = _extract_model_from_chunks(sse_chunks) or data.get("model", "unknown")

        duration_ms = (time.perf_counter() - start) * 1000
        self.store.update(
            span_id,
            name=f"llm.{model}",
            status="ok",
            output=full_content[:2000] if full_content else "(empty stream)",
            duration_ms=duration_ms,
        )

    def _build_upstream_request(self, data: dict) -> Request:
        """Builds the upstream API request."""
        url = f"{self.upstream_url}{self.path}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.headers.get("Authorization", ""),
        }
        # Forward relevant headers
        for h in ("HTTP-Referer", "X-Title", "User-Agent"):
            v = self.headers.get(h)
            if v:
                headers[h] = v

        return Request(url, data=json.dumps(data).encode(), headers=headers, method="POST")

    def do_GET(self):
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        stats = self.store.count()
        response = {
            "service": "treval-gateway",
            "status": "running",
            "upstream": self.upstream_url,
            "spans_traced": stats,
            "models": ["any compatible OpenAI/OpenRouter model"],
            "streaming": True,
        }
        self.wfile.write(json.dumps(response, indent=2).encode())

    def log_message(self, format, *args):
        pass  # Silenciar logs HTTP


def run_gateway(host: str = "127.0.0.1", port: int = 9090,
                upstream: str = "openrouter") -> None:
    """Starts the gateway server with multi-thread support.

    Args:
        host: Listen address
        port: Port
        upstream: "openrouter" or "openai"
    """
    GatewayHandler.upstream_url = OPENROUTER_BASE if upstream == "openrouter" else OPENAI_BASE
    server = GatewayServer((host, port), GatewayHandler)
    print(f"🚇 Treval Gateway at http://{host}:{port}")
    print(f"   Upstream: {GatewayHandler.upstream_url}")
    print(f"   Streaming: ✅ active")
    print(f"   Multi-thread: ✅ active")
    print(f"   CORS: ✅ active")
    print()
    print(f"   Use in your code:")
    print(f'     OpenAI(base_url="http://{host}:{port}/v1")')
    print(f'     AsyncOpenAI(base_url="http://{host}:{port}/v1")')
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGateway stopped.")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Treval Gateway")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--upstream", choices=["openrouter", "openai"], default="openrouter")
    args = parser.parse_args()
    run_gateway(port=args.port, upstream=args.upstream)