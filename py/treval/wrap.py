"""
wrap(client): wraps an existing OpenAI/Anthropic client to trace its calls.

Alternative to treval.instrument() when you want explicit control.

Usage:
    client = OpenAI(api_key="...")
    client = treval.wrap(client)
    response = client.chat.completions.create(...)  # ← traced automatically
"""
from __future__ import annotations

import functools
import time

from treval.db import SpanStore


def wrap(client):
    """Wraps an OpenAI client to trace all completions calls.

    Args:
        client: Instance of openai.OpenAI (or any client with
                chat.completions.create)

    Returns:
        The same client with the create method patched.
    """
    original_create = client.chat.completions.create

    @functools.wraps(original_create)
    def traced_create(*args, **kwargs):
        store = SpanStore()
        start = time.perf_counter()
        status = "ok"
        output = None

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        input_text = _summarize(messages)

        try:
            result = original_create(*args, **kwargs)
            output = _extract_response(result)
            return result
        except BaseException as e:
            status = "error"
            output = f"{type(e).__name__}: {e}"
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            store.save(
                name=f"llm.{model}",
                type="LLM",
                status=status,
                input=input_text,
                output=output,
                duration_ms=duration_ms,
            )

    client.chat.completions.create = traced_create
    return client


def wrap_anthropic(client):
    """Wraps an Anthropic client to trace calls to messages.create()."""
    original_create = client.messages.create

    @functools.wraps(original_create)
    def traced_create(*args, **kwargs):
        store = SpanStore()
        start = time.perf_counter()
        status = "ok"
        output = None

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        input_text = _summarize(messages)

        try:
            result = original_create(*args, **kwargs)
            # Anthropic response has .content as a list of blocks
            if hasattr(result, "content"):
                output = " ".join(
                    b.text if hasattr(b, "text") else str(b)
                    for b in result.content
                )
            else:
                output = str(result)
            return result
        except BaseException as e:
            status = "error"
            output = f"{type(e).__name__}: {e}"
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            store.save(
                name=f"llm.{model}",
                type="LLM",
                status=status,
                input=input_text,
                output=output,
                duration_ms=duration_ms,
            )

    client.messages.create = traced_create
    return client


def _summarize(messages: list) -> str:
    """Converts messages to a readable string."""
    parts = []
    for m in (messages or []):
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str):
            parts.append(f"{role}: {content[:200]}")
        else:
            parts.append(f"{role}: <complex content>")
    return "\n".join(parts)


def _extract_response(response) -> str:
    """Extracts text from the first choice of an OpenAI response."""
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        return str(response)