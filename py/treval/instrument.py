"""Auto-instrumentation to capture LLMs without decorators.

Usage:
    import treval
    treval.instrument()  # one line and it already traces OpenAI

Supports:
    - openai.OpenAI (chat.completions.create)
    - openai.AsyncOpenAI (chat.completions.create)
"""

import functools
import time
from typing import Any

from treval.context import current_span_id
from treval.db import SpanStore

_instrumented = False


def instrument() -> None:
    """Patches OpenAI clients to emit LLM spans automatically.

    Safe to call multiple times (only patches the first time).
    """
    global _instrumented
    if _instrumented:
        return

    _patch_openai_init()
    _patch_async_openai_init()
    _instrumented = True


def _patch_openai_init() -> None:
    try:
        import openai
    except ImportError:
        return

    original_init = openai.OpenAI.__init__

    @functools.wraps(original_init)
    def traced_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Now self.chat exists (cached_property was evaluated on access)
        original_create = self.chat.completions.create
        self.chat.completions.create = _make_traced_create(original_create)

    openai.OpenAI.__init__ = traced_init


def _patch_async_openai_init() -> None:
    try:
        import openai
    except ImportError:
        return

    original_init = openai.AsyncOpenAI.__init__

    @functools.wraps(original_init)
    def traced_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        original_create = self.chat.completions.create
        self.chat.completions.create = _make_traced_async_create(original_create)

    openai.AsyncOpenAI.__init__ = traced_init


def _make_traced_create(original_create):
    """Creates a synchronous wrapper for chat.completions.create."""

    @functools.wraps(original_create)
    def traced_create(*args, **kwargs):
        store = SpanStore()
        parent_id = current_span_id()
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        start = time.perf_counter()
        status = "ok"
        output = None
        usage_info = {}

        try:
            response = original_create(*args, **kwargs)
            output = _extract_response(response)
            usage_info = _extract_usage(response)
            return response
        except BaseException as e:
            status = "error"
            output = f"{type(e).__name__}: {e}"
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            metadata = {
                "model": model,
                "prompt_tokens": usage_info.get("prompt_tokens"),
                "completion_tokens": usage_info.get("completion_tokens"),
                "total_tokens": usage_info.get("total_tokens"),
            }
            store.save(
                name=f"llm.{model}",
                type="LLM",
                status=status,
                parent_id=parent_id,
                input=_summarize_messages(messages),
                output=output,
                duration_ms=duration_ms,
                metadata=metadata,
            )

    return traced_create


def _make_traced_async_create(original_create):
    """Creates an asynchronous wrapper for chat.completions.create."""

    @functools.wraps(original_create)
    async def traced_create(*args, **kwargs):
        store = SpanStore()
        parent_id = current_span_id()
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        start = time.perf_counter()
        status = "ok"
        output = None
        usage_info = {}

        try:
            response = await original_create(*args, **kwargs)
            output = _extract_response(response)
            usage_info = _extract_usage(response)
            return response
        except BaseException as e:
            status = "error"
            output = f"{type(e).__name__}: {e}"
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            metadata = {
                "model": model,
                "prompt_tokens": usage_info.get("prompt_tokens"),
                "completion_tokens": usage_info.get("completion_tokens"),
                "total_tokens": usage_info.get("total_tokens"),
            }
            store.save(
                name=f"llm.{model}",
                type="LLM",
                status=status,
                parent_id=parent_id,
                input=_summarize_messages(messages),
                output=output,
                duration_ms=duration_ms,
                metadata=metadata,
            )

    return traced_create


def _extract_response(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return repr(response)


def _extract_usage(response: Any) -> dict:
    try:
        if hasattr(response, "usage") and response.usage:
            return {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }
    except Exception:
        pass
    return {}


def _summarize_messages(messages: list) -> str:
    summary = []
    total_len = 0
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str):
            snippet = content[:300]
        else:
            snippet = repr(content)[:300]
        total_len += len(snippet)
        summary.append(f"{role}: {snippet}")
        if total_len > 2000:
            summary.append(f"... ({len(messages)} total messages)")
            break
    return "\n".join(summary)