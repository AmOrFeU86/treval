"""
Callbacks and hooks for integration with agent frameworks (LangChain, CrewAI, etc.).

Usage with LangChain:
    from treval.callbacks import TrevalCallbackHandler
    from langchain.callbacks import CallbackManager

    manager = CallbackManager([TrevalCallbackHandler()])
    chain = LLMChain(llm=llm, callbacks=manager)

Manual usage:
    from treval.callbacks import on_tool_start, on_tool_end, on_llm_start, on_llm_end

    @on_tool_start
    def my_tool(query):
        pass
"""
from __future__ import annotations

import json
import time
from typing import Any

from treval.db import SpanStore
from treval.context import current_span_id, pop_span, push_span


# ─── Hooks globales ───

_tool_start_hooks: list = []
_tool_end_hooks: list = []
_llm_start_hooks: list = []
_llm_end_hooks: list = []


def on_tool_start(func):
    """Registra un hook que se ejecuta cuando un tool comienza."""
    _tool_start_hooks.append(func)
    return func


def on_tool_end(func):
    """Registra un hook que se ejecuta cuando un tool termina."""
    _tool_end_hooks.append(func)
    return func


def on_llm_start(func):
    """Registra un hook que se ejecuta cuando una llamada LLM comienza."""
    _llm_start_hooks.append(func)
    return func


def on_llm_end(func):
    """Registra un hook que se ejecuta cuando una llamada LLM termina."""
    _llm_end_hooks.append(func)
    return func


# ─── LangChain Callback Handler ───

try:
    from langchain_core.callbacks import BaseCallbackHandler

    class TrevalCallbackHandler(BaseCallbackHandler):
        """Callback handler for LangChain that traces everything to treval.

        Usage:
            from treval.callbacks import TrevalCallbackHandler
            chain = LLMChain(llm=llm, callbacks=[TrevalCallbackHandler()])
        """

        def __init__(self):
            super().__init__()
            self.store = SpanStore()
            self._run_spans: dict[str, int] = {}

        def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
            parent_id = current_span_id()
            span_id = self.store.save(
                name=f"llm.{serialized.get('name', 'unknown')}",
                type="LLM",
                status="pending",
                parent_id=parent_id,
                input=json.dumps(prompts[:3]) if prompts else "",
            )
            self._run_spans[id(serialized)] = span_id

        def on_llm_end(self, response, **kwargs) -> None:
            spans = self.store.list(limit=1)
            if spans:
                self.store.update(spans[0]["id"], status="ok",
                                  output=str(response.llm_output or response.generations))

        def on_llm_error(self, error: Exception, **kwargs) -> None:
            spans = self.store.list(limit=1)
            if spans:
                self.store.update(spans[0]["id"], status="error",
                                  output=f"{type(error).__name__}: {error}")

        def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
            parent_id = current_span_id()
            span_id = self.store.save(
                name=serialized.get("name", "tool"),
                type="TOOL",
                status="pending",
                parent_id=parent_id,
                input=str(input_str)[:2000],
            )
            self._run_spans[id(serialized)] = span_id

        def on_tool_end(self, output: str, **kwargs) -> None:
            spans = self.store.list(limit=1)
            if spans:
                self.store.update(spans[0]["id"], status="ok",
                                  output=str(output)[:2000])

        def on_tool_error(self, error: Exception, **kwargs) -> None:
            spans = self.store.list(limit=1)
            if spans:
                self.store.update(spans[0]["id"], status="error",
                                  output=f"{type(error).__name__}: {error}")

        @property
        def always_verbose(self) -> bool:
            return True

        @property
        def ignore_agent(self) -> bool:
            return False

        @property
        def ignore_chain(self) -> bool:
            return False

except ImportError:
    TrevalCallbackHandler = None
    # If langchain-core is not installed, the handler is not available


# ─── Context Manager for manual tracing ───

import builtins
from contextlib import contextmanager


def trace(name: str, type: str = "OPERATION", metadata: dict | None = None):
    """Context manager to trace code blocks manually.

    Usage:
        with treval.trace("process_data", type="OPERATION"):
            data = process(data)
            result = transform(data)
    """
    store = SpanStore()
    parent_id = current_span_id()
    span_id = store.save(name=name, type=type, parent_id=parent_id,
                         metadata=metadata)
    push_span(span_id)
    try:
        yield span_id
    except BaseException as e:
        store.update(span_id, status="error",
                     output=f"{builtins.type(e).__name__}: {e}")
        raise
    finally:
        pop_span()


trace = contextmanager(trace)