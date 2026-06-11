"""Decorator @treval.tool for tracing agent tool calls."""

import functools
import time

from treval.context import current_span_id
from treval.db import SpanStore


def tool(func=None, *, name=None, metadata_fn=None):
    """Decorator that marks a function as a traceable tool.

    Basic usage:
        @treval.tool
        def my_tool(query: str) -> str:
            return search(query)

    With custom name:
        @treval.tool(name="search")
        def my_tool(query: str) -> str:
            return search(query)

    With metadata callback:
        @treval.tool(metadata_fn=lambda args, kwargs, result: {
            "query": kwargs.get("query"),
            "num_results": len(result),
        })
        def my_tool(query: str) -> list:
            ...

    The metadata_fn is called after a successful return with
    (args, kwargs, result) and must return a dict (serialized as JSON in
    the span's metadata column). On exception it is NOT called, leaving
    metadata as NULL.
    """
    if func is None:
        return lambda f: tool(f, name=name, metadata_fn=metadata_fn)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        store = SpanStore()
        parent_id = current_span_id()
        start = time.perf_counter()
        status = "ok"
        output = None
        metadata = None
        try:
            result = func(*args, **kwargs)
            output = _serialize(result)
            if metadata_fn is not None:
                try:
                    metadata = metadata_fn(args, kwargs, result)
                except Exception:
                    metadata = None
            return result
        except BaseException as e:
            status = "error"
            output = f"{type(e).__name__}: {e}"
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            store.save(
                name=name or func.__name__,
                type="TOOL",
                status=status,
                parent_id=parent_id,
                input=_serialize((args, kwargs)),
                output=output,
                duration_ms=duration_ms,
                metadata=metadata,
            )

    wrapper._treval_tool = True
    wrapper._treval_name = name or func.__name__
    return wrapper


def _serialize(obj) -> str:
    try:
        return repr(obj)
    except Exception:
        return "<unserializable>"