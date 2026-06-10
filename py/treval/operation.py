"""Decorator @treval.operation for methods inside an agent."""

import functools
import time

from treval.agent import get_agent_span_id
from treval.context import current_span_id, pop_span, push_span
from treval.db import SpanStore


def operation(func=None, *, name=None):
    """Decorator that marks a method as a traceable operation inside an agent."""
    if func is None:
        return lambda f: operation(f, name=name)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        store = SpanStore()
        parent_id = current_span_id() or get_agent_span_id(self)
        start = time.perf_counter()
        status = "ok"
        output = None

        # Create span and push it so nested @tool/@operation can see it
        span_id = store.save(
            name=name or func.__name__,
            type="OPERATION",
            parent_id=parent_id,
            input=_serialize((args, kwargs)),
        )
        push_span(span_id)
        try:
            result = func(self, *args, **kwargs)
            output = _serialize(result)
            return result
        except BaseException as e:
            status = "error"
            output = f"{type(e).__name__}: {e}"
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            store.update(span_id, status=status, output=output,
                         duration_ms=duration_ms)
            pop_span()

    wrapper._treval_operation = True
    return wrapper


def _serialize(obj) -> str:
    try:
        return repr(obj)
    except Exception:
        return "<unserializable>"