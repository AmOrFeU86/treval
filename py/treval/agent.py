"""Decorator @treval.agent for agent classes."""

import functools

from treval.context import pop_span, push_span
from treval.db import SpanStore


_SPAN_ID_ATTR = "_treval_span_id"


def agent(cls=None, *, name=None):
    """Decorator that marks a class as a traceable agent."""
    if cls is None:
        return lambda c: agent(c, name=name)

    original_init = cls.__init__

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        store = SpanStore()
        span_id = store.save(
            name=name or cls.__name__,
            type="AGENT",
            input=_serialize((args, kwargs)),
        )
        # Store the span_id on the instance so @operation can use it
        setattr(self, _SPAN_ID_ATTR, span_id)
        push_span(span_id)
        try:
            original_init(self, *args, **kwargs)
        except BaseException as e:
            store.save(name=name or cls.__name__, type="AGENT",
                       status="error", output=f"{type(e).__name__}: {e}")
            store.update(span_id, status="error",
                         output=f"{type(e).__name__}: {e}")
            raise
        finally:
            pop_span()

    cls.__init__ = new_init
    cls._treval_agent = True
    return cls


def get_agent_span_id(instance) -> int | None:
    """Gets the agent span_id from an instance."""
    return getattr(instance, _SPAN_ID_ATTR, None)


def _serialize(obj) -> str:
    try:
        return repr(obj)
    except Exception:
        return "<unserializable>"