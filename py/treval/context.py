"""Thread-local context for span hierarchy.

Maintains a stack of span_ids to establish
the parent-child relationship between spans.
"""

import threading

_local = threading.local()


def current_span_id() -> int | None:
    """ID of the active span at this moment, or None."""
    stack = getattr(_local, "span_stack", None)
    return stack[-1] if stack else None


def push_span(span_id: int) -> None:
    """Push a span_id onto the stack (new active parent)."""
    if not hasattr(_local, "span_stack"):
        _local.span_stack = []
    _local.span_stack.append(span_id)


def pop_span() -> int | None:
    """Pop the current span_id from the stack."""
    stack = getattr(_local, "span_stack", None)
    return stack.pop() if stack else None