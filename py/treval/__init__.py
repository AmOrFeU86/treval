"""treval - Trace, evaluate and improve AI agents from the terminal."""

from treval.agent import agent
from treval.instrument import instrument
from treval.operation import operation
from treval.tool import tool
from treval.wrap import wrap, wrap_anthropic
from treval.callbacks import trace, on_tool_start, on_tool_end, on_llm_start, on_llm_end
from treval.eval import LLMEvaluator, EvalStore

# Export LangChain handler if available
try:
    from treval.callbacks import TrevalCallbackHandler
except ImportError:
    TrevalCallbackHandler = None

__all__ = [
    "agent", "operation", "tool", "instrument",
    "wrap", "wrap_anthropic",
    "trace",
    "on_tool_start", "on_tool_end", "on_llm_start", "on_llm_end",
    "LLMEvaluator", "EvalStore",
    "TrevalCallbackHandler",
]