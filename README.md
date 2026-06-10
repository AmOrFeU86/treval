# treval ⚡

<p align="center">
  <a href="https://pypi.org/project/treval/"><img src="https://img.shields.io/pypi/v/treval?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/treval/"><img src="https://img.shields.io/pypi/pyversions/treval" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/AmOrFeU86/treval" alt="License: MIT"></a>
</p>

<p align="center">
  <picture>
    <img alt="treval logo" src="https://raw.githubusercontent.com/AmOrFeU86/treval/main/treval_logo_2.png" width="200" height="200">
  </picture>
</p>

[![en](https://img.shields.io/badge/lang-en-red.svg)](README.md)
[![es](https://img.shields.io/badge/lang-es-blue.svg)](README.es.md)

> Trace, evaluate and improve AI agents from the terminal.

Treval is an observability and evaluation framework for AI agents. With one line (`import treval; treval.instrument()`) you get full tracing of every LLM call, tool, and operation. Plus: LLM-as-judge evaluation, multi-model comparison with statistics, API costs, span replay, native agent tests, web dashboard, OpenTelemetry export, and standalone HTML reports.

---

## Installation

```bash
pip install treval
```

**Dependencies:** `openai`, `rich` (the rest are Python 3.11+ stdlib).

You need an API key from [OpenRouter](https://openrouter.ai/keys/) (or OpenAI if using OpenAI directly).

```bash
# In your ~/.bashrc or before running treval
export OPENROUTER_API_KEY=sk-or-v1-...
```

```bash
# Verify installation
treval --help           # 15 commands available
treval prices           # Updated OpenRouter prices
```

---

## Basic Tracing

### Auto-instrumentation (one line)

```python
import treval

treval.instrument()   # Patches OpenAI sync/async → automatic LLM spans

# From now on, ALL OpenAI calls are traced automatically
```

### @agent Decorator

```python
from treval import agent, operation, tool

@agent(name="WeatherBot")
class WeatherAgent:
    def __init__(self, api_key: str):
        from openai import OpenAI
        # OpenRouter as default provider
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    @operation
    def get_forecast(self, city: str) -> str:
        """Each @operation call is recorded as a child span of the agent."""
        return self._call_llm(f"weather in {city}")

    @operation(name="call_llm")
    def _call_llm(self, prompt: str) -> str:
        """LLM calls via OpenAI are traced automatically if you called instrument()."""
        resp = self.client.chat.completions.create(
            model="deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
```

### @tool Decorator

```python
@tool(name="get_weather")
def get_weather(city: str) -> str:
    """Each tool is recorded as a TOOL span."""
    return f"28°C, sunny in {city}"
```

### View Spans

```bash
treval spans                # List the 20 most recent spans
treval spans -t LLM         # Only LLM spans
treval spans -l 50          # 50 spans
treval span 42              # Full span detail (input, output, children)
treval metrics              # Aggregate metrics by type
treval count                # Total stored spans
treval clear                # Delete all spans
```

Spans have 4 types, represented as colored badges in the dashboard:

| Type | Color | Meaning |
|------|-------|---------|
| **AGENT** | 🔵 Blue | Complete agent instance |
| **OPERATION** | 🟢 Green | Operation inside the agent |
| **TOOL** | 🟡 Yellow | Executed tool or function |
| **LLM** | 🟣 Purple | Language model call |

Spans are organized in a parent → child hierarchy automatically via `parent_id`.

---

## LLM-as-Judge Evaluation

```bash
# Evaluate recent spans with DeepSeek as judge
treval eval                             # Default: correctness
treval eval -c conciseness              # Conciseness
treval eval -c helpfulness              # Helpfulness
treval eval -t LLM -c correctness       # Only LLM spans
treval evals                            # Evaluation history
```

Also from Python:

```python
from treval import LLMEvaluator, EvalStore

evaluator = LLMEvaluator(
    model="deepseek/deepseek-v4-flash",
    criteria="The response must be correct and helpful",
)
results = evaluator.evaluate(spans)

store = EvalStore()
store.save(results[0])
stats = store.get_stats()  # mean, min, max
```

The judge uses a tolerant JSON parser that handles malformed JSON (unclosed strings, markdown, extra text). If it fails, it automatically retries up to 2 times.

---

## Model Comparison (`treval compare`)

Compare **N models** on the **same prompt**, each run **M times**, with statistics (mean σ) and real costs from the OpenRouter API.

```bash
# 2 models, 3 runs each
treval compare \
  -p "Explain the difference between CNN and Transformer" \
  -m deepseek/deepseek-v4-flash,deepseek/deepseek-v4-pro \
  -r 3

# 4 models, 5 runs, export to HTML
treval compare \
  -p "what is fine-tuning?" \
  -m deepseek/deepseek-v4-flash,deepseek/deepseek-v4-pro,anthropic/claude-sonnet-4,xiaomi/mimo-v2.5-pro \
  -r 5 \
  -o comparison.html

# With custom criteria
treval compare -p "summarize this" -m m1,m2 -c conciseness
```

**Terminal output:** table with #, model, mean score, σ, duration, cost/run, tokens, runs. Winner marked with 🏆.

**Exported HTML** includes:
- Winner banner with score
- Sortable summary table
- Per-model detail with each individual run
- Expandable output per run
- **Trace tree** (agent mode): full span hierarchy with colored types

### Agent Mode

Compare full runs of an agent script instrumented with treval:

```bash
treval compare --agent "python my_agent.py 'question'" -r 5 -o agents.html
```

Each run:
1. Runs the script as a subprocess
2. Captures stdout (as output)
3. Reads the new spans the agent saved to the DB
4. Evaluates the output with LLM-as-judge
5. Renders the **hierarchical trace tree** in the HTML

---

## Replay (`treval replay`)

Re-execute a saved span by changing model, temperature, or input:

```bash
treval replay 42                          # Re-execute with same params
treval replay 42 --model anthropic/claude-sonnet-4  # Change model
treval replay 42 --input "new question"              # Change input
treval replay 42 --temperature 0.5                   # Change temperature
```

Shows a comparison table: original vs new output, duration, and token usage.

---

## Agent Testing

Define tests for agents using LLM-as-judge:

```python
# tests/test_my_agent.py
from treval.testing import case, TestSuite

suite = TestSuite(name="WeatherTests")

@case(suite,
      input="What's the weather like in Madrid?",
      criteria="The response must mention Madrid's weather")
def test_madrid(response: str) -> None:
    assert "Madrid" in response
    assert "28" in response or "sunny" in response
```

```bash
treval test run tests/test_my_agent.py
```

Each test runs the agent, evaluates the output with LLM-as-judge, and shows ✅/❌ with score and reason.

---

## Dashboard

```bash
treval dashboard                     # Web server at http://127.0.0.1:8080
treval dashboard --port 3000         # Custom port
treval dashboard --no-open           # Don't open browser
treval dashboard --export report.html  # Standalone HTML (works from file://)
```

The exported dashboard is 100% standalone (no server), responsive, with:
- Stats (total, agents, operations, tools, LLMs, errors)
- Sortable table by any column
- Detail panel with input/output and child hierarchy
- Color-coded duration bars
- Span type legend
- Dark mode mobile-friendly design

---

## Gateway Proxy

Intercept LLM traffic to trace it without modifying code:

```bash
treval gateway                       # Proxy on :9090 → OpenRouter
treval gateway --port 9090 --upstream openai   # → OpenAI
```

Useful for agents you can't modify: point their calls to the gateway and treval logs everything.

---

## OpenTelemetry Export

```bash
treval export --console              # Export spans to console (OTEL format)
treval export --endpoint http://localhost:4317  # Send to OTEL collector
```

---

## A/B Comparison (legacy)

```bash
treval ab "my question" --model-a flash --model-b pro
```

Simple comparison of 2 models on the same input. Recommended to use `treval compare` for 2+ models with statistics.

---

## Real-time Prices (`treval prices`)

Fetches updated OpenRouter API prices automatically, without hardcoding:

```bash
treval prices                          # All available models
treval prices --search flash           # Filter by name
treval prices --search deepseek        # Only DeepSeek models
treval prices --search xiaomi          # Only Xiaomi MiMo
```

Prices are cached for 1 hour in memory. If the API doesn't respond, a local fallback with ~20 common models is used. Costs in `treval compare` use these prices automatically.

---

## Public API (Python)

```python
import treval

# Decorators
treval.instrument()               # Auto-instrumentation OpenAI
treval.agent                      # @treval.agent — marks a class as an agent
treval.operation                  # @treval.operation — marks a method as an operation
treval.tool                       # @treval.tool — marks a function as a tool
treval.wrap(client)               # Wraps an existing OpenAI client
treval.wrap_anthropic(client)     # Wraps an existing Anthropic client

# Evaluation
treval.LLMEvaluator               # LLM-as-judge evaluator
treval.EvalStore                  # SQLite evaluation store

# Callbacks
treval.trace                      # Tracing callback
treval.on_tool_start / on_tool_end
treval.on_llm_start / on_llm_end

# Comparison (from Python)
from treval.compare import compare_models, compare_agents, build_report_html
results = compare_models(prompt="...", models=["m1", "m2"], runs=3)
html = build_report_html(results, prompt="...", criteria="correctness")
```

---

## Demo: ReAct Agent

```bash
export OPENROUTER_API_KEY=sk-or-...
cd py
python demo_react.py "What's the weather like in Madrid?"
python demo_react.py "3 * 7 + 12"
python demo_react.py "What is the capital of Spain?"
```

Functional demo of a ReAct agent with 3 tools (weather, calculator, search) instrumented with treval. After running it:

```bash
treval spans         # View all generated spans
treval span 1        # Agent detail
treval eval          # Evaluate with LLM-as-judge
```

---

## Commands (15)

| Command | Description |
|---------|-------------|
| `treval spans` | List recent spans (filter by type) |
| `treval span <id>` | Span detail with children |
| `treval count` | Total stored spans |
| `treval clear` | Delete all spans |
| `treval eval` | Evaluate spans with LLM-as-judge |
| `treval evals` | Evaluation history |
| `treval compare` | Compare N models × M runs |
| `treval ab` | Simple A/B comparison (legacy) |
| `treval replay <id>` | Re-execute a span with new params |
| `treval test run <file>` | Run agent tests |
| `treval dashboard` | Web dashboard / HTML export |
| `treval metrics` | Aggregate metrics |
| `treval prices` | OpenRouter API prices |
| `treval export` | Export spans to OTEL |
| `treval gateway` | Proxy to intercept LLM traffic |

---

## Storage

Everything is saved locally in `~/.treval/`:

```
~/.treval/
├── spans.db       # Traces (spans with parent→child hierarchy)
└── evals.db       # LLM-as-judge evaluations
```

SQLite, thread-safe, no server. You can delete the files at any time or use `treval clear` (only clears spans; evaluations are in a separate `evals.db`).

---

## Architecture

```
treval/
├── py/
│   ├── treval/
│   │   ├── __init__.py    # Public API (decorators + instrument + eval)
│   │   ├── agent.py       # @agent — decorator for agent classes
│   │   ├── operation.py   # @operation — decorator for methods
│   │   ├── tool.py        # @tool — decorator for functions
│   │   ├── instrument.py  # OpenAI sync/async auto-instrumentation
│   │   ├── wrap.py        # Wrappers for existing clients
│   │   ├── context.py     # Thread-local span_id stack
│   │   ├── db.py          # Local SQLite (~/.treval/spans.db)
│   │   ├── eval.py        # LLM-as-judge (tolerant JSON parser) + EvalStore
│   │   ├── compare.py     # Multi-model + agent comparison + HTML report
│   │   ├── replay.py      # Re-execute spans with modified params
│   │   ├── testing.py     # Native TestRunner with @case and TestSuite
│   │   ├── callbacks.py   # Tracing callbacks (LangChain compatible)
│   │   ├── otel.py        # OpenTelemetry exporter
│   │   ├── gateway.py     # HTTP proxy to intercept LLM traffic
│   │   ├── dashboard.py   # Web dashboard + standalone HTML export
│   │   └── cli.py         # CLI with Rich (15 commands)
│   ├── tests/             # 88 tests, all passing
│   └── demo_react.py      # Demo: functional ReAct agent with 3 tools
├── ts/                    # TypeScript skeleton (future)
└── pyproject.toml         # Package configuration
```

### Data Flow

```
LLM call
  │
  ├─ instrument() patches OpenAI → LLM span saved in SpanStore
  ├─ @agent / @operation / @tool → AGENT/OPERATION/TOOL span
  │
  ▼
SpanStore (SQLite) ─→ CLI (treval spans / span / metrics)
                  ─→ Dashboard (localhost:8080 or standalone HTML)
                  ─→ LLM-as-judge → EvalStore
                  ─→ compare_models() → HTML report with stats and costs
                  ─→ OTEL export (console or collector)
                  ─→ Replay (re-execute with new params)
```

---

## Tests

```bash
cd py
python -m pytest tests/ -v
```

**88 tests**, all passing. Developed with strict TDD: every new feature starts with a RED test, then GREEN implementation, then refactor.

Coverage: decorators (`@agent`, `@operation`, `@tool`), auto-instrumentation, storage, evaluation, comparison (models + agent + API prices), replay, testing, HTML generation, tolerant JSON parser.

---

## License

MIT