# Contributing to treval

Thank you for your interest in contributing! This guide covers how to set up your environment, what to work on, and how to get your PR merged.

## Development Setup

```bash
git clone https://github.com/AmOrFeU86/treval.git
cd treval
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest py/tests/ -v

# With coverage
pytest py/tests/ -v --cov=treval --cov-report=term
```

## Project Structure

```
treval/
├── py/
│   ├── treval/          # Library source
│   │   ├── agent.py         # @agent decorator
│   │   ├── tool.py          # @tool decorator
│   │   ├── operation.py     # @operation decorator
│   │   ├── instrument.py    # Auto-instrumentation
│   │   ├── wrap.py          # Explicit wrapping
│   │   ├── eval.py          # LLM-as-judge
│   │   ├── compare.py       # Multi-model comparison
│   │   ├── testing.py       # Test suites for agents
│   │   ├── replay.py        # Span replay
│   │   ├── dashboard.py     # Web dashboard
│   │   ├── gateway.py       # HTTP proxy gateway
│   │   ├── otel.py          # OpenTelemetry export
│   │   ├── db.py            # SQLite store
│   │   ├── cli.py           # CLI (15 commands)
│   │   ├── context.py       # Thread-local context
│   │   └── callbacks.py     # Post-save hooks
│   └── tests/           # Test suite
├── pyproject.toml       # Package config
├── README.md            # English docs
├── README.es.md         # Spanish docs
└── LICENSE              # MIT
```

## What to Work On

- **Bug fixes** — always welcome
- **New decorator types** — e.g., `@guardrail`, `@human_in_the_loop`
- **New evaluation criteria** — add to `treval/eval.py`
- **Additional LLM judge models** — support for more providers
- **Documentation** — fixes, examples, tutorials

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Write or update tests for your changes
3. Run `pytest py/tests/ -v` — all tests must pass
4. Keep code and comments in English
5. Open a PR with a clear description

## Code Style

- Python 3.11+ with type hints
- Comments and docstrings in English
- Follow existing patterns (decorators, context managers)

## Questions?

Open a [discussion](https://github.com/AmOrFeU86/treval/discussions) or an issue.
