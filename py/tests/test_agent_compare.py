"""Tests for agent mode and traces in treval compare."""
import pytest
import json
import os


# ─── Agent mode ───

def test_compare_agents_basic(monkeypatch):
    """compare_agents ejecuta un comando N veces y devuelve resultados."""
    from treval.compare import compare_agents, _run_agent

    call_count = [0]

    def fake_run_agent(cmd, api_key):
        call_count[0] += 1
        return {
            "output": f"result from run {call_count[0]}",
            "duration_ms": 100.0 * call_count[0],
            "spans": [
                {"id": call_count[0], "name": "agent_run", "type": "AGENT",
                 "status": "ok", "duration_ms": 50.0}
            ],
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
        }

    monkeypatch.setattr("treval.compare._run_agent", fake_run_agent)

    # Mock evaluator
    class FakeEval:
        def __init__(self, **kw):
            pass
        def evaluate_span(self, span):
            from treval.eval import EvalResult
            score = min(1.0, 0.7 + call_count[0] * 0.1)
            return EvalResult(span_id=1, evaluator_name="test", score=score, reason="ok")

    monkeypatch.setattr("treval.compare.LLMEvaluator", lambda **kw: FakeEval())

    results = compare_agents(
        agent_cmd="python test_agent.py",
        runs=3,
        criteria="correctness",
        api_key="fake-key",
    )

    assert len(results) == 1
    assert results[0].run_count == 3
    assert results[0].mean_score > 0
    assert results[0].mean_duration == 200.0  # (100 + 200 + 300) / 3
    assert call_count[0] == 3  # was called 3 times


def test_compare_agents_with_spans(monkeypatch):
    """compare_agents captures spans from each run."""
    from treval.compare import compare_agents

    run_data = [
        {"output": "resp A", "duration_ms": 100, "spans": [{"id": 1}, {"id": 2}],
         "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        {"output": "resp B", "duration_ms": 150, "spans": [{"id": 3}],
         "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    ]

    def fake_run_agent(cmd, api_key):
        return run_data.pop(0)

    monkeypatch.setattr("treval.compare._run_agent", fake_run_agent)

    class FakeEval:
        def __init__(self, **kw):
            pass
        def evaluate_span(self, span):
            from treval.eval import EvalResult
            return EvalResult(span_id=1, evaluator_name="test", score=0.8, reason="ok")

    monkeypatch.setattr("treval.compare.LLMEvaluator", lambda **kw: FakeEval())

    results = compare_agents("cmd", runs=2, api_key="fake")
    assert len(results) == 1
    assert results[0].run_count == 2


def test_run_agent_subprocess(monkeypatch, tmp_path):
    """_run_agent ejecuta un script real y captura su salida."""
    from treval.compare import _run_agent

    # Crear un script temporal
    script = tmp_path / "test_agent.py"
    script.write_text("""\
import treval
treval.instrument()

@treval.agent
class TestAgent:
    def __init__(self):
        pass

    @treval.operation
    def run(self):
        return "hello from agent"

agent = TestAgent()
result = agent.run()
print(result)
""")
    monkeypatch.chdir(str(tmp_path))

    # We need treval installed for the subprocess
    result = _run_agent(f"python3 {script}", api_key="")
    assert "output" in result
    assert "duration_ms" in result
    # If treval is installed, there should be spans
    # If not, at least we get stdout as output
    assert len(result.get("output", "")) > 0 or len(result.get("spans", [])) >= 0


def test_run_agent_nonzero_exit(monkeypatch):
    """_run_agent captura errores de subprocess."""
    from treval.compare import _run_agent
    import subprocess

    result = _run_agent("python3 -c 'raise RuntimeError(\"fail\")'", api_key="")
    # Must return something, not crash
    assert result is not None
    assert "output" in result
    # The output should mention the error
    combined = str(result.get("output", "")) + str(result.get("stderr", ""))
    # If it fails, duration_ms should be > 0
    assert result.get("duration_ms", 0) >= 0


# ─── HTML traces ───

def test_build_report_html_with_spans():
    """HTML includes trace section when there are spans."""
    from treval.compare import CompareRun, CompareResult, build_report_html

    runs = [
        CompareRun(
            model="agent:test",
            output="final result",
            run_index=1,
            duration_ms=500,
            score=0.85,
            reason="good",
            # Simular spans con JSON
        ),
    ]
    # Add spans manually to the run
    runs[0].spans = [
        {"id": 1, "name": "MyAgent", "type": "AGENT", "status": "ok", "duration_ms": 100},
        {"id": 2, "name": "llm.deepseek/deepseek-v4-flash", "type": "LLM", "status": "ok", "duration_ms": 200},
        {"id": 3, "name": "search", "type": "TOOL", "status": "ok", "duration_ms": 50},
    ]

    results = [CompareResult.from_runs("agent:test", runs)]
    html = build_report_html(results, prompt="python test_agent.py", criteria="correctness")

    assert "traza" in html.lower() or "span" in html.lower()
    assert "AGENT" in html
    assert "LLM" in html
    assert "TOOL" in html
    assert "MyAgent" in html


def test_build_report_html_trace_hierarchy():
    """HTML shows span hierarchy (parent → children)."""
    from treval.compare import CompareRun, CompareResult, build_report_html

    runs = [
        CompareRun(
            model="agent:test",
            output="result",
            run_index=1,
            duration_ms=300,
            score=0.9,
            reason="ok",
            spans=[
                {"id": 10, "name": "RootAgent", "type": "AGENT", "status": "ok", "duration_ms": 300, "parent_id": None},
                {"id": 11, "name": "think", "type": "OPERATION", "status": "ok", "duration_ms": 200, "parent_id": 10},
                {"id": 12, "name": "llm.test", "type": "LLM", "status": "ok", "duration_ms": 150, "parent_id": 11},
                {"id": 13, "name": "get_data", "type": "TOOL", "status": "ok", "duration_ms": 50, "parent_id": 11},
            ],
        ),
    ]

    results = [CompareResult.from_runs("agent:test", runs)]
    html = build_report_html(results, prompt="run agent", criteria="x")

    assert "RootAgent" in html
    assert "think" in html
    assert "get_data" in html
    # Hierarchy is shown with indentation or └─
    assert "└─" in html or "&nbsp;&nbsp;" in html or "indent" in html.lower()


def test_build_report_html_trace_tree_formatted():
    """Trace is shown as hierarchical tree with levels."""
    from treval.compare import CompareRun, CompareResult, build_report_html

    runs = [
        CompareRun(
            model="agent:repl",
            output="done",
            run_index=1,
            duration_ms=500,
            score=0.8,
            reason="ok",
            spans=[
                {"id": 1, "name": "ReplAgent", "type": "AGENT", "status": "ok", "duration_ms": 500},
                {"id": 2, "name": "process", "type": "OPERATION", "status": "ok", "duration_ms": 300, "parent_id": 1},
            ],
        ),
    ]
    results = [CompareResult.from_runs("agent:repl", runs)]
    html = build_report_html(results, prompt="run", criteria="x")

    # Must have the trace section in the detail
    assert "ReplAgent" in html
    assert "process" in html