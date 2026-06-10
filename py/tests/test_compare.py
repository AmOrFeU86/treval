"""Tests for treval compare — multi-model comparison with stats and costs."""
import pytest
from treval.compare import (
    _FALLBACK_PRICES,
    calculate_cost,
    format_cost,
    compute_stats,
    CompareRun,
    CompareResult,
    build_report_html,
    compare_models,
    fetch_model_prices,
    get_model_price,
    list_available_models,
)


# ─── Model pricing / cost ───

def test_calculate_cost_known_model(monkeypatch):
    """Coste para modelo conocido usa sus precios."""
    from treval.compare import get_model_price
    monkeypatch.setattr("treval.compare.get_model_price", lambda m, k=None: (0.30, 0.60))
    cost = calculate_cost("deepseek/deepseek-v4-flash", 1000, 500)
    assert abs(cost - 0.00060) < 0.00001


def test_calculate_cost_unknown_model(monkeypatch):
    """Unknown model uses default price (more expensive to be conservative)."""
    monkeypatch.setattr("treval.compare.get_model_price", lambda m, k=None: (5.00, 15.00))
    cost = calculate_cost("unknown/model", 1000, 500)
    assert cost > 0


def test_calculate_cost_zero_tokens(monkeypatch):
    """Sin tokens, coste cero."""
    monkeypatch.setattr("treval.compare.get_model_price", lambda m, k=None: (0.30, 0.60))
    assert calculate_cost("deepseek/deepseek-v4-flash", 0, 0) == 0.0


def test_calculate_cost_only_input(monkeypatch):
    """Solo tokens de input."""
    monkeypatch.setattr("treval.compare.get_model_price", lambda m, k=None: (0.30, 0.60))
    cost = calculate_cost("deepseek/deepseek-v4-flash", 2000, 0)
    assert abs(cost - 0.00060) < 0.00001


def test_format_cost_zero():
    assert format_cost(0.0) == "$0.00"


def test_format_cost_small():
    # $0.00060 con 8 decimales
    assert format_cost(0.00060) == "$0.00060000"


def test_format_cost_dollar():
    assert format_cost(1.50) == "$1.50"


# ─── Stats computation ───

def test_compute_stats_empty():
    """Empty list → zero stats."""
    s = compute_stats([])
    assert abs(s["mean"] - 0.0) < 0.001
    assert abs(s["std"] - 0.0) < 0.001
    assert s["min"] == 0.0
    assert s["max"] == 0.0


def test_compute_stats_single_value():
    """Un solo valor → std = 0."""
    s = compute_stats([0.85])
    assert abs(s["mean"] - 0.85) < 0.001
    assert abs(s["std"] - 0.0) < 0.001
    assert abs(s["min"] - 0.85) < 0.001
    assert abs(s["max"] - 0.85) < 0.001


def test_compute_stats_multiple():
    """Multiple values with variation."""
    s = compute_stats([0.9, 0.8, 0.7])
    assert abs(s["mean"] - 0.80) < 0.001
    assert abs(s["min"] - 0.70) < 0.001
    assert abs(s["max"] - 0.90) < 0.001
    assert s["std"] > 0  # must have deviation


def test_compute_stats_identical():
    """Identical values → std = 0."""
    s = compute_stats([0.75, 0.75, 0.75])
    assert abs(s["std"] - 0.0) < 0.001
    assert abs(s["mean"] - 0.75) < 0.001


# ─── CompareRun dataclass ───

def test_compare_run_defaults():
    """CompareRun con valores por defecto."""
    r = CompareRun(model="test/model", output="ok")
    assert r.run_index == 0
    assert r.duration_ms == 0.0
    assert r.score == 0.0
    assert r.prompt_tokens == 0
    assert r.completion_tokens == 0
    assert r.total_tokens == 0
    assert r.cost == 0.0
    assert r.reason == ""


def test_compare_run_with_cost():
    """CompareRun calculates cost automatically."""
    r = CompareRun(
        model="deepseek/deepseek-v4-flash",
        output="respuesta",
        prompt_tokens=1000,
        completion_tokens=500,
        run_index=1,
    )
    assert r.cost > 0
    assert r.run_index == 1
    assert r.output == "respuesta"


# ─── CompareResult ───

def test_compare_result_from_runs():
    """CompareResult calcula stats desde runs."""
    runs = [
        CompareRun(model="m1", output="a", score=0.9),
        CompareRun(model="m1", output="b", score=0.8),
        CompareRun(model="m1", output="c", score=0.7),
    ]
    result = CompareResult.from_runs("m1", runs)
    assert result.model == "m1"
    assert abs(result.mean_score - 0.80) < 0.001
    assert result.mean_duration == 0.0
    assert result.run_count == 3
    assert result.mean_cost == 0.0


def test_compare_result_from_runs_with_durations():
    runs = [
        CompareRun(model="m1", output="a", score=0.8, duration_ms=100.0, prompt_tokens=500, completion_tokens=200),
        CompareRun(model="m1", output="b", score=0.9, duration_ms=200.0, prompt_tokens=500, completion_tokens=200),
    ]
    result = CompareResult.from_runs("m1", runs)
    assert abs(result.mean_duration - 150.0) < 0.001
    assert result.run_count == 2
    assert result.mean_cost > 0


# ─── HTML report ───

def test_build_report_html_basic():
    """El HTML se genera sin errores y contiene datos clave."""
    runs_m1 = [
        CompareRun(model="m1", output="resp A", score=0.85, duration_ms=100, prompt_tokens=500, completion_tokens=200),
        CompareRun(model="m1", output="resp B", score=0.90, duration_ms=150, prompt_tokens=500, completion_tokens=200),
    ]
    runs_m2 = [
        CompareRun(model="m2", output="resp C", score=0.75, duration_ms=200, prompt_tokens=500, completion_tokens=200),
    ]
    results = [
        CompareResult.from_runs("m1", runs_m1),
        CompareResult.from_runs("m2", runs_m2),
    ]
    html = build_report_html(results, prompt="test prompt", criteria="correctness")
    assert "<!DOCTYPE html>" in html
    assert "test prompt" in html
    assert "m1" in html
    assert "m2" in html
    assert "0.85" in html
    assert "0.75" in html
    assert "0.90" in html  # best score
    assert html.count("<!DOCTYPE html>") == 1  # well-formed


def test_build_report_html_no_results():
    """Empty list produces HTML with info message."""
    html = build_report_html([], prompt="test", criteria="x")
    assert "<!DOCTYPE html>" in html
    assert "no" in html.lower() or "No hay" in html or "sin datos" in html.lower()


def test_build_report_html_highlights_winner():
    """El mejor modelo debe aparecer destacado."""
    runs_best = [CompareRun(model="best-model", output="best", score=0.95, duration_ms=100, prompt_tokens=500, completion_tokens=200)]
    runs_worst = [CompareRun(model="worst-model", output="worst", score=0.30, duration_ms=200, prompt_tokens=500, completion_tokens=200)]
    results = [
        CompareResult.from_runs("best-model", runs_best),
        CompareResult.from_runs("worst-model", runs_worst),
    ]
    html = build_report_html(results, prompt="test", criteria="x")
    assert "best-model" in html
    assert "worst-model" in html
    # El mejor score debe aparecer
    assert "0.95" in html
    assert "0.30" in html


# ─── compare_models con mocking (sin LLM real) ───

def test_compare_models_single_model_mocked(monkeypatch):
    """Sin llamadas reales, verify estructura del resultado."""
    import treval.compare as cmp

    def fake_llm(prompt, model, api_key):
        return {
            "output": f"respuesta from {model}",
            "duration_ms": 150.0,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    monkeypatch.setattr(cmp, "_call_llm", fake_llm)

    # Mock del evaluador para que devuelva scores deterministicos
    class FakeEvaluator:
        def __init__(self, **kw):
            pass
        def evaluate_span(self, span):
            from treval.eval import EvalResult
            return EvalResult(span_id=1, evaluator_name="test", score=0.85, reason="ok")

    monkeypatch.setattr(cmp, "LLMEvaluator", lambda **kw: FakeEvaluator())

    results = compare_models(
        prompt="test prompt",
        models=["test/model"],
        runs=2,
        criteria="correctness",
        api_key="fake-key",
    )
    assert len(results) == 1
    assert results[0].model == "test/model"
    assert results[0].run_count == 2
    assert results[0].mean_score > 0


def test_compare_models_multiple_models_mocked(monkeypatch):
    """Dos modelos, verify que ambos aparecen en resultados."""
    import treval.compare as cmp

    call_count = {"count": 0}

    def fake_llm(prompt, model, api_key):
        call_count["count"] += 1
        return {
            "output": f"resp from {model}",
            "duration_ms": 100.0 + call_count["count"] * 10,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    monkeypatch.setattr(cmp, "_call_llm", fake_llm)

    class FakeEvaluator:
        def __init__(self, **kw):
            pass
        def evaluate_span(self, span):
            from treval.eval import EvalResult
            return EvalResult(span_id=1, evaluator_name="test", score=0.80, reason="ok")

    monkeypatch.setattr(cmp, "LLMEvaluator", lambda **kw: FakeEvaluator())

    results = compare_models(
        prompt="test",
        models=["model-a", "model-b"],
        runs=1,
        api_key="fake",
    )
    assert len(results) == 2
    model_names = [r.model for r in results]
    assert "model-a" in model_names
    assert "model-b" in model_names


# ─── Edge cases ───

def test_compare_models_no_runs(monkeypatch):
    """runs=0 should not make calls."""
    import treval.compare as cmp
    called = []

    def fake_llm(prompt, model, api_key):
        called.append(True)
        return {"output": "", "duration_ms": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    monkeypatch.setattr(cmp, "_call_llm", fake_llm)

    results = compare_models(prompt="test", models=["m1"], runs=0, api_key="fake")
    assert len(called) == 0
    assert len(results) == 0 or (len(results) == 1 and results[0].run_count == 0)


def test_model_prices_has_common_models():
    """Los modelos que usamos deben estar en el fallback de precios."""
    assert "deepseek/deepseek-v4-flash" in _FALLBACK_PRICES
    assert "deepseek/deepseek-v4-pro" in _FALLBACK_PRICES


# ─── API price fetching (mocked) ───

def test_fetch_model_prices_success(monkeypatch):
    """fetch_model_prices parsea correctamente la respuesta de la API."""
    import json
    import urllib.request
    fake_response = json.dumps({
        "data": [
            {
                "id": "test/model-a",
                "pricing": {"prompt": "0.0000003", "completion": "0.0000006"},
            },
            {
                "id": "test/model-b",
                "pricing": {"prompt": "0.000002", "completion": "0.000008"},
            },
        ]
    })
    class FakeResponse:
        def read(self):
            return fake_response.encode()
        def __exit__(self, *a):
            pass
        def __enter__(self):
            return self

    class FakeUrlopen:
        def __call__(self, req, **kw):
            return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", FakeUrlopen())

    prices = fetch_model_prices(api_key="fake-key")
    assert "test/model-a" in prices
    assert "test/model-b" in prices
    assert abs(prices["test/model-a"][0] - 0.30) < 0.01  # 0.0000003 * 1M
    assert abs(prices["test/model-a"][1] - 0.60) < 0.01
    assert abs(prices["test/model-b"][0] - 2.00) < 0.01
    assert abs(prices["test/model-b"][1] - 8.00) < 0.01


def test_fetch_model_prices_network_error(monkeypatch):
    """If API fails, returns empty dict."""
    import urllib.request
    def fake_fail(*a, **kw):
        raise OSError("Network error")
    monkeypatch.setattr(urllib.request, "urlopen", fake_fail)

    prices = fetch_model_prices()
    assert prices == {}


def test_get_model_price_uses_api(monkeypatch):
    """get_model_price debe priorizar datos de API sobre fallback."""
    import treval.compare as cmp

    # Limpiar cache
    cmp._API_PRICES_CACHE = {"test/api-model": (1.00, 2.00)}
    cmp._API_CACHE_TIME = 9999999999.0  # futuro (no expira)

    inp, out = get_model_price("test/api-model")
    assert abs(inp - 1.00) < 0.01
    assert abs(out - 2.00) < 0.01


def test_get_model_price_fallback(monkeypatch):
    """If not in API cache, should fall back to fallback."""
    import treval.compare as cmp
    # Empty cache (simulate API didn't respond)
    cmp._API_PRICES_CACHE = {}
    cmp._API_CACHE_TIME = 9999999999.0

    inp, out = get_model_price("deepseek/deepseek-v4-flash")
    assert abs(inp - 0.30) < 0.01
    assert abs(out - 0.60) < 0.01


def test_get_model_price_default(monkeypatch):
    """Unknown model without API or fallback → conservative price."""
    import treval.compare as cmp
    cmp._API_PRICES_CACHE = {}
    cmp._API_CACHE_TIME = 9999999999.0

    inp, out = get_model_price("completely/unknown-model")
    assert inp > 0
    assert out > 0


def test_list_available_models_from_api(monkeypatch):
    """list_available_models devuelve modelos desde API cache."""
    import treval.compare as cmp
    cmp._API_PRICES_CACHE = {"m1": (0.5, 1.0), "m2": (1.0, 2.0)}
    cmp._API_CACHE_TIME = 9999999999.0

    models = list_available_models()
    assert len(models) == 2
    ids = [m["id"] for m in models]
    assert "m1" in ids
    assert "m2" in ids
    assert all(m["source"] == "api" for m in models)