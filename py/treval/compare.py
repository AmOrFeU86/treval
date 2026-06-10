"""
treval compare — Multi-model comparison with stats and costs.

Compares N models on the same prompt, each run M times,
with LLM-as-judge scoring, statistics (mean, std dev), estimated cost,
and export to standalone HTML.

Usage (via CLI):
    treval compare --prompt "question" --models m1,m2,m3 --runs 3 --export report.html
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

from treval.eval import EvalResult, LLMEvaluator

# ─── Model pricing ───────────────────────────────────────────────────────────
# Prices are fetched from the OpenRouter API automatically.
# This dictionary is only a fallback if the API does not respond.

_FALLBACK_PRICES: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-v4-flash":          (0.30,   0.60),
    "deepseek/deepseek-v4-pro":            (2.00,   8.00),
    "deepseek/deepseek-chat":              (0.27,   1.10),
    "deepseek/deepseek-r1":                (0.55,   2.19),
    "anthropic/claude-sonnet-4":           (3.00,  15.00),
    "anthropic/claude-3.5-sonnet":         (3.00,  15.00),
    "anthropic/claude-3-haiku":            (0.25,   1.25),
    "anthropic/claude-opus-4":             (15.00, 75.00),
    "openai/gpt-4o":                       (2.50,  10.00),
    "openai/gpt-4o-mini":                  (0.15,   0.60),
    "openai/o3-mini":                      (1.10,   4.40),
    "openai/o1":                           (15.00, 60.00),
    "meta/llama-3.3-70b-instruct":         (0.26,   1.04),
    "meta/llama-3.1-8b-instruct":          (0.06,   0.24),
    "google/gemini-2.0-flash-001":         (0.10,   0.40),
    "google/gemini-2.0-pro-exp-02-05":     (2.00,  10.00),
    "mistral/mistral-large-2411":          (2.00,   6.00),
    "mistral/mistral-small-latest":        (0.20,   0.60),
    "cohere/command-r-plus":               (2.50,  10.00),
    "qwen/qwen-2.5-72b-instruct":          (0.35,   1.40),
    "qwen/qwen-2.5-32b-instruct":          (0.35,   1.40),
    "xiaomi/mimo-v2.5-pro":                (1.50,   5.00),
    "xiaomi/mimo-v2.5":                    (0.50,   1.50),
}

# Cache de precios desde la API
_API_PRICES_CACHE: dict[str, tuple[float, float]] | None = None
_API_CACHE_TIME: float = 0.0
_API_CACHE_TTL: float = 3600.0  # 1 hour

_DEFAULT_INPUT_PRICE = 5.00   # $/1M tokens — conservative for unlisted models
_DEFAULT_OUTPUT_PRICE = 15.00


def fetch_model_prices(api_key: str | None = None) -> dict[str, tuple[float, float]]:
    """Obtiene los precios de todos los modelos desde la API de OpenRouter.

    Returns:
        Dict {model_id: (input_price_per_M, output_price_per_M)}.
        Los precios de la API vienen por token, los convertimos a $/1M tokens.
    """
    import urllib.request
    import json as _json

    url = "https://openrouter.ai/api/v1/models"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode())

        prices: dict[str, tuple[float, float]] = {}
        for model in data.get("data", []):
            mid = model.get("id", "")
            pricing = model.get("pricing", {})
            if mid and pricing:
                inp = float(pricing.get("prompt", 0)) * 1_000_000
                out = float(pricing.get("completion", 0)) * 1_000_000
                if inp > 0 or out > 0:
                    prices[mid] = (inp, out)
        return prices
    except Exception:
        return {}


def get_model_price(model_id: str, api_key: str | None = None) -> tuple[float, float]:
    """Returns the price (input, output) in $/1M tokens for a model.

    Priority:
    1. API cache (if not expired)
    2. OpenRouter API (live)
    3. Hardcoded fallback
    4. Conservative default price
    """
    global _API_PRICES_CACHE, _API_CACHE_TIME

    now = time.time()
    # Refresh cache if expired
    if _API_PRICES_CACHE is None or (now - _API_CACHE_TIME) > _API_CACHE_TTL:
        try:
            fetched = fetch_model_prices(api_key)
            if fetched:
                _API_PRICES_CACHE = fetched
                _API_CACHE_TIME = now
        except Exception:
            pass

    # 1. API cache
    if _API_PRICES_CACHE and model_id in _API_PRICES_CACHE:
        return _API_PRICES_CACHE[model_id]

    # 2. Hardcoded fallback
    if model_id in _FALLBACK_PRICES:
        return _FALLBACK_PRICES[model_id]

    # 3. Conservative default
    return (_DEFAULT_INPUT_PRICE, _DEFAULT_OUTPUT_PRICE)


def list_available_models(api_key: str | None = None) -> list[dict]:
    """Returns list of models with their prices, from API or fallback.

    Each entry: {"id": str, "input_price": float, "output_price": float, "source": str}
    """
    global _API_PRICES_CACHE, _API_CACHE_TIME

    now = time.time()
    models: list[dict] = []

    # Try API first
    if _API_PRICES_CACHE is None or (now - _API_CACHE_TIME) > _API_CACHE_TTL:
        try:
            fetched = fetch_model_prices(api_key)
            if fetched:
                _API_PRICES_CACHE = fetched
                _API_CACHE_TIME = now
        except Exception:
            pass

    if _API_PRICES_CACHE:
        for mid, (inp, out) in sorted(_API_PRICES_CACHE.items()):
            models.append({"id": mid, "input_price": inp, "output_price": out, "source": "api"})
        return models

    # Fallback to hardcoded
    for mid, (inp, out) in sorted(_FALLBACK_PRICES.items()):
        models.append({"id": mid, "input_price": inp, "output_price": out, "source": "fallback"})
    return models


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int,
                   api_key: str | None = None) -> float:
    """Calculates estimated cost in USD for a call, using the API if available."""
    input_price, output_price = get_model_price(model, api_key)
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


def format_cost(cost: float) -> str:
    """Format a cost in USD with adaptive precision."""
    if cost <= 0:
        return "$0.00"
    if cost >= 1.0:
        return f"${cost:.2f}"
    if cost >= 0.01:
        return f"${cost:.4f}"
    # For very small prices (e.g. $0.0000003 / token)
    return f"${cost:.8f}"


# ─── Statistics ────────────────────────────────────────────────────────────

def compute_stats(values: list[float]) -> dict[str, float]:
    """Calculates mean, standard deviation, min and max of a list."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        std = 0.0
    else:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance)
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "n": n,
    }


# ─── Data classes ──────────────────────────────────────────────────────────

@dataclass
class CompareRun:
    """Result of a single run within a comparison."""
    model: str
    output: str
    run_index: int = 0
    duration_ms: float = 0.0
    score: float = 0.0
    reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    spans: list[dict] | None = None  # Trace of captured spans (agent mode)
    stderr: str = ""                 # Stderr from subprocess (agent mode)

    def __post_init__(self):
        if self.cost == 0.0 and (self.prompt_tokens or self.completion_tokens):
            self.cost = calculate_cost(self.model, self.prompt_tokens, self.completion_tokens)


@dataclass
class CompareResult:
    """Aggregated results for a model after M runs."""
    model: str
    runs: list[CompareRun] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    mean_duration: float = 0.0
    std_duration: float = 0.0
    mean_cost: float = 0.0
    total_cost: float = 0.0
    run_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_runs(cls, model: str, runs: list[CompareRun]) -> "CompareResult":
        """Aggregates individual runs into a CompareResult with statistics."""
        if not runs:
            return cls(model=model)

        scores = [r.score for r in runs]
        durations = [r.duration_ms for r in runs]
        costs = [r.cost for r in runs]

        score_stats = compute_stats(scores)
        dur_stats = compute_stats(durations)
        cost_stats = compute_stats(costs)

        return cls(
            model=model,
            runs=runs,
            mean_score=score_stats["mean"],
            std_score=score_stats["std"],
            min_score=score_stats["min"],
            max_score=score_stats["max"],
            mean_duration=dur_stats["mean"],
            std_duration=dur_stats["std"],
            mean_cost=cost_stats["mean"],
            total_cost=sum(costs),
            run_count=len(runs),
            total_prompt_tokens=sum(r.prompt_tokens for r in runs),
            total_completion_tokens=sum(r.completion_tokens for r in runs),
            total_tokens=sum(r.total_tokens for r in runs),
        )


# ─── Core comparison logic ─────────────────────────────────────────────────

def _call_llm(prompt: str, model: str, api_key: str) -> dict[str, Any]:
    """Calls an LLM via OpenRouter and returns result + metrics.

    This function is mockable in tests.
    """
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://treval.dev",
            "X-Title": "treval-compare",
        },
    )
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1000,
    )
    duration_ms = (time.perf_counter() - start) * 1000
    output = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

    return {
        "output": output,
        "duration_ms": duration_ms,
        "prompt_tokens": prompt_tokens or 0,
        "completion_tokens": completion_tokens or 0,
        "total_tokens": total_tokens or 0,
    }


def _score_output(output: str, prompt: str, evaluator: LLMEvaluator, span_id: int, retries: int = 2) -> tuple[float, str]:
    """Evaluates an output using LLM-as-judge, with retries on failure.

    Args:
        output: Text generated by the model to evaluate.
        prompt: Original prompt.
        evaluator: LLMEvaluator instance.
        span_id: ID for the dummy span.
        retries: Number of retries if judge returns score ~0.5 with error.

    Returns:
        Tuple (score, reason).
    """
    fake_span = {
        "id": span_id,
        "input": prompt,
        "output": output,
        "type": "LLM",
    }
    result = evaluator.evaluate_span(fake_span)
    if result is None:
        return 0.0, "Could not evaluate"

    # If the result looks failed and retries remain, retry
    if retries > 0 and ("Error" in result.reason or "Could not" in result.reason):
        for attempt in range(retries):
            import time as _time
            _time.sleep(1)  # wait 1s between retries
            result = evaluator.evaluate_span(fake_span)
            if result and "Error" not in result.reason and "Could not" not in result.reason:
                break

    if result:
        return result.score, result.reason
    return 0.0, "Could not evaluate after retries"


# ─── Agent mode ───────────────────────────────────────────────────────────

def _run_agent(agent_cmd: str, api_key: str | None = None,
               timeout: int = 180) -> dict:
    """Runs an agent script as subprocess and captures its result + spans.

    The script must be instrumented with treval so that spans
    are saved in the DB. This function retrieves them after execution.

    Returns:
        Dict with output, duration_ms, spans, prompt_tokens, completion_tokens.
    """
    import subprocess
    import time as _time

    from treval.db import SpanStore

    # Record current spans before execution
    store = SpanStore()
    before_count = store.count()
    # List existing spans to know the last ID
    existing = store.list_spans(limit=1)
    before_max_id = existing[0]["id"] if existing else 0

    start = _time.perf_counter()
    try:
        proc = subprocess.run(
            agent_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration_ms = (_time.perf_counter() - start) * 1000
        output = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired:
        duration_ms = (_time.perf_counter() - start) * 1000
        return {
            "output": "",
            "duration_ms": duration_ms,
            "spans": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "stderr": f"TIMEOUT after {timeout}s",
        }
    except Exception as e:
        duration_ms = (_time.perf_counter() - start) * 1000
        return {
            "output": "",
            "duration_ms": duration_ms,
            "spans": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "stderr": str(e),
        }

    # Retrieve new spans (created during this execution)
    new_spans = []
    try:
        all_spans = store.list_spans(limit=5000)
        new_spans = [s for s in all_spans if s["id"] > before_max_id]
        # Calculate tokens from LLM spans
        total_pt = sum(
            json.loads(s.get("metadata", "{}")).get("prompt_tokens", 0)
            for s in new_spans if s.get("metadata")
        )
        total_ct = sum(
            json.loads(s.get("metadata", "{}")).get("completion_tokens", 0)
            for s in new_spans if s.get("metadata")
        )
        total_tt = total_pt + total_ct
    except Exception:
        new_spans = []
        total_pt = 0
        total_ct = 0
        total_tt = 0

    return {
        "output": output.strip(),
        "duration_ms": duration_ms,
        "spans": new_spans,
        "prompt_tokens": total_pt,
        "completion_tokens": total_ct,
        "total_tokens": total_tt,
        "stderr": stderr,
    }


def compare_agents(
    agent_cmd: str,
    runs: int = 3,
    criteria: str = "correctness",
    api_key: str | None = None,
) -> list[CompareResult]:
    """Compares runs of an agent script, each one M times.

    The agent must be instrumented with treval to capture traces.
    Each run is evaluated with LLM-as-judge.

    Args:
        agent_cmd: Shell command to execute the agent.
        runs: Number of executions.
        criteria: Evaluation criteria.
        api_key: OpenRouter API key.

    Returns:
        List with one CompareResult (contains all runs of the agent).
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not configured")

    evaluator = LLMEvaluator(name="agent-compare", criteria=criteria, api_key=api_key)
    runs_for_agent: list[CompareRun] = []
    total_spans = 0

    for i in range(runs):
        result = _run_agent(agent_cmd, api_key)
        output = result.get("output", "")
        spans = result.get("spans", [])
        duration_ms = result.get("duration_ms", 0)
        stderr = result.get("stderr", "")

        span_id = i + 1
        score, reason = _score_output(output, agent_cmd, evaluator, span_id)

        # Calculate tokens from real spans
        pt = result.get("prompt_tokens", 0)
        ct = result.get("completion_tokens", 0)
        tt = result.get("total_tokens", 0)

        cr = CompareRun(
            model=agent_cmd,
            output=output[:2000] if output else "",
            run_index=i + 1,
            duration_ms=duration_ms,
            score=score,
            reason=reason,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=tt,
            spans=spans,
            stderr=stderr,
        )
        runs_for_agent.append(cr)
        total_spans += len(spans)

    result = CompareResult.from_runs(f"agent:{agent_cmd}", runs_for_agent)
    return [result]


def compare_models(
    prompt: str,
    models: list[str],
    runs: int = 3,
    criteria: str = "correctness",
    api_key: str | None = None,
) -> list[CompareResult]:
    """Compares N models on the same prompt, each one M times.

    Args:
        prompt: The prompt/input to send to all models.
        models: List of model names (e.g. ["deepseek/deepseek-v4-flash", ...]).
        runs: Number of runs per model.
        criteria: LLM-as-judge evaluation criteria.
        api_key: OpenRouter API key (default: OPENROUTER_API_KEY env).

    Returns:
        List of CompareResult (one per model) sorted by mean score descending.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not configured")

    evaluator = LLMEvaluator(name="compare", criteria=criteria, api_key=api_key)
    all_results: list[CompareResult] = []
    run_id = 0

    for model in models:
        runs_for_model: list[CompareRun] = []
        for i in range(runs):
            run_id += 1
            result = _call_llm(prompt, model, api_key)
            score, reason = _score_output(result["output"], prompt, evaluator, run_id)
            cr = CompareRun(
                model=model,
                run_index=i + 1,
                output=result["output"],
                duration_ms=result["duration_ms"],
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                total_tokens=result["total_tokens"],
                score=score,
                reason=reason,
            )
            runs_for_model.append(cr)

        cr_result = CompareResult.from_runs(model, runs_for_model)
        all_results.append(cr_result)

    # Sort by mean score descending
    all_results.sort(key=lambda r: r.mean_score, reverse=True)
    return all_results


# ─── HTML report generation ────────────────────────────────────────────────

def _render_duration_bar(dur: float, max_dur: float) -> str:
    if max_dur <= 0:
        return ""
    pct = max(2, (dur / max_dur) * 100)
    color = "#34d399" if dur < max_dur * 0.5 else "#fbbf24" if dur < max_dur * 0.8 else "#f87171"
    return f'<div class="bar-wrap"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div></div>'


def _render_span_trace(spans: list[dict]) -> str:
    """Renders a list of spans as an HTML hierarchy tree.

    Builds the parent→child hierarchy from parent_id.
    Spans without parent_id are roots. Each child is indented with └─.
    """
    if not spans:
        return ""

    # Build children map: parent_id → [children]
    children_map: dict[int, list[dict]] = {}
    span_map: dict[int, dict] = {}
    roots: list[dict] = []

    for s in spans:
        sid = s.get("id", 0)
        pid = s.get("parent_id")
        span_map[sid] = s
        if pid is None:
            roots.append(s)
        else:
            children_map.setdefault(pid, []).append(s)

    # Sort children by ID
    for pid in children_map:
        children_map[pid].sort(key=lambda x: x.get("id", 0))
    roots.sort(key=lambda x: x.get("id", 0))

    def _render_node(node: dict, depth: int = 0) -> str:
        indent = "&nbsp;&nbsp;" * depth
        prefix = "└─" if depth > 0 else ""
        name = esc_html(node.get("name", "?"))
        stype = node.get("type", "?")
        status = node.get("status", "ok")
        dur = node.get("duration_ms")
        dur_str = f"{dur:.0f}ms" if dur is not None else "—"
        status_dot = "🟢" if status == "ok" else "🔴"
        type_colors = {
            "AGENT": "#60a5fa", "OPERATION": "#34d399",
            "LLM": "#a78bfa", "TOOL": "#fbbf24",
        }
        color = type_colors.get(stype, "#8899b4")
        html = f"""\
  <div class="trace-node" style="padding-left:{depth * 16}px">
    <span class="trace-indent">{indent}{prefix}</span>
    <span class="trace-type" style="color:{color};font-weight:600;font-size:.7rem">{stype}</span>
    <span class="trace-name">{name}</span>
    <span class="trace-status">{status_dot}</span>
    <span class="trace-dur">{dur_str}</span>
  </div>"""
        for child in children_map.get(node.get("id", 0), []):
            html += _render_node(child, depth + 1)
        return html

    trace_html = ""
    for root in roots:
        trace_html += _render_node(root, 0)

    n_spans = len(spans)
    return f"""\
  <details class="trace-section" open>
    <summary class="trace-summary">📊 Trace ({n_spans} spans)</summary>
    <div class="trace-tree">
      {trace_html}
    </div>
  </details>"""


def build_report_html(
    results: list[CompareResult],
    prompt: str,
    criteria: str = "",
) -> str:
    """Generates a standalone HTML page with the model comparison.

    Args:
        results: List of comparison results.
        prompt: The prompt used in the comparison.
        criteria: Evaluation criteria used.

    Returns:
        Complete HTML string, ready to open in browser or --export.
    """
    if not results:
        return _EMPTY_HTML_TEMPLATE.replace("__PROMPT__", esc_html(prompt))

    # Determine winner
    best = max(results, key=lambda r: r.mean_score)

    # Build HTML rows
    rows_html = ""
    max_dur = max((r.mean_duration for r in results if r.run_count > 0), default=0)
    for i, r in enumerate(results):
        is_winner = r.model == best.model and r.mean_score == best.mean_score
        winner_badge = (
            '<span class="winner-badge" title="Best avg score">🏆</span>'
            if is_winner else ""
        )
        rank = i + 1

        score_str = f"{r.mean_score:.3f}"
        std_str = f"±{r.std_score:.3f}" if r.run_count > 1 else ""
        dur_str = f"{r.mean_duration:.0f}ms" if r.run_count > 0 else "—"
        dur_bar = _render_duration_bar(r.mean_duration, max_dur) if r.run_count > 0 else ""
        cost_str = format_cost(r.mean_cost) if r.run_count > 0 else "—"
        tokens_str = f"{r.total_tokens}" if r.run_count > 0 else "—"

        rows_html += f"""\
<tr class="{'winner-row' if is_winner else ''}">
  <td class="rank">{rank}</td>
  <td><strong>{esc_html(r.model)}</strong> {winner_badge}</td>
  <td class="score-cell"><span class="score-val">{score_str}</span> <span class="std">{std_str}</span></td>
  <td>{dur_str}<div class="bar-cell">{dur_bar}</div></td>
  <td>{cost_str}</td>
  <td>{tokens_str}</td>
  <td>{r.run_count}</td>
</tr>"""

    # Detalle de runs
    detail_html = ""
    for r in results:
        detail_html += f"""\
<div class="model-section">
  <h3>{esc_html(r.model)} — {r.run_count} run(s)</h3>
  <div class="model-meta">
    <span>Score: <strong>{r.mean_score:.3f}</strong> ±{r.std_score:.3f}</span>
    <span>Duration: <strong>{r.mean_duration:.0f}ms</strong> ±{r.std_duration:.0f}ms</span>
    <span>Cost: <strong>{format_cost(r.mean_cost)}</strong>/run</span>
    <span>Tokens: <strong>{r.total_tokens}</strong> total</span>
  </div>"""
        for run in r.runs:
            detail_html += f"""\
  <div class="run-card">
    <div class="run-header">
      <span class="run-num">#{run.run_index}</span>
      <span class="run-score" style="color:{_score_color(run.score)}">{run.score:.3f}</span>
      <span class="run-dur">{run.duration_ms:.0f}ms</span>
      <span class="run-cost">{format_cost(run.cost)}</span>
    </div>
    <div class="run-reason">{esc_html(run.reason)}</div>
    <details class="run-output">
      <summary>View output</summary>
      <pre>{esc_html(run.output[:2000])}</pre>
    </details>
    {_render_span_trace(run.spans) if run.spans else ''}
  </div>"""
        detail_html += "</div>"

    # Stats row
    total_runs = sum(r.run_count for r in results)
    total_cost = sum(r.total_cost for r in results)

    return _REPORT_HTML_TEMPLATE.replace("__PROMPT__", esc_html(prompt)) \
        .replace("__CRITERIA__", esc_html(criteria)) \
        .replace("__NUM_MODELS__", str(len(results))) \
        .replace("__TOTAL_RUNS__", str(total_runs)) \
        .replace("__TOTAL_COST__", format_cost(total_cost)) \
        .replace("__WINNER__", esc_html(best.model)) \
        .replace("__WINNER_SCORE__", f"{best.mean_score:.3f}") \
        .replace("__ROWS__", rows_html) \
        .replace("__DETAILS__", detail_html)


def _score_color(score: float) -> str:
    if score >= 0.80:
        return "#34d399"
    elif score >= 0.50:
        return "#fbbf24"
    return "#f87171"


def esc_html(s: str) -> str:
    if not s:
        return ""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") \
            .replace('"', "&quot;").replace("'", "&#39;")


# ─── HTML Templates ────────────────────────────────────────────────────────

_REPORT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>treval compare — report</title>
<link rel="icon" type="image/x-icon" href="data:image/x-icon;base64,AAABAAEAICAAAAAAIADgBAAAFgAAAIlQTkcNChoKAAAADUlIRFIAAAAgAAAAIAgCAAAA/BjtowAABKdJREFUeJx9VktMXFUY/v5zzmVmgOHNKAhWKViEkBKp1QYlxmjaULDWnfGx0cSNcaOufSyMK000MXHFUk00bhpXldqNLSkm0kojgikxVNQGC8zwutxzfxf33HvPuUM9mZncM+fc//H93/8gmR8AE6IvQAAAhlkUP7C9dS+ZtwjMzivRkSIQI7souydi6xZbcgFjGZsHVxqJSDcdJDbVxXzAn6k4iqyMVBIoVkwAhKXRoHSQobZuc4viD1fbHdsFQMTbxF1znR0d9rK3FMNOieEWeABYue8wkA1sldBqZZHEFMco3pGcxANwRnXqaep0oDnQFqZAHADHAst7EgA4Vk6W1xm2MVjr8PGh7qdH7teaY1hs56qoBwBQAGKmRmo44Q2lVwkAEd26XfaUJEc7bPMjb+xjUc3ByIlMMoW+j1DvNPdWmh9AGIS+b5lJsZokpSimCYsoPlEqJLTL2A4pG86Oo8arKzYUG5tYyYaz41DKwAymmIcU408uRCnWlAiNQ87M2t+jja2G9u7luVmAmtq7aXMr8HclCYKIYkRxfrhhIJGFxwYGYA5zhWLf0NjGhYtBZbO7/9ihgePBzvb69MXewcdyhSJzGBmRRCI21/ggjGfG4BQ7ZgT7ATN7uXzPkUdq69sq/642tXY0t3eW11YK9a09/Se8XJ6ZARJSpkanvwyAVG7QJT0zwMw1nnekv2dubgGBDxJ1hbsDbIcBSSGYtCeLlcoKpACopqbm7bde+ejjKd/fd/kFBikT4wQSgAAdBE+eHPvwgzffee/T4aP9N26sbGxujZ4YufbL757yzp//8XBfp+f1PXxsqK216bPPv+zsLGW4m5Qs4eiL80tIOT+/dOnyz1evLvQePnRpZu6N119cXFyamBhtaaudPDN6enxsfaM8NfXNue9+eOH5iZ2dXXZEG6AoYlGkimHzgMrlcrFYVyq1rv516+bK6vr6hvTU9PTMF1+d+/brT76/MHNPR+nR40crle3m5sYwDK3YOsJI5QZgigmRnXTMz555qrK1s367PDMzOzw8Mnn61OxPc9PTl0+dfGLht4XlP5Zefuk5rcP564tSyitXrulQZyhugswWOCn9wXrPj8pwbV0zaU8DpdbefC6/+s+vrAMNf2drDSAICbDyVAI9W8kkYsyy5ZAAlasRUnT0PPjMa+8XS22qwIzdPb/Myq9ta5l89d2u/odICeUp5XlpHTDSDTNJ5QfBdnlwQGTmXKGusdS19udyqPfrC+1Eorz9txCqpeO+zbXVve0yEcHUOHtwMMKiPLCAd0qF+YtDLaRi1sXau4jExtaqIKF1IIQkElZIyW08QJwHpgklFcvubEQgqQAwc1NDlxByvXITJKX0rAyNDc+UZorywEhkjuQZ0jKD7VrPHLY23dvS2A0ObRfvsEw3IpUfjJ4yjcKKR7pC1iAIyOxFZzmIiUR6JNEdzpLmaowVJETitHMHbltOn9O5iExVIgtZuzTaPnEMYyKO4/6cHRyicp0ZuRJG21NJ9SlZW/vUmdXSwYvSY3Yv2cqqXHG2lo64zYiEMweMQtn1/7Rx1ZsRxVTTA61IhNKdtR74irUl/AfoQRBytDgiIQAAAABJRU5ErkJggg=="/>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600;14..32,700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0b0e14; --surface: #14181f; --surface2: #1c2330;
  --border: #2a3348; --text: #e2e8f0; --text2: #8899b4;
  --accent: #60a5fa; --green: #34d399; --red: #f87171; --yellow: #fbbf24; --purple: #a78bfa;
  --radius: 10px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', -apple-system, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; padding: 16px; }
.container { max-width: 1000px; margin: 0 auto; }
h1 { font-size: 1.3rem; font-weight: 700; letter-spacing: -.02em; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px; }
.subtitle { font-size: .78rem; color: var(--text2); margin-bottom: 16px; }
.prompt-box { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 12px 16px; margin-bottom: 16px; }
.prompt-box .label { font-size: .65rem; color: var(--text2); text-transform: uppercase; letter-spacing: .06em; font-weight: 600; margin-bottom: 4px; }
.prompt-box .text { font-size: .85rem; line-height: 1.5; color: var(--text); }
.winner-banner { background: linear-gradient(135deg, rgba(96,165,250,.12), rgba(167,139,250,.12)); border: 1px solid var(--accent); border-radius: var(--radius); padding: 14px 18px; margin-bottom: 16px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.winner-banner .trophy { font-size: 1.5rem; }
.winner-banner .label { font-size: .7rem; color: var(--text2); text-transform: uppercase; letter-spacing: .06em; }
.winner-banner .name { font-weight: 700; font-size: 1rem; color: var(--accent); }
.winner-banner .score { font-size: .85rem; color: var(--green); }
.summary-bar { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; font-size: .75rem; color: var(--text2); }
.summary-bar span { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 6px 12px; }
.summary-bar strong { color: var(--text); }

/* Table */
.table-wrap { overflow-x: auto; border-radius: var(--radius); border: 1px solid var(--border); background: var(--surface); margin-bottom: 20px; }
table { width: 100%; border-collapse: collapse; min-width: 500px; }
thead { background: var(--surface2); }
th { padding: 10px 12px; text-align: left; font-size: .68rem; color: var(--text2); text-transform: uppercase; letter-spacing: .06em; font-weight: 600; border-bottom: 1px solid var(--border); white-space: nowrap; }
th:first-child, td:first-child { padding-left: 16px; }
th:last-child, td:last-child { padding-right: 16px; }
td { padding: 10px 12px; border-top: 1px solid var(--border); font-size: .8rem; vertical-align: middle; }
.winner-row { background: rgba(96,165,250,.06); }
.winner-row td { border-top-color: rgba(96,165,250,.2); }
.rank { font-weight: 700; color: var(--text2); font-size: .75rem; }
.winner-row .rank { color: var(--accent); }
.winner-badge { font-size: 1rem; margin-left: 4px; }
.score-cell { font-family: 'JetBrains Mono', monospace; }
.score-val { font-weight: 600; }
.std { font-size: .7rem; color: var(--text2); }
.bar-cell { margin-top: 3px; }
.bar-wrap { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; max-width: 80px; }
.bar-fill { height: 100%; border-radius: 2px; }

/* Detail sections */
.model-section { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 16px; margin-bottom: 12px; }
.model-section h3 { font-size: .85rem; font-weight: 600; margin-bottom: 4px; }
.model-meta { display: flex; gap: 14px; flex-wrap: wrap; font-size: .72rem; color: var(--text2); margin-bottom: 10px; }
.model-meta strong { color: var(--text); }
.run-card { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; margin-bottom: 6px; }
.run-header { display: flex; gap: 12px; align-items: center; font-size: .78rem; margin-bottom: 4px; }
.run-num { font-family: 'JetBrains Mono', monospace; color: var(--text2); }
.run-score { font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.run-dur { font-size: .72rem; color: var(--text2); }
.run-cost { font-size: .72rem; color: var(--text2); margin-left: auto; }
.run-reason { font-size: .75rem; color: var(--text2); margin-bottom: 4px; font-style: italic; }
.run-output summary { font-size: .72rem; color: var(--accent); cursor: pointer; }
.run-output pre { background: #06080c; padding: 8px 10px; border-radius: 6px; font-size: .72rem; line-height: 1.4; overflow-x: auto; white-space: pre-wrap; word-break: break-all; margin-top: 4px; border: 1px solid var(--border); max-height: 200px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; }

/* Trace tree */
.trace-section { margin-top: 8px; }
.trace-summary { font-size: .75rem; font-weight: 600; color: var(--accent); cursor: pointer; padding: 4px 0; }
.trace-tree { background: #06080c; border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; margin-top: 4px; }
.trace-node { display: flex; align-items: center; gap: 6px; padding: 2px 0; font-size: .72rem; line-height: 1.5; }
.trace-indent { color: var(--text2); opacity: .4; font-family: monospace; white-space: pre; }
.trace-type { font-family: 'JetBrains Mono', monospace; min-width: 32px; }
.trace-name { color: var(--text); flex: 1; }
.trace-status { font-size: .65rem; }
.trace-dur { font-family: 'JetBrains Mono', monospace; color: var(--text2); font-size: .68rem; min-width: 40px; text-align: right; }
@media (max-width: 640px) {
  .trace-node { font-size: .65rem; gap: 3px; flex-wrap: wrap; }
  .trace-type { min-width: 24px; }
}
footer { text-align: center; font-size: .7rem; color: var(--text2); margin-top: 24px; padding: 12px; border-top: 1px solid var(--border); }
@media (max-width: 640px) {
  body { padding: 8px; }
  .winner-banner { flex-direction: column; text-align: center; }
  .summary-bar { flex-direction: column; gap: 6px; }
  .model-meta { flex-direction: column; gap: 4px; }
  td, th { font-size: .72rem; padding: 8px; }
  .run-header { flex-wrap: wrap; gap: 6px; }
}
</style>
</head>
<body>
<div class="container">
<h1>⚡ treval compare</h1>
<div class="subtitle">Model Comparison &middot; criteria: __CRITERIA__</div>

<div class="prompt-box">
  <div class="label">Prompt</div>
  <div class="text">__PROMPT__</div>
</div>

<div class="winner-banner">
  <span class="trophy">🏆</span>
  <div>
    <div class="label">Best model</div>
    <div class="name">__WINNER__</div>
  </div>
  <div class="score">Avg score: __WINNER_SCORE__</div>
</div>

<div class="summary-bar">
  <span>Models: <strong>__NUM_MODELS__</strong></span>
  <span>Runs: <strong>__TOTAL_RUNS__</strong></span>
  <span>Total cost: <strong>__TOTAL_COST__</strong></span>
</div>

<div class="table-wrap">
<table>
<thead><tr>
  <th>#</th><th>Model</th><th>Avg score</th><th>Duration</th><th>Cost/run</th><th>Tokens</th><th>Runs</th>
</tr></thead>
<tbody>
__ROWS__
</tbody>
</table>
</div>

<h2 style="font-size:.95rem;margin-bottom:10px;color:var(--text2)">Detail by model</h2>
<div class="details">
__DETAILS__
</div>

<footer>Generated by treval compare</footer>
</div>
</body>
</html>"""

_EMPTY_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>treval compare — no data</title>
<link rel="icon" type="image/x-icon" href="data:image/x-icon;base64,AAABAAEAICAAAAAAIADgBAAAFgAAAIlQTkcNChoKAAAADUlIRFIAAAAgAAAAIAgCAAAA/BjtowAABKdJREFUeJx9VktMXFUY/v5zzmVmgOHNKAhWKViEkBKp1QYlxmjaULDWnfGx0cSNcaOufSyMK000MXHFUk00bhpXldqNLSkm0kojgikxVNQGC8zwutxzfxf33HvPuUM9mZncM+fc//H93/8gmR8AE6IvQAAAhlkUP7C9dS+ZtwjMzivRkSIQI7souydi6xZbcgFjGZsHVxqJSDcdJDbVxXzAn6k4iqyMVBIoVkwAhKXRoHSQobZuc4viD1fbHdsFQMTbxF1znR0d9rK3FMNOieEWeABYue8wkA1sldBqZZHEFMco3pGcxANwRnXqaep0oDnQFqZAHADHAst7EgA4Vk6W1xm2MVjr8PGh7qdH7teaY1hs56qoBwBQAGKmRmo44Q2lVwkAEd26XfaUJEc7bPMjb+xjUc3ByIlMMoW+j1DvNPdWmh9AGIS+b5lJsZokpSimCYsoPlEqJLTL2A4pG86Oo8arKzYUG5tYyYaz41DKwAymmIcU408uRCnWlAiNQ87M2t+jja2G9u7luVmAmtq7aXMr8HclCYKIYkRxfrhhIJGFxwYGYA5zhWLf0NjGhYtBZbO7/9ihgePBzvb69MXewcdyhSJzGBmRRCI21/ggjGfG4BQ7ZgT7ATN7uXzPkUdq69sq/642tXY0t3eW11YK9a09/Se8XJ6ZARJSpkanvwyAVG7QJT0zwMw1nnekv2dubgGBDxJ1hbsDbIcBSSGYtCeLlcoKpACopqbm7bde+ejjKd/fd/kFBikT4wQSgAAdBE+eHPvwgzffee/T4aP9N26sbGxujZ4YufbL757yzp//8XBfp+f1PXxsqK216bPPv+zsLGW4m5Qs4eiL80tIOT+/dOnyz1evLvQePnRpZu6N119cXFyamBhtaaudPDN6enxsfaM8NfXNue9+eOH5iZ2dXXZEG6AoYlGkimHzgMrlcrFYVyq1rv516+bK6vr6hvTU9PTMF1+d+/brT76/MHNPR+nR40crle3m5sYwDK3YOsJI5QZgigmRnXTMz555qrK1s367PDMzOzw8Mnn61OxPc9PTl0+dfGLht4XlP5Zefuk5rcP564tSyitXrulQZyhugswWOCn9wXrPj8pwbV0zaU8DpdbefC6/+s+vrAMNf2drDSAICbDyVAI9W8kkYsyy5ZAAlasRUnT0PPjMa+8XS22qwIzdPb/Myq9ta5l89d2u/odICeUp5XlpHTDSDTNJ5QfBdnlwQGTmXKGusdS19udyqPfrC+1Eorz9txCqpeO+zbXVve0yEcHUOHtwMMKiPLCAd0qF+YtDLaRi1sXau4jExtaqIKF1IIQkElZIyW08QJwHpgklFcvubEQgqQAwc1NDlxByvXITJKX0rAyNDc+UZorywEhkjuQZ0jKD7VrPHLY23dvS2A0ObRfvsEw3IpUfjJ4yjcKKR7pC1iAIyOxFZzmIiUR6JNEdzpLmaowVJETitHMHbltOn9O5iExVIgtZuzTaPnEMYyKO4/6cHRyicp0ZuRJG21NJ9SlZW/vUmdXSwYvSY3Yv2cqqXHG2lo64zYiEMweMQtn1/7Rx1ZsRxVTTA61IhNKdtR74irUl/AfoQRBytDgiIQAAAABJRU5ErkJggg=="/>
<style>
body { font-family: system-ui, sans-serif; background: #0b0e14; color: #e2e8f0; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
.box { text-align: center; max-width: 400px; }
h2 { color: #8899b4; font-weight: 500; }
p { font-size: .85rem; color: #8899b4; margin-top: 8px; }
.prompt { font-size: .8rem; color: #60a5fa; margin-top: 12px; }
</style>
</head>
<body>
<div class="box">
<h2>📊 No comparison data</h2>
<p>Run <code>treval compare</code> to generate results.</p>
<div class="prompt">Prompt: __PROMPT__</div>
</div>
</body>
</html>"""