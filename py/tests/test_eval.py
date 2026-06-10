"""
RED: Evaluaciones LLM-as-judge deben funcionar.
"""
from treval.eval import EvalResult, EvalStore


def test_eval_store_save_and_list():
    """Save and list evaluations."""
    store = EvalStore()
    store.clear()

    result = EvalResult(
        span_id=1,
        evaluator_name="correctness",
        score=0.85,
        reason="Correct answer",
        metadata={"span_type": "TOOL"},
    )
    store.save(result)

    results = store.list()
    assert len(results) == 1
    assert results[0]["span_id"] == 1
    assert results[0]["score"] == 0.85


def test_eval_store_stats():
    """Stats calculation."""
    store = EvalStore()
    store.clear()

    for score in [0.9, 0.8, 0.7]:
        store.save(EvalResult(span_id=1, evaluator_name="test", score=score))

    stats = store.get_stats()
    assert stats["count"] == 3
    assert abs(stats["avg_score"] - 0.8) < 0.001


def test_eval_result_validation():
    """Scores must be between 0 and 1."""
    r = EvalResult(span_id=1, evaluator_name="test", score=0.5, reason="ok")
    assert 0.0 <= r.score <= 1.0


# ─── _parse_judge_json tests ───

def test_parse_clean_json():
    """Perfectly formed JSON."""
    from treval.eval import _parse_judge_json
    score, reason = _parse_judge_json('{"score": 0.85, "reason": "Correct answer"}')
    assert abs(score - 0.85) < 0.001
    assert "correct" in reason.lower()


def test_parse_with_markdown():
    """JSON wrapped in ```json ... ```."""
    from treval.eval import _parse_judge_json
    text = '```json\n{"score": 0.75, "reason": "Almost correct"}\n```'
    score, reason = _parse_judge_json(text)
    assert abs(score - 0.75) < 0.001
    assert "almost" in reason.lower()


def test_parse_with_plain_backticks():
    """JSON wrapped in ``` ... ``` without json."""
    from treval.eval import _parse_judge_json
    text = '```\n{"score": 0.9, "reason": "Excellent"}\n```'
    score, reason = _parse_judge_json(text)
    assert abs(score - 0.9) < 0.001


def test_parse_malformed_json_unterminated():
    """JSON with unterminated string (real failure case)."""
    from treval.eval import _parse_judge_json
    text = '{"score": 0.80, "reason": "The response is correct in explaining that'
    score, reason = _parse_judge_json(text)
    assert abs(score - 0.80) < 0.001
    # Parser extracts reason as far as it can (regex captures until end of line)
    assert reason != ""


def test_parse_malformed_no_closing_brace():
    """JSON without closing brace."""
    from treval.eval import _parse_judge_json
    text = '{"score": 0.65, "reason": "Acceptable'
    score, reason = _parse_judge_json(text)
    assert abs(score - 0.65) < 0.001


def test_parse_extra_text_before_json():
    """Extra text before JSON."""
    from treval.eval import _parse_judge_json
    text = 'Here is my evaluation:\n\n{"score": 0.70, "reason": "Good response"}'
    score, reason = _parse_judge_json(text)
    assert abs(score - 0.70) < 0.001


def test_parse_empty_text():
    """Empty text → fallback."""
    from treval.eval import _parse_judge_json
    score, reason = _parse_judge_json("")
    assert abs(score - 0.5) < 0.001


def test_parse_gibberish():
    """Nonsense text → fallback."""
    from treval.eval import _parse_judge_json
    score, reason = _parse_judge_json("hello world this is not json")
    assert abs(score - 0.5) < 0.001


def test_parse_score_only():
    """Numeric score only, no JSON."""
    from treval.eval import _parse_judge_json
    text = 'The score is 0.88 out of 1.0'
    score, reason = _parse_judge_json(text)
    assert abs(score - 0.88) < 0.001


def test_parse_score_out_of_range():
    """Score > 1.0 should clamp to 1.0."""
    from treval.eval import _parse_judge_json
    score, reason = _parse_judge_json('{"score": 1.5, "reason": "too high"}')
    assert abs(score - 1.0) < 0.001


def test_parse_negative_score():
    """Negative score should clamp to 0.0."""
    from treval.eval import _parse_judge_json
    score, reason = _parse_judge_json('{"score": -0.5, "reason": "very bad"}')
    assert abs(score - 0.0) < 0.001