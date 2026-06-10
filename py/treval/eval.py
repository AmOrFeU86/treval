"""LLM-as-judge evaluations for treval spans.

Usage:
    import treval
    evaluator = treval.LLMEvaluator(model="deepseek/deepseek-v4-flash",
                                    criteria="The response must be correct")
    scores = evaluator.evaluate(spans)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_judge_json(text: str) -> tuple[float, str]:
    """Extracts score and reason from an LLM judge JSON, tolerating malformed JSON.

    Strategies in order:
    1. Direct json.loads (ideal case)
    2. json.loads after cleaning markdown/backticks
    3. Regex to extract score + reason from loose JSON
    4. Regex to extract just the score
    """
    text = text.strip()

    # Strategy 1: Direct JSON
    try:
        data = json.loads(text)
        score = _clamp_score(data.get("score", 0.5))
        reason = str(data.get("reason", ""))
        return score, reason
    except json.JSONDecodeError:
        pass

    # Strategy 2: Clean markdown and retry
    cleaned = text
    # Remove ```json ... ``` or ``` ... ``` blocks
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1).strip()
    # Remove inline markdown
    cleaned = re.sub(r"^[#>*\-]\s*", "", cleaned, flags=re.MULTILINE).strip()
    # Remove text before first { and after last }
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        cleaned = cleaned[brace_start : brace_end + 1]
    try:
        data = json.loads(cleaned)
        return _clamp_score(data.get("score", 0.5)), str(data.get("reason", ""))
    except json.JSONDecodeError:
        pass

    # Strategy 3: Regex to extract score from partially broken JSON
    score_match = re.search(r'"?score"?\s*:\s*([0-9]*\.?[0-9]+)', text)
    reason_match = re.search(r'"?reason"?\s*:\s*"(.+?)(?:"|\n|$)', text, re.DOTALL)
    if score_match:
        score = _clamp_score(float(score_match.group(1)))
        reason = reason_match.group(1).strip() if reason_match else "Malformed JSON, score extracted by regex"
        return score, reason

    # Strategy 4: Search for any number in score context
    score_match = re.search(r"([0-9]*\.?[0-9]+)", text)
    if score_match:
        score = _clamp_score(float(score_match.group(1)))
        return score, "Score estimated by numeric pattern"

    return 0.5, "Could not parse evaluation"


def _clamp_score(s: float) -> float:
    return max(0.0, min(1.0, float(s)))

from treval.db import SpanStore

EVAL_DB_PATH = Path.home() / ".treval" / "evals.db"


@dataclass
class EvalResult:
    span_id: int
    evaluator_name: str
    score: float  # 0.0 - 1.0
    reason: str = ""
    metadata: dict = field(default_factory=dict)
    created_at: str = ""


class EvalStore:
    """Local storage for evaluation results in SQLite."""

    _lock = threading.Lock()

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path) if db_path else EVAL_DB_PATH
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    span_id         INTEGER NOT NULL,
                    evaluator_name  TEXT NOT NULL,
                    score           REAL NOT NULL,
                    reason          TEXT,
                    metadata        TEXT,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.commit()
            conn.close()

    def save(self, result: EvalResult) -> int:
        with self._lock:
            conn = self._conn()
            cur = conn.execute(
                """INSERT INTO evaluations (span_id, evaluator_name, score, reason, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (result.span_id, result.evaluator_name, result.score,
                 result.reason, json.dumps(result.metadata) if result.metadata else None),
            )
            conn.commit()
            conn.close()
            return cur.lastrowid

    def list(self, limit: int = 50) -> list[dict]:
        conn = self._conn()
        rows = conn.execute(
            """SELECT e.*, NULL as span_name, NULL as span_type
               FROM evaluations e
               ORDER BY e.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        # Try to enrich with span name if spans DB is accessible
        try:
            from treval.db import SpanStore, DB_PATH
            sconn = sqlite3.connect(str(DB_PATH))
            sconn.row_factory = sqlite3.Row
            for row in rows:
                srow = sconn.execute("SELECT name, type FROM spans WHERE id = ?",
                                     (row["span_id"],)).fetchone()
                if srow:
                    row["span_name"] = srow["name"]
                    row["span_type"] = srow["type"]
            sconn.close()
        except Exception:
            pass
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) as count, AVG(score) as avg_score, MIN(score) as min, MAX(score) as max FROM evaluations"
        ).fetchone()
        conn.close()
        return dict(row) if row else {"count": 0, "avg_score": None, "min": None, "max": None}

    def clear(self) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute("DELETE FROM evaluations")
            conn.commit()
            conn.close()


class LLMEvaluator:
    """Evaluates spans using an LLM as judge.

    Common criteria:
    - "correctness": The answer is correct with respect to the question
    - "conciseness": The answer is concise without irrelevant information
    - "helpfulness": The answer is helpful for the user
    - "groundedness": The answer uses tools correctly
    """

    def __init__(self, name: str = "", criteria: str = "correctness",
                 api_key: str | None = None, base_url: str | None = None,
                 model: str = "deepseek/deepseek-v4-flash"):
        self.name = name or criteria
        self.criteria = criteria
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url or "https://openrouter.ai/api/v1"
        self.model = model

    def evaluate(self, spans: list[dict]) -> list[EvalResult]:
        """Evaluates a list of spans and returns results.

        For TOOL and OPERATION spans, evaluates output against input.
        For LLM spans, evaluates the response.
        """
        results = []
        for span in spans:
            result = self._evaluate_one(span)
            if result:
                results.append(result)
        return results

    def evaluate_span(self, span: dict) -> EvalResult | None:
        """Evaluates a single span."""
        return self._evaluate_one(span)

    def _evaluate_one(self, span: dict) -> EvalResult | None:
        if not span.get("input") or not span.get("output"):
            return None

        score, reason = self._llm_judge(
            criteria=self.criteria,
            input_text=span["input"],
            output_text=span["output"],
        )
        return EvalResult(
            span_id=span["id"],
            evaluator_name=self.name,
            score=score,
            reason=reason,
            metadata={"criteria": self.criteria, "model": self.model, "span_type": span.get("type")},
            created_at=datetime.now().isoformat(),
        )

    def _llm_judge(self, criteria: str, input_text: str, output_text: str) -> tuple[float, str]:
        """Calls the LLM to judge an output according to the criteria."""
        prompt = f"""You are a fair and precise evaluator. Evaluate the following response according to the specified criteria.

CRITERIA: {criteria}

INPUT (question/instruction):
{input_text[:1500]}

OUTPUT (response to evaluate):
{output_text[:1500]}

Return your evaluation ONLY in the following JSON format (no markdown, no extra explanation):
{{"score": <number between 0.0 and 1.0>, "reason": "<brief 1-2 sentence explanation>"}}"""
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            text = response.choices[0].message.content or "{}"
            return _parse_judge_json(text)
        except Exception as e:
            return 0.5, f"Error evaluating: {e}"


# Predefined evaluators
CORRECTNESS_EVALUATOR = LLMEvaluator(
    name="correctness",
    criteria="""The answer is correct, accurate, and factual.
Score high (0.8-1.0) if the answer correctly addresses the question.
Score low (0.0-0.3) if the answer contains incorrect information or does not respond.""",
)

CONCISENESS_EVALUATOR = LLMEvaluator(
    name="conciseness",
    criteria="""The answer is concise and direct, without irrelevant information.
Score high if it's brief and to the point.
Score low if it's verbose or contains unnecessary details.""",
)

HELPFULNESS_EVALUATOR = LLMEvaluator(
    name="helpfulness",
    criteria="""The answer is helpful, actionable, and addresses the user's need.
Score high if the user can act based on the answer.
Score low if it's generic, vague, or does not solve the problem.""",
)