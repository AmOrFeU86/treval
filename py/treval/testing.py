"""
Testing native support for treval.

Lets you define tests for agents and run them with:
    treval test run tests/test_my_agent.py

Test file format:
    from treval.testing import case, TestSuite

    suite = TestSuite(name="MySuite")

    @case(suite, input="what's the weather in Madrid?",
               expected_output="Madrid",
               criteria="The response must mention the weather in Madrid")
    def test_madrid(response: str) -> None:
        assert "Madrid" in response
        assert "28" in response or "sunny" in response

    @test_case(suite, input="what's 2+2?",
               criteria="The response must be mathematically correct")
    def test_math(response: str) -> None:
        assert "4" in response or "four" in response
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Prevent pytest from picking up functions from this module as tests
__test__ = False

from treval.eval import EvalResult, EvalStore, LLMEvaluator


@dataclass
class TestCaseDef:
    """Definition of a test case."""
    name: str
    input: str
    criteria: str = "correctness"
    expected_output: str = ""
    assertions: Callable[[str], None] | None = None


@dataclass
class TestResult:
    """Result of running a test case."""
    name: str
    input: str
    output: str = ""
    score: float = 0.0
    reason: str = ""
    assertions_passed: bool = True
    error: str = ""
    duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return self.assertions_passed and self.score >= 0.5


class TestRunner:
    """Runs a test suite and evaluates the results."""

    def __init__(self, agent_fn: Callable[[str], str] | None = None,
                 evaluator: LLMEvaluator | None = None):
        self.agent_fn = agent_fn
        self.evaluator = evaluator or LLMEvaluator(
            name="test",
            criteria="correctness",
        )

    def run_test(self, test: TestCaseDef) -> TestResult:
        """Runs a test case and returns the result."""
        start = time.perf_counter()
        result = TestResult(name=test.name, input=test.input)

        try:
            # 1. Run the agent (if function provided)
            if self.agent_fn:
                output = self.agent_fn(test.input)
                result.output = output
            else:
                output = test.expected_output or ""

            # 2. LLM evaluation
            eval_result = self.evaluator._llm_judge(
                criteria=test.criteria,
                input_text=test.input,
                output_text=output,
            )
            result.score, result.reason = eval_result

            # 3. Run assertions (if any)
            if test.assertions and output:
                try:
                    test.assertions(output)
                    result.assertions_passed = True
                except AssertionError as e:
                    result.assertions_passed = False
                    result.error = str(e)

        except Exception as e:
            result.error = f"Error running test: {e}"
            result.assertions_passed = False

        result.duration_ms = (time.perf_counter() - start) * 1000
        return result

    def run_suite(self, suite: TestSuite) -> list[TestResult]:
        """Runs all tests in a suite."""
        results = []
        for test in suite.tests:
            r = self.run_test(test)
            results.append(r)
        return results


class TestSuite:
    """Collection of test cases."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.tests: list[TestCaseDef] = []

    def add(self, test: TestCaseDef) -> None:
        """Adds a test case to the suite."""
        self.tests.append(test)

    def __len__(self) -> int:
        return len(self.tests)


def case(suite: TestSuite, input: str,
              name: str | None = None,
              criteria: str = "correctness",
              expected_output: str = ""):
    """Decorator to define a test case in a suite.

    Usage:
        @test_case(suite, input="hello", criteria="helpfulness")
        def test_something(response: str) -> None:
            assert "hello" in response
    """
    def decorator(func):
        case = TestCaseDef(
            name=name or func.__name__,
            input=input,
            criteria=criteria,
            expected_output=expected_output,
            assertions=func,
        )
        suite.add(case)
        return func
    return decorator


def load_test_file(path: str) -> TestSuite | None:
    """Loads a .py test file and returns the TestSuite defined in it.

    The file must define a 'suite' variable of type TestSuite.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load: {path}")

    mod = importlib.util.module_from_spec(spec)
    # Add test directory to path for relative imports
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)

    suite = getattr(mod, "suite", None)
    if suite is None:
        raise ValueError(
            f"The file {path} must define a 'suite' variable of type TestSuite"
        )
    return suite