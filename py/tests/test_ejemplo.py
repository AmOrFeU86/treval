"""
Tests de ejemplo para treval test run.
Ejecutar con: treval test run py/tests/test_ejemplo.py
"""
__test__ = False

from treval.testing import case, TestSuite

suite = TestSuite(name="Examples")


@case(suite, input="What is 2+2?",
       criteria="The response must be mathematically correct")
def test_suma(response: str) -> None:
    assert "4" in response or "four" in response


@case(suite, input="What's the weather in Madrid?",
       expected_output="Madrid",
       criteria="The response should mention Madrid's weather")
def test_madrid(response: str) -> None:
    assert "Madrid" in response


@case(suite, input="Say hello please",
       criteria="The response should be friendly and greet")
def test_saludo(response: str) -> None:
    assert "hello" in response.lower() or "hi" in response.lower() or "greetings" in response.lower()