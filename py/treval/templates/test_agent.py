"""Tests for the ReAct agent using LLM-as-judge."""
from treval.testing import case, TestSuite

suite = TestSuite(name="WeatherTests")


@case(suite,
      input="what's the weather in Madrid?",
      criteria="The response should mention Madrid's weather")
def test_madrid(response: str) -> None:
    assert "Madrid" in response
    assert "28" in response or "soleado" in response


@case(suite,
      input="what is 2+2?",
      criteria="The response must be mathematically correct")
def test_math(response: str) -> None:
    assert "4" in response or "four" in response