"""
Test of the native treval testing module.
"""
from treval.testing import TestCaseDef, TestResult, TestRunner, TestSuite, case


def test_suite_adds_tests():
    """A suite should be able to add tests."""
    suite = TestSuite(name="test")
    assert len(suite) == 0

    suite.add(TestCaseDef(name="t1", input="hello"))
    assert len(suite) == 1
    assert suite.tests[0].name == "t1"

    suite.add(TestCaseDef(name="t2", input="world"))
    assert len(suite) == 2


def test_decorator_adds_to_suite():
    """The @case decorator should add to the suite automatically."""
    suite = TestSuite(name="decorated")

    @case(suite, input="hello", name="test_hola")
    def check(response):
        assert "hello" in response

    assert len(suite) == 1
    assert suite.tests[0].name == "test_hola"
    assert suite.tests[0].assertions is not None


def test_runner_no_agent():
    """Without agent function, the test uses expected_output."""
    runner = TestRunner()
    test = TestCaseDef(name="simple", input="question",
                       expected_output="answer")
    result = runner.run_test(test)
    assert result.name == "simple"
    assert result.input == "question"


def test_result_passed_logic():
    """A test passes if score >= 0.5 and there's no assertion error."""
    r = TestResult(name="a", input="x", output="y", score=0.8,
                   assertions_passed=True)
    assert r.passed

    r.score = 0.3
    assert not r.passed  # score bajo

    r.score = 0.8
    r.assertions_passed = False
    assert not r.passed  # assertion failed


def test_result_defaults():
    """Valores por defecto de TestResult."""
    r = TestResult(name="a", input="x")
    assert r.score == 0.0
    assert r.assertions_passed
    assert not r.error
    assert not r.output