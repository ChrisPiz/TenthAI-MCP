"""Phase 3 scoping pipeline: ScopingResult, CanonicalContext, feature flags."""
import pytest

from henge.scoping import (
    ScopingResult,
    CanonicalContext,
    ScopedQuestion,
)


def test_scoping_result_is_dataclass_like():
    questions = [
        ScopedQuestion(id="q_001", text="Q?", source="scoping", challenges_assumption=None),
    ]
    r = ScopingResult(
        questions=questions,
        adversarial_count=0,
        version="v0.6",
        haiku_usage={"model": "anthropic/haiku-4-5", "input_tokens": 0, "output_tokens": 0},
        gpt5_usage=None,
    )
    assert r.adversarial_count == 0
    assert r.version == "v0.6"
    assert r.questions[0].source == "scoping"


def test_canonical_context_is_dataclass_like():
    c = CanonicalContext(
        summary="The user is a 30-year-old considering X.",
        flags=[],
        opus_usage={"model": "anthropic/opus-4-7", "input_tokens": 0, "output_tokens": 0},
    )
    assert c.summary.startswith("The user")
    assert c.flags == []


def test_scoped_question_source_must_be_known():
    ScopedQuestion(id="q_002", text="Q", source="adversarial", challenges_assumption="X assumes...")
    ScopedQuestion(id="q_003", text="Q", source="scoping", challenges_assumption=None)
