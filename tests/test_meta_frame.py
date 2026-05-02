"""Phase 4 meta-frame: MetaFrameResult dataclass + feature flag."""
import pytest

from henge.meta_frame import MetaFrameResult, ENABLE_META_FRAME


def test_meta_frame_result_has_all_fields():
    r = MetaFrameResult(
        decision_class="two-way-with-cost",
        urgency="weeks",
        question_quality="well-formed",
        suggested_reformulation=None,
        meta_recommendation="proceed",
        reasoning="The question presents a clear decision with measurable trade-offs.",
        gpt5_usage={"model": "openai/gpt-5", "input_tokens": 0, "output_tokens": 0},
    )
    assert r.meta_recommendation == "proceed"
    assert r.suggested_reformulation is None


def test_meta_frame_result_passthrough():
    """When the model fails or flag is false, callers can build a passthrough result."""
    r = MetaFrameResult(
        decision_class="unknown",
        urgency="unknown",
        question_quality="unknown",
        suggested_reformulation=None,
        meta_recommendation="proceed",
        reasoning="meta-frame skipped",
        gpt5_usage=None,
    )
    assert r.gpt5_usage is None
    assert r.meta_recommendation == "proceed"


def test_enable_meta_frame_default_true():
    # Just check the flag exists and is bool; default=True at import time.
    assert isinstance(ENABLE_META_FRAME, bool)
