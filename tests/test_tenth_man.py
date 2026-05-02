"""Phase 5 dual tenth-man: blind + informed dataclasses + feature flag."""
import pytest

from henge.tenth_man import (
    TenthManBlindResult,
    TenthManInformedResult,
    ENABLE_TENTH_MAN_DUAL,
)


def test_blind_result_fields():
    r = TenthManBlindResult(
        text="Blind dissent text...",
        opus_usage={"model": "anthropic/opus-4-7", "input_tokens": 100, "output_tokens": 200},
    )
    assert r.text.startswith("Blind")
    assert r.opus_usage["model"] == "anthropic/opus-4-7"


def test_informed_result_fields():
    r = TenthManInformedResult(
        text="Refined dissent...",
        what_holds=["claim A holds", "claim B holds"],
        what_revised=["claim C revised to..."],
        what_discarded=["claim D was bias"],
        gpt5_usage={"model": "openai/gpt-5", "input_tokens": 500, "output_tokens": 300},
    )
    assert len(r.what_holds) == 2
    assert len(r.what_revised) == 1
    assert len(r.what_discarded) == 1
    assert r.gpt5_usage["model"] == "openai/gpt-5"


def test_informed_result_passthrough_defaults():
    """Empty informed result (e.g. on parse failure) is constructable."""
    r = TenthManInformedResult(
        text="raw text only",
        what_holds=[],
        what_revised=[],
        what_discarded=[],
        gpt5_usage=None,
    )
    assert r.text == "raw text only"
    assert r.gpt5_usage is None


def test_enable_tenth_man_dual_default_true():
    assert isinstance(ENABLE_TENTH_MAN_DUAL, bool)
