"""Real-call smoke test for both providers. Skipped automatically without API keys.

Each call is minimal (max_tokens=10) — under USD 0.01 per run. Verifies that
the canonical interface works end-to-end against live SDKs.
"""
import os

import pytest

from henge.providers import CompletionRequest, complete

_HAS_ANTHROPIC = bool(os.getenv("ANTHROPIC_API_KEY"))
_HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))


@pytest.mark.asyncio
@pytest.mark.skipif(not _HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")
async def test_smoke_anthropic_haiku():
    req = CompletionRequest(
        system="Reply with just the digit 4.",
        user="What is 2+2?",
        max_tokens=10,
    )
    resp = await complete("anthropic/haiku-4-5", req)
    assert resp.text.strip() != ""
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    assert resp.raw_model.startswith("claude-haiku-4-5")


@pytest.mark.asyncio
@pytest.mark.skipif(not _HAS_OPENAI, reason="OPENAI_API_KEY not set")
async def test_smoke_openai_gpt5():
    # gpt-5 is a reasoning model; it consumes ~100+ tokens internally before
    # producing any output, so max_tokens=500 is the practical minimum.
    req = CompletionRequest(
        system="Reply with just the digit 4.",
        user="What is 2+2?",
        max_tokens=500,
    )
    resp = await complete("openai/gpt-5", req)
    assert resp.text.strip() != ""
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    # OpenAI returns dated releases (e.g. "gpt-5-2025-08-07"); check prefix only.
    assert resp.raw_model.startswith("gpt-5")
