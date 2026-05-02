"""Token pricing for Henge cost accounting (USD per 1M tokens).

Versioned data: every report records ``PRICING_VERSION`` so historical totals
remain interpretable after a price card change. Bump the version when any
``in``/``out`` value changes.

Sources:
- https://docs.anthropic.com/en/docs/about-claude/pricing
- https://platform.openai.com/docs/pricing
"""
from __future__ import annotations

PRICING_VERSION = "2026-05"

PRICING: dict[str, dict[str, float]] = {
    "anthropic/haiku-4-5":  {"in": 1.00,  "out":  5.00},
    "anthropic/sonnet-4-6": {"in": 3.00,  "out": 15.00},
    "anthropic/opus-4-7":   {"in": 15.00, "out": 75.00},
    "openai/gpt-5":         {"in": 5.00,  "out": 20.00},
}

EMBEDDING_PRICING: dict[str, float] = {
    "openai/text-embedding-3-small": 0.02,
    "openai/text-embedding-3-large": 0.13,
    "voyage/voyage-3-large":         0.18,
}


def cost_for(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD for one completion. Returns 0.0 for unknown models or
    negative inputs (degrade gracefully instead of failing the run)."""
    rates = PRICING.get(model_id)
    if not rates:
        return 0.0
    inp = max(0, int(input_tokens or 0))
    out = max(0, int(output_tokens or 0))
    return (inp * rates["in"] + out * rates["out"]) / 1_000_000


def embedding_cost(embedding_model_id: str, n_input_tokens: int) -> float:
    rate = EMBEDDING_PRICING.get(embedding_model_id, 0.0)
    return (max(0, int(n_input_tokens or 0)) * rate) / 1_000_000
