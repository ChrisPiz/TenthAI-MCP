"""Tenth-man dual dissenter — Opus blind + gpt-5 informed.

v0.6 design:
  - Blind: Opus 4.7 sees only question + canonical_context, no 9 advisors.
    Produces the strongest pure dissent. Distance metric uses this.
  - Informed: gpt-5 sees the 9 + the blind, reconciles. Output:
      what_holds   — claims from blind that survive after seeing the 9
      what_revised — claims that get refined
      what_discarded — claims that turn out to be Opus bias

Cross-lab: blind on Anthropic, informed on OpenAI. If gpt-5 keeps most of
the blind, dissent is robust. If gpt-5 discards much, the blind was lab bias
masquerading as insight.

Implementation of ``run_tenth_man_blind`` and ``run_tenth_man_informed``
lands in Tasks 5.2 / 5.3.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _flag(name: str, default: bool = True) -> bool:
    val = os.getenv(name, "true" if default else "false").strip().lower()
    return val in ("1", "true", "yes", "on")


ENABLE_TENTH_MAN_DUAL = _flag("HENGE_ENABLE_TENTH_MAN_DUAL", True)


@dataclass
class TenthManBlindResult:
    text: str
    opus_usage: dict | None


@dataclass
class TenthManInformedResult:
    text: str
    what_holds: list[str] = field(default_factory=list)
    what_revised: list[str] = field(default_factory=list)
    what_discarded: list[str] = field(default_factory=list)
    gpt5_usage: dict | None = None
