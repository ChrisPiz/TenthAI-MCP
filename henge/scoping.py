"""Scoping phase — multi-step pipeline.

v0.6 pipeline:
  1. Haiku 4.5 [Anthropic] generates 4-7 base scoping questions.
  2. gpt-5 [OpenAI] adversarial review: 2-4 additional questions that
     challenge unexamined assumptions in the original question.
  3. Combined set returns to user.
  4. User responds (handled by server.py).
  5. Opus 4.7 [Anthropic] canonicalizes the user's responses into a
     summary + flags any inconsistencies.

Feature flags (env, default ``true``):
  - ``HENGE_ENABLE_ADVERSARIAL_SCOPING`` — skip step 2 if set to ``false``.
  - ``HENGE_ENABLE_CANONICAL_CONTEXT`` — skip step 5 if set to ``false``.

Cross-lab rationale: Haiku and Opus are Anthropic; gpt-5 is OpenAI. If Haiku
has a systematic bias about what counts as a "good scoping question", gpt-5
catches it from a different reasoning lineage.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Literal


# Feature flags read at import time. Override via .env or env vars.
def _flag(name: str, default: bool = True) -> bool:
    val = os.getenv(name, "true" if default else "false").strip().lower()
    return val in ("1", "true", "yes", "on")


ENABLE_ADVERSARIAL = _flag("HENGE_ENABLE_ADVERSARIAL_SCOPING", True)
ENABLE_CANONICAL_CONTEXT = _flag("HENGE_ENABLE_CANONICAL_CONTEXT", True)


@dataclass
class ScopedQuestion:
    id: str
    text: str
    source: Literal["scoping", "adversarial"]
    challenges_assumption: str | None


@dataclass
class ScopingResult:
    questions: list[ScopedQuestion]
    adversarial_count: int
    version: str
    haiku_usage: dict | None
    gpt5_usage: dict | None


@dataclass
class CanonicalContext:
    summary: str
    flags: list[str] = field(default_factory=list)
    opus_usage: dict | None = None


# Pipeline functions (run_scoping, finalize_context) are added in Tasks 3.2 / 3.3.
# Back-compat shim ``generate_questions(client, question)`` lives at the bottom
# (added in Task 3.4) so server.py keeps booting until its call-sites migrate.
