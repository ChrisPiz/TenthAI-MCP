"""Meta-frame — audits the question itself before spending on the 9 advisors.

v0.6 design: gpt-5 [OpenAI] runs cross-lab audit on the question + canonical
context. If the question is exploration disguised as decision, or a proxy for
the real question, the meta-frame recommends reformulation and the server
short-circuits the run before the 9 frames fire. This saves ~$1.50 per
malformed run.

The full implementation of ``evaluate_question_quality`` lands in Task 4.2.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


def _flag(name: str, default: bool = True) -> bool:
    val = os.getenv(name, "true" if default else "false").strip().lower()
    return val in ("1", "true", "yes", "on")


ENABLE_META_FRAME = _flag("HENGE_ENABLE_META_FRAME", True)


DecisionClass = Literal["reversible", "one-way-door", "two-way-with-cost", "unknown"]
Urgency = Literal["now", "weeks", "months", "fake-urgency", "unknown"]
QuestionQuality = Literal[
    "well-formed",
    "proxy-for-other-question",
    "exploration-disguised-as-decision",
    "unknown",
]
MetaRecommendation = Literal[
    "proceed",
    "reformulate",
    "postpone",
    "this-is-not-a-decision",
]


@dataclass
class MetaFrameResult:
    decision_class: DecisionClass
    urgency: Urgency
    question_quality: QuestionQuality
    suggested_reformulation: str | None
    meta_recommendation: MetaRecommendation
    reasoning: str
    gpt5_usage: dict | None
