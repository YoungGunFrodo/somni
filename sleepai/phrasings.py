"""Question phrasings that do not come from the teacher.

Real-night finding (2026-11-10): the learning gate passed at 95.7% on probes the teacher wrote,
but recall on phrasings the teacher never wrote was 70%: the teacher wrote both the training
questions and the probes, with the same habits. A user asks in their own words. So:

  TRAIN_STYLES  mixed into training (with the teacher's varied pairs)
  GATE_STYLES   used ONLY by the "learning-outside" gate (never trained on)

The two families share no template. (scripts/real_night.py keeps a third, independent set:
milestone-0 PROBES.) Applies to memories whose summary is canonical ("The <attribute> of
<subject> is <value>."), which structured memory formation now always writes for facts.
"""
from __future__ import annotations

import re

TRAIN_STYLES = [
    "What is the {a} of {s}?", "Tell me the {a} of {s}.", "{s}: what's its {a}?",
    "Do you know the {a} for {s}?", "Remind me of {s}'s {a}.", "Quick question, {s} {a}?",
    "yo whats the {a} for {s}", "{a} of {s} pls",
]

GATE_STYLES = [
    "Which {a} does {s} have?", "I forgot the {a} of {s}. What was it?",
    "hey, {s} {a}? need it", "What's {s}'s {a} again?",
]

_CANON = re.compile(r"^The (?P<attr>.+?) of (?P<subj>.+?) is (?P<val>.+?)\.?$")


def parse(memory) -> tuple[str, str] | None:
    """(attribute, subject) of a canonical fact summary, or None."""
    m = _CANON.match((memory.summary or "").strip())
    return (m["attr"], m["subj"]) if m else None


def questions(memory, styles: list[str], n: int | None = None) -> list[str]:
    p = parse(memory)
    if not p:
        return []
    a, s = p
    qs = [t.format(a=a, s=s) for t in styles]
    return qs[:n] if n else qs
