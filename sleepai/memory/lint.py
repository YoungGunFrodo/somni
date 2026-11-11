"""Lint memories before storage and again before training.

Memories are data, never instructions. Anything that looks like an instruction
to the model, a secret, or private data is rejected here.
"""
from __future__ import annotations

import re

KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*){1,3}$")
TYPES = {"fact", "skill", "lesson", "preference", "decision", "precedent"}
STRENGTHS = {"verified", "user-confirmed", "inferred"}
STATUSES = {"active", "disputed", "superseded"}
MAX_SUMMARY_WORDS = 40
MAX_BODY_WORDS = 250

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules)",
    r"disregard (your|all|the) (instructions|rules|guidelines)",
    r"\byou are now\b",
    r"\bsystem prompt\b",
    r"\bnew instructions?\b",
    r"\bjailbreak\b",
    r"\bdo anything now\b",
    r"\boverride (the )?(evaluation|gates?|safety)\b",
]
SECRET_PATTERNS = [
    r"\bsk-[A-Za-z0-9]{16,}\b",            # API keys
    r"\bAKIA[0-9A-Z]{16}\b",               # AWS access keys
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"\b(password|passwd|secret|token)\s*[:=]\s*\S+",
]
PRIVATE_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",              # US SSN shape
    r"\b(?:\d[ -]?){13,16}\b",             # card-number shape
]

_inj = [re.compile(p, re.I) for p in INJECTION_PATTERNS]
_sec = [re.compile(p, re.I) for p in SECRET_PATTERNS]
_prv = [re.compile(p) for p in PRIVATE_PATTERNS]


def scan_text(text: str) -> list[str]:
    """Return problems found in free text (used for memories and training data)."""
    problems = []
    if any(p.search(text) for p in _inj):
        problems.append("injection-pattern")
    if any(p.search(text) for p in _sec):
        problems.append("secret-pattern")
    if any(p.search(text) for p in _prv):
        problems.append("private-data-pattern")
    return problems


def lint_memory(mem) -> list[str]:
    """Return a list of problems; empty list means the memory passes."""
    problems = []
    if not KEY_RE.match(mem.key or ""):
        problems.append(f"bad-key:{mem.key!r}")
    if mem.type not in TYPES:
        problems.append(f"bad-type:{mem.type}")
    if mem.strength not in STRENGTHS:
        problems.append(f"bad-strength:{mem.strength}")
    if mem.status not in STATUSES:
        problems.append(f"bad-status:{mem.status}")
    if not mem.sources:
        problems.append("no-sources")
    if len((mem.summary or "").split()) > MAX_SUMMARY_WORDS:
        problems.append("summary-too-long")
    if len((mem.body or "").split()) > MAX_BODY_WORDS:
        problems.append("body-too-long")
    problems += scan_text(" ".join([mem.summary or "", mem.body or "", mem.value or ""]))
    return problems
