"""Salience, condensation and fading: the forgetting policy.

Human-like forgetting: memories that are used stay sharp in the weights; memories that are not
used are first CONDENSED (kept in the weights as a light, low-detail trace, replayed rarely and
grouped by subject) and only after a long time without use do they FADE from the weights.
Nothing is ever deleted from the memory store: retrieval still finds every memory in full detail.

    salience = (1 + uses) * 0.5 ** ((today - last_used_day) / half_life)   (+1 for corrections)

Tiers (a memory created within `grace_days` is always protected):
    protected   salience >= protect_min_salience    full replay; tested by the retention gate
    condensed   forget_below_salience <= salience   light replay (one low-weight example,
                < protect_min_salience              sampled by salience); reported, not gated
    faded       salience < forget_below_salience    no replay; may leave the weights

With the defaults (half-life 30 days, protect 1.5, forget below 0.1, grace 7 days) a memory that
is never used is condensed after 7 days and fades after about 100 days; each use moves it back
to protected and pushes fading further out. Using a condensed or faded memory again protects it
and it is relearned.

Salience comes only from wake: `uses` and `last_used_day` are written by the orchestrator when
an answer actually used the memory (MemoryStore.touch). Nothing in the sleep pipeline writes
them, so a sleep cannot make its own job easier by changing which memories are protected.
"""
from __future__ import annotations

PROTECTED, CONDENSED, FADED = "protected", "condensed", "faded"


def salience(memory, today: int, half_life_days: float) -> float:
    last = memory.last_used_day if memory.last_used_day >= 0 else memory.created_day
    age = max(0, today - last)
    s = (1 + memory.retrieval_count) * 0.5 ** (age / half_life_days)
    if memory.supersedes:          # a correction: the user cared enough to fix it
        s += 1.0
    return s


def tier(memory, today: int, cfg) -> str:
    """cfg is a SleepConfig. With forgetting off, every memory is protected."""
    if not getattr(cfg, "forgetting", False):
        return PROTECTED
    if today - memory.created_day < cfg.grace_days:
        return PROTECTED
    s = salience(memory, today, cfg.half_life_days)
    if s >= cfg.protect_min_salience:
        return PROTECTED
    return CONDENSED if s >= cfg.forget_below_salience else FADED


def protected(memory, today: int, cfg) -> bool:
    return tier(memory, today, cfg) == PROTECTED


def split(memories, today: int, cfg) -> tuple[list, list]:
    """(protected, not protected)"""
    keep, rest = [], []
    for m in memories:
        (keep if protected(m, today, cfg) else rest).append(m)
    return keep, rest


def tiers(memories, today: int, cfg) -> dict[str, list]:
    out = {PROTECTED: [], CONDENSED: [], FADED: []}
    for m in memories:
        out[tier(m, today, cfg)].append(m)
    return out
