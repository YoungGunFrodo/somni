"""Sleep stage 3, dreaming: checked self-practice.

Seeds: memories consolidated this sleep, memories whose last probe failed, and memories
that answered an open question. For each seed the teacher writes practice questions,
the candidate student attempts them WITHOUT retrieval, and a deterministic checker grades
the attempt against the memory store. Failed attempts become corrective training data,
mixed with replay; practice stops once the pass rate reaches 95%. (Self-test finding:
training on already-mastered practice only erodes other knowledge.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..memory.salience import salience, split
from ..models.base import TrainingExample
from .checkers import exact


@dataclass
class DreamResult:
    seeds: int = 0
    attempts: int = 0
    passed: int = 0
    rounds: list = field(default_factory=list)   # pass rate per round
    samples: list = field(default_factory=list)  # a few dreams for the report
    slipped: int = 0      # protected memories that failed their practice check after training
    repaired: int = 0     # of those, re-practiced this sleep (most salient first)
    train: list = field(default_factory=list)


def _answers(student, questions):
    """All answers in one batch when the student supports it (same answers, one model load)."""
    if not questions:
        return []
    if hasattr(student, "answer_many"):
        return student.answer_many(questions)
    return [student.answer(q, context=None) for q in questions]


def run_dreaming(system, student, new_memory_ids: list[str], rng: np.random.Generator,
                 day: int = 0) -> DreamResult:
    cfg = system.config.sleep
    store = system.store
    res = DreamResult()
    seeds = {mid: store.get(mid) for mid in new_memory_ids}
    for m in store.eligible_for_consolidation():
        if m.last_probe_result == "fail":
            seeds[m.id] = m
    resolved = store.db.execute("SELECT resolved_by FROM open_questions WHERE status='resolved'").fetchall()
    for (mid,) in resolved:
        m = store.get(mid)
        if m and m.eligible:
            seeds[mid] = m
    seeds = {k: v for k, v in seeds.items() if v is not None and v.eligible}
    if cfg.repair:
        # Repair (self-test finding): a night's training damages protected memories at random,
        # used and unused alike. Check every protected memory with one PRACTICE question (never a
        # gate probe) and re-practice the ones that slipped, most salient first, so any damage
        # that remains falls on the least useful memories.
        prot, _ = split([m for m in store.eligible_for_consolidation()
                         if m.consolidated_in and m.id not in seeds], day, cfg)
        # Teacher phase, then student phase (batched): on one GPU the two models take turns.
        asked = [(m, q[0]) for m in prot for q in [system.teacher.dream_questions(m, 1)] if q]
        got = _answers(student, [q for _, q in asked])
        slipped = [m for (m, _), a in zip(asked, got) if not exact(a.value, m.value)]
        slipped.sort(key=lambda m: salience(m, day, cfg.half_life_days), reverse=True)
        res.repaired = len(slipped[:cfg.repair_max])
        res.slipped = len(slipped)
        for m in slipped[:cfg.repair_max]:
            seeds[m.id] = m
    res.seeds = len(seeds)
    if not seeds:
        return res

    old = [m for m in store.eligible_for_consolidation() if m.consolidated_in and m.id not in seeds]
    old, _rest = split(old, day, cfg)   # dream replay covers protected memories only
    for r in range(cfg.dream_rounds):
        data, passed, attempts = [], 0, 0
        asked = [(m, q) for m in seeds.values() for q in system.teacher.dream_questions(m, 3)]  # teacher
        got = _answers(student, [q for _, q in asked])                                          # student
        for (m, q), ans in zip(asked, got):
            ok = exact(ans.value, m.value)
            attempts += 1
            passed += ok
            if not ok:  # only failures are trained: practice on mastered material just
                data.append(TrainingExample(q, m.value, 1.5, "dream", m.id))  # erodes other knowledge
            if len(res.samples) < 5 and r == 0:
                res.samples.append({"memory": m.id, "question": q, "attempt": ans.value,
                                    "expected": m.value, "passed": bool(ok)})
        res.attempts += attempts
        res.passed += passed
        res.rounds.append(round(passed / attempts, 3) if attempts else None)
        if not data or (attempts and passed / attempts >= 0.95):
            break  # nothing left to practice
        if old and cfg.replay:
            picks = rng.choice(len(old), size=min(len(old), len(data) // 3 + 1), replace=False)
            for i in picks:
                for q in system.teacher.training_questions(old[i], 1):
                    data.append(TrainingExample(q, old[i].value, 1.0, "replay", old[i].id))
        if cfg.replay and system.base_replay:
            k = min(len(system.base_replay), max(1, int(cfg.base_replay_ratio * len(data))))
            from ..rehearsal import rehearsal_weights, sample
            data += sample(system.base_replay, k, rng, rehearsal_weights(system, day))
        res.train.append(system.trainer.train(student, data, epochs=max(2, cfg.epochs // 3)))
    return res
