"""Sleep stage 4, evaluation gates.

All gates answer WITHOUT retrieval: they test what lives in the weights.
Comparisons are PAIRED (same probes, current vs candidate checkpoint) with a bootstrap
confidence interval, so a gate fails on a real regression, not on noise. Gates with
fewer probes than `min_probes_for_power` are flagged "underpowered" in the dream report.

Probe questions are written apart from training questions (held-out templates or a
different prompt) and cached so every sleep uses the same probes for a memory.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from ..memory.salience import CONDENSED, PROTECTED, tiers
from ..models.base import ABSTAIN, answers_match


@dataclass
class Gate:
    name: str
    passed: bool
    n: int
    current: float | None
    candidate: float
    diff: float | None = None
    ci: tuple | None = None
    underpowered: bool = False
    note: str = ""


@dataclass
class EvalResult:
    gates: list = field(default_factory=list)
    passed: bool = True
    margin_ok: bool = True
    info: dict = field(default_factory=dict)   # measured and reported, never gating

    def to_dict(self):
        return {"passed": self.passed, "margin_ok": self.margin_ok,
                "gates": [asdict(g) for g in self.gates], "info": self.info}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, samples: int, rng) -> tuple[float, tuple]:
    """Mean of (b - a) with a 95% bootstrap CI."""
    d = b.astype(float) - a.astype(float)
    if len(d) == 0:
        return 0.0, (0.0, 0.0)
    idx = rng.integers(0, len(d), size=(samples, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


class ProbeCache:
    def __init__(self, path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else {}

    def get(self, teacher, memory):
        if memory.id not in self.data:
            self.data[memory.id] = [asdict(p) for p in teacher.probes(memory)]
            if self.path:
                self.path.write_text(json.dumps(self.data), encoding="utf-8")
        return self.data[memory.id]


def extract_code(text: str) -> str:
    import re
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return "\n\n".join(blocks) if blocks else text


def _score(student, probes, mode="correct") -> np.ndarray:
    out = []
    if mode == "python":                        # coding skill: run the answer against unit tests
        from .checkers import python_tests
        if hasattr(student, "answer_many"):
            answers = student.answer_many([p["question"] + " Reply with only the code." for p in probes],
                                          max_new_tokens=384)
        else:
            answers = [student.answer(p["question"]) for p in probes]
        return np.array([python_tests(extract_code(a.text), p["tests"])[0]
                         for p, a in zip(probes, answers)], dtype=bool)
    from ..rehearsal import context_from          # reading probes carry their passage
    ctxs = [context_from(p.get("context")) for p in probes]
    if hasattr(student, "answer_many"):         # real student: batched generation, same answers
        answers = student.answer_many([p["question"] for p in probes], ctxs if any(ctxs) else None)
    else:
        answers = [student.answer(p["question"], context=c) for p, c in zip(probes, ctxs)]
    for p, ans in zip(probes, answers):
        if mode == "correct":
            out.append(answers_match(ans.value, p["answer"]))
        elif mode == "asserts":       # says this (wrong) value as fact
            out.append(answers_match(ans.value, p["answer"]) and not ans.caveat)
        elif mode == "abstains":
            out.append(ans.value == ABSTAIN)
    return np.array(out, dtype=bool)


def probe_key(p) -> str:
    return f"{p['memory_id']}|{p['question']}"


def load_reference(system) -> dict:
    """Best result each probe ever had in an ADOPTED checkpoint. Written only by morning review
    (review.morning.approve), read here; the sleep pipeline never writes it."""
    path = system.home / "retention_reference.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def evaluate(system, current, candidate, new_memory_ids, rng, day: int = 0) -> EvalResult:
    g = system.config.gates
    store, cache = system.store, system.probe_cache
    res = EvalResult()

    def probes_for(mems, limit=None):
        ps = []
        for m in mems:
            ps += cache.get(system.teacher, m)
        if limit and len(ps) > limit:
            ps = [ps[i] for i in rng.choice(len(ps), size=limit, replace=False)]
        return ps

    # Teacher phase first: write (and cache) every probe this evaluation can need, so on one GPU
    # the teacher and the student take turns once instead of alternating per memory.
    for m in store.all():
        cache.get(system.teacher, m)

    # Learning: new memories, candidate only (absolute threshold).
    new = [store.get(i) for i in new_memory_ids]
    p_new = probes_for([m for m in new if m])
    results = {}            # candidate's per-probe results; morning review keeps the best ever
    if p_new:
        c = _score(candidate, p_new)
        results.update({probe_key(p): bool(ok) for p, ok in zip(p_new, c)})
        res.gates.append(Gate("learning", bool(c.mean() >= g.learning_min), len(c), None, float(c.mean()),
                              underpowered=len(c) < g.min_probes_for_power))
        # Learning on phrasings the teacher did not write (user-approved 2026-11-10): the
        # teacher's own probes read optimistic (95.7% vs 70% on outside phrasings).
        from ..phrasings import GATE_STYLES, questions
        # Real models only (teachers that write training pairs): the toy brain is a pipeline
        # simulator with its own templates, and the self-test stays comparable.
        real = hasattr(system.teacher, "training_pairs")
        p_out = [{"memory_id": m.id, "question": q, "answer": m.value, "kind": "outside"}
                 for m in new if m and real for q in questions(m, GATE_STYLES)]
        if p_out:
            c2 = _score(candidate, p_out)
            results.update({probe_key(p): bool(ok) for p, ok in zip(p_out, c2)})
            res.gates.append(Gate("learning-outside", bool(c2.mean() >= g.learning_min), len(c2), None,
                                  float(c2.mean()), underpowered=len(c2) < g.min_probes_for_power))
        # remember per-memory results for dreaming and replay weighting
        by_mem = {}
        for p, ok in zip(p_new, c):
            by_mem.setdefault(p["memory_id"], []).append(ok)
        for mid, oks in by_mem.items():
            store.update_fields(mid, last_probe_result="pass" if np.mean(oks) >= 0.5 else "fail")

    def paired_gate(name, probes, tol, mode="correct", higher_is_better=True, reference=None):
        if not probes:
            return
        a, b = _score(reference or current, probes, mode), _score(candidate, probes, mode)
        if name == "retention":
            results.update({probe_key(p): bool(ok) for p, ok in zip(probes, b)})
        diff, ci = paired_bootstrap(a, b, g.bootstrap_samples, rng)
        signed, signed_ci = (diff, ci) if higher_is_better else (-diff, (-ci[1], -ci[0]))
        failed = signed < -tol and signed_ci[1] < 0          # a real, significant regression
        res.gates.append(Gate(name, bool(not failed), len(a), float(a.mean()), float(b.mean()),
                              round(diff, 4), (round(ci[0], 4), round(ci[1], 4)),
                              underpowered=len(a) < g.min_probes_for_power,
                              note="" if not (signed < -tol and not failed) else "drop not significant"))

    old = [m for m in store.eligible_for_consolidation() if m.consolidated_in and m.id not in new_memory_ids]
    # Forgetting policy: retention covers PROTECTED memories only. Condensed memories are kept
    # as a lighter trace and measured below for the dream report (not a gate); faded ones may
    # leave the weights. All stay in the memory store. Salience is written only by wake
    # (orchestrator touches), so the sleep pipeline cannot shrink the protected set itself.
    by_tier = tiers(old, day, system.config.sleep)
    paired_gate("retention", probes_for(by_tier[PROTECTED], limit=300), g.retention_tolerance)
    # Cumulative floor (user-approved, session 4): retention vs the CURRENT checkpoint lets small
    # nightly losses add up (1.00 -> 0.85 over a month in the toy runs). Each protected probe is
    # also compared with the best it ever did in an adopted checkpoint.
    ref = load_reference(system)
    known = [p for p in probes_for(by_tier[PROTECTED]) if ref.get(probe_key(p))]
    if len(known) > 300:
        known = [known[i] for i in rng.choice(len(known), size=300, replace=False)]
    if known and g.retention_floor_tolerance is not None:
        b = _score(candidate, known)
        results.update({probe_key(p): bool(ok) for p, ok in zip(known, b)})
        a = np.ones(len(b), dtype=bool)
        diff, ci = paired_bootstrap(a, b, g.bootstrap_samples, rng)
        failed = diff < -g.retention_floor_tolerance and ci[1] < 0
        res.gates.append(Gate("retention-floor", bool(not failed), len(b), 1.0, float(b.mean()),
                              round(diff, 4), (round(ci[0], 4), round(ci[1], 4)),
                              underpowered=len(b) < g.min_probes_for_power,
                              note="" if not (diff < -g.retention_floor_tolerance and not failed)
                              else "drop not significant"))
    p_cond = probes_for(by_tier[CONDENSED], limit=300)
    if p_cond:
        a, b = _score(current, p_cond), _score(candidate, p_cond)
        res.info["condensed_recall"] = {"n": len(a), "current": float(a.mean()), "candidate": float(b.mean())}
    paired_gate("general", system.base_probes, g.general_tolerance)
    # Drift: per-sleep gates miss slow decline (1% a night passes every night), so also
    # compare with the original base model using a cumulative tolerance.
    if system.registry.base and system.registry.active != system.registry.base:
        paired_gate("drift-vs-base", system.base_probes, g.drift_tolerance,
                    reference=system.load_student(system.registry.base))

    stale = []
    for m in store.all("superseded"):
        stale += [{**p, "answer": m.value} for p in cache.get(system.teacher, m)]
    paired_gate("stale-beliefs", stale[:300], 0.0, mode="asserts", higher_is_better=False)

    disputed = []
    for m in store.all("disputed"):
        active = store.active_by_key(m.key)
        if active is None or active.value != m.value:
            disputed += [{**p, "answer": m.value} for p in cache.get(system.teacher, m)]
    paired_gate("disputed-not-asserted", disputed[:300], 0.0, mode="asserts", higher_is_better=False)

    paired_gate("calibration", system.unknown_probes, g.calibration_tolerance, mode="abstains")
    # Other skill packs with exact answers (e.g. reasoning): same rule as general knowledge.
    for name, (probes, mode) in getattr(system, "skill_probes", {}).items():
        paired_gate(f"skill:{name}", probes, g.general_tolerance, mode=mode)
    # Coding skill pack (opt-in): no task that passed before may fail now.
    coding = getattr(system, "coding_probes", None)
    if coding:
        a, b = _score(current, coding, "python"), _score(candidate, coding, "python")
        broke = int(np.sum(a & ~b))
        res.gates.append(Gate("coding", broke == 0, len(a), float(a.mean()), float(b.mean()),
                              round(float(b.mean() - a.mean()), 4),
                              note=f"{broke} previously passing task(s) now fail" if broke else ""))

    res.info["probe_results"] = results
    res.passed = bool(all(x.passed for x in res.gates))
    # "Clear margin" for autonomy level 2: passed, and no gate needed the significance escape.
    res.margin_ok = res.passed and not any(x.note for x in res.gates) and not any(
        x.underpowered for x in res.gates)
    return res
