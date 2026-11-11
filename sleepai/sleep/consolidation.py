"""Sleep stage 2, consolidation: verified memories -> weight updates.

  * new memories      -> varied training questions from the teacher (paraphrases)
  * superseded facts  -> correction examples, weighted higher, so the new value
                         overrides the old one in the weights
  * replay            -> a sample of previously consolidated memories, weighted toward
                         ones whose last probe failed, plus optional base-knowledge
                         examples, mixed in at `replay_ratio`. With the forgetting policy
                         (memory/salience.py) PROTECTED memories get this full replay,
                         CONDENSED ones a light, low-weight trace sampled by salience
                         (`condensed_replay_ratio`), FADED ones none (all stay in the store)
  * adapter policy    -> incremental training from the active checkpoint, with a full
                         rebuild from the base model every `rebuild_every` sleeps on a
                         capped, importance-sampled set of all eligible memories
  * abstain examples  -> "I'm not sure" on questions about invented subjects, at
                         `abstain_ratio`, so training does not erode calibration
  * data hygiene      -> every example is scanned again for injection/secret/private data
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..memory.lint import scan_text
from ..memory.salience import CONDENSED, FADED, PROTECTED, salience, tiers
from ..models.base import TrainingExample


@dataclass
class ConsolidationResult:
    new_memories: list = field(default_factory=list)
    corrections: int = 0
    new_examples: int = 0
    replay_examples: int = 0
    dropped_by_hygiene: int = 0
    rebuild: bool = False
    rebuild_memories: list = field(default_factory=list)  # old memories retrained by a rebuild
    protected: int = 0      # consolidated memories with full replay (and the retention gate)
    condensed: int = 0      # consolidated memories kept as a light, low-detail trace
    faded: int = 0          # consolidated memories no longer replayed (still in the store)
    condensed_examples: int = 0
    train: dict = field(default_factory=dict)


def _importance(m) -> float:
    return 1.0 + (2.0 if m.last_probe_result == "fail" else 0.0) + 0.1 * min(m.retrieval_count, 20)


def _teacher_pairs(system, m, n: int) -> list[tuple[str, str]]:
    """Rich (question, answer) pairs from the teacher, cached per memory so replay on later
    nights reuses them instead of calling the teacher again. Generated in full once (at least
    `paraphrases_per_memory`), then sampled."""
    import json
    path = system.home / "training_pairs.json"
    cache = getattr(system, "_pairs_cache", None)
    if cache is None:
        cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        system._pairs_cache = cache
    key = f"{m.id}|{m.value}"
    if key not in cache:
        cache[key] = system.teacher.training_pairs(m, max(n, system.config.sleep.paraphrases_per_memory))
        path.write_text(json.dumps(cache), encoding="utf-8")
    pairs = cache[key]
    if len(pairs) <= n:
        return [tuple(p) for p in pairs]
    # Stable seed (Python's hash() of a str changes every run, so training data did too).
    import zlib
    idx = np.random.default_rng(zlib.crc32(f"{key}|{len(cache)}".encode())).choice(len(pairs), size=n,
                                                                                  replace=False)
    return [tuple(pairs[i]) for i in idx]


def build_examples(system, memories, n_per_memory: int, origin: str) -> list[TrainingExample]:
    out = []
    rich = hasattr(system.teacher, "training_pairs")
    for m in memories:
        weight = 1.0
        if m.supersedes and origin == "new":
            weight, kind = 2.0, "correction"
        else:
            kind = origin
        if rich:
            pairs = _teacher_pairs(system, m, n_per_memory)
            # Plus phrasings the teacher did not write (phrasings.py): recall on outside
            # phrasings was 70% when training used the teacher's questions only.
            from ..phrasings import TRAIN_STYLES, questions
            k = len(TRAIN_STYLES) if origin == "new" else 2
            styled = questions(m, TRAIN_STYLES)
            if styled:
                start = len(out) % len(styled)               # rotate which ones replay gets
                styled = (styled[start:] + styled[:start])[:k]
            if pairs or styled:
                out += [TrainingExample(q, a, weight, kind, m.id) for q, a in pairs]
                out += [TrainingExample(q, m.value, weight, kind, m.id) for q in styled]
                if origin == "new":
                    out += lookup_examples(system, m, [q for q, _ in pairs] + styled, weight, kind)
                continue
        for q in system.teacher.training_questions(m, n_per_memory):
            out.append(TrainingExample(q, m.value, weight, kind, m.id))
    return out


def lookup_examples(system, m, questions: list, weight: float, kind: str) -> list[TrainingExample]:
    """Lookup practice: the question with exactly what wake's recall returns for it. The answer
    cites the memory (or the memory file) holding the value, which is what salience counts at
    wake; if recall missed it, the answer comes from the weights, uncited, as it would at wake."""
    n = getattr(system.config.sleep, "lookup_practice", 0)
    if not n or not questions:
        return []
    import zlib
    from ..wake.orchestrator import Orchestrator
    orch = Orchestrator(system.store, None)
    out, val = [], str(m.value).lower()
    start = zlib.crc32(m.id.encode()) % len(questions)
    for q in (questions[start:] + questions[:start])[:n]:
        ctx = orch.recall(q)
        cite = next((c.key for c in ctx if c.id == m.id), "") or \
            next((c.key for c in ctx if c.type == "file" and val in c.summary.lower()), "")
        ans = f"{m.value}\n[memory: {cite}]" if cite else str(m.value)
        out.append(TrainingExample(q, ans, weight, kind, m.id, context=ctx))
    return out


def condensed_examples(system, memories, weight: float) -> list[TrainingExample]:
    """A light trace for each condensed memory. A teacher may lump memories of one subject
    into one example (less detail, less training cost); the default is one plain question each."""
    if not memories:
        return []
    if hasattr(system.teacher, "condensed_examples"):
        return system.teacher.condensed_examples(memories, weight)
    return [TrainingExample(q, m.value, weight, "condensed", m.id)
            for m in memories for q in system.teacher.training_questions(m, 1)]


def run_consolidation(system, sleep_no: int, rng: np.random.Generator, force_rebuild: bool = False,
                      day: int = 0, full_store: bool = False, exclude: set | None = None,
                      rehearsal_boost: float = 1.0):
    """full_store=True (brain transfer): rebuild every eligible memory in full, whatever its tier.
    A transfer moves to a new, usually bigger brain; condensing is for the capacity of this one."""
    cfg = system.config.sleep
    store = system.store
    eligible = store.eligible_for_consolidation()
    new = [m for m in eligible if not m.consolidated_in and m.id not in (exclude or ())]
    by_tier = tiers([m for m in eligible if m.consolidated_in], day, cfg)
    old, condensed = by_tier[PROTECTED], by_tier[CONDENSED]
    # Sleep budget: corrections first, then most-used, then oldest. The rest waits.
    new.sort(key=lambda m: (not m.supersedes, -m.retrieval_count, m.created_day))
    backlog = new[cfg.max_new_per_sleep:]
    new = new[:cfg.max_new_per_sleep]
    res = ConsolidationResult(new_memories=[m.id for m in new], protected=len(old),
                              condensed=len(condensed), faded=len(by_tier[FADED]))
    rebuild = sleep_no > 0 and cfg.rebuild_every > 0 and sleep_no % cfg.rebuild_every == 0
    # Stuck detector (self-test finding): after repeated gate rejections the active checkpoint
    # stops moving and every incremental candidate fails the same way. A rebuild from base
    # trains all memories together, interleaved, and breaks the loop.
    recent = [c.get("status") for c in list(system.registry.data["candidates"].values())[-cfg.rebuild_after_rejections:]]
    if (cfg.rebuild_after_rejections and len(recent) == cfg.rebuild_after_rejections
            and all(st == "rejected" for st in recent)):
        rebuild = True
    rebuild = rebuild or force_rebuild
    res.rebuild = rebuild

    examples = build_examples(system, new, cfg.paraphrases_per_memory, "new")
    res.corrections = sum(1 for e in examples if e.origin == "correction")
    res.new_examples = len(examples)

    if rebuild:
        # Full rebuild from base: every eligible memory, capped by importance sampling.
        # With forgetting: protected memories (and protected backlog) in full, condensed ones as
        # a light trace, faded ones not at all.
        cap = getattr(system, "rebuild_cap", 2000)
        if full_store:
            pool = sorted(old + condensed + by_tier[FADED] + backlog, key=_importance, reverse=True)[:cap]
            light = []
        else:
            back = tiers(backlog, day, cfg)
            pool = sorted(old + back[PROTECTED], key=_importance, reverse=True)[:cap]
            light = condensed + back[CONDENSED]
        res.rebuild_memories = [m.id for m in pool + light]
        examples += build_examples(system, pool, cfg.paraphrases_per_memory, "replay")
        # A rebuild starts from the base model, so a light trace cannot MAINTAIN a condensed
        # memory here; it has to be relearned. (Self-test finding: with one low-weight example,
        # every rebuild wiped the condensed memories, 0.92 -> 0.49 recall in two nights.)
        cond = [e for e in build_examples(system, light, cfg.condensed_rebuild_paraphrases, "condensed")]
        res.condensed_examples = len(cond)
        examples += cond
    elif cfg.replay and old:
        if cfg.replay_coverage:
            # Self-test finding: a replay SAMPLE sized to the new material leaves a growing share
            # of old memories unreplayed each night as the store grows, and they erode until the
            # retention gate refuses every sleep. Coverage: every protected memory, once.
            examples += build_examples(system, old, 1, "replay")
        n_replay = int(round(cfg.replay_ratio * max(1, res.new_examples)))
        weights = np.array([_importance(m) for m in old])
        picks = rng.choice(len(old), size=min(len(old), max(1, n_replay // 2)), replace=False,
                           p=weights / weights.sum())
        replay = build_examples(system, [old[i] for i in picks], 2, "replay")
        examples += replay[:n_replay]
    if not rebuild and cfg.replay and condensed:
        # Light trace: a few condensed memories each sleep, the more salient the more often.
        k = min(len(condensed), int(round(cfg.condensed_replay_ratio * max(1, res.new_examples))))
        w = np.array([salience(m, day, cfg.half_life_days) for m in condensed]) + 1e-6
        picks = rng.choice(len(condensed), size=k, replace=False, p=w / w.sum()) if k else []
        cond = condensed_examples(system, [condensed[i] for i in picks], cfg.condensed_weight)
        res.condensed_examples = len(cond)
        examples += cond
    n_abstain = int(round(cfg.abstain_ratio * len(examples)))
    if n_abstain and hasattr(system.teacher, "abstain_questions"):
        from ..models.base import ABSTAIN
        examples += [TrainingExample(q, ABSTAIN, 1.0, "abstain")
                     for q in system.teacher.abstain_questions(eligible, n_abstain)]
    if rebuild and system.base_replay:
        examples += list(system.base_replay)       # a rebuild replays ALL base knowledge
    elif cfg.replay and system.base_replay:
        k = min(len(system.base_replay), max(1, int(cfg.base_replay_ratio * rehearsal_boost * len(examples))))
        from ..rehearsal import rehearsal_weights, sample
        examples += sample(system.base_replay, k, rng, rehearsal_weights(system, day))
    res.replay_examples = len(examples) - res.new_examples

    clean = [e for e in examples if not scan_text(e.question + " " + e.answer)]
    res.dropped_by_hygiene = len(examples) - len(clean)

    student = system.load_student(system.registry.active)
    if clean or rebuild:
        res.train = system.trainer.train(student, clean, from_base=rebuild,
                                         epochs=cfg.rebuild_epochs if rebuild else cfg.epochs)
    return student, res
