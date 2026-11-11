"""Sleep stage 1, memory formation: episodes -> memories.

Conflict policy (deterministic, by key, never by text similarity):
  * new key                          -> new memory
  * same key, same value             -> update: refresh last_verified, keep the stronger source
  * same key, different value:
      new source >= old source       -> supersede (old memory archived, correction trained later)
      new source <  old source       -> new claim stored as DISPUTED, old stays active,
                                        flagged in the dream report for morning review
  * claims the user rejected before  -> not proposed again
  * lint failure                     -> rejected, recorded in the report
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..memory.lint import lint_memory
from ..memory.store import STRENGTH_RANK, Memory, MemoryStore


@dataclass
class FormationResult:
    episodes: int = 0
    created: list = field(default_factory=list)
    updated: list = field(default_factory=list)
    superseded: list = field(default_factory=list)   # (old_id, new_id)
    disputed: list = field(default_factory=list)     # new ids stored as disputed
    rejected: list = field(default_factory=list)     # (key, problems)
    skipped_previously_rejected: int = 0
    resolved_questions: int = 0


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower().replace("-", " ").replace("/", " ")))


def run_formation(store: MemoryStore, teacher, day: int = 0) -> FormationResult:
    res = FormationResult()
    open_qs = store.open_questions()
    for ep in store.unformed_episodes():
        res.episodes += 1
        formed_ids = []
        for p in teacher.form_memories(ep):
            mem = Memory(key=p.key, type=p.type, summary=p.summary, value=p.value, body=p.body,
                         sources=p.sources, strength=p.strength, created_day=day,
                         last_verified_day=day, never_consolidate=ep.never_consolidate)
            problems = lint_memory(mem)
            if problems:
                res.rejected.append((p.key, problems))
                continue
            if any(old.value == mem.value and old.rejection_reason.startswith("rejected at review")
                   for old in store.by_key_any(mem.key) if old.status == "disputed"):
                res.skipped_previously_rejected += 1
                continue
            existing = store.active_by_key(mem.key)
            if existing is None:
                mem = store.insert(mem)
                res.created.append(mem.id)
            elif existing.value == mem.value:
                fields = {"last_verified_day": day,
                          "sources": sorted(set(existing.sources) | set(mem.sources))}
                if STRENGTH_RANK[mem.strength] > existing.strength_rank:
                    fields["strength"] = mem.strength
                store.update_fields(existing.id, **fields)
                res.updated.append(existing.id)
                mem = existing
            elif STRENGTH_RANK[mem.strength] >= existing.strength_rank:
                mem = store.supersede(existing.id, mem)
                res.superseded.append((existing.id, mem.id))
            else:
                mem.status = "disputed"
                mem.rejection_reason = f"weaker claim conflicts with {existing.id}"
                mem = store.insert(mem)
                res.disputed.append(mem.id)
            formed_ids.append(mem.id)
            if mem.status == "active":
                need = _tokens(mem.key)
                for q in open_qs:
                    if q["status"] == "open" and need <= _tokens(q["question"]):
                        store.db.execute("UPDATE open_questions SET status='resolved', resolved_by=? "
                                         "WHERE id=?", (mem.id, q["id"]))
                        q["status"] = "resolved"
                        res.resolved_questions += 1
                store.db.commit()
        store.mark_formed(ep.id, formed_ids)
    return res
