"""Subject tags for memories (user's idea, Zettelkasten-style).

Each memory gets three subject tags, broad to specific:
  field    one of a fixed list (history, science, medicine, ...)    teacher, once per source
  topic    the source's main subject (normans, nikola-tesla)         teacher, once per source
  subject  what the memory itself is about (rollo, the key's first part)   no model call
Any combination finds it (MemoryStore.by_tags); memories sharing two tags are linked
(MemoryStore.related). Runs after memory formation, so it also tags memories formed before
tagging existed (backfill). Fact extraction itself is unchanged.

First version (per-fact tags written by the teacher) was too fragmented with a 1.7B teacher
(30 passages: "school-type" and "type-of-school"; attributes such as "potency" as subjects; whole
sources lost to broken replies). Tag hygiene: topics already in use are offered for reuse, and
tags are normalised (lowercase, hyphens, simple plurals folded onto an existing tag).
"""
from __future__ import annotations

import re
from collections import defaultdict

from ..models.llm import slug



def normalise(tag: str, vocab) -> str:
    t = slug(str(tag))[:32]
    if not t or t.isdigit():
        return ""
    for alt in (t[:-1] if t.endswith("s") else None, t + "s", t[:-2] if t.endswith("es") else None):
        if alt and alt in vocab:
            return alt
    return t


def _source(m) -> str:
    for s in m.sources:
        if str(s).startswith("episode:"):
            return str(s)
    return "none"


def tag_untagged(store, teacher, log=print) -> dict:
    """Tag every active memory that has no tags yet. Returns counts for the dream report."""
    if not hasattr(teacher, "source_tags"):
        return {"tagged": 0, "skipped": "teacher cannot tag"}
    groups = defaultdict(list)
    for m in store.untagged():
        groups[_source(m)].append(m)
    vocab = dict(store.tag_counts())
    topics = dict(store.topic_counts()) if hasattr(store, "topic_counts") else {}
    tagged = calls = 0
    for src, mems in groups.items():
        title = ""
        if src.startswith("episode:"):
            f = store.file_text(int(src.split(":", 1)[1]))
            title = f[0] if f else ""
        field_, topic = teacher.source_tags(mems, list(topics), title=title)
        calls += 1
        field_, topic = normalise(field_, vocab) or "general", normalise(topic, vocab)
        if topic:
            topics[topic] = topics.get(topic, 0) + 1
        for m in mems:
            subject = normalise(m.key.split("/")[0], vocab)
            clean = list(dict.fromkeys(t for t in (field_, topic, subject) if t))
            store.set_tags(m.id, clean)
            tagged += 1
            for t in clean:
                vocab[t] = vocab.get(t, 0) + 1
    if calls:
        log(f"[tags] tagged {tagged} memories in {calls} calls; {len(vocab)} tags in use")
    return {"tagged": tagged, "calls": calls, "tags_in_use": len(vocab)}


def question_tags(question: str, vocab) -> list[str]:
    """Tags named by the question itself (word or two-word match, plurals folded)."""
    words = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2]
    cands = set(words) | {f"{a}-{b}" for a, b in zip(words, words[1:])}
    out = []
    for c in cands:
        t = normalise(c, vocab)
        if t in vocab:
            out.append(t)
    return out
