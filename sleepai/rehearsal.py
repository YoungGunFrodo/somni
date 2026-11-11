"""Rehearsal for real models: the base model's OWN answers to general and coding prompts, made
once and replayed every sleep, so learning new memories does not erode what the model could
already do. (Real-model finding: a LoRA trained on 200 facts without rehearsal cut coding from
95% to 45% of tasks passing, and one real night to 10%.)

The prompts are disjoint from `evalsets` (the gate probes), so the gates stay held out.
"""
from __future__ import annotations

import json
from pathlib import Path

from .models.base import TrainingExample

GENERAL = [
    "What is the capital of Germany?", "What is the capital of Spain?", "What is the capital of Brazil?",
    "What is the capital of Kenya?", "What is the chemical symbol for iron?", "What is the chemical symbol for oxygen?",
    "How many sides does a hexagon have?", "How many minutes are in an hour?", "Who wrote '1984'?",
    "Who discovered penicillin?", "What is the tallest mountain on Earth?", "What is the largest desert in the world?",
    "What is photosynthesis, in one sentence?", "Explain what DNS does in one sentence.",
    "What is the difference between RAM and storage?", "What is an API?", "What does TCP stand for?",
    "What does RAM stand for?", "What is recursion in programming?", "What is a hash table?",
    "What is Big-O notation used for?", "Explain the difference between a list and a tuple in Python.",
    "What does the `git commit` command do?", "What is a pull request?", "What is unit testing?",
    "What is the speed of light in a vacuum, approximately?", "What causes the seasons on Earth?",
    "What is the boiling point of water in Fahrenheit?", "What is 12 squared?", "What is 7 times 8?",
    "Translate 'thank you' into Spanish.", "What is the plural of 'mouse'?",
    "Give a synonym for 'quick'.", "What is the opposite of 'ancient'?",
    "Summarize what a CPU cache is in one sentence.", "What is a virtual environment in Python for?",
    "What is the purpose of a .gitignore file?", "What is JSON used for?",
    "What is the difference between HTTP GET and POST?", "What is an operating system?",
]

CODING = [
    "Write a Python function `square(n)` that returns n squared.",
    "Write a Python function `is_odd(n)` that returns True if n is odd.",
    "Write a Python function `last_item(xs)` that returns the last element of a list.",
    "Write a Python function `count_words(s)` that returns the number of words in a string.",
    "Write a Python function `to_upper(s)` that returns s in uppercase.",
    "Write a Python function `sum_list(xs)` that returns the sum of a list of numbers.",
    "Write a Python function `min_max(xs)` that returns a tuple (smallest, largest).",
    "Write a Python function `remove_duplicates(xs)` that keeps the first occurrence of each item, in order.",
    "Write a Python function `power(base, exp)` that computes base**exp with a loop.",
    "Write a Python function `average(xs)` that returns the mean of a non-empty list.",
    "Write a Python function `starts_with_vowel(s)` that returns True if s starts with a vowel.",
    "Write a Python function `invert_dict(d)` that swaps keys and values.",
    "Write a Python function `lcm(a, b)` that returns the least common multiple.",
    "Write a Python function `count_char(s, c)` that counts occurrences of character c in s.",
    "Write a Python function `reverse_words(s)` that reverses the order of words in a sentence.",
    "Write a Python function `is_anagram(a, b)` that checks whether two strings are anagrams.",
    "Write a Python function `binary_search(xs, target)` that returns the index of target in a sorted list, or -1.",
    "Write a Python function `transpose(m)` that transposes a matrix given as a list of lists.",
    "Write a Python function `read_lines(path)` that returns the lines of a text file without newline characters.",
    "Write a Python function `safe_divide(a, b)` that returns None when b is zero.",
    "Write a Python class `Stack` with push, pop and is_empty methods.",
    "Write a Python function `parse_csv_line(line)` that splits a comma-separated line and strips spaces.",
    "Write a Python function `title_case(s)` that capitalizes each word.",
    "Write a Python function `running_total(xs)` that returns the cumulative sums.",
    "Write a Python function `filter_even(xs)` that returns only the even numbers.",
    "Write a JavaScript function `add(a, b)` that returns the sum.",
    "Write a SQL query that selects all rows from a table `users` where `age` is over 30.",
    "Write a bash command that counts the lines in file.txt.",
    "Explain what this Python code does: `[x * 2 for x in range(5)]`.",
    "Fix the bug: `def is_positive(n): return n > 0 if n else True`",
]


def build_rehearsal(home: Path, base_model: str, skills: list[str] | None = None) -> list[TrainingExample]:
    """The base model's own answers to every enabled skill pack's rehearsal prompts (made once,
    cached in <home>/rehearsal-<packs>.json). Each example records its pack in `memory_id`
    ("skill:<name>") so replay can sample the packs evenly.

    `skills=None` keeps the original general + coding set (cached as rehearsal.json), which the
    forgetting-lab results were measured with."""
    from .skills import get
    if skills is None:
        path, packs = Path(home) / "rehearsal.json", None
    else:
        path, packs = Path(home) / f"rehearsal-{'-'.join(skills)}.json", get(skills)
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        from .models.hf import HFStudent
        s = HFStudent(base_model)
        rows = []
        if packs is None:
            gen = s.answer_many(GENERAL, max_new_tokens=96)
            code = s.answer_many(CODING, max_new_tokens=384)
            rows = [{"q": q, "a": a.text, "skill": k} for k, qs, ans in (("general", GENERAL, gen), ("coding", CODING, code))
                    for q, a in zip(qs, ans)]
        else:
            for pack in packs:
                ans = s.answer_many(pack.rehearsal, max_new_tokens=pack.max_new_tokens)
                rows += [{"q": q, "a": a.text, "skill": pack.name} for q, a in zip(pack.rehearsal, ans)]
        rows = [r for r in rows if r["a"].strip()]
        path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return [TrainingExample(r["q"], r["a"], 1.0, "base", f"skill:{r.get('skill', '')}") for r in rows]


def sample(base_replay: list[TrainingExample], k: int, rng, weights: dict | None = None) -> list[TrainingExample]:
    """k rehearsal examples split across skill packs by `weights` (pack name -> share; see
    skills.usage_weights), evenly when no weights are given. Examples with no pack (the toy
    brain's base knowledge) are drawn exactly as before: uniformly."""
    k = min(k, len(base_replay))
    groups: dict[str, list[int]] = {}
    for i, e in enumerate(base_replay):
        if e.memory_id.startswith("skill:"):
            groups.setdefault(e.memory_id[6:], []).append(i)
    if not groups:
        return [base_replay[i] for i in rng.choice(len(base_replay), size=k, replace=False)]
    names = sorted(groups)
    w = {g: max(0.0, float((weights or {}).get(g, 1.0 if not weights else 0.0))) for g in names}
    if not sum(w.values()):
        w = {g: 1.0 for g in names}
    tot = sum(w.values())
    want = {g: k * w[g] / tot for g in names}
    take = {g: min(len(groups[g]), int(want[g])) for g in names}
    for g in sorted(names, key=lambda g: want[g] - int(want[g]), reverse=True):
        if sum(take.values()) < k and take[g] < len(groups[g]):
            take[g] += 1
    while sum(take.values()) < k:              # a pack ran out: give the rest to the others
        g = max((g for g in names if take[g] < len(groups[g])), key=lambda g: want[g] - take[g])
        take[g] += 1
    out = []
    for g in names:
        out += [base_replay[int(i)] for i in rng.permutation(groups[g])[:take[g]]]
    return out


def rehearsal_weights(system, day: int) -> dict | None:
    """Usage-based pack shares for this sleep (None for the toy brain, which has no packs)."""
    names = getattr(system.config, "skills", None)
    if not names or not any(e.memory_id.startswith("skill:") for e in system.base_replay):
        return None
    from .skills import usage_weights
    w = usage_weights(system.store, day, list(names))
    system.rehearsal_mix = w                     # shown in the dream report
    return w
