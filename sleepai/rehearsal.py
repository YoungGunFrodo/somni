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

# More general rehearsal for the skill pack (v3 SQuAD finding: 40 items repeated ~23x in a big
# night did not hold general knowledge: 92.5 -> 75). Disjoint from evalsets' gate probes.
GENERAL_MORE = [
    'What is the capital of Mexico?',
    'What is the capital of India?',
    'What is the capital of Argentina?',
    'What is the capital of Norway?',
    'What is the capital of South Korea?',
    'What is the capital of Turkey?',
    'What is the capital of Portugal?',
    'What is the capital of Thailand?',
    'What is the chemical symbol for silver?',
    'What is the chemical symbol for potassium?',
    'What is the chemical symbol for carbon?',
    'What is the chemical formula for water?',
    'How many legs does an insect have?',
    'How many hours are in a week?',
    'How many days are in a non-leap year?',
    'How many planets are in the solar system?',
    'What is the closest star to Earth?',
    'What is the smallest planet in the solar system?',
    'Which planet has the most prominent rings?',
    'What is the largest mammal?',
    'What is the fastest land animal?',
    'What is the largest organ of the human body?',
    'How many bones are in the adult human body?',
    'What organ pumps blood through the body?',
    'What vitamin does the body make from sunlight?',
    "What is the main gas in Earth's atmosphere?",
    'What gas do humans exhale?',
    'What is the chemical symbol for helium?',
    "Who wrote 'Pride and Prejudice'?",
    "Who wrote 'The Odyssey'?",
    "Who painted 'The Starry Night'?",
    "Who composed the 'Moonlight Sonata'?",
    'Who was the first person to walk on the Moon?',
    'Who invented the telephone?',
    'Who proposed the theory of evolution by natural selection?',
    'Who formulated the laws of motion and universal gravitation?',
    'In what year did World War II end?',
    'What ancient civilization built the pyramids of Giza?',
    'What is the largest country by area?',
    'What is the most populous country in the world?',
    'What is the longest river in Africa?',
    'What is the highest waterfall in the world?',
    'What is the largest lake in Africa?',
    'On which continent is the Sahara Desert?',
    'What is the deepest ocean trench?',
    'What is the currency of Japan?',
    'What is the currency of the United Kingdom?',
    'What language is mainly spoken in Brazil?',
    'How many strings does a standard guitar have?',
    'How many players are on a soccer team on the field?',
    'What is the square root of 81?',
    'What is 9 multiplied by 7?',
    'What is 100 divided by 4?',
    'What is 2 to the power of 10?',
    'What is the sum of the angles in a triangle?',
    'What is the value of pi to two decimal places?',
    'What is 25% of 80?',
    'What is a prime number?',
    'What is the binary representation of the decimal number 3?',
    'How many bytes are in a kilobyte, in the binary convention?',
    'What does GPU stand for?',
    'What does URL stand for?',
    'What does HTML stand for?',
    'What does CSS stand for?',
    'What does USB stand for?',
    'What does AI stand for?',
    'What does PDF stand for?',
    'What does SSD stand for?',
    'What is the default port for HTTP?',
    'What is the default port for SSH?',
    'What does DNS stand for?',
    'What is an IP address?',
    'What is the keyword for a loop over items in Python?',
    'What keyword defines a class in Python?',
    'Which Python keyword returns a value from a function?',
    'What does the `len()` function do in Python?',
    'Which Git command saves staged changes to the repository?',
    'Which Git command downloads and merges changes from a remote?',
    'Which Git command shows the state of the working tree?',
    'What data structure stores key-value pairs?',
    'What is a linked list?',
    'What sorting algorithm repeatedly swaps adjacent elements?',
    'What is the time complexity of binary search?',
    'What does an operating system kernel do?',
    'Which language runs natively in web browsers?',
    'Which HTML tag creates a paragraph?',
    'Which HTML tag inserts an image?',
    'What is open-source software?',
    'What is a compiler?',
    'What is the difference between a compiler and an interpreter?',
    'What is encryption?',
    'What is a firewall?',
    'What is machine learning, in one sentence?',
    'What is the cloud, in computing?',
    'Why is the sky blue?',
    'Why do we have day and night?',
    'What causes tides?',
    'What is gravity?',
    'What is an atom made of?',
    'What is the difference between weather and climate?',
    'What is a volcano?',
    'What causes earthquakes?',
    'What is evaporation?',
    'How do vaccines work, briefly?',
    'What is the difference between a virus and a bacterium?',
    'What is inflation, in economics?',
    'What is a democracy?',
    'What is the purpose of a central bank?',
    'What does a thermometer measure?',
    'What does a barometer measure?',
    'What is the freezing point of water in Celsius?',
    'How many centimeters are in a meter?',
    'How many grams are in a kilogram?',
    'How many feet are in a mile?',
    'How many ounces are in a pound?',
    'How many seconds are in an hour?',
    'How many weeks are in a year?',
    "What is the past tense of 'run'?",
    "What is the plural of 'child'?",
    'What is a noun?',
    'What is a verb?',
    'What is an adjective?',
    "Give an antonym of 'generous'.",
    "What is the past tense of 'write'?",
    "Translate 'good morning' into Spanish.",
    "Translate 'cat' into French.",
    'What is the main ingredient of guacamole?',
    'What is tofu made from?',
    'At what temperature in Celsius is water densest, approximately?',
    'How many teaspoons are in a tablespoon?',
    'Which instrument has 88 keys?',
    'What is the largest bone in the human body?',
    'How many chambers does the human heart have?',
    'What do bees make?',
    'What is the largest bird?',
    'What is a group of lions called?',
    'Explain what a spreadsheet is used for.',
    'What is the difference between a debit card and a credit card?',
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
        packs = get(skills)
        # The cache name follows the prompts, so a pack that grows is rebuilt, not silently reused.
        import zlib
        sig = zlib.crc32(json.dumps([[p.name, p.rehearsal, p.max_new_tokens] for p in packs],
                                    sort_keys=True, default=str).encode()) & 0xFFFFFF
        path = Path(home) / f"rehearsal-{'-'.join(skills)}-{sig:06x}.json"
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
            from .reading import prompt_parts
            for pack in packs:
                items = [prompt_parts(r) if isinstance(r, dict) else (r, None) for r in pack.rehearsal]
                ans = s.answer_many([q for q, _ in items], [c for _, c in items],
                                    max_new_tokens=pack.max_new_tokens)
                rows += [{"q": q, "a": a.text, "skill": pack.name,
                          "ctx": [{"key": m.key, "summary": m.summary, "id": m.id} for m in c or []]}
                         for (q, c), a in zip(items, ans)]
        rows = [r for r in rows if r["a"].strip()]
        path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return [TrainingExample(r["q"], r["a"], 1.0, "base", f"skill:{r.get('skill', '')}",
                            context=context_from(r.get("ctx"))) for r in rows]


def context_from(rows):
    """Stored context (key/summary/id dicts) back into memory-file objects, or None."""
    if not rows:
        return None
    from .memory.store import Memory
    return [Memory(key=c["key"], type="file", summary=c["summary"], strength="verified", status="active",
                   id=c.get("id", c["key"])) for c in rows]


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
