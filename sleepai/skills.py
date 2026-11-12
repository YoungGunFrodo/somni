"""Skill packs: what a real model must keep being able to do while it learns new memories.

Each pack brings
  rehearsal  prompts whose BASE-model answers are replayed every sleep (stay-out examples: they
             teach a gated expert to stay closed, and keep a plain adapter from drifting)
  probes     an optional held-out evaluation set with its own gate (never trained on)

No settings needed: every pack is protected for everyone (an unused skill must not silently
disappear), and how much each pack is REHEARSED follows what the user actually does. Each sleep
measures the recent episodes (code, calculations, writing requests, other) and splits rehearsal
half evenly across packs, half by that usage. A programmer's model rehearses more code, a
writer's more writing, and a new user's model rehearses everything evenly. The loss is a
per-example mean, so long answers (code) do not outweigh short ones.

Add your own pack with `register(SkillPack(...))` (e.g. medicine, law, a game, a language) and
give it a `detect` function so its usage can be measured.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

DEFAULT_SKILLS = ["general", "reasoning", "writing", "coding", "reading"]


@dataclass
class SkillPack:
    name: str
    rehearsal: list            # prompts (str), or reading items (dict: passage + question)
    max_new_tokens: int = 128                       # length of the base answers kept for rehearsal
    probes: Callable[[], list[dict]] | None = None  # held-out gate probes (None = no gate)
    probe_mode: str = "correct"                     # "correct" (contains answer) | "python" (unit tests)
    detect: Callable[[str], bool] | None = None     # does an episode's text use this skill?
    notes: str = ""


REASONING_REHEARSAL = [
    "A train leaves at 3 pm and travels for 2.5 hours. When does it arrive?",
    "If 4 pencils cost $1.20, how much do 10 pencils cost?",
    "Sam is older than Ana, and Ana is older than Leo. Who is the youngest?",
    "What comes next in the sequence 2, 6, 18, 54?",
    "A rectangle is 7 m long and 3 m wide. What is its area?",
    "If today is Wednesday, what day will it be in 10 days?",
    "A shirt costs $40 and is 25% off. What is the sale price?",
    "There are 3 red and 5 blue marbles in a bag. What fraction are red?",
    "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?",
    "Convert 2.5 hours into minutes.",
    "A recipe for 4 people needs 300 g of flour. How much for 6 people?",
    "Is 91 a prime number? Explain briefly.",
    "What is 15% of 260?",
    "Which is larger, 3/8 or 2/5?",
    "A car uses 6 liters per 100 km. How much fuel for 250 km?",
    "If x + 7 = 19, what is x?",
    "Two numbers add up to 20 and differ by 4. What are they?",
    "How many seconds are in a day?",
    "If a meeting starts at 10:45 and lasts 95 minutes, when does it end?",
    "A box has 12 eggs and 5 break. What percent are left?",
]

WRITING_REHEARSAL = [
    "Rewrite this sentence to be more formal: 'hey, can u send me the file asap?'",
    "Summarize in one sentence: Regular exercise improves sleep, mood and heart health, and even short walks help.",
    "Write a two-sentence thank-you note to a colleague who helped with a project.",
    "Give three tips for writing a clear email.",
    "Turn this into a bulleted list: eggs, milk, bread and coffee.",
    "Write a short friendly reminder that the team meeting moved to Friday.",
    "Fix the grammar: 'Their going to the store tomorrow and they is bringing snacks.'",
    "Describe a sunset in one vivid sentence.",
    "Write a polite message declining an invitation to a party.",
    "Explain what a metaphor is, with one example.",
    "Write a one-line product description for a reusable water bottle.",
    "Make this sentence shorter: 'Due to the fact that it was raining, we made the decision to stay inside.'",
    "Write a haiku about autumn.",
    "Translate into French: 'Where is the train station?'",
    "Write an opening line for a story set on a space station.",
    "Paraphrase: 'The results were not as good as we had hoped.'",
    "Write a short apology for replying late to a message.",
    "List the main parts of a persuasive essay.",
    "Write a headline for an article about a new city park.",
    "Give a synonym and an antonym for 'generous'.",
]

REASONING_PROBES = [   # held out (never in rehearsal); exact answers, checked by containment
    ("If 3 notebooks cost $4.50, how much do 7 notebooks cost? Answer with the number only.", "10.5"),
    ("What is 18% of 350? Answer with the number only.", "63"),
    ("If x - 9 = 23, what is x? Answer with the number only.", "32"),
    ("How many minutes are in 3.5 hours? Answer with the number only.", "210"),
    ("What comes next: 3, 9, 27, 81? Answer with the number only.", "243"),
    ("A square has a perimeter of 36 cm. What is its side length in cm? Number only.", "9"),
    ("If today is Monday, what day is it in 9 days? One word.", "wednesday"),
    ("Mia is taller than Jo, and Jo is taller than Kai. Who is the shortest? One word.", "kai"),
    ("What is 144 divided by 12? Number only.", "12"),
    ("A $80 jacket is 15% off. What is the sale price in dollars? Number only.", "68"),
    ("How many hours are in a week? Number only.", "168"),
    ("Two numbers add to 30 and differ by 6. What is the larger one? Number only.", "18"),
    ("What is 7 cubed? Number only.", "343"),
    ("A trip of 180 km takes 2 hours. What is the average speed in km/h? Number only.", "90"),
    ("Which is larger, 5/8 or 3/5? Answer with the fraction.", "5/8"),
    ("If 5 machines make 5 parts in 5 minutes, how many minutes do 100 machines need for 100 parts? Number only.", "5"),
    ("What is the next prime number after 31? Number only.", "37"),
    ("A class has 12 girls and 18 boys. What percent are girls? Number only.", "40"),
    ("How many days are in 6 weeks? Number only.", "42"),
    ("If a film starts at 7:40 pm and lasts 125 minutes, at what time does it end? Use h:mm pm.", "9:45"),
]


def _reasoning_probes() -> list[dict]:
    return [{"memory_id": f"reasoning:{i}", "question": q, "answer": a, "kind": "recall"}
            for i, (q, a) in enumerate(REASONING_PROBES)]


_CODE = re.compile(r"```|\bdef \w+\(|\bclass \w+|\bimport \w+|\bfunction\b|Traceback|\bconst \w+ =|"
                   r"\.(py|js|ts|java|cpp|rs|go)\b|\bSELECT\b.+\bFROM\b|\bgit \w+|\bnpm \w+", re.I)
_MATH = re.compile(r"\d+\s*[-+*/x%^]\s*\d+|\b(calculate|how many|how much|percent|average|probability|"
                   r"solve|equation)\b", re.I)
_READ = re.compile(r"\[document:|here is a (document|text|passage|article)|according to (the|this)|"
                   r"\b(read|summari[sz]e) (this|the following)\b", re.I)
_WRITE = re.compile(r"\b(rewrite|reword|summari[sz]e|draft|essay|letter|email|paragraph|proofread|"
                    r"translate|story|poem|tone|grammar)\b", re.I)


def _packs() -> dict[str, SkillPack]:
    from . import evalsets, reading, rehearsal
    return {
        "general": SkillPack("general", rehearsal.GENERAL + rehearsal.GENERAL_MORE, 96, evalsets.general_probes,
                             notes="gated by 'general' and 'drift-vs-base'"),
        "reasoning": SkillPack("reasoning", REASONING_REHEARSAL, 160, _reasoning_probes,
                               detect=lambda t: bool(_MATH.search(t))),
        "writing": SkillPack("writing", WRITING_REHEARSAL, 160, None, detect=lambda t: bool(_WRITE.search(t)),
                             notes="no automatic gate: free-form writing has no exact answer"),
        "coding": SkillPack("coding", rehearsal.CODING, 384, evalsets.coding_probes, "python",
                            detect=lambda t: bool(_CODE.search(t)),
                            notes="gated by 'coding': no previously passing task may fail"),
        # Reading (sleepai/reading.py): answer from a passage you are given. Memory lookup,
        # memory files and pasted documents all depend on it.
        "reading": SkillPack("reading", reading.rehearsal_items(), 96, reading.probes,
                             detect=lambda t: bool(_READ.search(t)) or len(t) > 1500,
                             notes="gated by 'skill:reading' (invented passages, never SQuAD)"),
    }


_EXTRA: dict[str, SkillPack] = {}


def register(pack: SkillPack) -> None:
    """Add your own skill pack (e.g. medicine, law, a game, a language)."""
    _EXTRA[pack.name] = pack


def get(names: list[str]) -> list[SkillPack]:
    packs = {**_packs(), **_EXTRA}
    unknown = [n for n in names if n not in packs]
    if unknown:
        raise ValueError(f"unknown skill pack(s) {unknown}; available: {sorted(packs)}")
    return [packs[n] for n in names]


def usage_weights(store, day: int, names: list[str], window_days: int = 30) -> dict[str, float]:
    """Rehearsal share per pack: half split evenly (every skill stays protected), half by how
    often the user's recent episodes used it. Episodes that match no other pack count as
    "general". No episodes yet -> even split."""
    packs = get(names)
    counts = {p.name: 0.0 for p in packs}
    texts = store.recent_texts(day - window_days) if store is not None else []
    for t in texts:
        hits = [p.name for p in packs if p.detect and p.detect(t)]
        if not hits and "general" in counts:
            hits = ["general"]
        for h in hits:
            counts[h] += 1.0 / len(hits)
    n, total = len(packs), sum(counts.values())
    if not total:
        return {k: 1.0 / n for k in counts}
    return {k: 0.5 / n + 0.5 * v / total for k, v in counts.items()}
