"""Reading skill pack data: answer a question from a passage you are given.

The whole framework relies on reading (memory lookup, memory files, documents the user pastes,
tool output), and the SQuAD run showed training can damage it: given the right passage, the
base model answered 82% and the trained model 45%. Rehearsal had no passages in it, so nothing
protected reading. This pack fixes that: rehearsal is the BASE model's own answers to reading
questions, and a held-out gate (`skill:reading`) rejects a night that reads worse.

Passages are generated from templates about INVENTED towns, companies, species, events and
people, so a question can only be answered by reading, never from the model's own knowledge.
Each passage carries several facts of the same kind (distractors). Rehearsal and gate probes
come from different seeds, so the gate is never trained on its own passages.

Two presentations, both used at wake:
  "file"   the passage arrives as a retrieved memory file (system message, wake's format)
  "pasted" the passage is pasted into the user's message
"""
from __future__ import annotations

import random

_SYL = ["ka", "lo", "mi", "ren", "tor", "va", "quel", "dun", "bri", "sha", "pol", "zem", "ari", "nok",
        "fen", "ost", "ul", "gra", "vin", "thal", "ess", "mor", "pin", "ket", "dra", "lum", "sar", "bel"]


def _name(rng, parts=2):
    return "".join(rng.choice(_SYL) for _ in range(parts)).capitalize()


def _town(rng):
    n, r, y, m = _name(rng, 3), _name(rng), rng.randint(1620, 1890), rng.choice(
        ["copper", "salt", "timber", "wool", "glass", "tin"])
    pop, pop2 = rng.randint(3, 90) * 1000, rng.randint(3, 90) * 1000
    river, lake = _name(rng) + " River", "Lake " + _name(rng)
    text = (f"{n} is a town on the {river}, founded in {y} by settlers from {r}. Its early economy "
            f"rested on {m} mining, and by the census of {y + 140} it had {pop:,} residents, while the "
            f"neighbouring village of {_name(rng, 3)} had {pop2:,}. The town hall, built in {y + 61}, faces "
            f"{lake}. Every autumn {n} holds the {_name(rng)} Fair, which began in {y + 97}.")
    qs = [(f"In what year was {n} founded?", str(y)),
          (f"What was {n}'s early economy based on?", m),
          (f"How many residents did {n} have at the census of {y + 140}?", f"{pop:,}"),
          (f"What does the town hall of {n} face?", lake),
          (f"Where did the founders of {n} come from?", r)]
    return text, qs


def _company(rng):
    c, f, city = _name(rng) + " " + rng.choice(["Instruments", "Foods", "Logistics", "Textiles", "Optics"]), \
        _name(rng) + " " + _name(rng), _name(rng, 3)
    y, emp, prod = rng.randint(1950, 2015), rng.randint(2, 60) * 100, rng.choice(
        ["water filters", "bicycle frames", "tea blends", "sailcloth", "microscope lenses", "garden tools"])
    rival = _name(rng) + " Group"
    text = (f"{c} was founded in {y} by {f} in the city of {city}. It is best known for its "
            f"{prod}, though it also sold kitchen scales until {y + 9}. Today the company employs about "
            f"{emp:,} people, roughly twice as many as its main rival, {rival}. In {y + 21} it moved its "
            f"headquarters from {city} to {_name(rng, 3)}.")
    qs = [(f"Who founded {c}?", f), (f"What is {c} best known for?", prod),
          (f"About how many people does {c} employ?", f"{emp:,}"),
          (f"Who is the main rival of {c}?", rival), (f"In which city was {c} founded?", city)]
    return text, qs


def _species(rng):
    s = _name(rng) + " " + rng.choice(["finch", "frog", "beetle", "orchid", "lizard", "moth"])
    reg, food, length, eggs, finder = _name(rng, 3) + " Highlands", rng.choice(
        ["fig seeds", "ants", "nectar", "fallen fruit", "small snails", "lichen"]), rng.randint(2, 40), \
        rng.randint(2, 9), _name(rng) + " " + _name(rng)
    text = (f"The {s} lives only in the cloud forests of the {reg}. Adults reach about {length} cm and "
            f"feed mainly on {food}, although juveniles also eat seeds. Females lay {eggs} eggs per season. "
            f"The species was first described in {rng.randint(1890, 2010)} by {finder}, who mistook it "
            f"for a close relative from the lowlands.")
    qs = [(f"Where does the {s} live?", reg), (f"What do adult {s}s mainly eat?", food),
          (f"How many eggs do {s} females lay per season?", str(eggs)),
          (f"Who first described the {s}?", finder), (f"About how long are adult {s}s, in cm?", str(length))]
    return text, qs


def _event(rng):
    e = "the " + _name(rng) + " " + rng.choice(["Treaty", "Flood", "Marathon", "Expedition", "Strike"])
    y, lead, days, place = rng.randint(1700, 2005), _name(rng) + " " + _name(rng), rng.randint(3, 80), \
        _name(rng, 3)
    count, cause = rng.randint(2, 900), rng.choice(
        ["a dispute over grazing rights", "heavy spring rains", "a bet between two clubs",
         "a search for a lost survey party", "a cut in wages"])
    text = (f"{e[0].upper() + e[1:]} took place in {y} near {place}. It was started by {cause} and lasted "
            f"{days} days. {lead} led the main group of {count} people; a second, smaller group led by "
            f"{_name(rng)} {_name(rng)} joined later. Historians date its end to the arrival of "
            f"{rng.choice(['winter', 'a new governor', 'the railway', 'a relief ship'])}.")
    qs = [(f"In what year did {e} take place?", str(y)), (f"Who led the main group in {e}?", lead),
          (f"How many days did {e} last?", str(days)), (f"What started {e}?", cause),
          (f"How many people were in the main group of {e}?", str(count))]
    return text, qs


_KINDS = [_town, _company, _species, _event]


def items(seed: int, n: int, per_passage: int = 2) -> list[dict]:
    """n reading items: {"passage", "title", "question", "answer", "presentation"}."""
    rng, out, k = random.Random(seed), [], 0
    while len(out) < n:
        text, qs = _KINDS[k % len(_KINDS)](rng)
        title = f"notes-{seed}-{k}"
        for q, a in rng.sample(qs, per_passage):
            out.append({"passage": text, "title": title, "question": q, "answer": a,
                        "presentation": "file" if len(out) % 2 == 0 else "pasted"})
        k += 1
    return out[:n]


REHEARSAL_SEED, PROBE_SEED = 101, 9001


def prompt_parts(item: dict):
    """(question text, context list) exactly as wake presents it."""
    from .memory.store import Memory
    if item["presentation"] == "pasted":
        return f"Here is a document:\n\n{item['passage']}\n\n{item['question']}", None
    return item["question"], [Memory(key=f"file/{item['title']}", type="file", summary=item["passage"],
                                     strength="verified", status="active", id=f"file:{item['title']}")]


def rehearsal_items(n: int = 48) -> list[dict]:
    return items(REHEARSAL_SEED, n)


def probes(n: int = 40) -> list[dict]:
    out = []
    for i, it in enumerate(items(PROBE_SEED, n)):
        q, ctx = prompt_parts(it)
        out.append({"memory_id": f"reading:{i}", "question": q, "answer": it["answer"], "kind": "recall",
                    "context": [{"key": c.key, "summary": c.summary, "id": c.id} for c in ctx or []]})
    return out
