"""Multi-night forgetting test: does learning night 2 and 3 erase what night 1 learned?

Single-night lab runs showed rehearsal + combine practice keep general knowledge and coding at
base level (run F). What one night cannot show is whether LATER nights overwrite EARLIER ones.
Three ways to carry learning across nights, same data, same lean trainer (models/experts.py):

  lora         one adapter, retrained every night            (the classic approach)
  stack        a new expert every night, older ones frozen    (no gate)
  stack-gated  a new expert every night, older ones frozen, gated (Google-style experts)

Every night: the new facts (6 phrasings), replay of every older fact (1 phrasing, as the sleep
pipeline's replay coverage does), rehearsal (base model's own general + coding answers), and
combine practice for the new facts. Measured after every night, WITHOUT context, on held-out
phrasings: recall per night's facts, general, coding, combine (on facts never combine-trained).

    .venv-gpu\\Scripts\\python.exe -u scripts\\forgetting_nights.py --variant lora
Results append to models/forgetting-lab/nights.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from forgetting_lab import COMBINE_TEST, COMBINE_TRAIN  # noqa: E402
from milestone0 import PROBES, TRAIN, invented_facts  # noqa: E402
from sleepai import evalsets  # noqa: E402
from sleepai.models.base import TrainingExample, answers_match  # noqa: E402

HELDOUT_PER_NIGHT = 15        # per night, facts whose combine task is never trained


def combine_ex(f):
    return TrainingExample(COMBINE_TRAIN.format(a=f["attr"], e=f["entity"]),
                           f'```python\ndef get_value():\n    return "{f["value"]}"\n```', 1.0, "dream")


def measure(student, groups, heldout):
    from sleepai.sleep.evaluation import _score
    out = {}
    for k, g in enumerate(groups):
        qs = [(f, PROBES[0].format(a=f["attr"], e=f["entity"])) for f in g]
        ans = student.answer_many([q for _, q in qs])
        out[f"night{k + 1}_facts"] = sum(answers_match(a.text, f["value"]) for (f, _), a in zip(qs, ans)) / len(qs)
    out["general"] = float(_score(student, evalsets.general_probes()).mean())
    out["coding"] = float(_score(student, evalsets.coding_probes(), "python").mean())
    probes = [{"memory_id": f"combine:{i}", "question": COMBINE_TEST.format(a=f["attr"], e=f["entity"]),
               "answer": "", "kind": "python",
               "tests": f"assert str(lookup()).strip().lower() == {f['value'].lower()!r}"}
              for i, f in enumerate(heldout)]
    out["combine"] = float(_score(student, probes, "python").mean()) if probes else None
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["lora", "stack", "stack-gated"])
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--nights", type=int, default=3)
    p.add_argument("--facts-per-night", type=int, default=60)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rehearsal", type=float, default=0.5)
    args = p.parse_args()
    from sleepai.resources import warn_if_restart_pending
    warn_if_restart_pending()

    from sleepai.models.experts import ExpertTrainer
    from sleepai.models.hf import HFStudent, clear_cache
    from sleepai.rehearsal import build_rehearsal
    from sleepai.resources import ResumableStop
    out = ROOT / "models" / "forgetting-lab"
    out.mkdir(parents=True, exist_ok=True)
    facts = invented_facts(args.nights * args.facts_per_night, seed=11)
    groups = [facts[k * args.facts_per_night:(k + 1) * args.facts_per_night] for k in range(args.nights)]
    reh = build_rehearsal(out, args.model)                  # same rehearsal set as runs E/F
    tr = ExpertTrainer(args.model, str(out), lr=args.lr, gated=(args.variant == "stack-gated"))

    class Student:
        adapter_dir = None
    st = Student()
    heldout = []
    for k, g in enumerate(groups):
        night = k + 1
        name = f"nights-{args.variant}-n{night}"
        done = out / name / "experts.json"
        new = [TrainingExample(t.format(a=f["attr"], e=f["entity"]), f["value"]) for f in g for t in TRAIN]
        old = [TrainingExample(TRAIN[(i + night) % len(TRAIN)].format(a=f["attr"], e=f["entity"]), f["value"], 1.0,
                               "replay") for gg in groups[:k] for i, f in enumerate(gg)]
        heldout += g[-HELDOUT_PER_NIGHT:]
        comb = [combine_ex(f) for f in g[:-HELDOUT_PER_NIGHT]]
        n_reh = int(round(args.rehearsal * len(new)))
        reh_part = [reh[(i + k * n_reh) % len(reh)] for i in range(n_reh)]
        examples = new + old + comb + reh_part
        if args.variant != "lora":
            tr.begin_sleep()                                 # stack: a new expert tonight
        if done.exists():                                    # resumed script: this night is trained
            st.adapter_dir = str(out / name)
            tr._sleep_dirs.add(st.adapter_dir)
        else:
            t0 = time.time()
            try:
                info = tr.train(st, examples, epochs=args.epochs, from_base=(night == 1), out_name=name)
            except ResumableStop as e:
                print(f"[nights] {name}: paused, run again to continue ({e})", flush=True)
                return 3
            print(f"[nights] {name}: trained {len(examples)} examples in {info['seconds']:.0f} s", flush=True)
        clear_cache()
        s = HFStudent(args.model)
        s.adapter_dir = st.adapter_dir
        r = measure(s, groups[:night], heldout)
        clear_cache()
        n_exp = json.loads((Path(st.adapter_dir) / "experts.json").read_text(encoding="utf-8"))["experts"]
        row = {"variant": args.variant, "night": night, "experts": n_exp, "examples": len(examples), **r}
        with open(out / "nights.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        facts_txt = "  ".join(f"n{j + 1} {r[f'night{j + 1}_facts']:.0%}" for j in range(night))
        print(f"[nights] {args.variant} after night {night} ({n_exp} expert(s)): facts {facts_txt}  "
              f"general {r['general']:.1%}  coding {r['coding']:.1%}  combine {r['combine']:.1%}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
