"""Forgetting lab: teach a real model 200 invented facts with one training recipe and measure
what it learned AND what it lost, in one run.

    .venv-gpu\\Scripts\\python.exe scripts\\forgetting_lab.py --name lr5e5 --lr 5e-5 --epochs 8 --rehearsal

Reports, all WITHOUT context:
  facts    recall of the 200 facts on held-out phrasings (600 probes)
  general  40 general-knowledge questions (sleepai/evalsets.py)
  coding   20 Python tasks run against unit tests
  combine  a new fact used inside code, on facts whose combine task was NEVER trained (the last
           50 facts): does the new knowledge work together with the old coding skill?
Results append to models/forgetting-lab/results.jsonl.
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

from milestone0 import PROBES, TRAIN, invented_facts  # noqa: E402
from sleepai import evalsets  # noqa: E402
from sleepai.models.base import TrainingExample, answers_match  # noqa: E402

TARGETS = {"all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
           "mlp": ["gate_proj", "up_proj", "down_proj"], "attn": ["q_proj", "k_proj", "v_proj", "o_proj"]}


COMBINE_TRAIN = "Write a Python function `get_value()` that returns the {a} of {e} as a string."
COMBINE_TEST = "Write a Python function `lookup()` that returns the {a} of {e}, as a string."
COMBINE_HELDOUT = 50                    # the last 50 facts: combine never trained on them


def combine_examples(facts):
    """Dream practice that ties a new fact into an old skill (coding)."""
    return [TrainingExample(COMBINE_TRAIN.format(a=f["attr"], e=f["entity"]),
                            f'```python\ndef get_value():\n    return "{f["value"]}"\n```', 1.0, "dream")
            for f in facts[:-COMBINE_HELDOUT]]


def combine_probes(facts):
    return [{"memory_id": f"combine:{i}", "question": COMBINE_TEST.format(a=f["attr"], e=f["entity"]),
             "answer": "", "kind": "python",
             "tests": f"assert str(lookup()).strip().lower() == {f['value'].lower()!r}"}
            for i, f in enumerate(facts[-COMBINE_HELDOUT:])]


def measure(student, facts):
    from sleepai.sleep.evaluation import _score
    qs = [(f, t.format(a=f["attr"], e=f["entity"])) for f in facts for t in PROBES]
    ans = student.answer_many([q for _, q in qs])
    fr = sum(answers_match(a.text, f["value"]) for (f, _), a in zip(qs, ans)) / len(qs)
    return {"facts": fr, "general": float(_score(student, evalsets.general_probes()).mean()),
            "coding": float(_score(student, evalsets.coding_probes(), "python").mean()),
            "combine": float(_score(student, combine_probes(facts), "python").mean())}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--facts", type=int, default=200)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--targets", default="all", choices=list(TARGETS))
    p.add_argument("--rehearsal", type=float, default=0.0, help="rehearsal examples per fact example")
    p.add_argument("--baseline", action="store_true", help="also measure the base model")
    p.add_argument("--method", default="lora", choices=["lora", "experts"],
                   help="lora: plain LoRA; experts: gated expert (sleepai/models/experts.py)")
    p.add_argument("--combine", action="store_true", help="add combine dream practice (fact inside code)")
    p.add_argument("--gate-penalty", type=float, default=1.0)
    p.add_argument("--no-gate", action="store_true", help="experts method with the gate pinned open (control)")
    args = p.parse_args()
    from sleepai.resources import warn_if_restart_pending
    warn_if_restart_pending()

    from sleepai.models.hf import HFStudent, clear_cache
    from sleepai.models.peft_trainer import PeftTrainer
    out = ROOT / "models" / "forgetting-lab"
    out.mkdir(parents=True, exist_ok=True)
    facts = invented_facts(args.facts)
    examples = [TrainingExample(t.format(a=f["attr"], e=f["entity"]), f["value"]) for f in facts for t in TRAIN]
    n_fact = len(examples)
    if args.combine:
        examples += combine_examples(facts)
    if args.rehearsal:
        from sleepai.rehearsal import build_rehearsal
        reh = build_rehearsal(out, args.model)
        k = int(round(args.rehearsal * n_fact))
        examples += [reh[i % len(reh)] for i in range(k)]
    if args.baseline:
        b = measure(HFStudent(args.model), facts)
        print(f"[lab] base: {b}", flush=True)
        clear_cache()
    t0 = time.time()
    if args.method == "experts":
        from sleepai.models.experts import ExpertTrainer
        tr = ExpertTrainer(args.model, str(out), rank=args.rank, lr=args.lr, targets=TARGETS[args.targets],
                           gate_penalty=args.gate_penalty, gated=not args.no_gate)
    else:
        tr = PeftTrainer(args.model, str(out), rank=args.rank, lr=args.lr, targets=TARGETS[args.targets])
    from sleepai.resources import ResumableStop
    try:
        info = tr.train(None, examples, epochs=args.epochs, from_base=True, out_name=f"adapter-{args.name}")
    except ResumableStop as e:
        print(f"[lab] {args.name}: paused, run again to continue ({e})", flush=True)
        return 3
    s = HFStudent(args.model)
    s.adapter_dir = info["adapter"]
    r = measure(s, facts)
    row = {"name": args.name, **{k: v for k, v in vars(args).items() if k not in ("name", "baseline")},
           "fact_examples": n_fact, "total_examples": len(examples), **r,
           "gate_new": info.get("gate_new"), "gate_stay_out": info.get("gate_stay_out"),
           "train_seconds": info["seconds"], "peak_vram_gb": info["peak_vram_gb"],
           "minutes": round((time.time() - t0) / 60, 1)}
    with open(out / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[lab] {args.name}: facts {r['facts']:.1%}  general {r['general']:.1%}  coding {r['coding']:.1%}  combine {r['combine']:.1%}  "
          f"({row['minutes']} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
