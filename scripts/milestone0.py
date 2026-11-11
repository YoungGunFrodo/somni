"""Milestone 0: prove this machine can teach a model a fact through a LoRA adapter.

    .venv-gpu\\Scripts\\python.exe scripts\\milestone0.py --model Qwen/Qwen3-0.6B

Steps (each prints its result; results also go to models/milestone0/<model>/result.json):
  1. Make 200 invented facts the base model cannot know (fixed seed), with 6 training
     paraphrases each and 3 HELD-OUT probe phrasings each (never trained on).
  2. Ask the base model the probes WITHOUT context  -> expect ~0% recall.
  3. Train a LoRA adapter (PeftTrainer) on the paraphrases; record speed and peak VRAM.
  4. Ask the base model + adapter the probes WITHOUT context -> the exit test.
  5. Optional (--gguf): convert the adapter to GGUF with llama.cpp's convert_lora_to_gguf.py,
     for loading into llama-server with --lora.

Exit test (design doc): an adapter trained locally teaches the model a fact it did not know.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sleepai.models.base import TrainingExample, answers_match  # noqa: E402

SYL = "ka lo mi ne ru ta vo zi pe sha dru fen gor hil jas kev lum nor pix quo".split()
THINGS = ["vault", "ship", "garden", "tower", "lab", "café", "station", "orchard", "bridge", "library"]
ATTRS = {
    "access code": lambda r: f"{r.randint(1000, 9999)}-{r.choice(SYL)}{r.choice(SYL)}",
    "keeper": lambda r: (r.choice(SYL) + r.choice(SYL) + r.choice(SYL)).capitalize(),
    "founding year": lambda r: str(r.randint(1703, 1899)),
    "color": lambda r: r.choice(["teal", "amber", "crimson", "indigo", "olive", "coral", "slate", "ochre"]),
}
TRAIN = ["What is the {a} of {e}?", "Tell me the {a} of {e}.", "{e}: what's its {a}?",
         "Do you know the {a} for {e}?", "Remind me of {e}'s {a}.", "Quick question, {e} {a}?"]
PROBES = ["What {a} is on record for {e}?", "Can you recall {e}'s {a}?", "For {e}, the {a} is what?"]


def invented_facts(n: int, seed: int = 0) -> list[dict]:
    r = random.Random(seed)
    facts, seen = [], set()
    while len(facts) < n:
        e = f"{r.choice(THINGS)} {r.choice(SYL)}{r.choice(SYL)}{r.choice(SYL)}"
        a = r.choice(list(ATTRS))
        if (e, a) in seen:
            continue
        seen.add((e, a))
        facts.append({"entity": e, "attr": a, "value": ATTRS[a](r)})
    return facts


def ask(model, tok, questions, device, max_new_tokens=24, batch=16) -> list[str]:
    import torch
    from sleepai.models.peft_trainer import chat_prompt
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for i in range(0, len(questions), batch):
        prompts = [chat_prompt(tok, q) for q in questions[i:i + batch]]   # same format as training
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        out += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return out


def recall(model, tok, facts, device) -> tuple[float, list]:
    qs = [(f, t.format(a=f["attr"], e=f["entity"])) for f in facts for t in PROBES]
    answers = ask(model, tok, [q for _, q in qs], device)
    hits = [answers_match(a, f["value"]) for (f, _), a in zip(qs, answers)]
    samples = [{"q": q, "expected": f["value"], "got": a.strip()[:80]} for (f, q), a in list(zip(qs, answers))[:6]]
    return sum(hits) / len(hits), samples


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--facts", type=int, default=200)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gguf", action="store_true", help="also convert the adapter with llama.cpp")
    p.add_argument("--rehearsal", action="store_true", help="mix in the base model's own answers")
    p.add_argument("--out-name", default="")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sleepai.models.peft_trainer import PeftTrainer, pick_device

    device, dtype = pick_device()
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "none (CPU)"
    print(f"[m0] torch {torch.__version__}, device {device}, GPU: {gpu}")
    out = ROOT / "models" / "milestone0" / (args.model.replace("/", "--") + args.out_name)
    out.mkdir(parents=True, exist_ok=True)

    facts = invented_facts(args.facts)
    examples = [TrainingExample(t.format(a=f["attr"], e=f["entity"]), f["value"])
                for f in facts for t in TRAIN]
    if args.rehearsal:
        from sleepai.rehearsal import build_rehearsal
        reh = build_rehearsal(out, args.model)
        examples += reh * max(1, len(examples) // (2 * len(reh)))   # ~1 rehearsal per 2 fact examples
        print(f"[m0] rehearsal: {len(reh)} base-model answers mixed in")
    print(f"[m0] {len(facts)} invented facts, {len(examples)} training examples, "
          f"{len(facts) * len(PROBES)} held-out probes")

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    before, s_before = recall(base, tok, facts, device)
    print(f"[m0] base model recall without context: {100 * before:.1f}%")
    del base
    if device == "cuda":
        torch.cuda.empty_cache()

    trainer = PeftTrainer(args.model, str(out), rank=args.rank, lr=args.lr, batch_size=args.batch_size)
    info = trainer.train(None, examples, epochs=args.epochs, from_base=True, out_name="adapter")
    print(f"[m0] trained: {info}")

    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(model, info["adapter"]).to(device)
    model.eval()
    after, s_after = recall(model, tok, facts, device)
    print(f"[m0] base + adapter recall without context (held-out phrasings): {100 * after:.1f}%")
    for s in s_after:
        print("     ", s)

    gguf = None
    if args.gguf:
        conv = ROOT / "tools" / "llama.cpp-src" / "convert_lora_to_gguf.py"
        if not conv.exists():
            print(f"[m0] skip GGUF: {conv} not found (see CLAUDE.md, Milestone 0)")
        else:
            from huggingface_hub import snapshot_download
            base_dir = snapshot_download(args.model)
            gguf = str(out / "adapter.gguf")
            subprocess.run([sys.executable, str(conv), info["adapter"], "--base", base_dir,
                            "--outfile", gguf], check=True)
            print(f"[m0] GGUF adapter: {gguf}")

    result = {"model": args.model, "device": device, "gpu": gpu, "torch": torch.__version__,
              "facts": len(facts), "epochs": args.epochs, "recall_before": before,
              "recall_after": after, "train": info, "gguf_adapter": gguf,
              "samples_before": s_before, "samples_after": s_after,
              "exit_test_passed": after >= 0.8 and before <= 0.05,
              "date": time.strftime("%Y-%m-%d %H:%M")}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[m0] exit test {'PASSED' if result['exit_test_passed'] else 'NOT passed'} "
          f"(needs >= 80% recall after, <= 5% before). Saved {out / 'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
