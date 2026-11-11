"""Knowledge incorporation on SQuAD: the factual benchmark Google's sleep paper reports (it
follows SEAL's setup). Passages are learned, then questions are answered WITHOUT the passage.

    .venv-gpu\\Scripts\\python.exe scripts\\squad_benchmark.py --model Qwen/Qwen3-1.7B --passages 200

Published numbers (Qwen2.5-7B in SEAL's setup; 200 passages, 974 questions): base 31.9, plain
fine-tuning 32.0, SEAL 43.2, Google Sleep 46.2. Our model is smaller (16 GB GPU), so compare
GAINS over each one's own base model, and read docs/related-work.md before quoting anything.

Like SEAL and Google, the model teaches itself: the teacher is the student's own base model
(served by llama-server with parallel slots, so generation is fast). Passages go through our
real pipeline: each passage is a document episode -> memory formation -> rich training pairs
-> consolidation (stacked expert) -> optional dreaming (checked practice) -> evaluation without context.

Grading: a question is correct if any gold answer appears in the model's answer (normalized).
This is stricter than an LLM judge (SEAL), so it can only understate our score.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sleepai.config import Config, SleepConfig  # noqa: E402
from sleepai.models.base import answers_match  # noqa: E402

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
BIN = ROOT / "tools" / "llama.cpp-bin" / "llama-server.exe"
SRC = ROOT / "tools" / "llama.cpp-src"


def load_squad(n: int, seed: int = 0):
    path = ROOT / "models" / "data" / "squad-dev-v1.1.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SQUAD_URL, path)
    data = json.loads(path.read_text(encoding="utf-8"))["data"]
    paras = [(art["title"], p) for art in data for p in art["paragraphs"]]
    random.Random(seed).shuffle(paras)
    out = []
    for title, p in paras[:n]:
        qs = [{"q": qa["question"], "answers": list({a["text"] for a in qa["answers"]})} for qa in p["qas"]]
        out.append({"title": title, "context": p["context"], "qas": qs})
    return out


def ensure_gguf(model: str) -> Path:
    gguf = ROOT / "models" / "gguf" / (model.replace("/", "--") + "-f16.gguf")
    if not gguf.exists():
        from huggingface_hub import snapshot_download
        gguf.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(SRC / "convert_hf_to_gguf.py"), snapshot_download(model),
                        "--outtype", "f16", "--outfile", str(gguf)], check=True)
    return gguf


def start_server(model: str, port: int, slots: int):
    gguf = ensure_gguf(model)
    log = open(ROOT / "models" / "teacher-server.log", "w", encoding="utf-8")
    proc = subprocess.Popen([str(BIN), "-m", str(gguf), "--port", str(port), "-ngl", "99", "--jinja",
                             "-np", str(slots), "-c", str(4096 * slots)], stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    for _ in range(180):
        try:
            urllib.request.urlopen(url + "/health", timeout=2)
            return proc, url
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("teacher llama-server did not start (see models/teacher-server.log)")


def ask_all(student, qas, batch=16):
    answers = student.answer_many([qa["q"] + " Answer briefly." for qa in qas])
    hits = [any(answers_match(a.text, g) for g in qa["answers"]) for qa, a in zip(qas, answers)]
    return sum(hits) / len(hits), [{"q": qa["q"], "gold": qa["answers"][:2], "got": a.text[:80]}
                                   for qa, a in list(zip(qas, answers))[:8]]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--passages", type=int, default=200)
    p.add_argument("--pairs", type=int, default=8, help="training pairs per memory")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--dreaming", action="store_true")
    p.add_argument("--trainer", default="experts", choices=["experts", "peft"])
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--eval-only", action="store_true",
                   help="score the experts a previous run already trained (e.g. after a Windows restart)")
    args = p.parse_args()
    from sleepai.resources import warn_if_restart_pending
    warn_if_restart_pending()

    from sleepai.memory.store import Episode
    from sleepai.models.hf import HFStudent, clear_cache
    from sleepai.models.llm import LLMTeacher
    from sleepai.sleep.consolidation import _teacher_pairs, run_consolidation
    from sleepai.sleep.formation import run_formation
    from sleepai.system import build_real_system

    out = ROOT / "models" / "squad" / f"{args.model.replace('/', '--')}-{args.passages}"
    out.mkdir(parents=True, exist_ok=True)
    passages = load_squad(args.passages)
    qas = [qa for ps in passages for qa in ps["qas"]]
    print(f"[squad] {len(passages)} passages, {len(qas)} questions")
    report = {"model": args.model, "passages": len(passages), "questions": len(qas), "date": time.strftime("%Y-%m-%d")}

    if args.eval_only:
        return eval_only(args, out, qas, report)

    # 1. base model, no context
    base = HFStudent(args.model, max_new_tokens=32)
    report["base"], report["base_samples"] = ask_all(base, qas)
    print(f"[squad] base model, no context: {100 * report['base']:.1f}%")

    # 2. our pipeline: wake (documents) -> memory formation -> consolidation
    home = out / "home"
    import shutil
    shutil.rmtree(home, ignore_errors=True)
    cfg = Config(home=str(home), backend="real", trainer=args.trainer, base_model=args.model,
                 sleep=SleepConfig(max_new_per_sleep=100000, paraphrases_per_memory=args.pairs,
                                   epochs=args.epochs, replay=False, dreaming=args.dreaming, forgetting=False))
    system = build_real_system(cfg)
    clear_cache()
    # On-demand teacher (GPU turn-taking): up for teacher phases, stopped while the student works.
    from sleepai.models.teacher_server import TeacherServer
    srv = TeacherServer(ensure_gguf(args.model), port=args.port, slots=args.slots)
    url = srv.url
    try:
        teacher = LLMTeacher(url, "", server=srv)
        system.teacher = teacher
        for ps in passages:
            system.store.log_episode("tool_call", [{"role": "tool", "content":
                                     f"[document: {ps['title']}] {ps['context']}"}], outcome="success",
                                     evidence="reference document", day=0)
        eps = system.store.unformed_episodes()
        t0 = time.time()
        with ThreadPoolExecutor(args.slots) as pool:          # parallel teacher calls
            formed = dict(zip([e.id for e in eps], pool.map(teacher.form_memories, eps)))

        class Cached(LLMTeacher):
            def form_memories(self, episode: Episode):
                return formed.get(episode.id, [])
        system.teacher = Cached(url, "", server=srv)
        fres = run_formation(system.store, system.teacher, day=0)
        mems = system.store.eligible_for_consolidation()
        print(f"[squad] memory formation: {len(mems)} memories from {len(eps)} passages "
              f"({len(fres.rejected)} rejected by lint), {time.time() - t0:.0f} s")
        from collections import Counter
        why = Counter(pr.split(":")[0] for _, probs in fres.rejected for pr in probs)
        print(f"[squad] lint rejections by reason: {dict(why)}; e.g. {fres.rejected[:3]}")
        t0 = time.time()
        with ThreadPoolExecutor(args.slots) as pool:
            pairs = list(pool.map(lambda m: (m, teacher.training_pairs(m, args.pairs)), mems))
        cache = {f"{m.id}|{m.value}": pr for m, pr in pairs}
        (home / "training_pairs.json").write_text(json.dumps(cache), encoding="utf-8")
        system._pairs_cache = cache
        print(f"[squad] training pairs: {sum(len(v) for v in cache.values())} for {len(mems)} memories, "
              f"{time.time() - t0:.0f} s")
        report["memories"] = len(mems)
        report["training_pairs"] = sum(len(v) for v in cache.values())

        # retrieval path (wake with the memory store, no training): our baseline to beat
        orch_student = HFStudent(args.model, max_new_tokens=32)
        from sleepai.wake.orchestrator import Orchestrator
        orch = Orchestrator(system.store, orch_student)
        ctx = [orch.recall(qa["q"]) for qa in qas]
    finally:
        pass                                   # the server stops when the student takes the GPU

    answers = orch_student.answer_many([qa["q"] + " Answer briefly." for qa in qas], ctx)
    report["retrieval"] = sum(any(answers_match(a.text, g) for g in qa["answers"])
                              for qa, a in zip(qas, answers)) / len(qas)
    print(f"[squad] memory store + retrieval (no training): {100 * report['retrieval']:.1f}%")
    clear_cache()

    t0 = time.time()
    import numpy as np
    student, cres = run_consolidation(system, 0, np.random.default_rng(0), day=0)
    print(f"[squad] consolidation: {cres.new_examples} examples, {cres.train.get('seconds')} s, "
          f"peak VRAM {cres.train.get('peak_vram_gb')} GB")
    report["consolidation"] = {"examples": cres.new_examples, "train": cres.train}
    report["ours"], report["ours_samples"] = ask_all(student, qas)
    print(f"[squad] ours (sleep: formation + rich pairs + {args.trainer}), no context: {100 * report['ours']:.1f}%")
    report["published"] = {"model": "Qwen2.5-7B", "base": 0.319, "sft": 0.320, "seal": 0.432, "google_sleep": 0.462}
    report["gain_over_base"] = {"ours": report["ours"] - report["base"], "seal": 0.432 - 0.319,
                                "google_sleep": 0.462 - 0.319}
    srv.stop()
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[squad] gain over own base: ours {100 * report['gain_over_base']['ours']:+.1f} pts; "
          f"SEAL +11.3; Google Sleep +14.3 (Qwen2.5-7B). Saved {out / 'report.json'}")
    return 0


def eval_only(args, out, qas, report) -> int:
    """Re-score saved experts: base and ours (no context), plus retrieval with memory files."""
    from sleepai.models.hf import HFStudent, clear_cache
    from sleepai.memory.store import MemoryStore
    from sleepai.wake.orchestrator import Orchestrator
    home = out / "home"
    dirs = sorted((home / "adapters").glob("experts-*"))
    assert dirs, f"no saved experts under {home / 'adapters'}"
    base = HFStudent(args.model, max_new_tokens=32)
    store = MemoryStore(home / "memory.db")
    old = json.loads((out / "report.json").read_text(encoding="utf-8")) if (out / "report.json").exists() else {}
    for name in ("base", "base_samples", "retrieval", "retrieval_with_files"):   # reuse finished scores
        if name in old:
            report[name] = old[name]
    if "base" not in report:
        report["base"], report["base_samples"] = ask_all(base, qas)
    print(f"[squad] base model, no context: {100 * report['base']:.1f}%", flush=True)
    for files, name in ((0, "retrieval"), (2, "retrieval_with_files")):
        if name in report:
            print(f"[squad] {name.replace('_', ' ')} (no training): {100 * report[name]:.1f}% (saved)", flush=True)
            continue
        orch = Orchestrator(store, base)
        ctx = [orch.recall(qa["q"], files=files) for qa in qas]
        ans = base.answer_many([qa["q"] + " Answer briefly." for qa in qas], ctx)
        report[name] = sum(any(answers_match(a.text, g) for g in qa["answers"]) for qa, a in zip(qas, ans)) / len(qas)
        print(f"[squad] {name.replace('_', ' ')} (no training): {100 * report[name]:.1f}%", flush=True)
    clear_cache()
    student = HFStudent(args.model, max_new_tokens=32)
    student.adapter_dir = str(dirs[-1])        # the experts folder itself (no checkpoint/adapter wrapper)
    from sleepai.models.hf import _load
    n = sum("Expert" in type(m).__name__ for m in _load(args.model, student.adapter_dir)[0].modules())
    assert n > 0, "experts did not load"
    print(f"[squad] loaded experts into {n} layers from {dirs[-1].name}", flush=True)
    report["ours"], report["ours_samples"] = ask_all(student, qas)
    print(f"[squad] ours (trained experts), no context: {100 * report['ours']:.1f}%", flush=True)
    orch = Orchestrator(store, student)
    ctx = [orch.recall(qa["q"]) for qa in qas]
    ans = student.answer_many([qa["q"] + " Answer briefly." for qa in qas], ctx)
    report["ours_with_lookup"] = sum(any(answers_match(a.text, g) for g in qa["answers"])
                                     for qa, a in zip(qas, ans)) / len(qas)
    report["routed_to_reading"] = getattr(student, "last_routed", 0) / len(qas)
    print(f"[squad] ours + memory lookup (memories and files): {100 * report['ours_with_lookup']:.1f}% "
          f"({100 * report['routed_to_reading']:.0f}% read with experts off)", flush=True)
    report["published"] = {"model": "Qwen2.5-7B", "base": 0.319, "sft": 0.320, "seal": 0.432, "google_sleep": 0.462}
    report["gain_over_base"] = {"ours": report["ours"] - report["base"], "seal": 0.432 - 0.319,
                                "google_sleep": 0.462 - 0.319}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[squad] gain over own base: ours {100 * report['gain_over_base']['ours']:+.1f} pts; "
          f"SEAL +11.3; Google Sleep +14.3 (Qwen2.5-7B). Saved {out / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
