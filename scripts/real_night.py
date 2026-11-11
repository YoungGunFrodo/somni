"""One real night end to end on the real backend (no servers needed):
wake (tell facts) -> sleep (memory formation, consolidation, dreaming, gates) -> dream report
-> approve -> ask again WITHOUT memory retrieval.

    .venv-gpu\\Scripts\\python.exe scripts\\real_night.py --model Qwen/Qwen3-1.7B --facts 20
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from milestone0 import PROBES, invented_facts  # noqa: E402
from sleepai.config import Config, SleepConfig  # noqa: E402
from sleepai.models.base import answers_match  # noqa: E402
from sleepai.review.morning import approve, pending  # noqa: E402
from sleepai.sleep.pipeline import run_sleep  # noqa: E402
from sleepai.system import build_real_system  # noqa: E402


def start_memory_log(path: Path, every_s: int = 15) -> None:
    """Background thread: free RAM, this process, its children (the teacher's llama-server) and
    the biggest other programs, every 15 s. A real night was stopped by Claude Code's
    low-memory safeguard while this process itself used only 1.5 GB; this shows who used it."""
    import threading
    import psutil

    def run():
        me = psutil.Process()
        with open(path, "w", encoding="utf-8") as fh:
            while True:
                try:
                    vm = psutil.virtual_memory()
                    kids = sum(c.memory_info().rss for c in me.children(recursive=True)) / 2**30
                    top = sorted(((p.info["memory_info"].rss, p.info["name"]) for p in
                                  psutil.process_iter(["name", "memory_info"]) if p.info["memory_info"]),
                                 reverse=True)[:4]
                    fh.write(f"{time.strftime('%H:%M:%S')} free {vm.available / 2**30:.1f} GB  self "
                             f"{me.memory_info().rss / 2**30:.1f} GB  children {kids:.1f} GB  top: "
                             + ", ".join(f"{n} {r / 2**30:.1f}" for r, n in top) + "\n")
                    fh.flush()
                except Exception as e:                      # never let logging break the night
                    fh.write(f"log error {e}\n")
                time.sleep(every_s)

    threading.Thread(target=run, daemon=True).start()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--facts", type=int, default=20)
    p.add_argument("--home", default=str(ROOT / "models" / "real-night"))
    p.add_argument("--trainer", default="experts", choices=["experts", "peft"])
    p.add_argument("--teacher-url", default="")
    p.add_argument("--teacher-model", default="")
    p.add_argument("--server-teacher", action="store_true",
                   help="serve the teacher (the base model) with llama-server: much faster generation")
    args = p.parse_args()
    from sleepai.resources import warn_if_restart_pending
    warn_if_restart_pending()
    home = Path(args.home)
    shutil.rmtree(home, ignore_errors=True)
    server = None
    start_memory_log(ROOT / "models" / "real-night-memory.log")
    if args.server_teacher and not args.teacher_url:
        # On-demand teacher: started for teacher phases, stopped while the student uses the GPU.
        from squad_benchmark import ensure_gguf
        from sleepai.models.teacher_server import TeacherServer
        server = TeacherServer(ensure_gguf(args.model))
        args.teacher_url = server.url
    try:
        return night(args, home, server)
    finally:
        if server:
            server.stop()


def night(args, home, server=None) -> int:
    cfg = Config(home=str(home), backend="real", trainer=args.trainer, base_model=args.model,
                 teacher_url=args.teacher_url, teacher_model=args.teacher_model,
                 sleep=SleepConfig(epochs=4, rebuild_epochs=4, paraphrases_per_memory=8, dream_rounds=1,
                                   second_take=True))
    system = build_real_system(cfg)
    if server is not None:
        system.teacher.server = server
    facts = invented_facts(args.facts, seed=7)
    t0 = time.time()
    orch = system.orchestrator()
    for f in facts:                                  # wake: the user tells facts
        orch.tell(f"Note for later. The {f['attr']} of {f['entity']} is {f['value']}.", day=0)
    print(f"[night] told {len(facts)} facts in {time.time() - t0:.0f} s")

    t1 = time.time()
    from sleepai.resources import ResumableStop
    for attempt in range(4):           # a sleep stopped to protect the PC resumes where it stopped
        try:
            result = run_sleep(system, day=0)
            break
        except ResumableStop as e:
            print(f"[night] sleep paused ({e}); resuming", flush=True)
    else:
        print("[night] sleep could not finish after 4 tries", flush=True)
        return 3
    f = result["formation"]
    print(f"[night] sleep {time.time() - t1:.0f} s: formed {len(f['created'])} memories from "
          f"{f['episodes']} episodes; status {result['status']}; recommendation {result['recommendation']}")
    for g in result["evaluation"]["gates"]:
        print(f"   gate {g['name']:22s} {'PASS' if g['passed'] else 'FAIL'}  current={g['current']}  "
              f"candidate={g['candidate']:.3f}  n={g['n']}  {g['note']}")
    print(f"[night] dream report: {result['report']}")
    if pending(system) and result["evaluation"]["passed"]:
        approve(system, note="real_night.py")
    cand = result.get("candidate")
    if cand and system.registry.active != cand:     # rejected: still show what it learned
        cs = system.load_student(cand)
        qs = [(fa, t.format(a=fa["attr"], e=fa["entity"])) for fa in facts for t in PROBES]
        ans = cs.answer_many([q for _, q in qs])
        rc = sum(answers_match(a.text, fa["value"]) for (fa, _), a in zip(qs, ans)) / len(qs)
        print(f"[night] candidate {cand} (not adopted): recall WITHOUT retrieval on held-out phrasings "
              f"{100 * rc:.1f}%")
    student = system.load_student(system.registry.active)
    qs = [(fa, t.format(a=fa["attr"], e=fa["entity"])) for fa in facts for t in PROBES]
    answers = student.answer_many([q for _, q in qs])
    hits = [answers_match(a.text, fa["value"]) for (fa, _), a in zip(qs, answers)]
    recall = sum(hits) / len(qs)
    # The sleep budget (max_new_per_sleep) leaves some memories for the next night: report recall
    # on the memories this checkpoint was trained on separately (night 4: 80% overall was 16/20
    # trained memories at 100% + 4 waiting in the backlog).
    from sleepai.models.llm import slug
    trained_keys = {m.key for m in system.store.all() if m.consolidated_in}
    is_trained = [f"{slug(fa['entity'])}/{slug(fa['attr'])}" in trained_keys for (fa, _) in qs]
    t_hits = [h for h, t in zip(hits, is_trained) if t]
    t_recall = sum(t_hits) / len(t_hits) if t_hits else 0.0
    print(f"[night] active checkpoint {system.registry.active}: recall WITHOUT retrieval on held-out "
          f"phrasings {100 * recall:.1f}% of all {len(facts)} facts; {100 * t_recall:.1f}% of the "
          f"{sum(is_trained) // len(PROBES)} memories trained tonight (the rest wait for the next sleep)")
    from sleepai import gpu
    print(f"[night] GPU switches between teacher and student: {len(gpu.switches)}", flush=True)
    (home / "night-summary.json").write_text(json.dumps({
        "model": args.model, "facts": len(facts), "memories": len(f["created"]),
        "gates": result["evaluation"]["gates"], "adopted": system.registry.active != system.registry.base,
        "recall_without_retrieval": recall, "recall_trained_memories": t_recall, "sleep_seconds": round(time.time() - t1)}, indent=2, default=str),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
