"""Lookup with and without subject tags, no training: re-tags a copy of a benchmark run's memory
store with the current tagging (sleep/tagging.py), then the base model answers every SQuAD
question from recall with tags off and on. Also reports how often the answer text is in what
recall returned.
Usage: .venv-gpu\\Scripts\\python.exe scripts\\tag_lookup_eval.py <run dir> [passages]"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from squad_benchmark import ensure_gguf, load_squad  # noqa: E402

from sleepai.memory.store import MemoryStore  # noqa: E402
from sleepai.models.base import answers_match, normalize  # noqa: E402
from sleepai.models.hf import HFStudent, clear_cache  # noqa: E402
from sleepai.models.llm import LLMTeacher  # noqa: E402
from sleepai.models.teacher_server import TeacherServer  # noqa: E402
from sleepai.sleep.tagging import tag_untagged  # noqa: E402
from sleepai.wake.orchestrator import Orchestrator  # noqa: E402

run, n = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 30
model = "Qwen/Qwen3-1.7B"
qas = [qa for ps in load_squad(n) for qa in ps["qas"]]
db = run / "home" / "memory-tags-eval.db"
shutil.copy2(run / "home" / "memory.db", db)
store = MemoryStore(db)
for m in store.all("active"):
    store.set_tags(m.id, [])
srv = TeacherServer(ensure_gguf(model), port=8093, slots=8)
res = tag_untagged(store, LLMTeacher(srv.url, "", server=srv))
srv.stop()
print("[tags]", res, list(store.tag_counts().items())[:25], flush=True)
student = HFStudent(model, max_new_tokens=32)
out = {"tagging": res}
for use in (False, True):
    orch = Orchestrator(store, student)
    orch.use_tags = use
    ctx = [orch.recall(qa["q"]) for qa in qas]
    found = sum(any(normalize(g) in normalize(" ".join(m.summary + " " + str(m.value) for m in c))
                    for g in qa["answers"]) for qa, c in zip(qas, ctx))
    ans = student.answer_many([qa["q"] + " Answer briefly." for qa in qas], ctx)
    acc = sum(any(answers_match(a.text, g) for g in qa["answers"]) for qa, a in zip(qas, ans)) / len(qas)
    name = "tags" if use else "no_tags"
    out[name] = {"lookup": round(acc, 4), "answer_in_context": round(found / len(qas), 4),
                 "items": round(sum(len(c) for c in ctx) / len(ctx), 2)}
    print(f"[lookup] {name}: {out[name]}", flush=True)
clear_cache()
(run / "tag_lookup_eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
