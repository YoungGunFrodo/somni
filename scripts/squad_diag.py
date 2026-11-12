"""Diagnose a SQuAD run: base vs trained answers with lookup, and general probes that broke."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from squad_benchmark import load_squad
from sleepai.models.base import answers_match
from sleepai.models.hf import HFStudent, clear_cache
from sleepai.memory.store import MemoryStore
from sleepai.wake.orchestrator import Orchestrator
from sleepai import evalsets

run, n = Path(sys.argv[1]), int(sys.argv[2])
qas = [qa for ps in load_squad(n) for qa in ps["qas"]]
store = MemoryStore(run / "home" / "memory.db")
orch = Orchestrator(store, None)
ctx = [orch.recall(qa["q"]) for qa in qas]
qs = [qa["q"] + " Answer briefly." for qa in qas]
gp = evalsets.general_probes()
res = {}
for name in ("base", "ours"):
    s = HFStudent("Qwen/Qwen3-1.7B", max_new_tokens=32)
    if name == "ours":
        s.adapter_dir = str(sorted((run / "home" / "adapters").glob("experts-*"))[-1])
    res[name] = [a.text for a in s.answer_many(qs, ctx)]
    res[name + "_g"] = [a.text for a in s.answer_many([p["question"] for p in gp])]
    clear_cache()
out = []
for i, qa in enumerate(qas):
    b = any(answers_match(res["base"][i], g) for g in qa["answers"])
    o = any(answers_match(res["ours"][i], g) for g in qa["answers"])
    if b != o:
        out.append({"q": qa["q"], "gold": qa["answers"][:2], "base": res["base"][i], "ours": res["ours"][i],
                    "base_ok": b, "ctx": [c.key for c in ctx[i]]})
g = [{"q": p["question"], "gold": p["answer"], "base": bg, "ours": og}
     for p, bg, og in zip(gp, res["base_g"], res["ours_g"])
     if answers_match(bg, p["answer"]) != answers_match(og, p["answer"])]
(run / "diag-lookup.json").write_text(json.dumps({"lookup": out, "general": g}, indent=1), encoding="utf-8")
print("base-only right", sum(x["base_ok"] for x in out), "ours-only right", sum(not x["base_ok"] for x in out))
