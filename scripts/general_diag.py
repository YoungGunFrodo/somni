"""General-probe answers, base vs trained experts (quick: 40 questions)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sleepai import evalsets
from sleepai.models.base import answers_match
from sleepai.models.hf import HFStudent, clear_cache
run = Path(sys.argv[1])
gp = evalsets.general_probes()
s = HFStudent("Qwen/Qwen3-1.7B", max_new_tokens=48)
s.adapter_dir = str(sorted((run / "home" / "adapters").glob("experts-*"))[-1])
ans = s.answer_many([p["question"] for p in gp])
rows = [{"q": p["question"], "gold": p["answer"], "ours": a.text, "ok": answers_match(a.text, p["answer"])}
        for p, a in zip(gp, ans)]
(run / "diag-general.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
for r in rows:
    if not r["ok"]:
        print(r["q"], "| gold", r["gold"], "|", repr(r["ours"][:110]))
