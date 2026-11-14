"""Mechanics check for WISE-style expert routing on a tiny random model (CPU, ~1 min; needs the
Qwen3-0.6B tokenizer cached). Trains one routed expert on vault facts vs arithmetic rehearsal,
then checks: a threshold is saved, memory prompts route on, rehearsal prompts route off, and an
off question gives exactly the base model's output.
Usage: .venv\\Scripts\\python.exe scripts\\check_route.py <empty temp dir>"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from sleepai.models import experts as X
from sleepai.models.base import TrainingExample
from sleepai.models.hf import HFStudent, _load, clear_cache

root = Path(sys.argv[1])
S = root / "tiny"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True)
torch.manual_seed(0)
Qwen3ForCausalLM(cfg).save_pretrained(S)
tok.save_pretrained(S)
facts = [TrainingExample(f"What is the code of vault {i}?", f"{1000 + i}-ka") for i in range(8)]
stay = [TrainingExample(f"What is {i} plus {i}?", str(2 * i), 1.0, "base") for i in range(8)]
tr = X.ExpertTrainer(str(S), str(root / "out"), rank=4, lr=1e-2, batch_size=8, gated=False, route=True)
info = tr.train(None, (facts + stay) * 4, epochs=4, from_base=True, out_name="e1")
print("route", info["route"])
meta = json.loads((Path(info["adapter"]) / "experts.json").read_text())
assert len(meta["route"]) == 1 and meta["route"][0] is not None, meta["route"]
assert info["route"]["train_memory_on"] >= 0.9 and info["route"]["train_rehearsal_on"] <= 0.1, info["route"]

base = HFStudent(str(S), max_new_tokens=6)
ours = HFStudent(str(S), max_new_tokens=6)
ours.adapter_dir = info["adapter"]
qs_new = ["What is the code of vault 3?", "What is the code of vault 6?"]
qs_old = ["What is 2 plus 2?", "What is 5 plus 5?"]
b = [a.text for a in base.answer_many(qs_old + qs_new)]
o = [a.text for a in ours.answer_many(qs_old + qs_new)]
print("routed on", ours.route_stats, "base", b, "ours", o)
assert ours.route_stats == (2, 4), ours.route_stats
assert o[:2] == b[:2], "routed-off questions must give exactly the base model's answer"
ectx = _load(str(S), info["adapter"])[0]._expert_ctx
assert ectx.get("route") is None and ectx.get("scores") is None
clear_cache()
print("route check ok")
