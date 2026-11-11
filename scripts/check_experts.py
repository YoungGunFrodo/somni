"""Mechanics check for gated experts on a tiny random model (CPU, ~1 min; needs the Qwen3-0.6B
tokenizer cached). Usage: .venv-gpu\Scripts\python.exe scripts\check_experts.py <empty temp dir>"""
import sys, torch, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from sleepai.models import experts as X
from sleepai.models.base import TrainingExample
S = Path(sys.argv[1]) / "tiny"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True)
torch.manual_seed(0)
m = Qwen3ForCausalLM(cfg); m.save_pretrained(S); tok.save_pretrained(S)
ids = tok("hello world", return_tensors="pt").input_ids
base_out = m(ids).logits
ctx = {}
w = X.wrap(m, X.DEFAULT_TARGETS, ctx); X.add_expert(w, 4, 8, 0.0)
assert len(w) == 14, len(w)
assert torch.allclose(m(ids).logits, base_out, atol=1e-5), "fresh expert must not change output"
facts = [TrainingExample(f"What is the code of vault {i}?", f"{1000+i}-ka") for i in range(8)]
stay = [TrainingExample(f"What is {i} plus {i}?", str(2*i), 1.0, "base") for i in range(8)]
tr = X.ExpertTrainer(str(S), str(Path(sys.argv[1]) / "out"), rank=4, lr=1e-2, batch_size=8, gate_penalty=1.0)
info = tr.train(None, (facts + stay) * 4, epochs=3, from_base=True, out_name="e1")
print("night1", {k: info[k] for k in ("loss", "gate_new", "gate_stay_out")})
assert info["gate_stay_out"] < info["gate_new"], "gate should be lower on stay-out examples"
# night 2 (a new sleep) stacks a new expert on the frozen first one
tr.begin_sleep()
class St: adapter_dir = info["adapter"]
st = St()
info2 = tr.train(st, facts[:2] * 4, epochs=1, out_name="e2")
import json
# same sleep: dreaming continues tonight's expert instead of stacking a third
n_before = json.loads((Path(info2["adapter"]) / "experts.json").read_text())["experts"]
st.adapter_dir = info2["adapter"]
info2b = tr.train(st, facts[:2] * 4, epochs=1, out_name="e2b")
assert json.loads((Path(info2b["adapter"]) / "experts.json").read_text())["experts"] == n_before
tr.begin_sleep()
assert json.loads((Path(info2["adapter"]) / "experts.json").read_text())["experts"] == 2
sd1 = torch.load(Path(info["adapter"]) / "experts.pt"); sd2 = torch.load(Path(info2["adapter"]) / "experts.pt")
k = next(iter(sd1))
assert torch.equal(sd1[k]["B"][0], sd2[k]["B"][0]), "old expert must stay frozen"
# the student loads it
from sleepai.models.hf import HFStudent
s = HFStudent(str(S)); s.adapter_dir = info2["adapter"]
print("answer:", s.answer_many(["What is the code of vault 1?"], max_new_tokens=4)[0].text[:40])
# resource governor: a tiny RAM limit stops training with progress saved; a rerun resumes
from sleepai.resources import ResumableStop
out = Path(sys.argv[1]) / "out"
try:
    X.ExpertTrainer(str(S), str(out), rank=4, lr=1e-2, batch_size=8, max_ram_gb=0.01).train(
        None, facts * 4, epochs=1, from_base=True, out_name="r1")
    raise AssertionError("expected a resumable stop")
except ResumableStop as e:
    print("stopped:", str(e)[:70])
assert (out / "r1" / "resume.pt").exists()
X.ExpertTrainer(str(S), str(out), rank=4, lr=1e-2, batch_size=8).train(
    None, facts * 4, epochs=1, from_base=True, out_name="r1")
assert not (out / "r1" / "resume.pt").exists() and (out / "r1" / "experts.pt").exists()
# micro-batches give the same gradients as a full batch (accumulation)
torch.manual_seed(0); a = X.ExpertTrainer(str(S), str(out), rank=4, lr=1e-2, batch_size=8, seed=5)
ia = a.train(None, facts + stay, epochs=1, from_base=True, out_name="m8")
import sleepai.resources as R
orig = R.ResourceGovernor.micro_batch_for_vram
R.ResourceGovernor.micro_batch_for_vram = staticmethod(lambda b: 2)
torch.manual_seed(0); b = X.ExpertTrainer(str(S), str(out), rank=4, lr=1e-2, batch_size=8, seed=5)
ib = b.train(None, facts + stay, epochs=1, from_base=True, out_name="m2")
R.ResourceGovernor.micro_batch_for_vram = orig
pa = torch.load(out / "m8" / "experts.pt"); pb = torch.load(out / "m2" / "experts.pt")
k = next(iter(pa))
diff = (pa[k]["B"][0] - pb[k]["B"][0]).abs().max().item()
print("micro-batch vs full-batch max weight diff:", diff)
assert diff < 1e-4, diff
print("OK")
