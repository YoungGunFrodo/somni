"""Real consolidation: LoRA training with Hugging Face PEFT.

Runs on the GPU when PyTorch sees one (ROCm or CUDA; on Windows the RX 6800 uses AMD's
TheRock wheels, see CLAUDE.md), otherwise on the CPU (slow but works for small models).

Install (Windows, RX 6800):
    pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ "torch[device-gfx1030]"
    pip install transformers peft accelerate

After training, convert the adapter for llama.cpp:
    python llama.cpp/convert_lora_to_gguf.py <adapter_dir> --base <base_model_dir>
"""
from __future__ import annotations

import random
import time
from pathlib import Path


def chat_prompt(tok, question: str, context=None) -> str:
    """The one prompt format used for training, wake and evaluation. With context, the retrieved
    memories go in a system message exactly as at wake, so reading them is a trained skill."""
    msgs = [{"role": "user", "content": question}]
    if context:
        from .llm import MEMORY_PREAMBLE
        lines = [f"- [{m.status}, {m.strength}] {m.key}: {m.summary}" for m in context]
        msgs.insert(0, {"role": "system", "content": MEMORY_PREAMBLE + "\n" + "\n".join(lines)})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def fit_context(tok, ex, max_len: int):
    """Prompt for an example, dropping retrieved items from the end until prompt + answer fit
    (truncation would cut the answer, which is the part that is trained)."""
    ctx = list(ex.context or [])
    n_ans = len(tok(ex.answer + tok.eos_token, add_special_tokens=False).input_ids)
    while True:
        prompt = chat_prompt(tok, ex.question, ctx)
        if not ctx or len(tok(prompt, add_special_tokens=False).input_ids) + n_ans <= max_len:
            return prompt
        ctx.pop()


def pick_device():
    import torch
    if torch.cuda.is_available():          # ROCm builds also report through torch.cuda
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


class PeftTrainer:
    def __init__(self, base_model: str, out_root: str, rank: int = 16, lr: float = 2e-4,
                 max_len: int = 256, batch_size: int = 8, seed: int = 0, targets: list | None = None):
        self.base_model, self.out_root = base_model, Path(out_root)
        self.targets = targets or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        self.rank, self.lr, self.max_len, self.batch_size = rank, lr, max_len, batch_size
        self.seed = seed
        self.runs = 0

    def train(self, student, examples, epochs: int = 2, from_base: bool = False,
              resume_adapter: str | None = None, out_name: str | None = None) -> dict:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device, dtype = pick_device()
        t0 = time.time()
        try:                                    # free the sleep-time inference models first
            from .hf import clear_cache
            clear_cache()
        except ImportError:
            pass
        from .. import gpu
        gpu.acquire("student")                  # the teacher's server (if any) leaves the GPU
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tok = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForCausalLM.from_pretrained(self.base_model, torch_dtype=dtype, low_cpu_mem_usage=True,
                                                     device_map=device)
        resume_adapter = resume_adapter or getattr(student, "adapter_dir", None)
        if resume_adapter and not from_base:
            model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                r=self.rank, lora_alpha=2 * self.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                target_modules=self.targets))
        if device == "cuda":
            # A normal out-of-memory error instead of spilling into shared memory (the Windows
            # AMD driver crashed the PC once when VRAM ran out).
            from ..resources import cap_vram
            cap_vram(0.7)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.lr)
        model.train()
        pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

        def encode(ex):
            # Same prompt format as wake and evaluation: thinking off (Qwen3 adds an empty
            # <think></think> block; templates without the option ignore it).
            prompt = fit_context(tok, ex, self.max_len)
            ids = tok(prompt + ex.answer + tok.eos_token, truncation=True, max_length=self.max_len,
                      add_special_tokens=False, return_tensors="pt").input_ids[0]
            n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
            labels = ids.clone()
            labels[:n_prompt] = -100            # train on the answer only
            return ids, labels, float(ex.weight)

        data = [encode(e) for e in examples]
        rng = random.Random(self.seed + self.runs)
        losses, tokens = [], 0
        for _ in range(epochs):
            rng.shuffle(data)                   # interleave new, replay and condensed examples
            for i in range(0, len(data), self.batch_size):
                chunk = data[i:i + self.batch_size]
                L = max(len(c[0]) for c in chunk)
                ids = torch.full((len(chunk), L), pad)
                lab = torch.full((len(chunk), L), -100)
                att = torch.zeros((len(chunk), L), dtype=torch.long)
                for j, (a, b, _) in enumerate(chunk):
                    ids[j, :len(a)], lab[j, :len(b)], att[j, :len(a)] = a, b, 1
                ids, lab, att = ids.to(device), lab.to(device), att.to(device)
                logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
                target = lab[:, 1:]
                tok_loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100,
                    reduction="none").view(target.shape)
                mask = (target != -100).float()
                per_ex = (tok_loss * mask).sum(1) / mask.sum(1).clamp(min=1)
                w = torch.tensor([c[2] for c in chunk], device=device)
                loss = (per_ex * w).sum() / w.sum()  # example weights: corrections 2x, condensed 0.5x
                loss.backward()
                opt.step()
                opt.zero_grad()
                losses.append(loss.item())
                tokens += int(att.sum())
        self.runs += 1
        out_dir = self.out_root / (out_name or f"adapter-{int(time.time())}-{self.runs:03d}")
        model.save_pretrained(out_dir)
        if student is not None and hasattr(student, "adapter_dir"):
            student.adapter_dir = str(out_dir)  # the candidate is now base + this adapter
        del model, opt
        if device == "cuda":
            torch.cuda.empty_cache()
        secs = time.time() - t0
        return {"loss": round(sum(losses[-20:]) / max(1, len(losses[-20:])), 4),
                "examples": len(examples), "adapter": str(out_dir), "from_base": from_base,
                "device": device, "seconds": round(secs, 1),
                "tokens_per_second": round(tokens / secs, 1) if secs else None,
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None}
