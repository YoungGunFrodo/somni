"""In-process student for SLEEP on the real backend: a Hugging Face base model plus the LoRA
adapter of a checkpoint, run with PyTorch (GPU when available).

Sleep needs two versions of the student at once (current checkpoint and candidate) to run the
paired evaluation gates; a server would have to swap adapters for every probe. Wake still uses
llama-server (models/llm.py) with the approved adapter converted to GGUF.

A checkpoint folder holds `adapter/` (PEFT) and, once approved, `adapter.gguf`. The base
checkpoint has no adapter.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .base import ABSTAIN, Answer, normalize
from .llm import MEMORY_PREAMBLE, cited_memory

_CACHE: dict = {}          # (base_model, adapter_dir) -> (model, tok); at most two kept


def _register_release() -> None:
    from .. import gpu
    gpu.register("student", clear_cache)


def clear_cache() -> None:
    # Free the weights themselves, not just our reference: a real night kept a whole 3.2 GB model
    # alive through a finished answer_many() frame that something still referenced, so dropping
    # the cache entry freed nothing. Moving the weights to the "meta" device frees the VRAM even
    # if a stale reference survives (traced 2026-11-10 with the ExpertTrainer diagnostic).
    for model, _tok in list(_CACHE.values()):
        try:
            model.to("meta")
        except Exception:
            pass
    _CACHE.clear()
    # Transformers models hold reference cycles: without a full collection the dropped models
    # stay in VRAM (a real night trained next to a leftover 3.4 GB copy and ran out of VRAM).
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _load(base_model: str, adapter_dir: str | None):
    key = (base_model, adapter_dir)
    if key in _CACHE:
        return _CACHE[key]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .. import gpu
    from .peft_trainer import pick_device
    gpu.acquire("student")                   # the teacher's server (if any) leaves the GPU first
    device, dtype = pick_device()
    while len(_CACHE) >= 2:
        _CACHE.pop(next(iter(_CACHE)))
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, low_cpu_mem_usage=True,
                                                     device_map=device)
    from .experts import is_expert_dir, load_into
    if adapter_dir and is_expert_dir(adapter_dir):      # gated experts (models/experts.py)
        model._expert_ctx = {}
        load_into(model, adapter_dir, model._expert_ctx)
    elif adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir).to(device)
    model.eval()
    _CACHE[key] = (model, tok)
    return _CACHE[key]


class HFTeacher:
    """In-process teacher: an HF model (by default the student's own base model, no adapter)
    reading the memory in context and writing training data for the student without it. This is
    context distillation, the design review's answer to a slow big teacher. Same prompts as
    LLMTeacher; only the transport differs."""

    def __new__(cls, model_name: str):
        from .llm import LLMTeacher

        class _T(LLMTeacher):
            def _chat(self, messages, temperature: float = 0.2, max_tokens: int = 1024) -> str:
                import torch
                model, tok = _load(model_name, None)
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
                enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=max_tokens, do_sample=temperature > 0.3,
                                         temperature=max(temperature, 0.01), top_p=0.95,
                                         pad_token_id=tok.pad_token_id)
                return tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

        return _T(url="in-process", model=model_name)


class HFStudent:
    # Lookup routing (SQuAD finding): trained experts make the model worse at READING memory
    # (given the right passage: base 82% vs trained 45%), so when recall found a strong match
    # the base model answers with the experts switched off. None = always use the experts.
    route_threshold: float | None = None

    def __init__(self, base_model: str, max_new_tokens: int = 48, batch_size: int = 16):
        self.base_model = base_model
        self.adapter_dir: str | None = None
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size

    # -------------------------------------------------- checkpoints
    def load_checkpoint(self, path: str) -> None:
        a = Path(path) / "adapter"
        ok = (a / "adapter_config.json").exists() or (a / "experts.json").exists()
        self.adapter_dir = str(a) if ok else None

    def save_checkpoint(self, path: str) -> None:
        dst = Path(path) / "adapter"
        Path(path).mkdir(parents=True, exist_ok=True)
        if self.adapter_dir and Path(self.adapter_dir).resolve() != dst.resolve():
            shutil.copytree(self.adapter_dir, dst, dirs_exist_ok=True)
            self.adapter_dir = str(dst)

    # -------------------------------------------------- inference
    def _prompts(self, items):
        from .peft_trainer import chat_prompt
        _, tok = _load(self.base_model, self.adapter_dir)
        out = []
        for q, context in items:
            out.append(chat_prompt(tok, q, context))   # exactly the training format
        return out

    def answer_many(self, questions, contexts=None, max_new_tokens: int | None = None) -> list[Answer]:
        import torch
        contexts = contexts or [None] * len(questions)
        max_new = max_new_tokens or self.max_new_tokens
        model, tok = _load(self.base_model, self.adapter_dir)
        prompts = self._prompts(list(zip(questions, contexts)))
        ectx = getattr(model, "_expert_ctx", None)
        read = [False] * len(prompts)
        if ectx is not None and self.route_threshold is not None:
            from ..wake.orchestrator import match_strength
            read = [bool(c) and match_strength(q, c) >= self.route_threshold for q, c in zip(questions, contexts)]
        self.last_routed = sum(read)
        texts = [""] * len(prompts)
        for mode in (False, True):                   # experts on, then off (reading memory)
            idx = [i for i in range(len(prompts)) if read[i] == mode]
            if ectx is not None:
                ectx["off"] = mode
            try:
                for j in range(0, len(idx), self.batch_size):
                    part = idx[j:j + self.batch_size]
                    enc = tok([prompts[i] for i in part], return_tensors="pt", padding=True,
                              add_special_tokens=False).to(model.device)
                    if ectx is not None and not mode:
                        self._route(model, ectx, enc)
                    with torch.no_grad():
                        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                             pad_token_id=tok.pad_token_id)
                    for i, t in zip(part, tok.batch_decode(gen[:, enc.input_ids.shape[1]:],
                                                           skip_special_tokens=True)):
                        texts[i] = t
            finally:
                if ectx is not None:
                    ectx["off"], ectx["route"], ectx["scores"] = False, None, None
        out = []
        for text, context in zip(texts, contexts):
            text, used = cited_memory(text.strip(), context)
            value = ABSTAIN if "not sure" in text.lower() else normalize(text)
            caveat = bool(used) and any(m.id == used and m.status == "disputed" for m in context or [])
            out.append(Answer(text, value, 1.0, used_memory=used, caveat=caveat))
        return out

    route_stats = (0, 0)      # (questions with an expert on, questions scored); reset by callers

    def _route(self, model, ectx, enc) -> None:
        """Per-question routing (models/experts.py): score the prompt once, then switch each
        expert on only for the questions where its score reaches its saved threshold."""
        import torch
        ectx["route"] = None
        th = ectx.get("thresholds") or []
        if not any(t is not None for t in th):
            return
        ectx["scores"], ectx["pmask"] = {}, enc.attention_mask.float()
        with torch.no_grad():
            model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        cols = []
        for k, t in enumerate(th):
            if t is None:
                cols.append(torch.ones(enc.input_ids.size(0), device=enc.input_ids.device))
            else:
                cols.append((torch.stack(ectx["scores"][k]).mean(0) >= t).float())
        ectx["scores"] = None
        ectx["route"] = torch.stack(cols, 1)                       # (batch, experts)
        on, n = self.route_stats
        on_now = (ectx["route"][:, [k for k, t in enumerate(th) if t is not None]].amax(1) > 0).sum().item()
        self.route_stats = (on + int(on_now), n + enc.input_ids.size(0))

    def answer(self, question: str, context=None) -> Answer:
        return self.answer_many([question], [context])[0]


_register_release()
