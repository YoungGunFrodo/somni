"""Gated experts: each sleep adds a NEW module and freezes every older one (and the base).

Why: a plain LoRA adds its change to EVERY prompt, so weights trained on short fact answers also
shift coding and general answers (real finding: coding 95% -> 10% after one night). Here every
target linear layer computes

    y = base(x) + sum_k  g_k(x) * B_k A_k x * scale        g_k(x) = sigmoid(w_k . rmsnorm(x) + b_k)

per token. The gate g_k decides how much expert k joins in. Training teaches it when:
  - recall examples (new memories): the expert must produce the fact -> gate opens
  - stay-out examples (rehearsal: the base model's own general/coding answers): the expert must
    leave the answer unchanged AND its gate is pushed to 0 (gate penalty)
  - combine examples (new fact used inside a coding task): the expert has to work WITH the old
    skills instead of replacing them
Old experts are frozen, so a new night cannot overwrite what an earlier night stored; the new
gate learns to stay closed where nothing new applies.

Saved as a folder with experts.json (config) and experts.pt (all experts' weights, oldest first).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def is_expert_dir(path) -> bool:
    return bool(path) and (Path(path) / "experts.json").exists()


def _module_class():
    import torch
    from torch import nn

    class GatedExpertLinear(nn.Module):
        def __init__(self, base: nn.Linear, ctx: dict):
            super().__init__()
            self.base, self.ctx = base, ctx
            self.A, self.B, self.w, self.b = nn.ParameterList(), nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
            self.scales: list[float] = []

        def add_expert(self, rank: int, alpha: float, gate_bias: float):
            dev = self.base.weight.device
            A = torch.empty(rank, self.base.in_features, device=dev)
            nn.init.kaiming_uniform_(A, a=5 ** 0.5)
            self.A.append(nn.Parameter(A))
            self.B.append(nn.Parameter(torch.zeros(self.base.out_features, rank, device=dev)))
            self.w.append(nn.Parameter(torch.zeros(self.base.in_features, device=dev)))
            self.b.append(nn.Parameter(torch.full((), float(gate_bias), device=dev)))
            self.scales.append(alpha / rank)

        def forward(self, x):
            y = self.base(x)
            if not len(self.A) or self.ctx.get("off"):    # off: the base model alone (reads memory)
                return y
            # Compute in the model's dtype (bf16 on the GPU): an fp32 copy of every layer input
            # used several GB of extra VRAM. Only the small gate dot product runs in fp32.
            # Memory: nothing layer-sized is stored for backward beyond x itself (which the base
            # layer keeps anyway). The RMS scale is a constant (no grad), the gate is one number
            # per token, and it multiplies the rank-sized middle, not the full output.
            dt = x.dtype
            with torch.no_grad():
                inv_rms = torch.rsqrt(x.float().pow(2).mean(-1) + 1e-6)          # (batch, tokens)
            for k in range(len(self.A)):
                g = torch.sigmoid((x @ self.w[k].to(dt)).float() * inv_rms + self.b[k])
                if self.ctx.get("record") and self.A[k].requires_grad:
                    self.ctx["gates"].append(g.mean(-1) if g.dim() > 1 else g)    # per example
                mid = (x @ self.A[k].to(dt).t()) * (g * self.scales[k]).to(dt).unsqueeze(-1)
                y = y + mid @ self.B[k].to(dt).t()
            return y

    return GatedExpertLinear


def wrap(model, targets, ctx: dict) -> list:
    """Replace every target linear layer with a gated-expert wrapper (no experts yet)."""
    cls = _module_class()
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in targets and child.__class__.__name__ == "Linear":
                w = cls(child, ctx)
                setattr(mod, child_name, w)
                wrapped.append((f"{name}.{child_name}", w))
    return wrapped


def add_expert(wrapped, rank: int, alpha: float, gate_bias: float) -> list:
    params = []
    for _, w in wrapped:
        w.add_expert(rank, alpha, gate_bias)
        params += [w.A[-1], w.B[-1], w.w[-1], w.b[-1]]
    return params


def state(wrapped) -> dict:
    return {name: {"A": [p.detach().cpu() for p in w.A], "B": [p.detach().cpu() for p in w.B],
                   "w": [p.detach().cpu() for p in w.w], "b": [p.detach().cpu() for p in w.b],
                   "scales": list(w.scales)} for name, w in wrapped}


def load_into(model, expert_dir, ctx: dict | None = None) -> list:
    """Wrap `model` and load every saved expert, frozen."""
    import torch
    from torch import nn
    cfg = json.loads((Path(expert_dir) / "experts.json").read_text(encoding="utf-8"))
    sd = torch.load(Path(expert_dir) / "experts.pt", map_location="cpu")
    wrapped = wrap(model, cfg["targets"], ctx if ctx is not None else {})
    for name, w in wrapped:
        s = sd[name]
        dev = w.base.weight.device
        for k in range(len(s["A"])):
            for key, plist in (("A", w.A), ("B", w.B), ("w", w.w), ("b", w.b)):
                plist.append(nn.Parameter(s[key][k].to(dev), requires_grad=False))
            w.scales.append(s["scales"][k])
    return wrapped


def save(wrapped, out_dir, targets, meta: dict) -> None:
    import torch
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(state(wrapped), out / "experts.pt")
    n = len(wrapped[0][1].A) if wrapped else 0
    (out / "experts.json").write_text(json.dumps({"targets": targets, "experts": n, **meta}, indent=1),
                                      encoding="utf-8")


class ExpertTrainer:
    """Same interface as PeftTrainer.train(); examples with origin "base" are stay-out examples."""

    def __init__(self, base_model: str, out_root: str, rank: int = 16, lr: float = 2e-4,
                 max_len: int = 384, batch_size: int = 4, seed: int = 0, targets: list | None = None,
                 gate_penalty: float = 1.0, gate_sparsity: float = 0.02, gate_bias: float = 0.0,
                 vram_fraction: float = 0.7, checkpointing: bool = False,
                 max_ram_gb: float | None = None, gated: bool = True):
        self.base_model, self.out_root = base_model, Path(out_root)
        self.targets = targets or DEFAULT_TARGETS
        self.rank, self.lr, self.max_len, self.batch_size, self.seed = rank, lr, max_len, batch_size, seed
        self.max_len_context = max(max_len, 768)     # lookup practice carries retrieved memories
        self.gate_penalty, self.gate_sparsity, self.gate_bias = gate_penalty, gate_sparsity, gate_bias
        self.vram_fraction, self.checkpointing = vram_fraction, checkpointing
        self.max_ram_gb = max_ram_gb
        # gated=False is the control: the gate is pinned open (plain LoRA, same code path).
        self.gated = gated
        if not gated:
            self.gate_bias, self.gate_penalty, self.gate_sparsity = 20.0, 0.0, 0.0
        self.runs = 0
        self._sleep_dirs: set[str] = set()

    def begin_sleep(self) -> None:
        """Called by the pipeline at the start of each sleep: the next train() adds a new expert;
        later train() calls in the same sleep (dreaming) continue that expert."""
        self._sleep_dirs = set()

    def train(self, student, examples, epochs: int = 2, from_base: bool = False,
              resume_adapter: str | None = None, out_name: str | None = None) -> dict:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .peft_trainer import fit_context, pick_device

        device, dtype = pick_device()
        t0 = time.time()
        try:
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
        ctx = {"record": False, "gates": []}
        prev = resume_adapter or getattr(student, "adapter_dir", None)
        if prev and not from_base and is_expert_dir(prev):
            wrapped = load_into(model, prev, ctx)          # older experts: frozen
        else:
            wrapped = wrap(model, self.targets, ctx)
        same_sleep = bool(prev) and not from_base and str(prev) in self._sleep_dirs and wrapped and             len(wrapped[0][1].A) > 0
        if same_sleep:
            # Dreaming after consolidation in the same sleep: keep training tonight's expert
            # instead of stacking another one (one expert per night).
            params = []
            for _, w in wrapped:
                for plist in (w.A, w.B, w.w, w.b):
                    plist[-1].requires_grad_(True)
                    params.append(plist[-1])
        else:
            params = add_expert(wrapped, self.rank, 2 * self.rank, self.gate_bias)
        if not self.gated:                                 # freeze the pinned-open gate
            gates = {id(w.w[-1]) for _, w in wrapped} | {id(w.b[-1]) for _, w in wrapped}
            for prm in params:
                if id(prm) in gates:
                    prm.requires_grad_(False)
            params = [prm for prm in params if id(prm) not in gates]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.0)
        model.train()
        if device == "cuda":
            # Fail with a normal out-of-memory error instead of spilling into shared memory: the
            # Windows AMD driver crashed the whole PC (bugcheck 0x7E) during an earlier run.
            from ..resources import cap_vram
            held = torch.cuda.memory_allocated() / 2**30
            print(f"[resources] training model loaded: this process holds {held:.1f} GB of VRAM", flush=True)
            if held > 1.5 * sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30:
                # More than the model itself: list what else is on the GPU (diagnostic).
                import gc
                mine = {id(p) for p in model.parameters()} | {id(b) for b in model.buffers()}
                other = [o for o in gc.get_objects() if torch.is_tensor(o) and o.is_cuda and id(o) not in mine]
                other.sort(key=lambda t: -t.numel() * t.element_size())
                total = sum(t.numel() * t.element_size() for t in other) / 2**30
                print(f"[resources] {len(other)} other GPU tensors, {total:.2f} GB; largest: "
                      + ", ".join(f"{tuple(t.shape)} {t.dtype}" for t in other[:5]), flush=True)
                # Who keeps the leftover MODEL alive? Every full model object except ours, and
                # what refers to it (two levels).
                def desc(o):
                    n = type(o).__name__
                    if n == "frame":
                        return f"frame({o.f_code.co_filename.rsplit(chr(92), 1)[-1]}:{o.f_code.co_name}:{o.f_lineno})"
                    if isinstance(o, dict):
                        return f"dict(keys {[str(k)[:30] for k in list(o)[:4]]})"
                    if isinstance(o, (list, tuple)):
                        return f"{n}(len {len(o)})"
                    return n
                models = [o for o in gc.get_objects() if type(o).__name__.endswith("ForCausalLM") and o is not model]
                for lm in models[:2]:
                    lvl1 = [r for r in gc.get_referrers(lm) if r is not models]
                    print(f"[resources] leftover {type(lm).__name__} referenced by: "
                          + "; ".join(desc(r) + " <- [" + ", ".join(desc(x) for x in gc.get_referrers(r)[:3]
                                                                   if x is not lvl1) + "]" for r in lvl1[:5]),
                          flush=True)
                del other
            cap_vram(self.vram_fraction)
            if self.checkpointing:              # saves VRAM, but made a run several times slower
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
            model.config.use_cache = False
        pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

        def encode(ex):
            # Lookup-practice examples get room for their context; the rest keep max_len.
            limit = self.max_len_context if ex.context else self.max_len
            prompt = fit_context(tok, ex, limit)
            ids = tok(prompt + ex.answer + tok.eos_token, truncation=True, max_length=limit,
                      add_special_tokens=False, return_tensors="pt").input_ids[0]
            n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
            labels = ids.clone()
            labels[:n_prompt] = -100
            return ids, labels, float(ex.weight), ex.origin == "base"

        data = [encode(e) for e in examples]
        rng = random.Random(self.seed + self.runs)

        # The whole schedule is fixed up front, so a stopped run can resume at any step.
        # Similar lengths are grouped (short fact answers vs long code answers), batch order random.
        schedule = []
        pool = 32 * self.batch_size
        for _ in range(epochs):
            order = list(range(len(data)))
            rng.shuffle(order)
            ep = []
            for p0 in range(0, len(order), pool):
                part = sorted(order[p0:p0 + pool], key=lambda i: len(data[i][0]))
                ep += [part[i:i + self.batch_size] for i in range(0, len(part), self.batch_size)]
            rng.shuffle(ep)
            schedule += ep

        out_dir = self.out_root / (out_name or f"experts-{int(time.time())}-{self.runs:03d}")
        resume_file = out_dir / "resume.pt"
        signature = [len(data), epochs, self.batch_size, self.rank, self.lr]
        losses, gate_in, gate_out, tokens, step = [], [], [], 0, 0
        if resume_file.exists():
            r = torch.load(resume_file, map_location="cpu")
            if r.get("signature") == signature:
                with torch.no_grad():
                    for prm, val in zip(params, r["params"]):
                        prm.copy_(val.to(prm.device))
                opt.load_state_dict(r["opt"])
                step, losses, gate_in, gate_out = r["step"], r["losses"], r["gate_in"], r["gate_out"]
                print(f"[train] resumed at step {step}/{len(schedule)}", flush=True)

        def save_progress():
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"signature": signature, "step": step, "params": [x.detach().cpu() for x in params],
                        "opt": opt.state_dict(), "losses": losses[-200:], "gate_in": gate_in[-200:],
                        "gate_out": gate_out[-200:]}, resume_file)

        from ..resources import ResourceGovernor, ResumableStop
        gov = ResourceGovernor(self.max_ram_gb)
        micro = self.batch_size                 # cap; lowered only by the RAM governor
        vram_micro = min(self.batch_size, gov.micro_batch_for_vram(self.batch_size))
        # Micro-batches sized by TOKENS, not a count (v3 SQuAD finding: one long lookup example ran
        # out of VRAM at step 115, the count dropped to 1 and stayed there for 11,300 steps, short
        # examples included). An out-of-VRAM halves the token budget; short steps still run whole.
        budget = vram_micro * self.max_len_context

        def micro_for(items) -> int:
            L = -(-max(len(c[0]) for c in items) // 64) * 64
            n = max(1, min(micro, budget // L))
            return 1 << (n.bit_length() - 1)    # powers of two: few distinct shapes (RAM)

        def run_micro(items, W, n_stay, n_new):
            """Forward/backward one micro-batch; terms scaled so a full batch gives the same loss."""
            nonlocal tokens
            L = -(-max(len(c[0]) for c in items) // 64) * 64             # few distinct shapes
            ids = torch.full((len(items), L), pad)
            lab = torch.full((len(items), L), -100)
            att = torch.zeros((len(items), L), dtype=torch.long)
            for j, (a, b, _, _) in enumerate(items):
                ids[j, :len(a)], lab[j, :len(b)], att[j, :len(a)] = a, b, 1
            ids, lab, att = ids.to(device), lab.to(device), att.to(device)
            ctx["gates"] = []
            ctx["record"] = True
            # Vocabulary scores only at answer positions (full logits were the largest allocation).
            hidden = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
            ctx["record"] = False
            target = lab[:, 1:]
            sel = target != -100
            h, t = hidden[sel], target[sel]
            n = len(t)
            pad_rows = -(-n // 64) * 64 - n
            if pad_rows:
                h = torch.cat([h, h.new_zeros(pad_rows, h.size(-1))])
                t = torch.cat([t, t.new_full((pad_rows,), -100)])
            tok_loss = torch.nn.functional.cross_entropy(model.lm_head(h).float(), t, ignore_index=-100,
                                                         reduction="none")[:n]
            per_ex = torch.zeros(len(items), device=device).index_add(0, sel.nonzero()[:, 0], tok_loss)
            per_ex = per_ex / sel.sum(1).clamp(min=1)
            w = torch.tensor([c[2] for c in items], device=device)
            loss = (per_ex * w).sum() / W
            g_ex = torch.stack(ctx["gates"]).mean(0)                     # (batch,) mean gate
            ctx["gates"] = []
            stay = torch.tensor([c[3] for c in items], device=device, dtype=torch.bool)
            if stay.any():
                loss = loss + self.gate_penalty * g_ex[stay].sum() / n_stay   # stay out of old skills
            loss = loss + self.gate_sparsity * g_ex.sum() / (n_stay + n_new)  # open only where needed
            loss.backward()
            tokens += int(att.sum())
            return (loss.item(), g_ex[stay].sum().item() if stay.any() else 0.0,
                    g_ex[~stay].sum().item() if (~stay).any() else 0.0)

        while step < len(schedule):
            if step % 20 == 0:
                gov.wait_if_pc_busy()
                if not gov.own_usage_ok():
                    if micro > 1:
                        micro = max(1, micro // 2)
                        print(f"[resources] training at {gov.rss():.1f} GB RAM: micro-batch -> {micro}",
                              flush=True)
                    if not gov.own_usage_ok():
                        save_progress()
                        raise ResumableStop(f"training process at {gov.rss():.1f} GB RAM (limit "
                                            f"{gov.ram_limit:.1f} GB); progress saved at step {step}")
            chunk = [data[i] for i in schedule[step]]
            W = sum(c[2] for c in chunk)
            n_stay = sum(1 for c in chunk if c[3])
            n_new = len(chunk) - n_stay
            try:
                tot, gs, gn = 0.0, 0.0, 0.0
                mb = micro_for(chunk)
                for m in range(0, len(chunk), mb):
                    a, b, c = run_micro(chunk[m:m + mb], W, max(1, n_stay), max(1, n_new))
                    tot, gs, gn = tot + a, gs + b, gn + c
            except torch.OutOfMemoryError:
                # Out of VRAM (another program, e.g. the teacher's server, holds some): drop this
                # step's partial gradients, free memory, retry the SAME step with half the
                # micro-batch. Same math, just slower. At micro-batch 1: save and stop.
                opt.zero_grad(set_to_none=True)
                ctx["gates"], ctx["record"] = [], False
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                if mb == 1:
                    save_progress()
                    raise ResumableStop(f"out of VRAM even at micro-batch 1; progress saved at step {step}")
                budget = max(64, budget // 2)
                print(f"[resources] out of VRAM at step {step}: token budget -> {budget}, retrying", flush=True)
                continue
            opt.step()
            opt.zero_grad()
            losses.append(tot)
            if n_stay:
                gate_out.append(gs / n_stay)
            if n_new:
                gate_in.append(gn / n_new)
            step += 1
            if step % 200 == 0:
                save_progress()
            if step % 100 == 0:
                print(f"[train] step {step}/{len(schedule)} loss {sum(losses[-100:]) / 100:.3f} "
                      f"gate new {sum(gate_in[-100:]) / max(1, len(gate_in[-100:])):.3f} "
                      f"stay-out {sum(gate_out[-100:]) / max(1, len(gate_out[-100:])):.3f} "
                      f"({time.time() - t0:.0f} s) RAM {gov.rss():.1f} GB micro {micro} budget {budget}", flush=True)
        self.runs += 1
        if resume_file.exists():
            resume_file.unlink()
        tail = lambda xs: round(sum(xs[-50:]) / max(1, len(xs[-50:])), 4)
        save(wrapped, out_dir, self.targets, {"rank": self.rank, "lr": self.lr,
                                               "gate_penalty": self.gate_penalty})
        self._sleep_dirs.add(str(out_dir))
        if student is not None and hasattr(student, "adapter_dir"):
            student.adapter_dir = str(out_dir)
        del model, opt
        if device == "cuda":
            torch.cuda.empty_cache()
        secs = time.time() - t0
        return {"loss": tail(losses), "gate_new": tail(gate_in), "gate_stay_out": tail(gate_out),
                "examples": len(examples), "adapter": str(out_dir), "from_base": from_base,
                "device": device, "seconds": round(secs, 1),
                "tokens_per_second": round(tokens / secs, 1) if secs else None,
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None}
