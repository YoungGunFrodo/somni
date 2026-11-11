"""Run the synthetic world through many days and sleeps under several conditions and score
every sleep against ground truth. This is how the project tests itself.

Conditions:
  retrieval-only        memory store + retrieval, weights never change (the baseline to beat)
  no-replay (ungated)   consolidation without replay, every candidate adopted -> shows forgetting
  no-replay (gated)     same, but morning review follows the gates -> shows gates catching it
  replay                consolidation with replay, gated
  replay + dreaming     the full sleep, gated, every old memory protected
  replay + dreaming + forgetting
                        the full sleep with the forgetting policy: unused memories fade from
                        the weights (still retrievable); replay and retention cover protected ones
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from ..config import Config, GateConfig, SleepConfig
from ..models.base import ABSTAIN, answers_match
from ..models.toy import PROBE_TEMPLATES
from ..review.morning import approve, pending, reject, reject_memories, resolve_dispute
from ..sleep.formation import run_formation
from ..sleep.pipeline import run_sleep
from ..system import build_toy_system
from .world import World

CONDITIONS = {
    "retrieval-only": dict(train=False),
    "no-replay (ungated)": dict(replay=False, dreaming=False, gated=False),
    "no-replay (gated)": dict(replay=False, dreaming=False, gated=True),
    "replay": dict(replay=True, dreaming=False, gated=True),
    "replay + dreaming": dict(replay=True, dreaming=True, gated=True),
    "replay + dreaming + forgetting": dict(replay=True, dreaming=True, gated=True, forgetting=True),
}
TRANSFER_SOURCE = "replay + dreaming + forgetting"   # the configuration the real system runs


def score(system, world: World, day: int, checkpoint: str | None = None) -> dict:
    """Ground-truth scores for a checkpoint (default: the ACTIVE one)."""
    student = system.load_student(checkpoint or system.registry.active)
    orch = system.orchestrator()

    def acc(facts, use_memory=False):
        if not facts:
            return None
        hits = []
        for f in facts:
            for t in PROBE_TEMPLATES:
                q = t.format(entity=f.entity, attr=f.attr)
                if use_memory:
                    a = student.answer(q, orch.recall(q))
                else:
                    a = student.answer(q)
                hits.append(answers_match(a.value, f.value))
        return float(np.mean(hits))

    told = world.told()
    old = [f for f in told if f.day_told <= day - 7]
    changed = [f for f in told if f.history]
    stale = []
    for f in changed:
        for t in PROBE_TEMPLATES:
            a = student.answer(t.format(entity=f.entity, attr=f.attr))
            stale.append(any(answers_match(a.value, h) for h in f.history))
    base = [student.answer(p["question"]) for p in world.base_probes()]
    unknown = [student.answer(p["question"]) for p in world.unknown_probes()]
    sample = [told[i] for i in np.random.default_rng(day).choice(len(told), size=min(60, len(told)), replace=False)]
    return {
        "day": day,
        "facts_told": len(told),
        "weights_all": acc(told),
        "weights_old": acc(old),
        "weights_frequent": acc(world.frequent()),   # the most-asked 10% of facts
        "weights_rare": acc(world.rare()),           # the least-asked half
        "with_memory": acc(sample, use_memory=True),
        "stale_rate": float(np.mean(stale)) if stale else None,
        "base": float(np.mean([answers_match(a.value, p["answer"]) for a, p in zip(base, world.base_probes())])),
        "abstain_unknown": float(np.mean([a.value == ABSTAIN for a in unknown])),
    }


def review_disputes(system, world: World, disputed_ids) -> None:
    """Simulated reviewer decides each new dispute against ground truth. (Self-test finding:
    a user's update to a fact first learned from a tool result is a weaker source, so it is
    stored as disputed; with nobody resolving disputes the outdated value stayed active forever.)"""
    for mid in disputed_ids:
        m = system.store.get(mid)
        if m is not None and m.status == "disputed":
            resolve_dispute(system, mid, accept=(world.truth(m.key) == m.value),
                            reason="simulated reviewer: contradicts ground truth")


def run_condition(name: str, opts: dict, days: int, seed: int, root: Path, dim: int,
                  sleep_overrides: dict | None = None, keep: bool = False,
                  gate_overrides: dict | None = None) -> dict:
    home = root / name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
    shutil.rmtree(home, ignore_errors=True)
    world = World(seed=seed)
    sleep_cfg = SleepConfig(replay=opts.get("replay", True), dreaming=opts.get("dreaming", False),
                            forgetting=opts.get("forgetting", False), **(sleep_overrides or {}))
    cfg = Config(home=str(home), seed=seed, sleep=sleep_cfg, gates=GateConfig(**(gate_overrides or {})))
    system = build_toy_system(cfg, world.base_examples(), world.base_probes(), world.unknown_probes(), dim=dim)
    timeline, sleeps = [], []
    t0 = time.time()
    for day in range(days):
        orch = system.orchestrator()
        world.live_day(day, orch)
        if not opts.get("train", True):
            review_disputes(system, world, run_formation(system.store, system.teacher, day).disputed)
        else:
            result = run_sleep(system, day)
            # Simulated morning reviewer: knows the truth, so it decides disputes, rejects wrong
            # memories, then follows the gates (or ignores them in the ungated condition).
            review_disputes(system, world, result["formation"]["disputed"])
            wrong = [mid for mid in result["formation"]["created"]
                     if world.truth(system.store.get(mid).key) not in (None, system.store.get(mid).value)
                     and system.store.get(mid).status == "active"]
            if wrong:
                reject_memories(system, wrong, "simulated reviewer: contradicts ground truth")
            if pending(system):
                if result["evaluation"]["passed"] or not opts.get("gated", True):
                    approve(system)
                else:
                    reject(system, reason="failed gates")
            sleeps.append({"day": day, "status": system.registry.data["candidates"].get(result.get("candidate") or "", {}).get("status", result["status"]),
                           "gates": {g["name"]: bool(g["passed"]) for g in result["evaluation"]["gates"]},
                           "adopted": bool(system.registry.active == result.get("candidate")),
                           "lint_rejected": len(result["formation"]["rejected"]),
                           "disputed": len(result["formation"]["disputed"]),
                           "superseded": len(result["formation"]["superseded"]),
                           "protected": result.get("consolidation", {}).get("protected"),
                           "condensed": result.get("consolidation", {}).get("condensed"),
                           "faded": result.get("consolidation", {}).get("faded")})
        timeline.append(score(system, world, day))
    out = {"condition": name, "timeline": timeline, "sleeps": sleeps,
           "store": system.store.stats(), "seconds": round(time.time() - t0, 1)}
    if keep:
        out["_system"], out["_world"] = system, world
    return out


def run_transfer(source: dict, seed: int, root: Path, big_dim: int) -> dict:
    """Brain transfer: move the source run's memory store into a bigger toy brain."""
    from ..transfer.brain_transfer import brain_transfer
    old, world = source.pop("_system"), source.pop("_world")
    home = root / f"transfer_dim{big_dim}"
    shutil.rmtree(home, ignore_errors=True)
    cfg = Config(home=str(home), seed=seed, sleep=old.config.sleep)
    new = build_toy_system(cfg, world.base_examples(), world.base_probes(), world.unknown_probes(), dim=big_dim)
    day = source["timeline"][-1]["day"]
    result = brain_transfer(old, new, day=day)
    if pending(new) and result["evaluation"]["passed"]:
        approve(new, note="brain transfer")
    # Score the transfer CANDIDATE, adopted or not. (Scoring the active checkpoint showed the
    # untouched base model whenever the gates blocked the transfer, which looked like a
    # failed transfer rather than a blocked one.)
    return {"before": score(old, world, day), "after": score(new, world, day, result.get("candidate")),
            "adopted": bool(new.registry.active == result.get("candidate")),
            "gates": {g["name"]: bool(g["passed"]) for g in result["evaluation"]["gates"]},
            "small_dim": old.student_factory().dim, "big_dim": big_dim}


def run_all(days: int = 30, seed: int = 0, out: str = "results", dim: int = 2048,
            conditions: list[str] | None = None, transfer_dim: int = 8192) -> dict:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "runs"
    results = {}
    for name in conditions or CONDITIONS:
        print(f"[selftest] running condition: {name}", flush=True)
        keep = bool(transfer_dim) and name == TRANSFER_SOURCE
        results[name] = run_condition(name, CONDITIONS[name], days, seed, work, dim, keep=keep)
    transfer = None
    if transfer_dim and "_system" in results.get(TRANSFER_SOURCE, {}):
        print(f"[selftest] brain transfer: dim {dim} -> {transfer_dim}", flush=True)
        transfer = run_transfer(results[TRANSFER_SOURCE], seed, work, transfer_dim)
        transfer["source"] = TRANSFER_SOURCE
    (out_dir / "results.json").write_text(json.dumps({"conditions": results, "transfer": transfer}, indent=2), encoding="utf-8")
    write_summary(results, out_dir, transfer)
    return {"conditions": results, "transfer": transfer}


def write_summary(results: dict, out_dir: Path, transfer: dict | None = None) -> None:
    def last(r, k):
        v = r["timeline"][-1][k]
        return "—" if v is None else f"{100 * v:.1f}%"

    strips = ["", "Adoption by night (A = adopted, . = rejected or nothing to learn):", ""]
    for name, r in results.items():
        if r["sleeps"]:
            strips.append(f"- `{''.join('A' if x['adopted'] else '.' for x in r['sleeps'])}` {name}")
    strips += ["", "Gate failures (number of nights each gate failed):", ""]
    for name, r in results.items():
        if r["sleeps"]:
            fails = {}
            for x in r["sleeps"]:
                for g, ok in x["gates"].items():
                    fails[g] = fails.get(g, 0) + (not ok)
            strips.append(f"- {name}: " + (", ".join(f"{g} {n}" for g, n in fails.items() if n) or "none"))
    rows = ["| Condition | Weights: all facts | Weights: facts ≥7 days old | Weights: frequent | "
            "Weights: rare | With memory | Stale beliefs | Base knowledge | Abstains on unknown | "
            "Sleeps adopted |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        adopted = sum(s["adopted"] for s in r["sleeps"])
        rows.append(f"| {name} | {last(r, 'weights_all')} | {last(r, 'weights_old')} | "
                    f"{last(r, 'weights_frequent')} | {last(r, 'weights_rare')} | {last(r, 'with_memory')} | "
                    f"{last(r, 'stale_rate')} | {last(r, 'base')} | {last(r, 'abstain_unknown')} | "
                    f"{adopted}/{len(r['sleeps'])} |")
    any_r = next(iter(results.values()))
    lint = sum(s["lint_rejected"] for s in any_r["sleeps"])
    text = ["# Self-test results", "",
            "Toy brain, not an LLM: these show pipeline dynamics, not real-model performance.", "",
            f"{len(any_r['timeline'])} simulated days, {any_r['timeline'][-1]['facts_told']} facts told, "
            "scored against ground truth after each night. \"Weights\" columns answer WITHOUT retrieval "
            "(what was consolidated); \"With memory\" uses retrieval. Questions follow a Zipf-like "
            "popularity: \"frequent\" = the 10% most-asked facts, \"rare\" = the least-asked half.",
            "", *rows, *strips, ""]
    if lint:
        text.append(f"Injected-instruction episodes rejected by lint (one run): {lint}.")
    if transfer:
        b, a = transfer["before"], transfer["after"]
        f = lambda v: "—" if v is None else f"{100 * v:.1f}%"
        text += ["", f"## Brain transfer (toy brain dim {transfer['small_dim']} → {transfer['big_dim']})", "",
                 f"Source: the \"{transfer.get('source', '?')}\" run. The new brain's candidate is scored "
                 "whether or not the gates adopted it.", "",
                 "| Measure | Old brain | New brain candidate after one transfer sleep |", "|---|---|---|",
                 f"| Facts in weights (all) | {f(b['weights_all'])} | {f(a['weights_all'])} |",
                 f"| Facts ≥7 days old | {f(b['weights_old'])} | {f(a['weights_old'])} |",
                 f"| Base knowledge | {f(b['base'])} | {f(a['base'])} |",
                 f"| Stale beliefs | {f(b['stale_rate'])} | {f(a['stale_rate'])} |",
                 f"| Abstains on unknown | {f(b['abstain_unknown'])} | {f(a['abstain_unknown'])} |", "",
                 f"Transfer adopted by gates: {transfer['adopted']}. Failed gates: "
                 f"{[k for k, v in transfer['gates'].items() if not v] or 'none'}."]
    (out_dir / "results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
        axes = axes.ravel()
        keys = ["weights_all", "weights_old", "base", "weights_frequent", "weights_rare"]
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, (name, r) in enumerate(results.items()):
            color = cycle[i % len(cycle)]   # same color for a condition in every panel
            days = [t["day"] for t in r["timeline"]]
            for ax, key in zip(axes, keys):
                ax.plot(days, [t[key] if t[key] is not None else np.nan for t in r["timeline"]],
                        label=name, color=color)
            if r["sleeps"]:
                axes[5].plot([s["day"] for s in r["sleeps"]],
                             np.cumsum([s["adopted"] for s in r["sleeps"]]) / len(r["sleeps"]),
                             label=name, color=color)
        for ax, title in zip(axes, ["Facts in weights (no retrieval)", "Facts told ≥7 days ago",
                                     "Base knowledge (general ability)", "Frequent facts (top 10% asked)",
                                     "Rare facts (least-asked half)", "Sleeps adopted (cumulative share)"]):
            ax.set_title(title)
            ax.set_xlabel("day (one sleep per night)")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("probe accuracy")
        axes[3].set_ylabel("probe accuracy")
        axes[2].legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(out_dir / "results.png", dpi=120)
    except Exception as e:  # charts are optional
        print("chart skipped:", e)
