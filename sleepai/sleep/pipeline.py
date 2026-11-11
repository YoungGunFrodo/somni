"""One sleep: memory formation -> consolidation -> dreaming -> evaluation -> dream report.

Nothing changes the active model here unless the autonomy level allows it
(level >= 2 and every gate passed by a clear margin). Otherwise the candidate waits
for morning review. Memories are marked as consolidated only when a checkpoint
containing them is adopted.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from .consolidation import run_consolidation
from .dreaming import run_dreaming
from .evaluation import evaluate
from .formation import run_formation
from .report import write_report
from .scheduler import wake_requested


SKILL_GATES = ("general", "drift-vs-base", "coding", "calibration")


def _second_take(system, sleep_no, rng, day, full_store, current, ev, learned, student, consolidation,
                 dreaming):
    """Retrain once more in the same sleep, keeping what worked (see SleepConfig.second_take)."""
    failed = [g.name for g in ev.gates if not g.passed]
    skill = any(n in SKILL_GATES or n.startswith("skill:") for n in failed)
    learn = any(n.startswith("learning") for n in failed)
    if not (skill or learn):              # retention-type failures: no second take
        return student, consolidation, dreaming, learned, ev, {"tried": False, "failed": failed}
    exclude = set()
    if learn:
        exclude = {mid for mid in learned if (m := system.store.get(mid)) and m.last_probe_result == "fail"}
        if len(exclude) >= len(learned):
            return student, consolidation, dreaming, learned, ev, {"tried": False, "failed": failed,
                                                                  "reason": "every memory failed"}
    boost = 2.0 if skill else 1.0
    print(f"[sleep] second take: failed {failed}; leaving out {len(exclude)} memories, "
          f"rehearsal x{boost:g}", flush=True)
    if hasattr(system.trainer, "begin_sleep"):
        system.trainer.begin_sleep()     # a fresh expert on top of the ACTIVE checkpoint
    student, consolidation = run_consolidation(system, sleep_no, rng, False, day=day, full_store=full_store,
                                               exclude=exclude, rehearsal_boost=boost)
    learned = list(consolidation.new_memories)
    if system.config.sleep.dreaming and learned:
        dreaming = run_dreaming(system, student, learned, rng, day=day)
    ev2 = evaluate(system, current, student, learned, rng, day=day)
    return student, consolidation, dreaming, learned, ev2, {
        "tried": True, "failed_first": failed, "left_out": sorted(exclude), "rehearsal_boost": boost,
        "passed": ev2.passed}


def run_sleep(system, day: int = 0, force_rebuild: bool = False, full_store: bool = False) -> dict:
    sleep_no = len(list(system.sleeps_dir.glob("sleep-*")))
    folder = system.sleeps_dir / f"sleep-{sleep_no:04d}"
    rng = np.random.default_rng(system.config.seed * 100003 + sleep_no)

    if hasattr(system.trainer, "begin_sleep"):     # gated experts: one new expert per sleep
        system.trainer.begin_sleep()
    formation = run_formation(system.store, system.teacher, day)
    if wake_requested(system.home):
        return {"sleep_no": sleep_no, "status": "paused for wake", "formation": asdict(formation)}
    student, consolidation = run_consolidation(system, sleep_no, rng, force_rebuild, day=day,
                                               full_store=full_store)
    learned = list(consolidation.new_memories)
    nothing_to_learn = not learned and not consolidation.rebuild

    dreaming = None
    if system.config.sleep.dreaming and not nothing_to_learn:
        dreaming = run_dreaming(system, student, learned, rng, day=day)

    data = {"sleep_no": sleep_no, "day": day, "formation": asdict(formation),
            "consolidation": asdict(consolidation),
            "dreaming": asdict(dreaming) if dreaming else None,
            "evaluation": {"gates": [], "passed": True}, "candidate": None,
            "status": "nothing to learn", "recommendation": "no change",
            "rehearsal_mix": getattr(system, "rehearsal_mix", None)}

    if not nothing_to_learn:
        current = system.load_student(system.registry.active)
        ev = evaluate(system, current, student, learned, rng, day=day)
        if not ev.passed and system.config.sleep.second_take and not consolidation.rebuild:
            student, consolidation, dreaming, learned, ev, take2 = _second_take(
                system, sleep_no, rng, day, full_store, current, ev, learned, student, consolidation, dreaming)
            data.update(consolidation=asdict(consolidation), dreaming=asdict(dreaming) if dreaming else None,
                        second_take=take2)
        cid = system.registry.new_id()
        student.save_checkpoint(str(system.registry.path(cid)))
        # A rebuild retrains from base, so the checkpoint holds exactly what it was trained on
        # (with forgetting, faded memories are left out and leave the weights).
        to_mark = learned if not consolidation.rebuild else learned + consolidation.rebuild_memories
        probe_file = folder / "probe_results.json"
        folder.mkdir(parents=True, exist_ok=True)
        probe_file.write_text(json.dumps(ev.info.pop("probe_results", {})), encoding="utf-8")
        system.registry.add_candidate(cid, {"sleep_no": sleep_no, "memories": to_mark,
                                            "passed": ev.passed, "status": "awaiting review",
                                            "probe_results": str(probe_file)})
        data.update(evaluation=ev.to_dict(), candidate=cid,
                    recommendation="adopt" if ev.passed else "reject",
                    status="awaiting review")
        if system.autonomy >= 2 and ev.margin_ok:
            from ..review.morning import approve
            approve(system, cid, note="auto-adopted (autonomy level >= 2, clear margin)")
            data["status"] = "auto-adopted"
        elif not ev.passed and system.autonomy >= 2:
            system.registry.reject(cid, "failed gates")
            data["status"] = "rejected by gates"
        # At levels 0-1 a failing candidate still waits for morning review, with a
        # "reject" recommendation: the human decides, not the pipeline.

    data["report"] = str(write_report(folder, data, system.store))
    return data
