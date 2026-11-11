"""Command line for the sleep-learning AI.

    python -m sleepai.cli [--home DIR] <command> ...

    chat                           talk through the orchestrator (wake); /right, /wrong <fix>, /quit
    tell <text>                    state something for later (logged as an episode)
    sleep                          run one sleep now and write a dream report
    review                         show what waits for morning review
    review approve [ckpt]          adopt the candidate checkpoint
    review reject [ckpt]           reject it
    review reject-memories <id>... mark memories wrong (kept as disputed, never consolidated)
    review resolve <id> accept|reject   decide a disputed memory
    review rollback                go back to the previous checkpoint (drops autonomy by one)
    status                         active checkpoint, memory store, autonomy, last dream report
    selftest [--days N ...]        the synthetic benchmark (toy brain, no GPU)
    doctor                         check the environment and the llama-server
    daemon                         sleep automatically when idle / in the nightly window

Configuration is `<home>/config.json` (created with defaults on first use). With
`"backend": "toy"` (the default) the toy brain is used, which is only good for trying the
commands; set `"backend": "llamacpp"` and `"trainer": "peft"` for the real system.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config


def _system(args):
    home = Path(args.home).expanduser()
    home.mkdir(parents=True, exist_ok=True)
    cfg_path = home / "config.json"
    cfg = Config.load(cfg_path, home=str(home))
    if not cfg_path.exists():
        cfg.save(cfg_path)
    if cfg.backend == "toy":
        from .system import build_toy_system
        return build_toy_system(cfg, [], [], [])
    from .system import build_real_system
    return build_real_system(cfg)


def _last_report(system) -> Path | None:
    reports = sorted(system.sleeps_dir.glob("sleep-*/report.md"))
    return reports[-1] if reports else None


# ---------------------------------------------------------------- wake
def cmd_chat(args) -> int:
    from .sleep.scheduler import today
    system = _system(args)
    orch = system.orchestrator()
    last = None
    print("Wake. Type a question or statement. /right, /wrong <correction>, /quit")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/right" and last:
            orch.feedback(last, True, day=today())
            print("(noted: correct)")
        elif line.startswith("/wrong") and last:
            orch.feedback(last, False, line[len("/wrong"):].strip(), day=today())
            print("(noted: correction logged for the next sleep)")
        elif line.endswith("?"):
            ans, last = orch.ask(line, day=today())
            print(ans.text)
        else:
            orch.tell(line, day=today())
            print("(noted)")
    return 0


def cmd_tell(args) -> int:
    from .sleep.scheduler import today
    eid = _system(args).orchestrator().tell(" ".join(args.text), day=today())
    print(f"logged episode {eid}")
    return 0


# ---------------------------------------------------------------- sleep
def cmd_sleep(args) -> int:
    from .sleep.pipeline import run_sleep
    from .sleep.scheduler import today
    system = _system(args)
    result = run_sleep(system, day=today())
    print(f"sleep {result['sleep_no']}: {result['status']}, recommendation: "
          f"{result.get('recommendation', '-')}")
    if result.get("report"):
        print(f"dream report: {result['report']}")
    return 0


def cmd_review(args) -> int:
    from .review import morning
    system = _system(args)
    action = args.action
    if action is None:
        waiting = morning.pending(system)
        print("awaiting review:", ", ".join(waiting) or "nothing")
        disputed = [m for m in system.store.all("disputed") if not m.rejection_reason.startswith("rejected at review")]
        for m in disputed:
            print(f"  disputed {m.id}: {m.key} = {m.value!r} ({m.rejection_reason})")
        report = _last_report(system)
        if report:
            print(f"latest dream report: {report}")
        return 0
    if action == "approve":
        print("adopted", morning.approve(system, args.ids[0] if args.ids else None))
    elif action == "reject":
        print("rejected", morning.reject(system, args.ids[0] if args.ids else None, args.reason))
    elif action == "reject-memories":
        if not args.ids:
            print("give one or more memory ids", file=sys.stderr)
            return 2
        morning.reject_memories(system, args.ids, args.reason)
        print(f"rejected {len(args.ids)} memories")
    elif action == "resolve":
        if len(args.ids) != 2 or args.ids[1] not in ("accept", "reject"):
            print("usage: review resolve <memory-id> accept|reject", file=sys.stderr)
            return 2
        active = morning.resolve_dispute(system, args.ids[0], args.ids[1] == "accept", args.reason)
        print(f"resolved; active memory for the key: {active or 'none'}")
    elif action == "rollback":
        print("active checkpoint is now", morning.rollback(system))
    return 0


def cmd_status(args) -> int:
    from .memory.salience import tiers
    from .sleep.scheduler import today
    from .review.morning import pending
    system = _system(args)
    reg = system.registry
    consolidated = [m for m in system.store.eligible_for_consolidation() if m.consolidated_in]
    by_tier = tiers(consolidated, today(), system.config.sleep)
    print(json.dumps({
        "home": str(system.home),
        "backend": system.config.backend,
        "active_checkpoint": reg.active,
        "base_checkpoint": reg.base,
        "sleeps_adopted": reg.adopted_count(),
        "awaiting_review": pending(system),
        "autonomy_level": system.autonomy,
        "memory_store": system.store.stats(),
        "consolidated_protected": len(by_tier["protected"]),
        "consolidated_condensed": len(by_tier["condensed"]),
        "consolidated_faded": len(by_tier["faded"]),
        "latest_dream_report": str(_last_report(system) or ""),
    }, indent=2))
    return 0


def cmd_selftest(args) -> int:
    from .sim.experiment import run_all
    run_all(days=args.days, out=args.out, dim=args.dim, transfer_dim=args.transfer_dim)
    print(f"results: {Path(args.out) / 'results.md'}")
    return 0


def cmd_doctor(args) -> int:
    import importlib
    import sqlite3
    import urllib.request
    ok = True

    def check(name, fn):
        nonlocal ok
        try:
            detail = fn()
            print(f"  OK    {name}{': ' + detail if detail else ''}")
        except Exception as e:  # report every problem, do not stop at the first
            ok = False
            print(f"  FAIL  {name}: {e}")

    print(f"python {sys.version.split()[0]}")
    for mod in ("numpy", "matplotlib"):
        check(mod, lambda m=mod: importlib.import_module(m).__version__)

    def fts5():
        sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        return f"sqlite {sqlite3.sqlite_version}"
    check("sqlite fts5", fts5)

    home = Path(args.home).expanduser()
    cfg = Config.load(home / "config.json", home=str(home))
    print(f"backend: {cfg.backend}, trainer: {cfg.trainer}")
    if cfg.backend != "toy":
        def get(path):
            with urllib.request.urlopen(cfg.llamacpp_url.rstrip("/") + path, timeout=5) as r:
                return r.read().decode("utf-8", "replace")
        def models():
            ids = [m["id"] for m in json.loads(get("/v1/models"))["data"]]
            for want in filter(None, {cfg.student_model, cfg.teacher_model}):
                if want not in ids:
                    raise RuntimeError(f"model {want!r} not offered by the server; it has: {', '.join(ids)}")
            return ", ".join(ids)
        check(f"server {cfg.llamacpp_url} /v1/models", models)

        def adapters():
            try:
                return f"{len(json.loads(get('/lora-adapters')))} adapters loaded (llama-server)"
            except urllib.error.HTTPError:
                return "no adapter hot-swap (LM Studio?): wake runs the base model, sleep still works"
        import urllib.error
        check("adapter hot-swap", adapters)
        if cfg.teacher_url and cfg.teacher_url != cfg.llamacpp_url:
            check(f"teacher {cfg.teacher_url} /v1/models",
                  lambda: ", ".join(m["id"] for m in json.loads(urllib.request.urlopen(
                      cfg.teacher_url.rstrip("/") + "/v1/models", timeout=5).read())["data"]))
    if cfg.trainer in ("peft", "experts"):
        for mod in ("torch", "peft", "transformers"):
            check(mod, lambda m=mod: importlib.import_module(m).__version__)

        def gpu():
            import torch
            if not torch.cuda.is_available():  # ROCm builds also report through torch.cuda
                raise RuntimeError("no GPU visible to PyTorch")
            return torch.cuda.get_device_name(0)
        check("GPU", gpu)
    return 0 if ok else 1


def cmd_daemon(args) -> int:
    from .sleep.scheduler import run_daemon
    run_daemon(_system(args), poll_seconds=args.poll)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sleepai", description="Sleep-learning AI")
    p.add_argument("--home", default="~/.sleep-ai", help="data folder (default ~/.sleep-ai)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("chat").set_defaults(fn=cmd_chat)
    t = sub.add_parser("tell")
    t.add_argument("text", nargs="+")
    t.set_defaults(fn=cmd_tell)
    sub.add_parser("sleep").set_defaults(fn=cmd_sleep)
    r = sub.add_parser("review")
    r.add_argument("action", nargs="?", choices=["approve", "reject", "reject-memories", "resolve", "rollback"])
    r.add_argument("ids", nargs="*")
    r.add_argument("--reason", default="rejected at morning review")
    r.set_defaults(fn=cmd_review)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    s = sub.add_parser("selftest")
    s.add_argument("--days", type=int, default=30)
    s.add_argument("--dim", type=int, default=4096)
    s.add_argument("--transfer-dim", type=int, default=16384)
    s.add_argument("--out", default="results")
    s.set_defaults(fn=cmd_selftest)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    d = sub.add_parser("daemon")
    d.add_argument("--poll", type=int, default=300, help="seconds between idle checks")
    d.set_defaults(fn=cmd_daemon)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
