"""When to sleep: the machine is idle, the GPU is free, and there is something to learn.
Wake always wins: while the file `<home>/WAKE` exists, a running sleep stops at the next
stage boundary (formed memories are kept; training is redone next sleep)."""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path


def should_sleep(now: dt.datetime, last_activity: dt.datetime, gpu_free: bool,
                 unformed_episodes: int, cfg) -> tuple[bool, str]:
    if not gpu_free:
        return False, "GPU busy"
    if unformed_episodes < cfg.min_new_episodes:
        return False, f"only {unformed_episodes} new episodes"
    in_window = cfg.nightly_start_hour <= now.hour < cfg.nightly_end_hour
    idle = (now - last_activity).total_seconds() >= cfg.idle_minutes * 60
    if in_window or idle:
        return True, "nightly window" if in_window else "idle"
    return False, "user active"


def today() -> int:
    """The day number used for episodes, memories and salience on a real machine (the sim
    counts days from 0 instead). Any monotonic day count works; differences are what matter."""
    return dt.date.today().toordinal()


def wake_requested(home: Path) -> bool:
    return (Path(home) / "WAKE").exists()


def run_daemon(system, gpu_free=lambda: True, poll_seconds: int = 300) -> None:  # pragma: no cover
    """Long-running loop for your PC. Last activity = newest episode or WAKE file time."""
    from .pipeline import run_sleep
    wake = system.home / "WAKE"
    while True:
        last = system.store.db.execute("SELECT MAX(created_at) FROM episodes").fetchone()[0] or 0
        if wake.exists():
            last = max(last, wake.stat().st_mtime)
        ok, why = should_sleep(dt.datetime.now(), dt.datetime.fromtimestamp(last), gpu_free(),
                               system.store.count_unformed(), system.config.sleep)
        if ok:
            print(f"[sleep] starting ({why})")
            result = run_sleep(system, day=today())
            print(f"[sleep] done: {result['status']} — report at {result['report']}")
        time.sleep(poll_seconds)
