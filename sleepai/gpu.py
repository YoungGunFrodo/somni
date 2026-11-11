"""GPU turn-taking: the teacher and the student never hold the GPU at the same time.

User's idea (after a real night ran out of VRAM with the teacher's llama-server holding ~5 GB
the whole night): load the teacher, pause the student; load the student, pause the teacher.
Each side registers a release function; `acquire(name)` releases every OTHER holder first.
The sleep pipeline is batched into teacher phases and student phases, so a night switches a
handful of times (seconds each) instead of per question.
"""
from __future__ import annotations

import time

_release: dict[str, callable] = {}
_owner: str | None = None
switches: list[tuple[float, str]] = []


def register(name: str, release) -> None:
    _release[name] = release


def acquire(name: str) -> None:
    global _owner
    if _owner == name:
        return
    for other, fn in list(_release.items()):
        if other != name:
            fn()
    if _owner is not None:
        switches.append((time.time(), name))
        print(f"[gpu] {name} takes the GPU (switch {len(switches)})", flush=True)
    _owner = name


def owner() -> str | None:
    return _owner
