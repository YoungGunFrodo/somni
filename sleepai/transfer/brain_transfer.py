"""Brain transfer: move everything learned into a new base model, from the memory store.

Adapters do not transfer between base models; memories do. The new brain starts from its
own base checkpoint, every eligible memory is marked unconsolidated, and one sleep with a
forced rebuild consolidates the whole store in full (condensed and faded memories included:
condensing is for the old brain's capacity), ordered by importance. The usual gates decide
whether the new brain is adopted.
"""
from __future__ import annotations

import shutil

from ..sleep.pipeline import run_sleep


def brain_transfer(old_system, new_system, day: int = 0) -> dict:
    old_system.store.db.commit()
    new_system.store.db.close()
    shutil.copy(old_system.home / "memory.db", new_system.home / "memory.db")
    from ..memory.store import MemoryStore
    new_system.store = MemoryStore(new_system.home / "memory.db")
    for m in new_system.store.all():
        new_system.store.update_fields(m.id, consolidated_in=[], last_probe_result="")
    return run_sleep(new_system, day=day, force_rebuild=True, full_store=True)
