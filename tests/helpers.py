"""Shared helpers: small, fast systems in temporary folders."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from sleepai.config import Config, SleepConfig
from sleepai.memory.store import Memory, MemoryStore


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sleepai-test-"))

    def tearDown(self):
        for attr in ("store", "system"):
            obj = getattr(self, attr, None)
            store = getattr(obj, "store", obj)
            if isinstance(store, MemoryStore):
                store.db.close()   # Windows cannot delete an open SQLite file
        shutil.rmtree(self.tmp, ignore_errors=True)


def fact(key, value, strength="user-confirmed", **kw) -> Memory:
    entity, attr = key.split("/")
    return Memory(key=key, type="fact", summary=f"The {attr} of {entity.replace('-', ' ')} is {value}.",
                  value=value, strength=strength, sources=["test"], **kw)


def toy_system(home: Path, dim: int = 512, **sleep):
    from sleepai.sim.world import World
    from sleepai.system import build_toy_system
    world = World(seed=0, base_facts=20, personal_facts=80)
    cfg = Config(home=str(home), sleep=SleepConfig(**sleep))
    return build_toy_system(cfg, world.base_examples(), world.base_probes(), world.unknown_probes(), dim=dim), world
