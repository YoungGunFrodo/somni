"""Skill packs: every pack protected, rehearsal mix follows the user's own episodes."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sleepai import skills
from sleepai.memory.store import MemoryStore
from sleepai.models.base import TrainingExample
from sleepai.rehearsal import sample

ALL = skills.DEFAULT_SKILLS


def replay():
    out = []
    for name, n in (("general", 40), ("reasoning", 20), ("writing", 20), ("coding", 30)):
        out += [TrainingExample(f"{name} q{i}", "a", 1.0, "base", f"skill:{name}") for i in range(n)]
    return out


class SkillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "m.db")

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def say(self, text, day=10):
        self.store.log_episode("conversation", [{"role": "user", "content": text}], day=day)

    def test_no_history_is_even(self):
        w = skills.usage_weights(self.store, 10, ALL)
        self.assertEqual(set(w), set(ALL))
        for v in w.values():
            self.assertAlmostEqual(v, 1 / len(ALL))

    def test_coder_gets_more_code_but_every_pack_keeps_a_floor(self):
        for i in range(20):
            self.say(f"Fix this: ```python\ndef f{i}(x):\n    return x +\n``` Traceback ...")
        self.say("Rewrite this email to sound friendlier.")
        w = skills.usage_weights(self.store, 10, ALL)
        self.assertGreater(w["coding"], 0.5)
        for name in ALL:
            self.assertGreaterEqual(w[name], 0.5 / len(ALL) - 1e-9)      # half of an even share
        self.assertAlmostEqual(sum(w.values()), 1.0)

    def test_writer_gets_more_writing(self):
        for _ in range(10):
            self.say("Please proofread my essay and fix the grammar.")
        w = skills.usage_weights(self.store, 10, ALL)
        self.assertGreater(w["writing"], w["coding"])

    def test_old_episodes_do_not_count(self):
        for _ in range(10):
            self.say("def f(): pass  # import os", day=0)
        w = skills.usage_weights(self.store, 100, ALL, window_days=30)
        self.assertAlmostEqual(w["coding"], 1 / len(ALL))

    def test_sample_follows_weights(self):
        rng = np.random.default_rng(0)
        out = sample(replay(), 40, rng, {"general": 0.1, "reasoning": 0.1, "writing": 0.1, "coding": 0.7})
        n = {k: sum(e.memory_id == f"skill:{k}" for e in out) for k in ALL}
        self.assertEqual(len(out), 40)
        self.assertEqual(n["coding"], 28)
        self.assertEqual(len({id(e) for e in out}), 40)              # no duplicates

    def test_sample_even_without_weights_and_handles_small_packs(self):
        rng = np.random.default_rng(1)
        out = sample(replay(), 100, rng)
        n = {k: sum(e.memory_id == f"skill:{k}" for e in out) for k in ALL}
        self.assertEqual(len(out), 100)
        self.assertEqual(n["reasoning"], 20)                         # capped at what exists
        self.assertEqual(n["writing"], 20)

    def test_toy_replay_unchanged(self):
        toy = [TrainingExample(f"q{i}", "a", 1.0, "base") for i in range(50)]
        a = sample(toy, 10, np.random.default_rng(3))
        idx = np.random.default_rng(3).choice(50, size=10, replace=False)
        self.assertEqual([e.question for e in a], [toy[i].question for i in idx])

    def test_unknown_pack_is_an_error(self):
        with self.assertRaises(ValueError):
            skills.get(["general", "astrology"])


if __name__ == "__main__":
    unittest.main()
