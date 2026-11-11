import unittest

from sleepai.config import SleepConfig
from sleepai.memory.salience import CONDENSED, FADED, PROTECTED, protected, salience, split, tier
from tests.helpers import fact


class SalienceTest(unittest.TestCase):
    cfg = SleepConfig(forgetting=True, half_life_days=10, protect_min_salience=1.5, grace_days=7)

    def test_decay_halves_per_half_life(self):
        m = fact("a/b", "1", created_day=0, retrieval_count=3, last_used_day=0)
        self.assertAlmostEqual(salience(m, 0, 10), 4.0)
        self.assertAlmostEqual(salience(m, 10, 10), 2.0)

    def test_never_used_counts_from_creation(self):
        m = fact("a/b", "1", created_day=5)
        self.assertAlmostEqual(salience(m, 15, 10), 0.5)

    def test_correction_bonus(self):
        m = fact("a/b", "1", created_day=0, supersedes="m0")
        self.assertAlmostEqual(salience(m, 10, 10), 1.5)

    def test_grace_then_fading_then_use_protects(self):
        m = fact("a/b", "1", created_day=0)
        self.assertTrue(protected(m, 6, self.cfg))      # grace
        self.assertFalse(protected(m, 7, self.cfg))     # unused -> fading
        m.retrieval_count, m.last_used_day = 1, 20      # used again
        self.assertTrue(protected(m, 20, self.cfg))

    def test_tiers_condense_before_fading(self):
        cfg = SleepConfig()   # defaults: half-life 30, grace 7, protect 1.5, forget below 0.1
        m = fact("a/b", "1", created_day=0)
        self.assertEqual(tier(m, 6, cfg), PROTECTED)     # grace
        self.assertEqual(tier(m, 7, cfg), CONDENSED)     # unused, but a light trace stays
        self.assertEqual(tier(m, 90, cfg), CONDENSED)    # still there after three months
        self.assertEqual(tier(m, 110, cfg), FADED)       # ~100 days unused: fades
        m.retrieval_count, m.last_used_day = 1, 110      # used again -> protected
        self.assertEqual(tier(m, 110, cfg), PROTECTED)

    def test_each_use_pushes_fading_out(self):
        cfg = SleepConfig()
        m = fact("a/b", "1", created_day=0, retrieval_count=3, last_used_day=0)
        self.assertEqual(tier(m, 150, cfg), CONDENSED)   # 4 * 0.5**5 = 0.125
        self.assertEqual(tier(m, 170, cfg), FADED)

    def test_forgetting_off_protects_everything(self):
        cfg = SleepConfig(forgetting=False)
        m = fact("a/b", "1", created_day=0)
        self.assertTrue(protected(m, 1000, cfg))
        keep, fade = split([m], 1000, cfg)
        self.assertEqual((len(keep), len(fade)), (1, 0))


if __name__ == "__main__":
    unittest.main()
