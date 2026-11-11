"""A 3-day run of the synthetic benchmark: the whole pipeline end to end, in seconds."""
import unittest

from sleepai.sim.experiment import run_condition
from tests.helpers import TempDirCase


class SimSmokeTest(TempDirCase):
    def test_three_days_full_sleep_with_forgetting(self):
        opts = dict(replay=True, dreaming=True, gated=True, forgetting=True)
        r = run_condition("smoke", opts, days=3, seed=0, root=self.tmp, dim=1024)
        self.assertEqual(len(r["timeline"]), 3)
        self.assertEqual(len(r["sleeps"]), 3)
        last = r["timeline"][-1]
        self.assertEqual(last["with_memory"], 1.0)            # retrieval answers everything told
        self.assertGreater(last["weights_all"], 0.5)          # something was consolidated
        self.assertGreaterEqual(sum(s["adopted"] for s in r["sleeps"]), 1)
        self.assertGreaterEqual(sum(s["lint_rejected"] for s in r["sleeps"]), 3)  # one injection a day
        self.assertTrue((self.tmp / "smoke" / "sleeps" / "sleep-0002" / "report.md").exists())

    def test_retrieval_only_never_trains(self):
        r = run_condition("retrieval-only", dict(train=False), days=2, seed=0, root=self.tmp, dim=512)
        self.assertEqual(r["sleeps"], [])
        self.assertEqual(r["timeline"][-1]["with_memory"], 1.0)


if __name__ == "__main__":
    unittest.main()
