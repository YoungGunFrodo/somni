"""Second take: a failed night retrains once, leaving out the memories that failed."""
from unittest import mock

from sleepai.sleep import pipeline
from sleepai.sleep.evaluation import EvalResult, Gate
from tests.helpers import TempDirCase, fact, toy_system


class SecondTakeTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.system, _ = toy_system(self.tmp, second_take=True, dreaming=False)
        self.ids = [self.system.store.insert(fact(f"server-n{i}/port", str(8000 + i))).id for i in range(4)]

    def fake_evaluate(self, results):
        calls = []

        def ev(system, current, candidate, learned, rng, day=0):
            calls.append(list(learned))
            ok = results[len(calls) - 1]
            if not ok:     # the first memory fails its probes
                system.store.update_fields(learned[0], last_probe_result="fail")
            r = EvalResult()
            r.gates = [Gate("learning", ok, 10, None, 0.9 if ok else 0.5)]
            r.passed = ok
            return r
        return ev, calls

    def test_learning_failure_retrains_without_the_failed_memory(self):
        ev, calls = self.fake_evaluate([False, True])
        with mock.patch.object(pipeline, "evaluate", ev):
            data = pipeline.run_sleep(self.system, day=0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[0]), 4)
        self.assertNotIn(calls[0][0], calls[1])              # left out
        self.assertEqual(len(calls[1]), 3)
        self.assertTrue(data["second_take"]["tried"])
        self.assertEqual(data["recommendation"], "adopt")
        # the left-out memory is not marked as consolidated in the candidate
        cand = self.system.registry.data["candidates"][data["candidate"]]
        self.assertNotIn(calls[0][0], cand["memories"])

    def test_off_by_default(self):
        self.system.config.sleep.second_take = False
        ev, calls = self.fake_evaluate([False, True])
        with mock.patch.object(pipeline, "evaluate", ev):
            data = pipeline.run_sleep(self.system, day=0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(data["recommendation"], "reject")
