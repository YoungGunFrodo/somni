import unittest

import numpy as np

from sleepai.models.base import TrainingExample
from sleepai.sleep.evaluation import evaluate, paired_bootstrap
from tests.helpers import TempDirCase, fact, toy_system


def gate(res, name):
    return next(g for g in res.gates if g.name == name)


class GateTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.system, _ = toy_system(self.tmp / "home", forgetting=False)
        store = self.system.store
        self.mems = []
        for i, port in enumerate(["8001", "8004", "8007", "8010", "8013", "8016", "8019", "8022", "8025", "8028"]):
            m = store.insert(fact(f"server-test{i}/port", port))
            store.mark_consolidated([m.id], "ckpt-x")
            self.mems.append(store.get(m.id))
        self.base = self.system.load_student(self.system.registry.base)
        self.trained = self.system.load_student(self.system.registry.base)
        ex = [TrainingExample(q, m.value) for m in self.mems
              for q in self.system.teacher.training_questions(m, 8)]
        self.trained.fit(ex + list(self.system.base_replay), epochs=30)

    def test_identical_candidate_passes(self):
        res = evaluate(self.system, self.trained, self.trained, [], np.random.default_rng(0))
        self.assertTrue(res.passed, [(g.name, g.passed) for g in res.gates])

    def test_forgetting_candidate_fails_retention(self):
        res = evaluate(self.system, self.trained, self.base, [], np.random.default_rng(0))
        self.assertFalse(gate(res, "retention").passed)
        self.assertFalse(res.passed)

    def test_condensed_memories_reported_not_gated(self):
        self.system.config.sleep.forgetting = True
        # created day 0, never used, today = 30: every memory is condensed
        res = evaluate(self.system, self.trained, self.base, [], np.random.default_rng(0), day=30)
        self.assertNotIn("retention", [g.name for g in res.gates])
        cr = res.info["condensed_recall"]
        self.assertGreater(cr["current"], cr["candidate"])   # the drop is visible in the report

    def test_faded_memories_neither_gated_nor_reported(self):
        self.system.config.sleep.forgetting = True
        res = evaluate(self.system, self.trained, self.base, [], np.random.default_rng(0), day=300)
        self.assertNotIn("retention", [g.name for g in res.gates])
        self.assertNotIn("condensed_recall", res.info)

    def test_learning_gate_uses_absolute_threshold(self):
        m = self.mems[0]
        res = evaluate(self.system, self.base, self.base, [m.id], np.random.default_rng(0))
        self.assertFalse(gate(res, "learning").passed)

    def test_paired_bootstrap_ci_contains_mean(self):
        a = np.array([1, 1, 0, 1, 0, 1] * 10, dtype=bool)
        b = np.array([1, 0, 0, 1, 0, 1] * 10, dtype=bool)
        mean, (lo, hi) = paired_bootstrap(a, b, 500, np.random.default_rng(0))
        self.assertLess(mean, 0)
        self.assertLessEqual(lo, mean)
        self.assertLessEqual(mean, hi)


if __name__ == "__main__":
    unittest.main()


class PhrasingFamiliesTest(unittest.TestCase):
    def test_training_and_gate_phrasings_never_overlap(self):
        from sleepai.phrasings import GATE_STYLES, TRAIN_STYLES
        self.assertFalse(set(GATE_STYLES) & set(TRAIN_STYLES))
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
        from milestone0 import PROBES
        probes = {t.replace("{e}", "{s}") for t in PROBES}
        self.assertFalse(probes & set(GATE_STYLES))
        self.assertFalse(probes & set(TRAIN_STYLES))

    def test_questions_from_canonical_summary(self):
        from sleepai.models.base import ProposedMemory
        from sleepai.phrasings import GATE_STYLES, questions
        m = ProposedMemory("cafe-rugorlo/access-code", "fact", "The access code of café rugorlo is 9779-nefen.",
                           "9779-nefen", "x")
        qs = questions(m, GATE_STYLES)
        self.assertEqual(qs[0], "Which access code does café rugorlo have?")
        self.assertEqual(questions(ProposedMemory("k/x", "fact", "free text", "v", "x"), GATE_STYLES), [])
