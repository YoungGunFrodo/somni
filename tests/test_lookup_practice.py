"""Lookup practice: some training questions carry what wake's recall retrieves, and the answer
cites the memory, so reading memory stays a trained skill."""
from sleepai.sleep.consolidation import lookup_examples
from tests.helpers import TempDirCase, fact, toy_system


class LookupPracticeTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.system, _ = toy_system(self.tmp)
        self.m = self.system.store.insert(fact("billing-api/port", "8443"))
        self.system.store.insert(fact("auth-api/port", "9001"))

    def test_context_is_what_recall_returns_and_answer_cites(self):
        exs = lookup_examples(self.system, self.m, ["What is the port of billing-api?"], 1.0, "new")
        self.assertEqual(len(exs), 1)
        ex = exs[0]
        self.assertTrue(ex.context)
        self.assertIn(self.m.id, [c.id for c in ex.context])
        self.assertEqual(ex.answer, f"8443\n[memory: {self.m.key}]")

    def test_missed_recall_answers_from_weights_uncited(self):
        exs = lookup_examples(self.system, self.m, ["zzz qqq?"], 1.0, "new")
        self.assertEqual(exs[0].answer, "8443")

    def test_off(self):
        self.system.config.sleep.lookup_practice = 0
        self.assertEqual(lookup_examples(self.system, self.m, ["What is the port?"], 1.0, "new"), [])
