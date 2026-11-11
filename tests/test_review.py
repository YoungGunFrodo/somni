import unittest

from sleepai.review.morning import approve, pending, reject_memories, resolve_dispute, rollback
from sleepai.sleep.pipeline import run_sleep
from tests.helpers import TempDirCase, fact, toy_system


class ReviewTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.system, _ = toy_system(self.tmp / "home")
        self.store = self.system.store

    def dispute(self):
        old = self.store.insert(fact("server-orion/port", "8042", strength="verified"))
        new = fact("server-orion/port", "9000", status="disputed",
                   rejection_reason=f"weaker claim conflicts with {old.id}")
        return old, self.store.insert(new)

    def test_accept_dispute_supersedes(self):
        old, new = self.dispute()
        self.assertEqual(resolve_dispute(self.system, new.id, accept=True), new.id)
        self.assertEqual(self.store.active_by_key("server-orion/port").value, "9000")
        self.assertEqual(self.store.get(old.id).status, "superseded")
        self.assertEqual(self.store.get(new.id).supersedes, old.id)

    def test_reject_dispute_keeps_old(self):
        old, new = self.dispute()
        self.assertEqual(resolve_dispute(self.system, new.id, accept=False), old.id)
        got = self.store.get(new.id)
        self.assertEqual(got.status, "disputed")
        self.assertTrue(got.never_consolidate)
        self.assertTrue(got.rejection_reason.startswith("rejected at review"))

    def test_resolve_requires_a_dispute(self):
        old, _ = self.dispute()
        with self.assertRaises(ValueError):
            resolve_dispute(self.system, old.id, accept=True)

    def test_sleep_approve_rollback(self):
        orch = self.system.orchestrator()
        orch.tell("The port of server orion is 8042.")
        orch.tell("The owner of server orion is alice.")
        result = run_sleep(self.system, day=0)
        self.assertEqual(pending(self.system), [result["candidate"]])
        approve(self.system)
        self.assertEqual(self.system.registry.active, result["candidate"])
        self.assertTrue(all(self.store.get(m).consolidated_in for m in result["consolidation"]["new_memories"]))
        rollback(self.system)
        self.assertEqual(self.system.registry.active, self.system.registry.base)
        self.assertFalse(any(self.store.get(m).consolidated_in for m in result["consolidation"]["new_memories"]))

    def test_reject_memories_rejects_candidate(self):
        orch = self.system.orchestrator()
        orch.tell("The port of server orion is 8042.")
        result = run_sleep(self.system, day=0)
        mid = result["formation"]["created"][0]
        reject_memories(self.system, [mid], "wrong")
        self.assertEqual(pending(self.system), [])
        self.assertFalse(self.store.get(mid).eligible)


if __name__ == "__main__":
    unittest.main()
