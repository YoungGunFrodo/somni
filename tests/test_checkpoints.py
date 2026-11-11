import unittest

from sleepai.checkpoints import CheckpointRegistry
from tests.helpers import TempDirCase


class CheckpointTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.reg = CheckpointRegistry(self.tmp / "checkpoints")
        self.base = self.reg.new_id()
        self.reg.register_base(self.base)

    def test_adopt_and_rollback(self):
        c1 = self.reg.new_id()
        self.reg.add_candidate(c1, {"status": "awaiting review"})
        self.reg.adopt(c1)
        self.assertEqual(self.reg.active, c1)
        self.assertEqual(self.reg.rollback(), self.base)
        self.assertEqual(self.reg.data["candidates"][c1]["status"], "rolled-back")

    def test_cannot_roll_back_past_base(self):
        with self.assertRaises(ValueError):
            self.reg.rollback()

    def test_only_candidates_can_be_adopted(self):
        with self.assertRaises(ValueError):
            self.reg.adopt("ckpt-9999")

    def test_registry_persists(self):
        c1 = self.reg.new_id()
        self.reg.add_candidate(c1, {})
        self.reg.adopt(c1)
        again = CheckpointRegistry(self.tmp / "checkpoints")
        self.assertEqual((again.active, again.base), (c1, self.base))


if __name__ == "__main__":
    unittest.main()
