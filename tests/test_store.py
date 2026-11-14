import sqlite3
import sqlite3 as _sq
import unittest

from sleepai.memory.store import SCHEMA, MemoryStore
from tests.helpers import TempDirCase, fact


class StoreTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.store = MemoryStore(self.tmp / "memory.db")

    def test_one_active_memory_per_key(self):
        self.store.insert(fact("server-orion/port", "8042"))
        with self.assertRaises(_sq.IntegrityError):
            self.store.insert(fact("server-orion/port", "9000"))

    def test_supersede_archives_old(self):
        old = self.store.insert(fact("server-orion/port", "8042"))
        new = self.store.supersede(old.id, fact("server-orion/port", "9000"))
        self.assertEqual(self.store.get(old.id).status, "superseded")
        self.assertEqual(self.store.get(old.id).superseded_by, new.id)
        self.assertEqual(self.store.active_by_key("server-orion/port").value, "9000")
        self.assertEqual(new.supersedes, old.id)

    def test_search_finds_but_does_not_count_as_use(self):
        m = self.store.insert(fact("server-orion/port", "8042"))
        found = self.store.search("What is the port of server orion?")
        self.assertEqual([x.id for x in found], [m.id])
        self.assertEqual(self.store.get(m.id).retrieval_count, 0)

    def test_touch_counts_use_and_day(self):
        m = self.store.insert(fact("server-orion/port", "8042", created_day=2))
        self.store.touch([m.id], 5)
        self.store.touch([m.id], 4)   # an older day never moves last_used_day back
        got = self.store.get(m.id)
        self.assertEqual((got.retrieval_count, got.last_used_day), (2, 5))

    def test_migrates_old_database(self):
        path = self.tmp / "old.db"
        old_schema = SCHEMA.split(",\n    last_used_day")[0] + "\n);" + SCHEMA.split("broad to specific\n);")[1]
        self.assertNotIn("last_used_day", old_schema)
        con = sqlite3.connect(path)
        con.executescript(old_schema)
        con.execute("INSERT INTO memories(id,key,type,summary,sources,strength,created_at) "
                    "VALUES ('m1','a/b','fact','s','[]','verified',0)")
        con.commit()
        con.close()
        store = MemoryStore(path)
        try:
            self.assertEqual(store.get("m1").last_used_day, -1)
            self.assertEqual(store.get("m1").tags, [])
        finally:
            store.db.close()

    def test_purge_marks_formed_memories(self):
        eid = self.store.log_episode("conversation", [{"role": "user", "content": "x"}])
        m = self.store.insert(fact("server-orion/port", "8042"))
        self.store.mark_formed(eid, [m.id])
        self.store.purge_episodes([eid])
        got = self.store.get(m.id)
        self.assertTrue(got.never_consolidate)
        self.assertEqual(got.status, "disputed")
        self.assertEqual(self.store.stats()["episodes"], 0)


if __name__ == "__main__":
    unittest.main()
