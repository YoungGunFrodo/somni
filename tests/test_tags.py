"""Subject tags (sleep/tagging.py), tag-based recall and implications training data."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sleepai.memory.store import Memory, MemoryStore
from sleepai.sleep import tagging
from sleepai.sleep.consolidation import implication_examples
from sleepai.wake.orchestrator import Orchestrator


def _mem(store, key, summary, value, tags=(), ep=1):
    return store.insert(Memory(key=key, type="fact", summary=summary, value=value, strength="verified",
                               sources=[f"episode:{ep}", "verified"], tags=list(tags)))


class FakeTeacher:
    url = "fake"

    def __init__(self, tags):
        self.tags, self.calls = tags, []

    def source_tags(self, memories, vocab, title=""):
        self.calls.append((title, list(vocab)))
        return self.tags.get(title, ("general", title))

    def implications(self, text, n=30):
        return ["The Normans gave their name to Normandy, a region in France.",
                "Rollo was the first ruler of the Normans in Normandy.",
                "Bananas are a popular fruit eaten worldwide.",          # not in the document
                "short"]


class TagStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")

    def test_any_tag_combination_finds_memory_most_matched_first(self):
        a = _mem(self.store, "normans/origin", "The origin of the Normans is Norse.", "Norse",
                 ["history", "europe", "normans"])
        b = _mem(self.store, "rome/founding", "The founding year of Rome is 753 BC.", "753 BC",
                 ["history", "europe", "rome"])
        got = [m.id for m in self.store.by_tags(["history", "normans"])]
        self.assertEqual(got[0], a.id)
        self.assertIn(b.id, got)                       # partial match still found, ranked lower
        self.assertEqual(self.store.tag_counts()["history"], 2)

    def test_related_needs_two_shared_tags(self):
        a = _mem(self.store, "normans/origin", "x", "Norse", ["history", "europe", "normans"])
        b = _mem(self.store, "normans/leader", "y", "Rollo", ["history", "normans"])
        _mem(self.store, "physics/c", "z", "c", ["science", "history"])
        self.assertEqual([m.id for m in self.store.related(a)], [b.id])

    def test_tags_survive_reload_and_old_databases_migrate(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.db"
            s = MemoryStore(p)
            m = _mem(s, "k/a", "s", "v", ["alpha", "beta"])
            s.db.close()
            s2 = MemoryStore(p)
            self.assertEqual(s2.get(m.id).tags, ["alpha", "beta"])
            s2.db.close()


class TaggingStageTests(unittest.TestCase):
    def test_field_topic_subject_with_topics_reused_and_plurals_folded(self):
        store = MemoryStore(":memory:")
        _mem(store, "normans/origin", "The origin of the Normans is Norse.", "Norse",
             ["history", "norman", "normans-origin"])
        eid = store.log_episode("tool_call", [{"role": "tool", "content": "[document: Normans] " + "x" * 300}])
        m = _mem(store, "rollo/title", "The title of Rollo is Duke.", "Duke", ep=eid)
        t = FakeTeacher({"Normans": ("History", "Normans")})
        res = tagging.tag_untagged(store, t, log=lambda *_: None)
        self.assertEqual(res["tagged"], 1)
        self.assertEqual(store.get(m.id).tags, ["history", "norman", "rollo"])
        self.assertEqual(t.calls[0], ("Normans", ["norman"]))   # topics in use offered for reuse

    def test_question_tags(self):
        vocab = {"history": 3, "norman": 2, "viking-age": 1}
        self.assertEqual(sorted(tagging.question_tags("Who led the Normans in the Viking age?", vocab)),
                         ["norman", "viking-age"])


class TagRecallTests(unittest.TestCase):
    def test_recall_adds_tagged_and_linked_memories_only_when_enabled(self):
        store = MemoryStore(":memory:")
        _mem(store, "normans/origin", "The Normans descend from Norse raiders.", "Norse",
             ["history", "europe", "norman"])
        linked = _mem(store, "rollo/title", "Rollo was crowned Duke.", "Duke",
                      ["history", "europe", "norman"])
        q = "Where did the Normans come from?"
        plain = Orchestrator(store, None)
        self.assertNotIn(linked.id, [m.id for m in plain.recall(q, files=0)])
        tagged = Orchestrator(store, None)
        tagged.use_tags = True
        self.assertIn(linked.id, [m.id for m in tagged.recall(q, files=0)])


class ImplicationTests(unittest.TestCase):
    def test_examples_from_documents_with_invented_lines_dropped_and_cached(self):
        with tempfile.TemporaryDirectory() as d:
            store = MemoryStore(Path(d) / "m.db")
            text = ("[document: Normans] The Normans were the people who gave their name to Normandy, a "
                    "region in France. They descended from Norse raiders led by Rollo, the first ruler. " * 3)
            eid = store.log_episode("tool_call", [{"role": "tool", "content": text}], outcome="success")
            m = _mem(store, "normans/origin", "The origin of the Normans is Norse.", "Norse", ep=eid)
            system = SimpleNamespace(config=SimpleNamespace(backend="real"), teacher=FakeTeacher({}),
                                     store=store, home=Path(d))
            ex = implication_examples(system, [m], per_example=3)
            self.assertEqual(len(ex), 1)
            self.assertIn("Normans", ex[0].question)
            self.assertIn("Rollo", ex[0].answer)
            self.assertNotIn("Bananas", ex[0].answer)
            self.assertEqual(ex[0].memory_id, f"doc:{eid}")
            self.assertIn(str(eid), json.loads((Path(d) / "implications.json").read_text()))
            store.db.close()

    def test_toy_backend_makes_none(self):
        system = SimpleNamespace(config=SimpleNamespace(backend="toy"), teacher=FakeTeacher({}), store=None)
        self.assertEqual(implication_examples(system, []), [])


if __name__ == "__main__":
    unittest.main()
