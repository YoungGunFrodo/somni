"""Reading skill pack: invented passages, held-out probes, and probes carry their passage."""
import unittest

from sleepai import reading
from sleepai.models.base import Answer
from sleepai.sleep.evaluation import _score


class ReadingTests(unittest.TestCase):
    def test_deterministic_and_held_out(self):
        self.assertEqual(reading.probes(), reading.probes())
        rehearsed = {r["passage"] for r in reading.rehearsal_items()}
        for p in reading.probes():
            text = p["context"][0]["summary"] if p["context"] else p["question"]
            self.assertFalse(any(r in text for r in rehearsed))

    def test_both_presentations_and_answer_in_passage(self):
        ps = reading.probes()
        self.assertTrue(any(p["context"] for p in ps))
        self.assertTrue(any(not p["context"] and "Here is a document" in p["question"] for p in ps))
        for p in ps:
            text = p["context"][0]["summary"] if p["context"] else p["question"]
            self.assertIn(p["answer"], text)

    def test_probes_are_scored_with_their_passage(self):
        seen = []

        class Reader:
            def answer_many(self, qs, contexts=None, max_new_tokens=None):
                seen.append(contexts)
                out = []
                for q, c in zip(qs, contexts or [None] * len(qs)):
                    text = c[0].summary if c else q
                    out.append(Answer(text, text.lower(), 1.0))
                return out
        ps = reading.probes()
        got = _score(Reader(), ps)
        self.assertTrue(seen[0] and any(seen[0]))
        self.assertGreater(got.mean(), 0.9)      # a reader that sees the passage contains the answer
