"""Real-backend parsing that needs no server."""
import json
import unittest

from sleepai.memory.store import Episode
from sleepai.models.llm import LLMTeacher, cited_memory
from tests.helpers import fact


class CitationTest(unittest.TestCase):
    def setUp(self):
        self.ctx = [fact("server-orion/port", "8042", id="m1"), fact("server-orion/owner", "alice", id="m2")]

    def test_citation_maps_to_memory_id_and_is_stripped(self):
        text, used = cited_memory("It is 8042.\n[memory: server-orion/port]", self.ctx)
        self.assertEqual((text, used), ("It is 8042.", "m1"))

    def test_key_not_in_context_is_ignored(self):
        _, used = cited_memory("It is 8042. [memory: server-vega/port]", self.ctx)
        self.assertEqual(used, "")

    def test_no_citation(self):
        self.assertEqual(cited_memory("I'm not sure.", self.ctx), ("I'm not sure.", ""))


class FakeTeacher(LLMTeacher):
    def __init__(self, reply):
        super().__init__("fake")
        self.reply = reply

    def _chat(self, messages, temperature=0.2, max_tokens=1024):
        return self.reply


def episode(role, outcome="unknown"):
    return Episode(1, 0, "conversation", [{"role": role, "content": "x"}], outcome, "", "", False)


class FormationParsingTest(unittest.TestCase):
    item = {"key": "Server Orion/Port", "type": "fact", "summary": "s", "value": "8042", "strength": "verified"}

    def test_model_cannot_claim_stronger_source_than_evidence(self):
        t = FakeTeacher(json.dumps([self.item]))
        self.assertEqual(t.form_memories(episode("user"))[0].strength, "user-confirmed")
        self.assertEqual(t.form_memories(episode("assistant"))[0].strength, "inferred")
        self.assertEqual(t.form_memories(episode("tool", "success"))[0].strength, "verified")

    def test_malformed_output_is_skipped_not_fatal(self):
        self.assertEqual(FakeTeacher('["just a string", 3]').form_memories(episode("user")), [])
        self.assertEqual(FakeTeacher("no json at all").form_memories(episode("user")), [])
        self.assertEqual(len(FakeTeacher(json.dumps(self.item)).form_memories(episode("user"))), 1)

    def test_json_with_extra_text_or_several_values(self):
        reply = 'Sure! ' + json.dumps([self.item]) + '\n\nAlso: {"note": 1} and more text ]'
        self.assertEqual(len(FakeTeacher(reply).form_memories(episode("user"))), 1)

    def test_key_is_normalized(self):
        self.assertEqual(FakeTeacher(json.dumps([self.item])).form_memories(episode("user"))[0].key,
                         "server-orion/port")


if __name__ == "__main__":
    unittest.main()


class StructuredFormationTest(unittest.TestCase):
    """Real-night finding: a 1.7B teacher wrote broken keys; keys are now built from fields."""

    def form(self, items):
        return FakeTeacher(json.dumps(items)).form_memories(episode("user"))

    def test_key_and_summary_built_from_fields(self):
        m = self.form([{"subject": "Café Rugorlo", "attribute": "Access Code", "value": "9779-nefen",
                        "type": "fact", "summary": "whatever", "strength": "user-confirmed"}])[0]
        self.assertEqual(m.key, "cafe-rugorlo/access-code")
        self.assertEqual(m.summary, "The access code of Café Rugorlo is 9779-nefen.")

    def test_inverted_keeper_fact_gets_a_clean_key(self):
        m = self.form([{"subject": "ship nelumhil", "attribute": "keeper", "value": "Drurukev",
                        "summary": "Drurukev is the keeper of the ship Nelumhil."}])[0]
        self.assertEqual(m.key, "ship-nelumhil/keeper")

    def test_old_key_format_still_accepted(self):
        m = self.form([{"key": "server-orion/port", "summary": "s", "value": "8042"}])[0]
        self.assertEqual(m.key, "server-orion/port")


class QuestionFilterTest(unittest.TestCase):
    def test_partial_answer_leak_is_caught(self):
        from sleepai.models.llm import leaks_answer
        self.assertTrue(leaks_answer("The code made of 9779 and nefen?", "9779-nefen"))
        self.assertFalse(leaks_answer("What is the access code of café rugorlo?", "9779-nefen"))

    def test_question_must_name_the_subject(self):
        from sleepai.models.base import ProposedMemory
        from sleepai.models.llm import about_subject
        m = ProposedMemory("orchard-norhildru/color", "fact", "The color of Orchard Norhildru is ochre.", "ochre", "x")
        self.assertTrue(about_subject("What color is Norhildru's orchard?", m))
        self.assertFalse(about_subject("What color is the cultural symbol?", m))

    def test_fallback_uses_the_memorys_own_words(self):
        from sleepai.models.base import ProposedMemory
        from sleepai.models.llm import fallback_question
        m = ProposedMemory("cafe-rugorlo/access-code", "fact", "The access code of café rugorlo is 9779-nefen.",
                           "9779-nefen", "x")
        self.assertEqual(fallback_question(m), "What is the access code of café rugorlo?")


class DocumentModeTest(unittest.TestCase):
    def test_documents_use_the_every_fact_prompt(self):
        seen = []

        class T(FakeTeacher):
            def _chat(self, messages, temperature=0.2, max_tokens=1024):
                seen.append(messages[0]["content"])
                return json.dumps([{"subject": "Denver Broncos", "attribute": "super bowl 50 result",
                                    "value": "won", "strength": "verified"}])
        ep = Episode(1, 0, "tool_call", [{"role": "tool", "content": "[document: SB50] The Broncos won."}],
                     "success", "", "", False)
        m = T("").form_memories(ep)[0]
        self.assertIn("EVERY distinct fact", seen[0])
        self.assertEqual(m.key, "denver-broncos/super-bowl-50-result")
        self.assertEqual(m.strength, "verified")

    def test_conversations_keep_the_selective_prompt(self):
        seen = []

        class T(FakeTeacher):
            def _chat(self, messages, temperature=0.2, max_tokens=1024):
                seen.append(messages[0]["content"])
                return "[]"
        T("").form_memories(episode("user"))
        self.assertIn("would change future behavior", seen[0])
