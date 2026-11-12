import unittest

from sleepai.sleep.consolidation import _echo


class EchoFilterTest(unittest.TestCase):
    def test_answer_inside_question_is_echo(self):
        self.assertTrue(_echo("What is the English spelling of the river Rhine?", "Rhine"))
        self.assertTrue(_echo("Did Tesla's 1906 demo reach 16,000 rpm?", "16,000 rpm"))

    def test_real_answers_kept(self):
        self.assertFalse(_echo("Which ocean is the largest?", "the Pacific Ocean"))
        self.assertFalse(_echo("What did settlers introduce?", "a hut tax"))
        self.assertFalse(_echo("Anything?", ""))


if __name__ == "__main__":
    unittest.main()
