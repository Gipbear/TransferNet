from __future__ import annotations

import unittest

from kgqa.retrieve.cwq_comparison_condition import (
    ComparisonQuery,
    OperatorExample,
    extract_threshold,
    predict_operators,
)


class TestCwqComparisonCondition(unittest.TestCase):
    def test_extracts_threshold_after_comparison_cue(self):
        question = "Who was the leader in 2012 who started the position after May 16, 2007?"

        threshold = extract_threshold(question)

        self.assertEqual(threshold, "2007-05-16")

    def test_extracts_month_year_and_signed_integer(self):
        self.assertEqual(extract_threshold("Which books were published before October 1980?"), "1980-10")
        self.assertEqual(extract_threshold("Which country has migration less than -1,437?"), "-1437")

    def test_abstains_when_multiple_numbers_have_no_comparison_cue(self):
        threshold = extract_threshold("Which of the 2012 and 2013 films was released?")

        self.assertIsNone(threshold)

    def test_abstains_when_two_cues_are_equally_close(self):
        threshold = extract_threshold("Which films were released before 2007 or after 2008?")

        self.assertIsNone(threshold)

    def test_predicts_operator_from_question_only(self):
        training = [
            OperatorExample("Which zoo opened before 1900?", "comparative", "lt"),
            OperatorExample("Which museum opened before 1950?", "comparative", "lt"),
            OperatorExample("Which zoo opened after 1900?", "comparative", "gt"),
            OperatorExample("Which museum opened after 1950?", "comparative", "gt"),
            OperatorExample("Which zoo opened earliest?", "superlative", "argmin"),
            OperatorExample("Which museum opened earliest?", "superlative", "argmin"),
            OperatorExample("Which zoo opened latest?", "superlative", "argmax"),
            OperatorExample("Which museum opened latest?", "superlative", "argmax"),
        ]
        queries = [
            ComparisonQuery("Which site opened before 2000?", "comparative"),
            ComparisonQuery("Which site opened latest?", "superlative"),
        ]

        predictions = predict_operators(training, queries)

        self.assertEqual([item.operator for item in predictions], ["lt", "argmax"])
        self.assertTrue(all(0 < item.confidence <= 1 for item in predictions))


if __name__ == "__main__":
    unittest.main()
