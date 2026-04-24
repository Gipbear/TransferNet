import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.common import (
    build_reverse_entity_map,
    expand_pred_answers_with_path_constraint,
    load_webqsp_qa_samples,
    parse_v2_output,
)


class SimpleAgentCommonTests(unittest.TestCase):
    def test_load_webqsp_qa_samples_parses_topic_and_dedupes_gold(self):
        qa_path = ROOT / "data" / "input" / "WebQSP" / "QA_data" / "WebQuestionsSP" / "qa_test_webqsp_fixed.txt"

        samples = load_webqsp_qa_samples(str(qa_path), limit=1)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].question, "what does jamaican people speak")
        self.assertEqual(samples[0].topic_mid, "m.03_r3")
        self.assertEqual(samples[0].gold_mids, ["m.01428y", "m.04ygk0"])

    def test_reverse_entity_map_and_path_constraint_disambiguate_names(self):
        reverse_map = build_reverse_entity_map(
            {
                "m.010hl7": "Salisbury",
                "m.0jgvy": "Salisbury",
                "m.02": "Other",
            }
        )

        expanded, constrained = expand_pred_answers_with_path_constraint(
            pred_answers=["Salisbury"],
            rev_entity_map=reverse_map,
            path_mid_entities={"m.0jgvy", "m.other"},
        )

        self.assertEqual(expanded, ["m.010hl7", "m.0jgvy"])
        self.assertEqual(constrained, ["m.0jgvy"])

    def test_parse_v2_output_extracts_citations_and_answers(self):
        parsed = parse_v2_output(
            "Supporting Paths: 1, 3\nAnswer: Blethen Maine Newspapers, Inc. | Frederick Stanley"
        )

        self.assertTrue(parsed.format_ok)
        self.assertEqual(parsed.cited_indices, [1, 3])
        self.assertEqual(
            parsed.answers,
            ["Blethen Maine Newspapers, Inc.", "Frederick Stanley"],
        )


if __name__ == "__main__":
    unittest.main()
