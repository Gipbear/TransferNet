from __future__ import annotations

import unittest

from kgqa.retrieve.cwq_main_relation import (
    MainRelationExample,
    PathEvidence,
    predict_relation_signatures,
    supported_names,
)


class TestCwqMainRelation(unittest.TestCase):
    def test_predicts_relation_signature_from_similar_training_question(self):
        examples = [
            MainRelationExample("Which university did Russell Wilson attend?", (("education.student",),)),
            MainRelationExample("Which movie featured Taylor Lautner?", (("film.performance",),)),
        ]

        predicted = predict_relation_signatures(examples, ["Which college did Russell Wilson attend?"], 1, 0.0)

        self.assertEqual(predicted, [(("education.student",),)])

    def test_filters_candidate_by_topic_and_relation(self):
        question = "Which university attended by Russell Wilson has the largest enrollment?"
        paths = (
            PathEvidence("Russell Wilson", "Wisconsin", ("education.student",)),
            PathEvidence("College/University", "Oxford", ("generic.type",)),
        )

        supported = supported_names(
            question, ("Russell Wilson", "College/University"), paths, (("education.student",),)
        )

        self.assertEqual(supported, ("Wisconsin",))

    def test_keeps_all_topics_when_question_has_no_topic_overlap(self):
        paths = (
            PathEvidence("John F. Kennedy", "Robert F. Kennedy", ("family.sibling",)),
            PathEvidence("Male", "Other", ("generic.type",)),
        )

        supported = supported_names("Which brother of JFK was earliest?", ("John F. Kennedy", "Male"), paths, None)

        self.assertEqual(supported, ("Robert F. Kennedy", "Other"))


if __name__ == "__main__":
    unittest.main()
