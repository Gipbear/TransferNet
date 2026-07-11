"""pfit.formats 与 llm_infer.kg_format 的文本级 parity(schema 系已废弃,不在范围)。"""
import unittest

from llm_infer import kg_format as legacy


_EDGES = [
    ["m.019nnl", "tv.tv_program.regular_cast", "m.05st1mv"],
    ["m.05st1mv", "tv.tv_actor.starring_roles_reverse", "m.03jldb"],
]
_EMAP = {"m.019nnl": "Family Guy", "m.03jldb": "Seth MacFarlane"}
_PATHS_WITH_META = [(_EDGES, -4.782481, 1), ([_EDGES[0]], -0.5, 2)]
_QUESTION = "who plays peter in family guy"

_PFIT_FORMATS = ["arrow", "nl", "tuple", "chain"]
_OUTPUT_FORMATS = ["v0", "v1", "v2", "v3", "v4", "v11"]


class TestFormatParity(unittest.TestCase):
    def test_path_formats_char_identical(self):
        from kgqa.pfit import formats as pfit
        for path_format in _PFIT_FORMATS:
            for show_score in (False, True):
                for emap in (None, _EMAP):
                    with self.subTest(path_format=path_format,
                                      show_score=show_score, mapped=emap is not None):
                        self.assertEqual(
                            pfit.build_user_content(
                                _PATHS_WITH_META, _QUESTION, show_score=show_score,
                                path_format=path_format, entity_map=emap),
                            legacy.build_user_content(
                                _PATHS_WITH_META, _QUESTION, show_score=show_score,
                                path_format=path_format, entity_map=emap),
                        )

    def test_schema_formats_removed(self):
        from kgqa.pfit import formats as pfit
        for dead in ("schema", "schema_gloss"):
            with self.assertRaises(ValueError):
                pfit.build_user_content(_PATHS_WITH_META, _QUESTION, path_format=dead)

    def test_system_prompts_char_identical(self):
        from kgqa.pfit import formats as pfit
        for fmt in _OUTPUT_FORMATS:
            for use_names in (False, True):
                with self.subTest(fmt=fmt, use_names=use_names):
                    self.assertEqual(
                        pfit.select_format_prompt(fmt, use_entity_names=use_names),
                        legacy.select_format_prompt(fmt, use_entity_names=use_names),
                    )
        # 拒答变体(Group F 功能保留)
        for use_names in (False, True):
            self.assertEqual(
                pfit.select_format_prompt("v2", use_entity_names=use_names,
                                          reject_prompt=True),
                legacy.select_format_prompt("v2", use_entity_names=use_names,
                                            reject_prompt=True),
            )

    def test_no_paths_parity(self):
        from kgqa.pfit import formats as pfit
        self.assertEqual(pfit.build_user_content_no_paths(_QUESTION),
                         legacy.build_user_content_no_paths(_QUESTION))
        self.assertEqual(pfit.FORMAT_PROMPTS["no_paths"],
                         legacy.FORMAT_PROMPTS["no_paths"])

    def test_entity_map_helpers_parity(self):
        from kgqa.pfit import formats as pfit
        self.assertEqual(pfit.apply_entity_map(_EDGES, _EMAP),
                         legacy.apply_entity_map(_EDGES, _EMAP))
        self.assertEqual(pfit.map_answers(["m.03jldb", "m.unknown"], _EMAP),
                         legacy.map_answers(["m.03jldb", "m.unknown"], _EMAP))
        self.assertEqual(pfit.build_reverse_entity_map(_EMAP),
                         legacy.build_reverse_entity_map(_EMAP))


if __name__ == "__main__":
    unittest.main()
