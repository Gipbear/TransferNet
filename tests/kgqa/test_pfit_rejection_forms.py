"""拒答表达的识别测试。

原实现只认字面 (none)。RoG-CWQ 拒答组 3531 个样本的实测分布:
None 231 / (no answer) 70 / (no answer provided) 6 / (none) 4,
共 311 个拒答里只认出 4 个,其余 307 个被当成幻觉实体计入指标。

放宽必须走「完整匹配」而非前缀匹配:用 n/?a 这类宽松模式会把 Navy Blue、
National Museum of Fine Arts、Nanny and the Professor 一并误判为拒答。
"""
import unittest

from kgqa.pfit.eval import REJECTION_SENTINEL, is_rejection_response, is_rejection_text


class TestIsRejectionText(unittest.TestCase):
    def test_forms_observed_in_real_output(self):
        for text in ["None", "(no answer)", "(no answer provided)", "(none)"]:
            with self.subTest(text=text):
                self.assertTrue(is_rejection_text(text))

    def test_common_variants(self):
        for text in ["none", "NONE", "null", "nil", "N/A", "n/a", "unknown",
                     "unanswerable", "(empty)", "no answer found", "cannot answer",
                     "unable to answer"]:
            with self.subTest(text=text):
                self.assertTrue(is_rejection_text(text))

    def test_entity_names_are_not_rejections(self):
        """防前缀匹配回归:这些都是实测输出里的真实实体名。"""
        for text in ["Navy Blue", "National Museum of Fine Arts, Malta",
                     "Nanny and the Professor", "Nulla in mundo pax sincera",
                     "Naturhistorisches Museum", "National Computing Centre",
                     "Nine Inch Nails", "Nonesuch Records", "Nullarbor Plain"]:
            with self.subTest(text=text):
                self.assertFalse(is_rejection_text(text))

    def test_wrapping_punctuation_tolerated(self):
        for text in ['"none"', "(None).", " none ", "None."]:
            with self.subTest(text=text):
                self.assertTrue(is_rejection_text(text))

    def test_empty_is_not_rejection(self):
        """空答案是格式错误,不是主动拒答,两者在指标上含义不同。"""
        self.assertFalse(is_rejection_text(""))
        self.assertFalse(is_rejection_text("   "))


class TestIsRejectionResponse(unittest.TestCase):
    def test_all_answers_rejection_forms(self):
        self.assertTrue(is_rejection_response({"answers": ["None"]}))
        self.assertTrue(is_rejection_response({"answers": ["(no answer)"]}))

    def test_mixed_answers_not_rejection(self):
        """只要有一个真实答案就不算拒答。"""
        self.assertFalse(is_rejection_response({"answers": ["None", "Barack Obama"]}))

    def test_empty_answers_not_rejection(self):
        self.assertFalse(is_rejection_response({"answers": []}))

    def test_sentinel_still_recognized(self):
        self.assertTrue(is_rejection_response({"answers": [REJECTION_SENTINEL]}))


if __name__ == "__main__":
    unittest.main()
