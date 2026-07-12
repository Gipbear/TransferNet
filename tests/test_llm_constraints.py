"""Tests for reject-list constrained decoding helpers."""

import unittest

from kgqa.llm_server.constraints import (
    build_reject_list_prefix_fn,
    reject_list_complete,
    valid_reject_list_prefix,
)


class ValidPrefixTests(unittest.TestCase):
    def test_empty_is_valid_prefix_but_not_complete(self):
        self.assertTrue(valid_reject_list_prefix("", 5))
        self.assertFalse(reject_list_complete("", 5))

    def test_none_prefixes(self):
        for text in ["N", "NO", "NON", "NONE"]:
            self.assertTrue(valid_reject_list_prefix(text, 5), text)
        self.assertTrue(reject_list_complete("NONE", 5))
        self.assertFalse(reject_list_complete("NON", 5))
        self.assertFalse(valid_reject_list_prefix("NONEE", 5))
        self.assertFalse(valid_reject_list_prefix("NA", 5))

    def test_single_and_multi_numbers(self):
        self.assertTrue(valid_reject_list_prefix("3", 5))
        self.assertTrue(reject_list_complete("3", 5))
        self.assertTrue(valid_reject_list_prefix("1,2,5", 5))
        self.assertTrue(reject_list_complete("1,2,5", 5))
        self.assertTrue(valid_reject_list_prefix("1,", 5))
        self.assertFalse(reject_list_complete("1,", 5))

    def test_out_of_range_rejected(self):
        self.assertFalse(valid_reject_list_prefix("6", 5))
        self.assertFalse(valid_reject_list_prefix("1,7", 5))
        # 部分数字超界后无法通过追加位数变合法
        self.assertTrue(valid_reject_list_prefix("1", 12))
        self.assertFalse(valid_reject_list_prefix("13", 12))

    def test_leading_zero_and_zero_rejected(self):
        self.assertFalse(valid_reject_list_prefix("0", 5))
        self.assertFalse(valid_reject_list_prefix("01", 15))

    def test_duplicates_rejected(self):
        self.assertFalse(valid_reject_list_prefix("2,2", 5))
        self.assertFalse(valid_reject_list_prefix("1,2,1", 5))

    def test_malformed_rejected(self):
        self.assertFalse(valid_reject_list_prefix(",1", 5))
        self.assertFalse(valid_reject_list_prefix("1,,2", 5))
        self.assertFalse(valid_reject_list_prefix("1 2", 5))
        self.assertFalse(valid_reject_list_prefix("a", 5))


class _FakeTokenizer:
    """Maps single characters and NONE to fixed ids."""

    eos_token_id = 99

    _id_to_text = {i: str(i) for i in range(10)}
    _id_to_text[10] = ","
    _id_to_text[11] = "NONE"

    def encode(self, text, add_special_tokens=False):
        if text == "NONE":
            return [11]
        if text == ",":
            return [10]
        return [int(text)]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._id_to_text.get(int(i), "") for i in ids)


class PrefixFnTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _FakeTokenizer()

    def fn(self, max_index, generated_ids):
        prefix_fn = build_reject_list_prefix_fn(self.tokenizer, 2, max_index)
        return prefix_fn(0, [0, 0] + generated_ids)

    def test_first_step_allows_indices_and_none_only(self):
        allowed = self.fn(3, [])
        self.assertEqual(sorted(allowed), [1, 2, 3, 11])

    def test_after_number_allows_comma_and_eos(self):
        allowed = self.fn(3, [2])
        self.assertEqual(sorted(allowed), [10, 99])

    def test_after_comma_excludes_used_number(self):
        allowed = self.fn(3, [2, 10])
        self.assertEqual(sorted(allowed), [1, 3])

    def test_after_none_only_eos(self):
        allowed = self.fn(3, [11])
        self.assertEqual(allowed, [99])

    def test_two_digit_index(self):
        # max=12 时首 token 允许 1..9（1 可扩展成 10..12）
        allowed = self.fn(12, [])
        self.assertEqual(sorted(allowed), [1, 2, 3, 4, 5, 6, 7, 8, 9, 11])
        # "1" 之后允许 0/1/2（→10/11/12）、逗号、EOS
        allowed = self.fn(12, [1])
        self.assertEqual(sorted(allowed), [0, 1, 2, 10, 99])

    def test_all_numbers_used_only_eos(self):
        allowed = self.fn(2, [1, 10, 2])
        self.assertEqual(sorted(allowed), [99])


if __name__ == "__main__":
    unittest.main()
