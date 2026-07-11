"""pfit.train 数据整形纯函数:智能截断保金路径 + prompt masking(免 GPU)。

legacy(train_sft.py)对应逻辑是嵌套闭包不可导入,此处用假 tokenizer
做行为等价断言:预算内不动、超长丢 distractor 保 golden、全 -100 兜底。
"""
import unittest


class FakeTokenizer:
    """确定性空白分词;apply_chat_template 用角色标记拼接。"""
    unk_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.vocab = {}

    def _ids(self, text):
        out = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = 10 + len(self.vocab)
            out.append(self.vocab[w])
        return out

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._ids(text)}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        parts = [f"[{m['role']}] {m['content']}" for m in messages]
        tail = " [assistant]" if add_generation_prompt else ""
        return "\n".join(parts) + tail

    def convert_tokens_to_ids(self, token):
        return 2 if token == "<|eot_id|>" else self.unk_token_id

    def decode_words(self, ids):
        rev = {v: k for k, v in self.vocab.items()}
        return [rev.get(i, "?") for i in ids]


def _rec(user_lines, assistant="Answer: GOLDTAIL", golden_indices=(1,)):
    user = "Question: q\n\nReasoning Paths:\n" + "\n".join(user_lines)
    return {
        "messages": [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "_meta": {"golden_path_indices": list(golden_indices)},
    }


class TestTokenizeRecord(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()

    def test_within_budget_masks_prompt_only(self):
        from kgqa.pfit.train import tokenize_record
        rec = _rec(["1: A -> r -> GOLDTAIL"])
        out = tokenize_record(rec, self.tok, max_seq_len=256)
        ids, labels = out["input_ids"], out["labels"]
        self.assertEqual(len(ids), len(labels))
        self.assertEqual(ids[-1], 2)                      # 末尾 <|eot_id|>
        asst_len = len(self.tok("Answer: GOLDTAIL")["input_ids"]) + 1
        self.assertTrue(all(l == -100 for l in labels[:-asst_len]))
        self.assertEqual(labels[-asst_len:], ids[-asst_len:])  # assistant 段监督
        self.assertFalse(out["truncated"])
        self.assertFalse(out["label_fallback"])

    def test_overlong_drops_distractors_keeps_golden(self):
        from kgqa.pfit.train import tokenize_record
        lines = ["1: TOPIC -> rel -> GOLDTAIL"] + [
            f"{i}: TOPIC -> rel{i} -> DIST{i} w{i}a w{i}b w{i}c w{i}d w{i}e"
            for i in range(2, 12)
        ]
        rec = _rec(lines)
        out = tokenize_record(rec, self.tok, max_seq_len=48)
        words = set(self.tok.decode_words(out["input_ids"]))
        self.assertTrue(out["truncated"])
        self.assertIn("GOLDTAIL", words)                  # golden 路径保留
        dropped = [w for w in words if w.startswith("DIST")]
        self.assertLess(len(dropped), 10)                 # 有 distractor 被丢弃
        self.assertLessEqual(len(out["input_ids"]), 48)

    def test_no_droppable_lines_token_truncates_keeps_assistant(self):
        from kgqa.pfit.train import tokenize_record
        # 无 distractor 可丢且 prompt 远超预算 → token 级截断,assistant 段仍被监督
        long_q = " ".join(f"q{i}" for i in range(200))
        rec = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"Question: {long_q}"},
                {"role": "assistant", "content": "Answer: X"},
            ],
            "_meta": {"golden_path_indices": []},
        }
        out = tokenize_record(rec, self.tok, max_seq_len=32)
        self.assertTrue(out["truncated"])
        self.assertLessEqual(len(out["input_ids"]), 32)
        asst_len = len(self.tok("Answer: X")["input_ids"]) + 1
        self.assertEqual(out["labels"][-asst_len:], out["input_ids"][-asst_len:])
        self.assertTrue(all(l == -100 for l in out["labels"][:-asst_len]))


if __name__ == "__main__":
    unittest.main()
