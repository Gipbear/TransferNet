import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.path_retrieve_server.service import CachedPathRetriever, normalize_question


def write_tiny_webqsp_input(root: Path):
    fb_dir = root / "fbwq_full"
    fb_dir.mkdir(parents=True)
    (fb_dir / "entities.dict").write_text("m.topic\nm.answer\nm.other\n", encoding="utf-8")
    (fb_dir / "relations.dict").write_text("rel.answer\t0\nrel.answer_reverse\t1\n", encoding="utf-8")
    (fb_dir / "train.txt").write_text("m.topic\trel.answer\tm.answer\n", encoding="utf-8")


def write_tiny_cache(path: Path):
    cache = {
        "version": 1,
        "meta": {
            "dataset": "WebQSP",
            "split": "test",
            "num_samples": 1,
            "num_entities": 3,
            "num_relations": 2,
            "num_steps": 1,
            "topk_entities": 3,
            "input_dir": "tiny",
            "qa_file": "tiny_qa.txt",
            "id2ent": {0: "m.topic", 1: "m.answer", 2: "m.other"},
            "id2rel": {0: "rel.answer", 1: "rel.answer_reverse"},
        },
        "samples": [
            {
                "question": "[CLS] what does jamaican people speak [SEP]",
                "topic_ids": [0],
                "gold_ids": [1],
                "hop_attn": torch.tensor([1.0]),
                "rel_probs": [torch.tensor([0.9, 0.0])],
                "ent_indices": [torch.tensor([1])],
                "ent_scores": [torch.tensor([0.8])],
                "e_score_indices": torch.tensor([1]),
                "e_score_values": torch.tensor([0.95]),
            }
        ],
    }
    torch.save(cache, path)


class CachedPathRetrieverTests(unittest.TestCase):
    def make_retriever(self, tmp: Path) -> CachedPathRetriever:
        input_dir = tmp / "input"
        cache_path = tmp / "cache.pt"
        write_tiny_webqsp_input(input_dir)
        write_tiny_cache(cache_path)
        return CachedPathRetriever(cache_path=str(cache_path), input_dir=str(input_dir))

    def test_normalize_question_matches_plain_and_cached_forms(self):
        self.assertEqual(
            normalize_question("[CLS] what does jamaican people speak [SEP]"),
            "what does jamaican people speak",
        )

    def test_retrieve_by_sample_index_uses_default_search_logic(self):
        with tempfile.TemporaryDirectory() as td:
            retriever = self.make_retriever(Path(td))
            result = retriever.retrieve(sample_index=0)

        self.assertEqual(result.sample_index, 0)
        self.assertEqual(result.topics, ["m.topic"])
        self.assertEqual(result.beam_size, 50)
        self.assertEqual(result.alpha_final, 1.0)
        self.assertEqual(result.lambda_val, 0.2)
        self.assertEqual(
            result.mmr_reason_paths[0]["path"],
            [["m.topic", "rel.answer", "m.answer"]],
        )
        self.assertEqual(result.prediction, {"m.answer": 0.95})

    def test_retrieve_by_plain_question_matches_cached_question(self):
        with tempfile.TemporaryDirectory() as td:
            retriever = self.make_retriever(Path(td))
            result = retriever.retrieve(question="what does jamaican people speak")

        self.assertEqual(result.sample_index, 0)

    def test_unknown_question_raises_key_error(self):
        with tempfile.TemporaryDirectory() as td:
            retriever = self.make_retriever(Path(td))
            with self.assertRaises(KeyError):
                retriever.retrieve(question="not in cache")

    def test_topic_mismatch_raises_value_error(self):
        with tempfile.TemporaryDirectory() as td:
            retriever = self.make_retriever(Path(td))
            with self.assertRaises(ValueError):
                retriever.retrieve(sample_index=0, topic_entities=["m.other"])


if __name__ == "__main__":
    unittest.main()
