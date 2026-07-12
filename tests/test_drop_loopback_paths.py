"""检索层剔除"绕回 topic"的无效路径(尾实体 == topic 本身)。

这类路径(node_ids[-1] == node_ids[0])在 WebQSP test 全集 9777 条(占 13.4%),
**0 条尾是 gold**——答案=被问的实体本身逻辑上不可能成立。源头剔除后 LLM 看不到,
既不会引用、也不会被诱导成自指答案。零损失(无 gold 反例)。
"""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kgqa.server.service import drop_loopback_paths
# legacy 集成对拍:CachedPathRetriever 只读保留,kgqa 侧等价覆盖见 tests/kgqa/test_server_full.py
from oh_my_agent.path_retrieve_server.service import CachedPathRetriever


class DropLoopbackPureFunctionTests(unittest.TestCase):
    def test_removes_path_whose_tail_equals_topic(self):
        # topic 0 --rel0--> 10 --rel1--> 0:尾绕回 topic,应剔除
        loopback = ([0, 10, 0], [0, 1], -1.0)
        valid = ([0, 10, 20], [0, 1], -2.0)
        kept = drop_loopback_paths([loopback, valid])
        self.assertEqual(kept, [valid])

    def test_keeps_paths_with_distinct_tail(self):
        paths = [([0, 1], [0], -1.0), ([0, 5, 9], [0, 1], -2.0)]
        self.assertEqual(drop_loopback_paths(paths), paths)

    def test_removes_one_hop_self_loop(self):
        # 1-hop 自环 topic --rel--> topic 也要剔除
        self_loop = ([0, 0], [3], -0.5)
        self.assertEqual(drop_loopback_paths([self_loop]), [])

    def test_empty_input_returns_empty(self):
        self.assertEqual(drop_loopback_paths([]), [])


def write_input(root: Path):
    fb = root / "fbwq_full"
    fb.mkdir(parents=True)
    (fb / "entities.dict").write_text("m.topic\nm.answer\nm.mid\n", encoding="utf-8")
    (fb / "relations.dict").write_text(
        "rel.fwd\t0\nrel.back\t1\n", encoding="utf-8"
    )
    # topic --rel.fwd--> m.mid --rel.back--> m.topic(绕回);topic --rel.fwd--> m.answer(有效)
    (fb / "train.txt").write_text(
        "m.topic\trel.fwd\tm.answer\n"
        "m.topic\trel.fwd\tm.mid\n"
        "m.mid\trel.back\tm.topic\n",
        encoding="utf-8",
    )


def write_cache(path: Path):
    cache = {
        "version": 1,
        "meta": {
            "dataset": "WebQSP", "split": "test", "num_samples": 1,
            "num_entities": 3, "num_relations": 2, "num_steps": 2,
            "topk_entities": 3, "input_dir": "tiny", "qa_file": "tiny_qa.txt",
            "id2ent": {0: "m.topic", 1: "m.answer", 2: "m.mid"},
            "id2rel": {0: "rel.fwd", 1: "rel.back"},
        },
        "samples": [
            {
                "question": "[CLS] q [SEP]",
                "topic_ids": [0],
                "gold_ids": [1],
                "hop_attn": torch.tensor([1.0, 0.0]),
                "rel_probs": [torch.tensor([0.9, 0.0]), torch.tensor([0.0, 0.9])],
                "ent_indices": [torch.tensor([1, 2]), torch.tensor([0])],
                "ent_scores": [torch.tensor([0.8, 0.8]), torch.tensor([0.8])],
                "e_score_indices": torch.tensor([0, 1, 2]),
                "e_score_values": torch.tensor([0.95, 0.95, 0.95]),
            }
        ],
    }
    torch.save(cache, path)


class DropLoopbackIntegrationTests(unittest.TestCase):
    def test_retrieve_excludes_loopback_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_input(root / "input")
            write_cache(root / "cache.pt")
            retriever = CachedPathRetriever(
                cache_path=str(root / "cache.pt"), input_dir=str(root / "input")
            )
            result = retriever.retrieve(sample_index=0)

        # 没有任何返回路径的尾 == topic
        for path in result.mmr_reason_paths:
            edges = path["path"]
            self.assertNotEqual(
                edges[-1][2], "m.topic",
                msg=f"绕回 topic 的路径未被剔除: {edges}",
            )

    def test_retrieve_keeps_loopback_when_disabled(self):
        # drop_loopback=False(消融对照):绕回 topic 的路径应保留
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_input(root / "input")
            write_cache(root / "cache.pt")
            retriever = CachedPathRetriever(
                cache_path=str(root / "cache.pt"), input_dir=str(root / "input")
            )
            kept = retriever.retrieve(sample_index=0, drop_loopback=True)
            unfiltered = retriever.retrieve(sample_index=0, drop_loopback=False)

        self.assertGreater(
            len(unfiltered.mmr_reason_paths), len(kept.mmr_reason_paths)
        )
        self.assertTrue(
            any(p["path"][-1][2] == "m.topic" for p in unfiltered.mmr_reason_paths)
        )


if __name__ == "__main__":
    unittest.main()
