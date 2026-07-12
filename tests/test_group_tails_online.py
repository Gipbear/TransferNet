"""在线 group_tails:用常驻 KG 邻接表实时算"(topic, 关系序列) → 全 KG 尾实体",
替代离线 sidecar 文件。key 格式与 sidecar 对齐:'topic_mid|rel_name1[|rel_name2]'。
"""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kgqa.server.service import group_tails_from_path
# legacy 集成对拍:CachedPathRetriever 只读保留,kgqa 侧等价覆盖见 tests/kgqa/test_server_full.py
from oh_my_agent.path_retrieve_server.service import CachedPathRetriever


class GroupTailsPureFunctionTests(unittest.TestCase):
    def test_one_hop_expands_to_all_kg_tails(self):
        # topic 0 --rel0--> {1, 2};beam 只检索到尾 1,group_tails 应补出 2
        valid_edges = {0: [(0, 1), (0, 2)]}
        id2ent = {0: "m.topic", 1: "m.answer", 2: "m.other"}
        id2rel = {0: "rel.answer"}
        key, tails = group_tails_from_path([0, 1], [0], valid_edges, id2ent, id2rel)
        self.assertEqual(key, "m.topic|rel.answer")
        self.assertEqual(tails, ["m.answer", "m.other"])

    def test_two_hop_traverses_intermediate_nodes(self):
        valid_edges = {0: [(0, 10)], 10: [(1, 20), (1, 21)]}
        id2ent = {0: "m.t", 10: "m.mid", 20: "m.p", 21: "m.q"}
        id2rel = {0: "rel.a", 1: "rel.b"}
        key, tails = group_tails_from_path(
            [0, 10, 20], [0, 1], valid_edges, id2ent, id2rel
        )
        self.assertEqual(key, "m.t|rel.a|rel.b")
        self.assertEqual(tails, ["m.p", "m.q"])

    def test_empty_relation_sequence_returns_none(self):
        self.assertIsNone(group_tails_from_path([0], [], {}, {}, {}))

    def test_dead_end_yields_empty_tail_list(self):
        # 关系在邻接表里查不到尾 → 空尾,但 key 仍可拼出
        key, tails = group_tails_from_path(
            [0], [5], {0: [(0, 1)]}, {0: "m.t"}, {5: "rel.x"}
        )
        self.assertEqual(key, "m.t|rel.x")
        self.assertEqual(tails, [])

    def test_prediction_filter_keeps_only_predicted_tails(self):
        # topic --rel0--> {1,2,3},但 prediction 只含 1、3 → 过滤掉 2(hub 尾)
        valid_edges = {0: [(0, 1), (0, 2), (0, 3)]}
        id2ent = {0: "m.t", 1: "m.a", 2: "m.b", 3: "m.c"}
        id2rel = {0: "rel.x"}
        key, tails = group_tails_from_path(
            [0, 1], [0], valid_edges, id2ent, id2rel, prediction_ids={1, 3}
        )
        self.assertEqual(key, "m.t|rel.x")
        self.assertEqual(tails, ["m.a", "m.c"])

    def test_prediction_filter_only_applies_to_final_hop(self):
        # 中间节点 10 不在 prediction 里,但不该被过滤(只过滤最后一跳的尾)
        valid_edges = {0: [(0, 10)], 10: [(1, 20), (1, 21)]}
        id2ent = {0: "m.t", 10: "m.mid", 20: "m.p", 21: "m.q"}
        id2rel = {0: "rel.a", 1: "rel.b"}
        key, tails = group_tails_from_path(
            [0, 10, 20], [0, 1], valid_edges, id2ent, id2rel, prediction_ids={20}
        )
        self.assertEqual(key, "m.t|rel.a|rel.b")
        self.assertEqual(tails, ["m.p"])  # 21 被过滤,中间节点 10 不受影响

    def test_prediction_none_returns_all_tails(self):
        # prediction_ids=None → 向后兼容,返回全尾
        valid_edges = {0: [(0, 1), (0, 2)]}
        key, tails = group_tails_from_path(
            [0, 1], [0], valid_edges, {0: "m.t", 1: "m.a", 2: "m.b"},
            {0: "rel.x"}, prediction_ids=None
        )
        self.assertEqual(tails, ["m.a", "m.b"])


def write_input(root: Path):
    fb = root / "fbwq_full"
    fb.mkdir(parents=True)
    (fb / "entities.dict").write_text(
        "m.topic\nm.answer\nm.other\nm.noise\n", encoding="utf-8"
    )
    (fb / "relations.dict").write_text(
        "rel.answer\t0\nrel.answer_reverse\t1\n", encoding="utf-8"
    )
    # 同组三条尾:beam 只检索到 m.answer;KG 里还有 m.other(预测内)与 m.noise(hub 噪声)
    (fb / "train.txt").write_text(
        "m.topic\trel.answer\tm.answer\n"
        "m.topic\trel.answer\tm.other\n"
        "m.topic\trel.answer\tm.noise\n",
        encoding="utf-8",
    )


def write_cache(path: Path):
    cache = {
        "version": 1,
        "meta": {
            "dataset": "WebQSP", "split": "test", "num_samples": 1,
            "num_entities": 4, "num_relations": 2, "num_steps": 1,
            "topk_entities": 4, "input_dir": "tiny", "qa_file": "tiny_qa.txt",
            "id2ent": {0: "m.topic", 1: "m.answer", 2: "m.other", 3: "m.noise"},
            "id2rel": {0: "rel.answer", 1: "rel.answer_reverse"},
        },
        "samples": [
            {
                "question": "[CLS] q [SEP]",
                "topic_ids": [0],
                "gold_ids": [1],
                "hop_attn": torch.tensor([1.0]),
                "rel_probs": [torch.tensor([0.9, 0.0])],
                "ent_indices": [torch.tensor([1])],
                "ent_scores": [torch.tensor([0.8])],
                # prediction(e_score≥0.9): m.answer、m.other;m.noise 不在
                "e_score_indices": torch.tensor([1, 2]),
                "e_score_values": torch.tensor([0.95, 0.92]),
            }
        ],
    }
    torch.save(cache, path)


class GroupTailsIntegrationTests(unittest.TestCase):
    def test_retrieve_returns_prediction_filtered_group_tails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_input(root / "input")
            write_cache(root / "cache.pt")
            retriever = CachedPathRetriever(
                cache_path=str(root / "cache.pt"), input_dir=str(root / "input")
            )
            result = retriever.retrieve(sample_index=0)

        # beam 只检索到 m.answer
        self.assertEqual(
            result.mmr_reason_paths[0]["path"],
            [["m.topic", "rel.answer", "m.answer"]],
        )
        # group_tails 扩展补出 m.other(beam 外但在 prediction),过滤掉 m.noise(不在 prediction)
        self.assertIn("m.topic|rel.answer", result.group_tails)
        self.assertEqual(
            result.group_tails["m.topic|rel.answer"], ["m.answer", "m.other"]
        )
        # 字段进入 to_dict 以便 HTTP 序列化
        self.assertIn("group_tails", result.to_dict())


if __name__ == "__main__":
    unittest.main()
