"""load_score_cache 的安全反序列化回归测试。"""
import os
import tempfile
import unittest

import torch

from kgqa.retrieve.cache.base import load_score_cache


def _write_cache(payload, suffix=".pt"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    torch.save(payload, path)
    return path


class _ArbitraryPicklable:
    """非白名单的可 pickle 对象:weights_only 加载必须拒绝。"""

    def __init__(self):
        self.tag = "not-a-basic-container"


class TestLoadScoreCacheWeightsOnly(unittest.TestCase):
    def _base_cache(self):
        return {
            "version": 1,
            "meta": {"dataset": "webqsp", "split": "test", "num_samples": 0,
                     "topk_entities": 500, "input_dir": "data/input/WebQSP",
                     "qa_file": "", "id2ent": {}, "id2rel": {}},
            "samples": [],
        }

    def test_loads_basic_container_cache(self):
        """合法缓存(基础容器 + tensor)可正常加载,防止误伤现役格式。"""
        cache = self._base_cache()
        cache["samples"] = [{
            "question": "q", "topic_ids": [1], "gold_ids": [],
            "hop_attn": torch.tensor([1.0, 0.0, 0.0]),
            "rel_probs": [], "ent_indices": [], "ent_scores": [],
            "e_score_indices": torch.tensor([]),
            "e_score_values": torch.tensor([]),
        }]
        path = _write_cache(cache)
        try:
            loaded = load_score_cache(path)
            self.assertEqual(loaded["version"], 1)
            self.assertEqual(len(loaded["samples"]), 1)
        finally:
            os.remove(path)

    def test_rejects_arbitrary_pickled_object(self):
        """嵌入任意类实例的缓存必须抛错,不允许任意代码执行。"""
        cache = self._base_cache()
        cache["meta"]["id2ent"] = {0: _ArbitraryPicklable()}
        path = _write_cache(cache)
        try:
            with self.assertRaises(Exception):
                load_score_cache(path)
        finally:
            os.remove(path)
