import os
import tempfile
import unittest

import torch

from kgqa.retrieve.cache.metaqa import MetaQAScoreLoader


class TestMetaQAScoreLoader(unittest.TestCase):
    def _write_cache(self):
        cache = {
            "version": 1,
            "meta": {"dataset": "MetaQA", "split": "test", "num_samples": 1,
                     "topk_entities": 500, "input_dir": "data/input/MetaQA_KB",
                     "qa_file": "data/input/MetaQA_KB/test.pt",
                     "id2ent": {0: "DUMMY", 1: "Movie A"}, "id2rel": {10: "starred_actors"}},
            "samples": [{
                "question": "what movie", "topic_ids": [1], "gold_ids": [1],
                "hop_attn": torch.tensor([1.0, 0.0, 0.0]),
                "rel_probs": [torch.tensor([0.0, 0.9]), torch.tensor([0.0, 0.0]),
                              torch.tensor([0.0, 0.0])],
                "ent_indices": [torch.tensor([1]), torch.tensor([], dtype=torch.long),
                                torch.tensor([], dtype=torch.long)],
                "ent_scores": [torch.tensor([0.8]), torch.tensor([]), torch.tensor([])],
                "e_score_indices": torch.tensor([1]),
                "e_score_values": torch.tensor([0.95]),
                "hop": 1,
            }],
        }
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.save(cache, path)
        return path

    def test_load_restores_hop(self):
        path = self._write_cache()
        try:
            bundle = MetaQAScoreLoader().load(path)
        finally:
            os.unlink(path)
        self.assertEqual(bundle.meta.dataset, "MetaQA")
        self.assertEqual(len(bundle.samples), 1)
        s = bundle.samples[0]
        self.assertEqual(s.hop, 1)
        self.assertEqual(s.sample_index, 0)
        self.assertEqual(s.gold_ids, [1])


if __name__ == "__main__":
    unittest.main()
