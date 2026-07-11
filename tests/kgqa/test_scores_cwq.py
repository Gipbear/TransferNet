import os
import tempfile
import unittest

import torch


def _write_fake_cache(path):
    cache = {
        "version": 1,
        "meta": {"dataset": "CWQ", "split": "test", "num_samples": 1,
                 "topk_entities": 500, "input_dir": "in", "qa_file": "qa",
                 "id2ent": {0: "m.0a", 1: "m.0b"}, "id2rel": {0: "r.loc"}},
        "samples": [{
            "question": "who?", "topic_ids": [0], "gold_ids": [1],
            "hop_attn": torch.tensor([1.0, 0.0]),
            "rel_probs": [torch.zeros(2), torch.zeros(2)],
            "ent_indices": [torch.tensor([1]), torch.tensor([1])],
            "ent_scores": [torch.tensor([0.5]), torch.tensor([0.5])],
            "e_score_indices": torch.tensor([1]),
            "e_score_values": torch.tensor([0.9]),
            "triples": [[0, 0, 1]],
        }],
    }
    torch.save(cache, path)


class TestCWQScoreLoader(unittest.TestCase):
    def test_load_restores_triples_and_meta(self):
        from kgqa.scores.cwq import CWQScoreLoader
        path = os.path.join(tempfile.mkdtemp(), "fake_cwq.pt")
        _write_fake_cache(path)
        bundle = CWQScoreLoader().load(path)
        self.assertEqual(bundle.meta.dataset, "CWQ")
        self.assertEqual(len(bundle.samples), 1)
        s = bundle.samples[0]
        self.assertEqual(s.triples, [[0, 0, 1]])
        self.assertIsNone(s.hop)
        self.assertEqual(s.sample_index, 0)
        self.assertEqual(s.topic_ids, [0])


if __name__ == "__main__":
    unittest.main()
