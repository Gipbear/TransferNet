import os
import unittest

CACHE = "data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt"
INPUT_DIR = "data/input/WebQSP"


def _rels_from(path_dict):
    return [e[1] for e in path_dict["path"]]


@unittest.skipUnless(os.path.isfile(CACHE), "缓存缺失，跳过")
class TestWebQSPRegression(unittest.TestCase):
    def test_offline_paths_match_legacy(self):
        from kgqa.kg.global_kg import GlobalKG
        from kgqa.retrieve.engine import (
            candidate_hop_numbers, candidate_to_tuple,
            reconstruct_ent_dict, reconstruct_rel_dict, search_path_candidates,
            select_path_candidates,
        )
        from kgqa.scores.base import load_score_cache
        from kgqa.datasets.registry import get_adapter
        from kgqa.retrieve.backends.offline import OfflineBackend

        params = dict(alpha_final=1.0, threshold=0.01,
                      beam_size=50, lambda_val=0.2)

        cache = load_score_cache(CACHE)
        id2rel = cache["meta"]["id2rel"]
        ved = GlobalKG.from_input_dir(INPUT_DIR).valid_edges_dict
        N = 50

        def legacy_rel_name_seqs(sample):
            hop_nums = candidate_hop_numbers(len(sample["rel_probs"]))
            rel_dicts, ent_dicts = [], []
            for t in range(max(hop_nums)):
                rel_dicts.append(reconstruct_rel_dict(sample["rel_probs"][t], 0.01))
                ent_dicts.append(reconstruct_ent_dict(sample["ent_indices"][t], sample["ent_scores"][t], 0.01))
            final_scores = {
                int(entity_id): float(score)
                for entity_id, score in zip(
                    sample["e_score_indices"].tolist(), sample["e_score_values"].tolist()
                )
            }
            cands = []
            for ch in hop_nums:
                cands.extend(search_path_candidates(sample["topic_ids"], rel_dicts, ent_dicts, ch,
                                                    ved, 50, final_ent_scores=final_scores,
                                                    order_start=len(cands)))
            selected = select_path_candidates(cands, 50, alpha_final=1.0, lambda_val=0.2)
            # 映射 rel id → 名称，与新后端序列化口径一致
            return [[id2rel.get(r, str(r)) for r in candidate_to_tuple(c)[1]] for c in selected]

        adapter = get_adapter("webqsp", input_dir=INPUT_DIR)
        backend = OfflineBackend(adapter, cache_path=CACHE)

        for i in range(N):
            legacy = legacy_rel_name_seqs(cache["samples"][i])
            r = backend.retrieve(i, drop_loopback=False, **params)
            new = [_rels_from(p) for p in r.paths]
            self.assertEqual(len(new), len(legacy), f"sample {i} 路径数不一致")
            self.assertEqual(new, legacy, f"sample {i} rels 序列不一致")


if __name__ == "__main__":
    unittest.main()
