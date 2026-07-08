import os
import unittest

CACHE = "data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt"
INPUT_DIR = "data/input/WebQSP"


def _rels_from(path_dict):
    return [e[1] for e in path_dict["path"]]


@unittest.skipUnless(os.path.isfile(CACHE), "缓存缺失，跳过")
class TestWebQSPRegression(unittest.TestCase):
    def test_offline_paths_match_legacy(self):
        from scripts.offline_path_search import (
            load_score_cache, rebuild_valid_edges_dict, _method_hop_numbers,
            reconstruct_rel_dict, reconstruct_ent_dict, LogNormStrategy,
            search_path_candidates, select_path_candidates, candidate_to_tuple,
            final_ent_score_dict,
        )
        from kgqa.datasets.registry import get_adapter
        from kgqa.retrieve.backends.offline import OfflineBackend

        params = dict(method="tail_blend", alpha_final=1.0, threshold=0.01,
                      beam_size=50, lambda_val=0.2)

        cache = load_score_cache(CACHE)
        id2rel = cache["meta"]["id2rel"]
        ved = rebuild_valid_edges_dict(INPUT_DIR)
        N = 50

        def legacy_rel_name_seqs(sample):
            hop_num = int(sample["hop_attn"].argmax().item()) + 1
            hop_nums = _method_hop_numbers("tail_blend", hop_num, len(sample["rel_probs"]))
            rel_dicts, ent_dicts = [], []
            for t in range(max(hop_nums)):
                rel_dicts.append(reconstruct_rel_dict(sample["rel_probs"][t], 0.01))
                ent_dicts.append(reconstruct_ent_dict(sample["ent_indices"][t], sample["ent_scores"][t], 0.01))
            scoring = LogNormStrategy()
            final_scores = final_ent_score_dict(sample)
            cands = []
            for ch in hop_nums:
                cands.extend(search_path_candidates(sample["topic_ids"], rel_dicts, ent_dicts, ch,
                                                    ved, scoring, 50, final_ent_scores=final_scores,
                                                    order_start=len(cands)))
            selected = select_path_candidates(cands, 50, method="tail_blend", alpha_final=1.0, lambda_val=0.2)
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
