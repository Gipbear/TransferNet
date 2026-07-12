"""demo_page 图变换与轨迹变换的单元测试。"""
import unittest

from kgqa.agent.web.service import paths_to_graph, shape_replay_result


def _paths():
    return [
        {"path": [["Jamaica", "location.country.official_language", "Jamaican English"]],
         "log_score": -7.9},
        {"path": [["Jamaica", "location.country.languages_spoken", "Jamaican English"]],
         "log_score": -8.1},
        {"path": [["Jamaica", "location.location.containedby", "North America"],
                  ["North America", "location.location.contains", "Canada"]],
         "log_score": -9.0},
    ]


class TestPathsToGraph(unittest.TestCase):
    def setUp(self):
        self.graph = paths_to_graph(_paths(), topics=["Jamaica"])

    def test_nodes_layered_by_first_hop(self):
        layers = {n["id"]: n["layer"] for n in self.graph["nodes"]}
        self.assertEqual(layers["Jamaica"], 0)
        self.assertEqual(layers["Jamaican English"], 1)
        self.assertEqual(layers["North America"], 1)
        self.assertEqual(layers["Canada"], 2)

    def test_edges_merged_with_path_ids(self):
        edges = {(e["source"], e["relation"], e["target"]): e["path_ids"]
                 for e in self.graph["edges"]}
        self.assertEqual(
            edges[("Jamaica", "location.country.official_language", "Jamaican English")],
            [1])
        # 两跳路径 3 的两条边都携带 path_id 3
        self.assertEqual(
            edges[("North America", "location.location.contains", "Canada")], [3])

    def test_paths_one_based_with_text(self):
        p1 = self.graph["paths"][0]
        self.assertEqual(p1["id"], 1)
        self.assertEqual(p1["tail"], "Jamaican English")
        self.assertEqual(
            p1["text"],
            "Jamaica -> location.country.official_language -> Jamaican English")
        p3 = self.graph["paths"][2]
        self.assertEqual(
            p3["text"],
            "Jamaica -> location.location.containedby -> North America"
            " -> location.location.contains -> Canada")

    def test_shared_triple_merges_path_ids(self):
        graph = paths_to_graph([
            {"path": [["Jamaica", "r.lang", "Jamaican English"]], "log_score": -7.9},
            {"path": [["Jamaica", "r.lang", "Jamaican English"]], "log_score": -8.3},
        ], topics=["Jamaica"])
        (edge,) = graph["edges"]
        self.assertEqual(edge["path_ids"], [1, 2])
        self.assertEqual([p["id"] for p in graph["paths"]], [1, 2])

    def test_extra_kg_path_uses_string_id(self):
        graph = paths_to_graph(
            [{"path": [["Jamaica", "r.lang", "Jamaican English"]], "log_score": -7.9}],
            topics=["Jamaica"],
            extra_paths=[{
                "id": "kg1",
                "label": "P_kg1",
                "path": [["Jamaica", "KG: works_written", "Kingston"]],
                "tail": "Kingston",
                "text": "P_kg1: Jamaica -> works_written -> Kingston",
                "synthetic": True,
            }],
        )
        by_id = {p["id"]: p for p in graph["paths"]}
        self.assertTrue(by_id["kg1"]["synthetic"])
        self.assertEqual(by_id["kg1"]["label"], "P_kg1")
        edges = {(e["source"], e["relation"], e["target"]): e["path_ids"]
                 for e in graph["edges"]}
        self.assertEqual(edges[("Jamaica", "KG: works_written", "Kingston")], ["kg1"])


def _result_dict():
    return {
        "named_mmr_reason_paths": [
            {"path": [["Jamaica", "r.lang", "Jamaican English"]], "log_score": -7.9},
            {"path": [["Jamaica", "r.lang2", "Jamaican Patois"]], "log_score": -8.1},
            {"path": [["Jamaica", "r.tz", "UTC-05:00"]], "log_score": -9.5},
        ],
        "iterations": [
            {"batch_index": 0, "batch_start_rank": 1, "batch_end_rank": 3,
             "batch_status": "mixed",
             "answer_names": ["Jamaican English", "Jamaican Patois", "UTC-05:00"],
             "global_cited_path_indices": [1, 2, 3],
             "accepted_path_indices": [1, 2]},
        ],
        "pred_answer_names": ["Jamaican English", "Jamaican Patois"],
        "final_accepted_path_indices": [1, 2],
        "relation_expanded_path_indices": [2],
        "large_answer_expanded_mids": ["m.0abc", "m.0nope"],
        "stop_reason": "mixed_ratio",
    }


class TestShapeReplayResult(unittest.TestCase):
    def setUp(self):
        self.shaped = shape_replay_result(
            _result_dict(), entity_map={"m.0abc": "Kingston"})

    def test_iteration_rejected_ids(self):
        it = self.shaped["iterations"][0]
        self.assertEqual(it["cited_path_ids"], [1, 2, 3])
        self.assertEqual(it["accepted_path_ids"], [1, 2])
        self.assertEqual(it["rejected_path_ids"], [3])

    def test_final_answers_support_and_via(self):
        by_name = {a["name"]: a for a in self.shaped["final_answers"]}
        self.assertEqual(by_name["Jamaican English"]["path_ids"], [1])
        self.assertEqual(by_name["Jamaican English"]["via"], "llm")
        # 路径 2 已在 final_accepted 中(校验接受),即使同时在关系扩展列表里,
        # 也按正常支撑处理,不重复计入 expansion_path_ids
        self.assertEqual(by_name["Jamaican Patois"]["path_ids"], [2])
        self.assertEqual(by_name["Jamaican Patois"]["expansion_path_ids"], [])
        self.assertEqual(by_name["Jamaican Patois"]["via"], "llm")

    def test_expansion_only_and_group_expansion_via(self):
        """扩展来源答案应带 expansion_path_ids;补全来源答案标 group_expansion。"""
        result = {
            "named_mmr_reason_paths": [
                {"path": [["Jamaica", "r.lang", "Jamaican English"]], "log_score": -7.9},
                {"path": [["Jamaica", "r.lang2", "Jamaican Patois"]], "log_score": -8.1},
            ],
            "iterations": [],
            "pred_answer_names": ["Jamaican English", "Jamaican Patois", "Kingston"],
            "final_accepted_path_indices": [1],
            "relation_expanded_path_indices": [2],   # 路径 2 被拒后经关系扩展收回
            "large_answer_expanded_mids": ["m.0abc"],
            "group_tails": {"m.topic|book.author.works_written": ["m.0abc"]},
            "stop_reason": "max_batches",
        }
        shaped = shape_replay_result(result, entity_map={"m.0abc": "Kingston"})
        by_name = {a["name"]: a for a in shaped["final_answers"]}
        self.assertEqual(by_name["Jamaican Patois"]["path_ids"], [])
        self.assertEqual(by_name["Jamaican Patois"]["expansion_path_ids"], [2])
        self.assertEqual(by_name["Jamaican Patois"]["via"], "relation_expansion")
        self.assertEqual(by_name["Kingston"]["path_ids"], [])
        self.assertEqual(by_name["Kingston"]["expansion_path_ids"], [])
        self.assertEqual(by_name["Kingston"]["kg_path_ids"], ["kg1"])
        self.assertEqual(by_name["Kingston"]["group_source_labels"], ["works_written"])
        self.assertEqual(by_name["Kingston"]["via"], "group_expansion")
        self.assertEqual(
            shaped["calibration"]["group_expanded_items"],
            [{"name": "Kingston", "source_labels": ["works_written"], "kg_path_ids": ["kg1"]}],
        )
        self.assertEqual(
            shaped["kg_completion_paths"],
            [{
                "id": "kg1",
                "label": "P_kg1",
                "path": [["m.topic", "KG: works_written", "Kingston"]],
                "raw_path": [],
                "tail": "Kingston",
                "text": "P_kg1: m.topic -> KG: works_written -> Kingston",
                "source_key": "m.topic|book.author.works_written",
                "source_label": "works_written",
                "relations": ["book.author.works_written"],
                "restored": False,
                "synthetic": True,
            }],
        )

    def test_group_completion_uses_restored_kg_path(self):
        class Resolver:
            def resolve_many(self, topic_mid, relations, tail_mids):
                self.args = (topic_mid, relations, tail_mids)
                return {
                    "m.tail": [
                        ["m.topic", "r.one", "m.mid"],
                        ["m.mid", "r.two", "m.tail"],
                    ],
                }

        resolver = Resolver()
        result = {
            "named_mmr_reason_paths": [],
            "iterations": [],
            "pred_answer_names": ["Tail"],
            "final_accepted_path_indices": [],
            "relation_expanded_path_indices": [],
            "large_answer_expanded_mids": ["m.tail"],
            "group_tails": {"m.topic|r.one|r.two": ["m.tail"]},
            "topic_mid": "m.topic",
            "stop_reason": "max_batches",
        }
        shaped = shape_replay_result(
            result,
            entity_map={"m.topic": "Topic", "m.mid": "Middle", "m.tail": "Tail"},
            kg_path_resolver=resolver,
        )
        self.assertEqual(resolver.args, ("m.topic", ["r.one", "r.two"], ["m.tail"]))
        kg_path = shaped["kg_completion_paths"][0]
        self.assertTrue(kg_path["restored"])
        self.assertEqual(
            kg_path["path"],
            [["Topic", "r.one", "Middle"], ["Middle", "r.two", "Tail"]],
        )
        self.assertEqual(
            kg_path["raw_path"],
            [["m.topic", "r.one", "m.mid"], ["m.mid", "r.two", "m.tail"]],
        )
        self.assertEqual(kg_path["text"], "P_kg1: Topic -> r.one -> Middle -> r.two -> Tail")
        by_name = {a["name"]: a for a in shaped["final_answers"]}
        self.assertEqual(by_name["Tail"]["kg_path_ids"], ["kg1"])

    def test_duplicate_final_names_deduped(self):
        """不同 MID 映射到同名实体时,最终答案按名字去重只保留一行。"""
        result = _result_dict()
        result["pred_answer_names"] = [
            "Jamaican English", "Jamaican English", "Jamaican Patois"]
        shaped = shape_replay_result(result, entity_map={})
        names = [a["name"] for a in shaped["final_answers"]]
        self.assertEqual(names, ["Jamaican English", "Jamaican Patois"])

    def test_calibration_summary(self):
        cal = self.shaped["calibration"]
        self.assertEqual(cal["dropped_answers"], ["UTC-05:00"])
        self.assertEqual(cal["relation_expanded_path_ids"], [2])
        self.assertEqual(cal["group_expanded_names"], ["Kingston", "m.0nope"])

    def test_stop_reason_passthrough(self):
        self.assertEqual(self.shaped["stop_reason"], "mixed_ratio")

    def test_out_of_range_pid_does_not_crash(self):
        """关系扩展可能引入超出范围的路径号,应无尾实体而非崩溃。"""
        result = {
            "named_mmr_reason_paths": [
                {"path": [["Jamaica", "r.lang", "Jamaican English"]], "log_score": -7.9},
                {"path": [["Jamaica", "r.lang2", "Jamaican Patois"]], "log_score": -8.1},
            ],
            "iterations": [
                {"batch_index": 0, "batch_start_rank": 1, "batch_end_rank": 2,
                 "batch_status": "mixed",
                 "answer_names": ["Jamaican English", "Jamaican Patois"],
                 "global_cited_path_indices": [1, 2, 99],
                 "accepted_path_indices": [1, 2]},
            ],
            "pred_answer_names": ["Jamaican English"],
            "final_accepted_path_indices": [1, 99],  # 99 超出范围
            "relation_expanded_path_indices": [99],
            "large_answer_expanded_mids": [],
            "stop_reason": "mixed_ratio",
        }
        shaped = shape_replay_result(result, entity_map={})
        # 应该不崩溃,99 超出范围对应答案 path_ids 为 []
        by_name = {a["name"]: a for a in shaped["final_answers"]}
        self.assertEqual(by_name["Jamaican English"]["path_ids"], [1])
        self.assertEqual(by_name["Jamaican English"]["via"], "llm")


if __name__ == "__main__":
    unittest.main()
