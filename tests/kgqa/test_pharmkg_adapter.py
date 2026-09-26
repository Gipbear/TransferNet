"""验证 PharmKG 在统一第三章检索框架中的独立适配契约。"""

import json
import tempfile
import unittest
from pathlib import Path


class TestPharmKGAdapter(unittest.TestCase):
    def test_registry_loads_pharmkg_qa_and_entity_names(self):
        # Given: 一个采用 PharmKG/WebQSP 文本边界格式的最小数据目录。
        from kgqa.retrieve.datasets.registry import get_adapter

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            graph_dir = root / "fbwq_full"
            graph_dir.mkdir()
            (graph_dir / "mid2name.txt").write_text("p.1\tdrug name\n", encoding="utf-8")
            qa_file = root / "qa.txt"
            qa_file.write_text("Question with [inner] text [p.1]\tp.2\n", encoding="utf-8")

            # When: 通过正式 pharmkg 名称获取并使用 adapter。
            adapter = get_adapter("pharmkg", input_dir=str(root))
            samples = adapter.load_qa(str(qa_file))
            entity_name = adapter.entity_name("p.1")

        # Then: 数据集身份、末尾主题实体解析和本地实体名称映射均属于 PharmKG。
        self.assertEqual(adapter.name, "pharmkg")
        self.assertEqual(adapter.max_hop, 2)
        self.assertEqual(samples[0].question, "Question with [inner] text")
        self.assertEqual(samples[0].topic_ids, ["p.1"])
        self.assertEqual(samples[0].gold_ids, ["p.2"])
        self.assertEqual(entity_name, "drug name")
        self.assertEqual(adapter.metric_spec().gold_key, "mid")

    def test_conservative_answer_type_inference_rejects_ambiguous_questions(self):
        # Given: 明确、冲突和易误判的三类医学问题。
        from kgqa.retrieve.datasets import pharmkg

        # When/Then: 只接受唯一且保守的答案类型线索。
        self.assertEqual(pharmkg.infer_answer_type("Which gene activates socs3?"), "gene")
        self.assertIsNone(pharmkg.infer_answer_type("Which gene treats a disease?"))
        self.assertIsNone(pharmkg.infer_answer_type("What disease has this biomarker?"))

    def test_type_aware_paths_are_stable_and_tail_deduplicated(self):
        # Given: 首条路径类型不符，后续含两个 gene 尾实体和一个重复尾实体。
        from kgqa.retrieve.datasets import pharmkg

        entity_types = {"p.d": "disease", "p.g1": "gene", "p.g2": "gene"}
        paths = [
            {"path": [["p.topic", "r", "p.d"]], "log_score": -0.1},
            {"path": [["p.topic", "r", "p.g1"]], "log_score": -0.2},
            {"path": [["p.topic", "r2", "p.g1"]], "log_score": -0.3},
            {"path": [["p.topic", "r", "p.g2"]], "log_score": -0.4},
        ]

        # When: 按答案类型稳定重排，再按尾实体去重并截断为两条。
        selected = pharmkg.rerank_and_dedupe_paths(
            "Which gene activates the topic?", paths, entity_types, limit=2,
        )

        # Then: 类型匹配路径内部次序不变，重复尾实体不占用固定预算。
        tails = [item["path"][-1][-1] for item in selected]
        self.assertEqual(tails, ["p.g1", "p.g2"])

    def test_unknown_answer_type_only_deduplicates_tails(self):
        # Given: 问题没有可靠类型线索。
        from kgqa.retrieve.datasets import pharmkg

        paths = [
            {"path": [["p.topic", "r", "p.d"]], "log_score": -0.1},
            {"path": [["p.topic", "r2", "p.d"]], "log_score": -0.2},
            {"path": [["p.topic", "r", "p.g"]], "log_score": -0.3},
        ]

        # When: 执行后处理。
        selected = pharmkg.rerank_and_dedupe_paths(
            "What is associated with the topic?", paths, {}, limit=20,
        )

        # Then: 仅去重，不改变首个尾实体的原始顺序。
        tails = [item["path"][-1][-1] for item in selected]
        self.assertEqual(tails, ["p.d", "p.g"])

    def test_postprocess_summary_separates_rescues_from_harms(self):
        # Given: 固定预算为一条时，原始首条错误而类型匹配路径正确。
        from kgqa.retrieve.cli.pharmkg_postprocess import process_records

        records = [{
            "sample_index": 0,
            "question": "Which gene activates the topic?",
            "topics": ["p.topic"],
            "hop": 1,
            "golden": ["p.g"],
            "mmr_reason_paths": [
                {"path": [["p.topic", "r", "p.d"]], "log_score": -0.1},
                {"path": [["p.topic", "r", "p.g"]], "log_score": -0.2},
            ],
            "prediction": {"p.d": 0.9},
        }]

        # When: 执行类型感知固定预算后处理。
        _output, summary = process_records(
            records, {"p.d": "disease", "p.g": "gene"}, path_limit=1,
        )

        # Then: summary 分开报告救回与损失，而不是只报告净变化。
        self.assertEqual(summary["answer_hit_gained"], 1)
        self.assertEqual(summary["answer_hit_lost"], 0)
        self.assertEqual(summary["top1_rescued"], 1)
        self.assertEqual(summary["top1_harmed"], 0)


class TestPharmKGDispatch(unittest.TestCase):
    def test_score_producer_dispatches_without_webqsp_alias(self):
        # Given: 第三章统一在线得分生产入口。
        from kgqa.retrieve.cli.retrieve import _make_producer

        # When: 使用正式 PharmKG 数据集名称分发。
        producer = _make_producer("pharmkg")

        # Then: 返回独立 PharmKG producer。
        self.assertEqual(type(producer).__name__, "PharmKGScoreProducer")
        self.assertEqual(type(producer).__module__, "kgqa.backbone.pharmkg")

    def test_produce_without_checkpoint_raises_runtime_error(self):
        # Given: 尚未绑定 checkpoint 的 PharmKG 得分生产器。
        from kgqa.backbone.pharmkg import PharmKGScoreProducer

        producer = PharmKGScoreProducer()

        # When/Then: 缺失 checkpoint 必须抛 RuntimeError，而不是可被 -O 剥离的 assert。
        with self.assertRaises(RuntimeError):
            producer.produce("unused-input-dir", "unused-qa-file", show_progress=False)

    def test_ch3_parser_accepts_pharmkg(self):
        # Given: 第三章正式实验编排入口。
        from experiments.ch3.run import _default_config, build_parser

        # When: 解析 PharmKG 数据集参数。
        args = build_parser().parse_args(["--dataset", "pharmkg", "--dry_run"])

        # Then: 编排器保留独立数据集名称。
        self.assertEqual(args.dataset, "pharmkg")
        self.assertTrue(_default_config(Path.cwd(), "pharmkg", "transfernet").is_file())

    def test_ch3_profile_uses_dev_selected_beam20_candidate(self):
        config_path = Path.cwd() / "experiments/configs/ch3/pharmkg_transfernet_v1.json"

        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["status"], "confirmed")
        self.assertEqual(config["selection_split"], "dev")
        self.assertEqual(config["selected_candidate"], "beam20_lambda03_eta05")
        self.assertEqual(config["retrieve"]["beam_size"], 20)
        self.assertEqual(config["retrieve"]["lambda_val"], 0.3)
        self.assertEqual(config["retrieve"]["eta"], 0.5)


if __name__ == "__main__":
    unittest.main()
