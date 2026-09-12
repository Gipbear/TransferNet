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
