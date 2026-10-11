"""验证名称资源的显式加载、内容指纹和恢复边界。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kgqa.agent.cli.run_workflow import execute
from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.resources import EntityNameResource
from kgqa.agent.workflow.run import WorkflowRunner
from kgqa.serving.graph_server import GraphScoringClient


class TestEntityNameResource(unittest.TestCase):
    def test_absent_configuration_produces_explicit_empty_resource(self):
        resource = EntityNameResource.load(None)
        self.assertEqual(resource.names, {})
        self.assertIsNone(resource.fingerprint)

    def test_changed_name_content_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "names.json"
            path.write_text(json.dumps({"m.a": "A"}))
            first = EntityNameResource.load(path)
            path.write_text(json.dumps({"m.a": "Changed"}))
            second = EntityNameResource.load(path)
            self.assertEqual(first.names, {"m.a": "A"})
            self.assertNotEqual(first.fingerprint["sha256"], second.fingerprint["sha256"])

    def test_configured_missing_resource_fails_instead_of_silent_empty(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(FileNotFoundError):
            EntityNameResource.load(Path(folder) / "missing.json")

    def test_invalid_resource_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "names.json"
            path.write_text(json.dumps({"m.a": ["A"]}))
            with self.assertRaises(ValueError):
                EntityNameResource.load(path)

    def test_resource_root_must_be_a_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "names.json"
            path.write_text("[]")
            with self.assertRaises(TypeError):
                EntityNameResource.load(path)

    def test_runner_has_no_implicit_historical_name_dependency(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = WorkflowRunner(WorkflowConfig(), None, None, Path(folder), Path(folder), "unused")
            self.assertEqual(runner.names, {})

    def test_changed_name_resource_is_rejected_when_resuming(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "adapter_model.safetensors").write_bytes(b"fake weights")
            (root / "templates").write_bytes(b"fake templates")
            (root / "input").write_text("")
            names = root / "names.json"
            names.write_text(json.dumps({"m.a": "A"}))
            raw = {"workflow": {}, "model": "unused", "adapter": str(root), "entity_names_file": str(names)}
            config = root / "config.json"
            config.write_text(json.dumps(raw))
            args = SimpleNamespace(config=str(config), phase="full", input=str(root / "input"), limit=0,
                                   templates=str(root / "templates"), cache_dir=str(root / "cache"),
                                   workers=1, wait_initial=False)
            with patch("kgqa.agent.cli.run_workflow.PfitAnswerer"), \
                    patch("kgqa.agent.cli.run_workflow.TemplateLibrary.load"), \
                    patch("kgqa.agent.cli.run_workflow.WorkflowRunner") as mocked:
                mocked.return_value.run.return_value = {"n": 0}
                execute(args, root / "output")
                names.write_text(json.dumps({"m.a": "Changed"}))
                with self.assertRaisesRegex(ValueError, "运行配置或输入发生变化"):
                    execute(args, root / "output")

    def test_graph_client_sends_only_scoring_fields_to_resident_server(self):
        client = GraphScoringClient("http://unused")
        response = Mock()
        response.json.return_value = {"paths": []}
        client.session.post = Mock(return_value=response)
        config = WorkflowConfig(allow_partial_rescue=True, allow_template_fallback=True)
        client.score("Q", ("m.a",), [["m.a", "r", "m.b"]], config)
        workflow = client.session.post.call_args.kwargs["json"]["workflow"]
        self.assertEqual(set(workflow), {"max_paths", "triple_budget", "lambda_val", "eta", "penalty_mode"})


if __name__ == "__main__":
    unittest.main()
