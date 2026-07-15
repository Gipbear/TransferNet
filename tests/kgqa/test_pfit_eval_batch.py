import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kgqa.pfit import eval_batch


class TestPfitEvalBatch(unittest.TestCase):
    def test_batch_loads_model_once_and_reuses_it_for_each_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [
                {
                    "condition": condition, "layer": "base_zeroshot", "input": str(root / f"{condition}.jsonl"),
                    "exp_dir": str(root / condition), "run_dir": str(root / condition),
                    "no_paths": condition == "no_path", "manifest": {"condition": condition},
                }
                for condition in ("no_path", "tarrs")
            ]
            jobs_file = root / "jobs.json"
            jobs_file.write_text(json.dumps(jobs), encoding="utf-8")
            with patch.object(eval_batch, "load_inference_model", return_value=(object(), object())) as load_model, \
                 patch.object(eval_batch, "run_eval", return_value={"overall": {"n": 1}}) as run_eval:
                eval_batch.main([
                    "--jobs_file", str(jobs_file), "--dataset", "webqsp", "--model", "model",
                    "--format", "v2", "--path_format", "chain", "--entity_repr", "name",
                    "--max_new_tokens", "2", "--batch_size", "1", "--no_progress",
                ])
            load_model.assert_called_once_with(model="model", max_seq_length=2048, adapter=None)
            self.assertEqual(run_eval.call_count, 2)
            self.assertTrue(run_eval.call_args_list[0].kwargs["no_paths"])
            self.assertFalse(run_eval.call_args_list[1].kwargs["no_paths"])
            self.assertIs(run_eval.call_args_list[0].kwargs["loaded_model"], run_eval.call_args_list[1].kwargs["loaded_model"])
