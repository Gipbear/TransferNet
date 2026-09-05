"""pfit eval 路径预算截断(truncate_paths_by_score)单元测试。"""
import contextlib
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from kgqa.pfit import eval as pfit_eval
from kgqa.pfit.eval import build_parser, truncate_paths_by_score


class TestTruncatePathsByScore(unittest.TestCase):
    def _paths(self, scores):
        return [{"path": [["a", "r", f"t{i}"]], "log_score": s}
                for i, s in enumerate(scores)]

    def test_keeps_top_k_by_log_score_even_if_input_unsorted(self):
        paths = self._paths([-12.0, -9.0, -15.0])
        out = truncate_paths_by_score(paths, 2)
        self.assertEqual([p["log_score"] for p in out], [-9.0, -12.0])

    def test_non_positive_max_paths_keeps_all(self):
        paths = self._paths([-9.0, -10.0])
        self.assertEqual(len(truncate_paths_by_score(paths, 0)), 2)
        self.assertEqual(len(truncate_paths_by_score(paths, -1)), 2)

    def test_max_paths_beyond_length_keeps_all(self):
        paths = self._paths([-9.0, -10.0])
        self.assertEqual(len(truncate_paths_by_score(paths, 5)), 2)

    def test_does_not_mutate_input(self):
        paths = self._paths([-9.0])
        original = list(paths)
        truncate_paths_by_score(paths, 1)
        self.assertEqual(paths, original)

    def test_shuffle_order_is_seeded_and_stable_across_python_processes(self):
        parser = build_parser()
        args = parser.parse_args([
            "--dataset", "webqsp", "--input", "input.jsonl", "--exp_dir", "exp", "--seed", "17",
        ])
        self.assertEqual(args.seed, 17)

        script = (
            "from kgqa.pfit.eval import shuffled_path_order; "
            "print(shuffled_path_order('same question', 6, seed=17, run_idx=0))"
        )
        outputs = [
            subprocess.check_output([sys.executable, "-c", script], text=True).strip()
            for _ in range(3)
        ]
        self.assertEqual(len(set(outputs)), 1)


class TestInferenceModelLoading(unittest.TestCase):
    def test_uses_16bit_text_only_and_restores_missing_architecture(self):
        model = mock.Mock()
        model.config.architectures = None
        tokenizer = mock.Mock(pad_token="<pad>")
        fast_language_model = mock.Mock()
        fast_language_model.from_pretrained.return_value = model, tokenizer
        fake_unsloth = types.SimpleNamespace(FastLanguageModel=fast_language_model)

        with mock.patch.dict(sys.modules, {"unsloth": fake_unsloth}), mock.patch(
            "utils.huggingface.resolve_model_path_local_first",
            return_value="/local/model/snapshot",
        ):
            pfit_eval.load_inference_model(
                model="unsloth/Qwen3.5-2B",
                max_seq_length=1280,
                adapter=None,
                model_precision="16bit",
                text_only=True,
            )

        load_kwargs = fast_language_model.from_pretrained.call_args.kwargs
        self.assertFalse(load_kwargs["load_in_4bit"])
        self.assertTrue(load_kwargs["load_in_16bit"])
        self.assertTrue(load_kwargs["text_only"])
        self.assertEqual(model.config.architectures, [type(model).__name__])

    def test_parser_accepts_qwen_loading_options(self):
        args = build_parser().parse_args([
            "--dataset", "webqsp",
            "--input", "input.jsonl",
            "--exp_dir", "exp",
            "--model_precision", "16bit",
            "--text_only",
        ])

        self.assertEqual(args.model_precision, "16bit")
        self.assertTrue(args.text_only)


class _DeviceBatch(dict):
    def to(self, device):
        return self


class _InputIds:
    shape = (1, 2)


class _OutputIds:
    def __getitem__(self, key):
        return [[3]]


class TestDirectAnswerMode(unittest.TestCase):
    def test_direct_answer_disables_thinking_in_chat_template(self):
        # Given: 启用直接回答模式的单条 WebQSP 推理配置。
        tokenizer = mock.Mock(eos_token_id=0)
        tokenizer.apply_chat_template.return_value = [1, 2]
        tokenizer.pad.return_value = _DeviceBatch({"input_ids": _InputIds()})
        tokenizer.batch_decode.return_value = ["Supporting Paths: 1\nAnswer: answer"]
        model = mock.Mock(device="cpu")
        model.generate.return_value = _OutputIds()
        spec = types.SimpleNamespace(clean_question=lambda question, topics: question)
        samples = [{
            "sample_index": 0,
            "question": "question",
            "topics": [],
            "golden": ["answer"],
            "mmr_reason_paths": [{"path": [["topic", "relation", "answer"]], "log_score": 0.0}],
        }]
        cfg = {
            "entity_map_dict": None, "rev_entity_map": None, "use_entity_names": True,
            "fmt": "v2", "path_format": "chain", "show_score": False,
            "system_prompt": None, "no_paths": False, "reject_prompt": False,
            "max_paths": 0, "intervention": None, "noise_paths": 0,
            "dedupe_tail_paths": False, "shuffle_paths": False, "seed": 17,
            "batch_size": 1, "max_new_tokens": 8, "show_progress": False,
            "progress_interval": 50, "run_dir": None, "direct_answer": True,
        }

        # When: 执行一轮真实的 prompt 构建与生成调用。
        with tempfile.TemporaryDirectory() as directory:
            predictions_path = str(Path(directory) / "predictions.jsonl")
            fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                pfit_eval.run_single(samples, model, tokenizer, cfg, spec, 0, predictions_path, 1)

        # Then: chat template 收到官方定义的关闭思考参数。
        self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
