import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.llm_server import server


class FakeTokenizer:
    eos_token_id = 7

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt"):
        return torch.tensor([[11, 12, 13]])

    def decode(self, new_tokens, skip_special_tokens=True):
        return "decoded"


class TrackingModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self._state_lock = threading.Lock()
        self.active_generates = 0
        self.max_concurrent_generates = 0
        self.disable_adapter_entries = 0

    def generate(self, **kwargs):
        with self._state_lock:
            self.active_generates += 1
            self.max_concurrent_generates = max(
                self.max_concurrent_generates,
                self.active_generates,
            )

        time.sleep(0.05)

        with self._state_lock:
            self.active_generates -= 1

        return torch.tensor([[11, 12, 13, 14]])

    def disable_adapter(self):
        model = self

        class _DisableAdapterCtx:
            def __enter__(self_inner):
                model.disable_adapter_entries += 1

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _DisableAdapterCtx()


class LLMServerGenerateTests(unittest.TestCase):
    def test_do_generate_serializes_shared_model_access(self):
        model = TrackingModel()
        tokenizer = FakeTokenizer()
        results = []
        errors = []
        barrier = threading.Barrier(3)

        def worker(use_adapter):
            try:
                barrier.wait(timeout=1)
                response = server._do_generate(
                    prompt="hello",
                    use_adapter=use_adapter,
                    max_new_tokens=8,
                    temperature=0.0,
                    system_prompt="sys",
                )
                results.append(response)
            except Exception as exc:  # pragma: no cover - failure path surfaces in assertions
                errors.append(exc)

        with patch.object(server, "_model", model), patch.object(server, "_tokenizer", tokenizer), patch.object(
            server, "_adapter_loaded", True
        ), patch.object(server, "_generate_lock", threading.Lock()):
            threads = [
                threading.Thread(target=worker, args=(True,)),
                threading.Thread(target=worker, args=(False,)),
            ]
            for thread in threads:
                thread.start()

            barrier.wait(timeout=1)

            for thread in threads:
                thread.join(timeout=1)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(model.max_concurrent_generates, 1)
        self.assertEqual(model.disable_adapter_entries, 1)
        self.assertEqual([response.text for response in results], ["decoded", "decoded"])

    def test_do_generate_logs_token_and_timing_metadata(self):
        model = TrackingModel()
        tokenizer = FakeTokenizer()

        with patch.object(server, "_model", model), patch.object(server, "_tokenizer", tokenizer), patch.object(
            server, "_adapter_loaded", False
        ), patch.object(server, "_generate_lock", threading.Lock()), self.assertLogs(server.log, level="INFO") as logs:
            response = server._do_generate(
                prompt="hello",
                use_adapter=False,
                max_new_tokens=8,
                temperature=0.0,
                system_prompt="sys",
            )

        self.assertEqual(response.tokens_generated, 1)
        log_text = "\n".join(logs.output)
        self.assertIn("生成完成", log_text)
        self.assertIn("input_tokens=3", log_text)
        self.assertIn("output_tokens=1", log_text)
        self.assertIn("use_adapter=False", log_text)
        self.assertIn("max_new_tokens=8", log_text)

    def test_submit_generate_uses_single_worker_queue(self):
        active = 0
        max_active = 0
        state_lock = threading.Lock()
        barrier = threading.Barrier(4)
        results = []
        errors = []

        def fake_run_generate_jobs(jobs):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)

            time.sleep(0.05)

            with state_lock:
                active -= 1

            return [
                server.GenerateResponse(
                    text=job.prompt,
                    used_adapter=job.use_adapter,
                    tokens_generated=1,
                    elapsed_ms=1.0,
                )
                for job in jobs
            ]

        def worker(idx):
            try:
                barrier.wait(timeout=1)
                result = server._submit_generate(
                    prompt=f"p{idx}",
                    use_adapter=bool(idx % 2),
                    max_new_tokens=8,
                    temperature=0.0,
                    system_prompt=None,
                )
                results.append(result)
            except Exception as exc:  # pragma: no cover - failure path surfaces in assertions
                errors.append(exc)

        with patch.object(server, "_run_generate_jobs", side_effect=fake_run_generate_jobs), patch.object(
            server, "_max_batch_size", 1
        ):
            threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(3)]
            for thread in threads:
                thread.start()

            barrier.wait(timeout=1)

            for thread in threads:
                thread.join(timeout=2)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 3)
        self.assertEqual(max_active, 1)

    def test_submit_generate_batches_compatible_jobs(self):
        batch_sizes = []
        barrier = threading.Barrier(4)
        results = []

        def fake_run_generate_jobs(jobs):
            batch_sizes.append(len(jobs))
            return [
                server.GenerateResponse(
                    text=job.prompt,
                    used_adapter=job.use_adapter,
                    tokens_generated=1,
                    elapsed_ms=1.0,
                )
                for job in jobs
            ]

        def worker(idx):
            barrier.wait(timeout=1)
            results.append(
                server._submit_generate(
                    prompt=f"p{idx}",
                    use_adapter=True,
                    max_new_tokens=8,
                    temperature=0.0,
                    system_prompt="sys",
                )
            )

        with patch.object(server, "_run_generate_jobs", side_effect=fake_run_generate_jobs), patch.object(
            server, "_max_batch_size", 4
        ), patch.object(server, "_batch_wait_seconds", 0.05):
            threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(3)]
            for thread in threads:
                thread.start()

            barrier.wait(timeout=1)

            for thread in threads:
                thread.join(timeout=2)

        self.assertEqual(len(results), 3)
        self.assertIn(3, batch_sizes)


if __name__ == "__main__":
    unittest.main()
