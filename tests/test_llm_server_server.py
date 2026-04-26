"""Tests for ModelEngine and BatchScheduler (refactored llm_server).

Replaces the old monkey-patch-based tests with dependency-injection style.
All tests construct ModelEngine / BatchScheduler directly and inject fakes.
"""

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

from oh_my_agent.llm_server.client import GenerateResponse
from oh_my_agent.llm_server.engine import ModelEngine
from oh_my_agent.llm_server.scheduler import BatchScheduler


class FakeTokenizer:
    eos_token_id = 7
    eos_token = "<eos>"
    pad_token = "<pad>"
    padding_side = "left"

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        if not tokenize:
            return "RENDERED"
        return {"input_ids": torch.tensor([[11, 12, 13]])}

    def __call__(self, texts, return_tensors="pt", padding=True):
        if isinstance(texts, str):
            # single-string call: truncation re-tokenisation in _generate_one_batch
            return {"input_ids": torch.arange(100, 100 + 256).unsqueeze(0)}
        batch = len(texts)
        return {
            "input_ids": torch.tensor([[11, 12, 13]] * batch),
            "attention_mask": torch.ones(batch, 3, dtype=torch.long),
        }

    def decode(self, tokens, skip_special_tokens=True):
        return "decoded"


class TrackingModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self._state_lock = threading.Lock()
        self.active_generates = 0
        self.max_concurrent_generates = 0
        self.disable_adapter_entries = 0

    def parameters(self):
        yield torch.tensor([1.0])

    def generate(self, **kwargs):
        with self._state_lock:
            self.active_generates += 1
            self.max_concurrent_generates = max(
                self.max_concurrent_generates, self.active_generates
            )
        time.sleep(0.05)
        with self._state_lock:
            self.active_generates -= 1
        batch_size = kwargs["input_ids"].shape[0]
        return torch.tensor([[11, 12, 13, 14]] * batch_size)

    def disable_adapter(self):
        model = self

        class _Ctx:
            def __enter__(self_inner):
                model.disable_adapter_entries += 1

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()


def _make_engine(model, tokenizer, *, adapter_loaded: bool = True) -> ModelEngine:
    engine = ModelEngine()
    engine._model = model
    engine._tokenizer = tokenizer
    engine._adapter_loaded = adapter_loaded
    engine._model_name = "fake-model"
    engine._adapter_path = "fake-adapter" if adapter_loaded else ""
    return engine


# ── ModelEngine tests ─────────────────────────────────────────────────────────

class ModelEngineTests(unittest.TestCase):
    def test_generate_batch_disable_adapter_once(self):
        """Concurrent calls: use_adapter=False triggers disable_adapter exactly once."""
        model = TrackingModel()
        engine = _make_engine(model, FakeTokenizer())
        barrier = threading.Barrier(3)
        results = []
        errors = []

        def worker(use_adapter):
            try:
                barrier.wait(timeout=1)
                out = engine.generate_batch(
                    prompts=["RENDERED"],
                    use_adapter=use_adapter,
                    max_new_tokens=8,
                    temperature=0.0,
                )
                results.append(out)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(True,)),
            threading.Thread(target=worker, args=(False,)),
        ]
        for t in threads:
            t.start()
        barrier.wait(timeout=1)
        for t in threads:
            t.join(timeout=2)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(model.disable_adapter_entries, 1)
        for texts, _counts in results:
            self.assertEqual(texts[0], "decoded")

    def test_generate_batch_logs_metadata(self):
        """generate_batch emits a log with input/output tokens, use_adapter, max_new_tokens."""
        from oh_my_agent.llm_server.engine import log as engine_log

        model = TrackingModel()
        engine = _make_engine(model, FakeTokenizer(), adapter_loaded=False)

        with self.assertLogs(engine_log, level="INFO") as logs:
            _texts, counts = engine.generate_batch(
                prompts=["RENDERED"],
                use_adapter=False,
                max_new_tokens=8,
                temperature=0.0,
            )

        self.assertEqual(counts[0], 1)  # output_ids[3:] = [14], 1 new token
        log_text = "\n".join(logs.output)
        self.assertIn("批量生成完成", log_text)
        self.assertIn("input_tokens=3", log_text)
        self.assertIn("output_tokens=[1]", log_text)
        self.assertIn("use_adapter=False", log_text)
        self.assertIn("max_new_tokens=8", log_text)


# ── BatchScheduler tests ──────────────────────────────────────────────────────

class BatchSchedulerTests(unittest.TestCase):
    def test_scheduler_serializes_jobs(self):
        """With max_batch_size=1 the worker processes one job at a time (max_active==1)."""
        active = 0
        max_active = 0
        state_lock = threading.Lock()
        barrier = threading.Barrier(4)
        results = []
        errors = []

        def fake_run_jobs(jobs):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1
            return [
                GenerateResponse(
                    text=j.prompt, used_adapter=j.use_adapter,
                    tokens_generated=1, elapsed_ms=1.0,
                )
                for j in jobs
            ]

        engine = _make_engine(TrackingModel(), FakeTokenizer())
        scheduler = BatchScheduler(engine, max_batch_size=1, batch_wait_seconds=0.0)
        scheduler.start()
        try:
            with patch.object(scheduler, "_run_jobs", side_effect=fake_run_jobs):

                def worker(idx):
                    try:
                        barrier.wait(timeout=1)
                        future = scheduler.submit(
                            prompt=f"p{idx}",
                            use_adapter=bool(idx % 2),
                            max_new_tokens=8,
                            temperature=0.0,
                            system_prompt=None,
                        )
                        results.append(future.result(timeout=5))
                    except Exception as exc:
                        errors.append(exc)

                threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
                for t in threads:
                    t.start()
                barrier.wait(timeout=1)
                for t in threads:
                    t.join(timeout=5)
        finally:
            scheduler.stop()

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 3)
        self.assertEqual(max_active, 1)

    def test_scheduler_batches_compatible_jobs(self):
        """Compatible jobs submitted concurrently are merged into a single batch."""
        batch_sizes = []
        barrier = threading.Barrier(4)
        results = []

        def fake_run_jobs(jobs):
            batch_sizes.append(len(jobs))
            return [
                GenerateResponse(
                    text=j.prompt, used_adapter=j.use_adapter,
                    tokens_generated=1, elapsed_ms=1.0,
                )
                for j in jobs
            ]

        engine = _make_engine(TrackingModel(), FakeTokenizer())
        scheduler = BatchScheduler(engine, max_batch_size=4, batch_wait_seconds=0.05)
        scheduler.start()
        try:
            with patch.object(scheduler, "_run_jobs", side_effect=fake_run_jobs):

                def worker(idx):
                    barrier.wait(timeout=1)
                    future = scheduler.submit(
                        prompt=f"p{idx}",
                        use_adapter=True,
                        max_new_tokens=8,
                        temperature=0.0,
                        system_prompt="sys",
                    )
                    results.append(future.result(timeout=5))

                threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
                for t in threads:
                    t.start()
                barrier.wait(timeout=1)
                for t in threads:
                    t.join(timeout=5)
        finally:
            scheduler.stop()

        self.assertEqual(len(results), 3)
        self.assertIn(3, batch_sizes)

    def test_scheduler_batches_with_max_new_tokens_widening(self):
        """Jobs with different max_new_tokens are batched; each response is truncated to its limit."""

        class CapturingModel(TrackingModel):
            def __init__(self):
                super().__init__()
                self.called_max_new_tokens: list[int] = []

            def generate(self, **kwargs):
                max_new = kwargs.get("max_new_tokens", 1)
                self.called_max_new_tokens.append(max_new)
                input_ids = kwargs["input_ids"]
                batch_size = input_ids.shape[0]
                extra = torch.arange(100, 100 + max_new).unsqueeze(0).expand(batch_size, -1)
                return torch.cat([input_ids, extra], dim=1)

        model = CapturingModel()
        engine = _make_engine(model, FakeTokenizer(), adapter_loaded=False)
        scheduler = BatchScheduler(engine, max_batch_size=4, batch_wait_seconds=0.05)
        scheduler.start()

        job_max_tokens = [2, 4, 8]
        barrier = threading.Barrier(len(job_max_tokens) + 1)
        futures: list[tuple[int, object]] = []

        def worker(max_new_tokens):
            barrier.wait(timeout=1)
            future = scheduler.submit(
                prompt="hello",
                use_adapter=False,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                system_prompt=None,
            )
            futures.append((max_new_tokens, future))

        threads = [threading.Thread(target=worker, args=(t,)) for t in job_max_tokens]
        for t in threads:
            t.start()
        barrier.wait(timeout=1)
        for t in threads:
            t.join(timeout=5)

        results = [(mn, f.result(timeout=5)) for mn, f in futures]
        scheduler.stop()

        self.assertEqual(len(results), 3)
        # effective_max = max(2, 4, 8) = 8 was passed to generate()
        self.assertIn(8, model.called_max_new_tokens)
        # each response is truncated to its original max_new_tokens
        for original_max, response in results:
            self.assertEqual(response.tokens_generated, original_max)

    def test_scheduler_stop_drains_pending_futures(self):
        """Futures enqueued after the shutdown SENTINEL receive RuntimeError."""
        first_job_started = threading.Event()
        release_first_job = threading.Event()

        class BlockingModel(TrackingModel):
            def generate(self, **kwargs):
                first_job_started.set()
                release_first_job.wait(timeout=5)
                batch_size = kwargs["input_ids"].shape[0]
                return torch.tensor([[11, 12, 13, 14]] * batch_size)

        engine = _make_engine(BlockingModel(), FakeTokenizer(), adapter_loaded=False)
        scheduler = BatchScheduler(engine, max_batch_size=1, batch_wait_seconds=0.0)
        scheduler.start()

        future1 = scheduler.submit(
            prompt="p1", use_adapter=False, max_new_tokens=8,
            temperature=0.0, system_prompt=None,
        )
        first_job_started.wait(timeout=2)

        # stop() runs in a thread: it enqueues SENTINEL then joins the worker
        stop_thread = threading.Thread(target=scheduler.stop)
        stop_thread.start()
        time.sleep(0.05)  # SENTINEL is enqueued essentially instantly; 50 ms is ample

        # job2/job3 land AFTER SENTINEL in the queue → worker exits before seeing them
        future2 = scheduler.submit(
            prompt="p2", use_adapter=False, max_new_tokens=8,
            temperature=0.0, system_prompt=None,
        )
        future3 = scheduler.submit(
            prompt="p3", use_adapter=False, max_new_tokens=8,
            temperature=0.0, system_prompt=None,
        )

        release_first_job.set()
        stop_thread.join(timeout=5)

        # job1 completes normally
        self.assertIsNotNone(future1.result(timeout=2))

        # job2 and job3 are drained with RuntimeError
        with self.assertRaises(RuntimeError):
            future2.result(timeout=2)
        with self.assertRaises(RuntimeError):
            future3.result(timeout=2)


if __name__ == "__main__":
    unittest.main()
