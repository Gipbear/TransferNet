"""BatchScheduler：Job 队列 + 单 worker 线程 + 动态批量收集 + graceful shutdown。"""

import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Optional

from .client import GenerateResponse
from .engine import ModelEngine

log = logging.getLogger(__name__)

_SENTINEL = object()  # worker 退出信号


@dataclass(frozen=True)
class _GenerateJob:
    prompt: str
    use_adapter: bool
    max_new_tokens: int    # 提交时的原始值，批内取 max 生成，输出时截断到此值
    temperature: float
    system_prompt: Optional[str]
    future: Future
    queued_at: float


class BatchScheduler:
    """封装生成请求的批量调度。单 worker 线程串行执行。"""

    def __init__(self, engine: ModelEngine, max_batch_size: int, batch_wait_seconds: float) -> None:
        self._engine = engine
        self._max_batch_size = max_batch_size
        self._batch_wait_seconds = batch_wait_seconds
        self._queue: queue.Queue = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._worker_lock = threading.Lock()

    def start(self) -> None:
        """启动 worker 线程（lifespan 调用）。"""
        with self._worker_lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="llm-generate-worker",
                daemon=True,
            )
            self._worker_thread.start()

    def stop(self) -> None:
        """停止 worker 线程，drain 未完成 future（lifespan 退出时调用）。"""
        self._queue.put(_SENTINEL)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=10)
        # drain 剩余 pending futures
        while True:
            try:
                item = self._queue.get_nowait()
                if isinstance(item, _GenerateJob) and not item.future.done():
                    item.future.set_exception(RuntimeError("server shutting down"))
            except queue.Empty:
                break

    def submit(
        self,
        prompt: str,
        use_adapter: bool,
        max_new_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> Future:
        """提交生成请求，返回 Future[GenerateResponse]。"""
        future: Future = Future()
        self._queue.put(
            _GenerateJob(
                prompt=prompt,
                use_adapter=use_adapter,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                future=future,
                queued_at=time.perf_counter(),
            )
        )
        return future

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _compatible(self, left: _GenerateJob, right: _GenerateJob) -> bool:
        """两个 job 是否可以合并到同一批次。

        use_adapter/temperature/system_prompt 必须相同。
        max_new_tokens 允许不同（批内取 max，各 job 输出按原值截断）。
        """
        return (
            left.use_adapter == right.use_adapter
            and left.temperature == right.temperature
            and left.system_prompt == right.system_prompt
        )

    def _collect_batch(self, first_job: _GenerateJob) -> list[_GenerateJob]:
        """从队列收集可合批的 job（最多等待 batch_wait_seconds）。"""
        if self._max_batch_size <= 1 or self._batch_wait_seconds <= 0:
            return [first_job]

        batch = [first_job]
        deferred = []
        deadline = time.perf_counter() + self._batch_wait_seconds

        while len(batch) < self._max_batch_size:
            timeout = max(0.0, deadline - time.perf_counter())
            if timeout == 0:
                break
            try:
                job = self._queue.get(timeout=timeout)
            except queue.Empty:
                break

            if job is _SENTINEL:
                # 收到 sentinel，放回并 task_done，让 worker_loop 下次 get 时看到
                self._queue.put(_SENTINEL)
                self._queue.task_done()
                break

            if self._compatible(first_job, job):
                batch.append(job)
            else:
                deferred.append(job)

        for job in deferred:
            self._queue.put(job)
            # task_done 撤销本次 get 的计数，因为该 job 被放回队列，
            # 将由后续迭代消费，不属于本批次
            self._queue.task_done()

        return batch

    def _run_jobs(self, jobs: list[_GenerateJob]) -> list[GenerateResponse]:
        """执行生成，失败时迭代回退到逐条。"""
        if len(jobs) == 1:
            return self._generate_one_batch(jobs)
        try:
            return self._generate_one_batch(jobs)
        except Exception:
            log.exception("批量生成失败，回退到逐条生成 batch_size=%s", len(jobs))
            results = []
            for job in jobs:
                results.extend(self._run_jobs([job]))
            return results

    def _generate_one_batch(self, jobs: list[_GenerateJob]) -> list[GenerateResponse]:
        """调用 engine.generate_batch，处理 max_new_tokens widening + 截断。"""
        t_queued = jobs[0].queued_at
        queue_wait_ms = (time.perf_counter() - t_queued) * 1000
        if queue_wait_ms >= 1:
            log.info(
                "等待推理队列 %.1f ms pending=%s batch_size=%s use_adapter=%s max_new_tokens=%s",
                queue_wait_ms,
                self._queue.qsize(),
                len(jobs),
                jobs[0].use_adapter,
                jobs[0].max_new_tokens,
            )

        first = jobs[0]
        # max_new_tokens 取批内最大值，各 job 输出截断到原始值
        effective_max = max(j.max_new_tokens for j in jobs)

        # render prompts
        rendered = [self._engine.render(j.prompt, j.system_prompt) for j in jobs]

        t0 = time.perf_counter()
        texts, token_counts = self._engine.generate_batch(
            rendered, first.use_adapter, effective_max, first.temperature
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        responses = []
        for i, job in enumerate(jobs):
            text = texts[i]
            token_count = token_counts[i]

            # 截断到原始 max_new_tokens（当批内取 max 导致生成超过单 job 上限时）
            if token_count > job.max_new_tokens:
                tokenizer = self._engine._tokenizer
                ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
                text = tokenizer.decode(ids[:job.max_new_tokens], skip_special_tokens=True)
                token_count = job.max_new_tokens

            responses.append(GenerateResponse(
                text=text,
                used_adapter=job.use_adapter and self._engine._adapter_loaded,
                tokens_generated=token_count,
                elapsed_ms=round(elapsed_ms, 1),
            ))
        return responses

    def _worker_loop(self) -> None:
        """Worker 线程主循环。收到 sentinel 时退出。"""
        while True:
            first_job = self._queue.get()

            if first_job is _SENTINEL:
                self._queue.task_done()
                break

            batch = self._collect_batch(first_job)
            running_jobs = []
            try:
                for job in batch:
                    if job.future.set_running_or_notify_cancel():
                        running_jobs.append(job)
                    else:
                        self._queue.task_done()

                if not running_jobs:
                    continue

                results = self._run_jobs(running_jobs)
                for job, result in zip(running_jobs, results):
                    job.future.set_result(result)
            except Exception as exc:
                for job in running_jobs:
                    if not job.future.done():
                        job.future.set_exception(exc)
            finally:
                for _ in running_jobs:
                    self._queue.task_done()
