"""
本地 LLM HTTP 服务器：单进程加载 base + adapter，单端点切换推理。

Usage:
    # 带 adapter 启动（默认 adapter: models/webqsp/ablation/groupJ_schema_name）
    conda run -n py312_t271_cuda python -m oh_my_agent.llm_server.server \
        --adapter models/webqsp/ablation/groupJ_schema_name --port 8788

    # 不加载 adapter（纯 base 模型）
    conda run -n py312_t271_cuda python -m oh_my_agent.llm_server.server \
        --port 8788

API:
    POST /generate
        body: {"prompt": "...", "use_adapter": true, "max_new_tokens": 256,
               "temperature": 0.0, "system_prompt": "..."}
        response: {"text": "...", "used_adapter": true, "tokens_generated": 42}

    GET  /health
    GET  /info
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import time
from contextlib import asynccontextmanager
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# transformers warning_once 用 % 格式但传了 FutureWarning 类作为参数，会触发 logging 内部异常
# 屏蔽该 logger 的 WARNING 级别以下噪音
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

# ── 全局模型状态 ──────────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_adapter_loaded: bool = False
_model_name: str = ""
_adapter_path: str = ""
_generate_lock = threading.Lock()
_generate_queue: queue.Queue["_GenerateJob"] = queue.Queue()
_generate_worker_lock = threading.Lock()
_generate_worker_thread: threading.Thread | None = None
_max_batch_size = max(1, int(os.environ.get("LLM_SERVER_MAX_BATCH_SIZE", "4")))
_batch_wait_seconds = max(0.0, float(os.environ.get("LLM_SERVER_BATCH_WAIT_MS", "20")) / 1000)


@dataclass(frozen=True)
class _GenerateJob:
    prompt: str
    use_adapter: bool
    max_new_tokens: int
    temperature: float
    system_prompt: Optional[str]
    future: Future
    queued_at: float


def _load_model(model_name: str, adapter_path: Optional[str]) -> None:
    global _model, _tokenizer, _adapter_loaded, _model_name, _adapter_path

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        sys.exit("[Error] unsloth 未安装。pip install unsloth")

    log.info("加载 base 模型: %s", model_name)
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        local_files_only=True,
    )

    if adapter_path:
        log.info("加载 LoRA adapter: %s", adapter_path)
        from peft import PeftModel
        _model = PeftModel.from_pretrained(_model, adapter_path)
        _adapter_loaded = True
        _adapter_path = adapter_path
    else:
        _adapter_loaded = False

    FastLanguageModel.for_inference(_model)
    _model.eval()

    _tokenizer.padding_side = "left"
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model_name = model_name
    log.info("模型加载完成。adapter_loaded=%s", _adapter_loaded)


# ── FastAPI 应用 ──────────────────────────────────────────────────────────────
app = FastAPI(title="LLM Local Server", version="1.0")


class GenerateRequest(BaseModel):
    prompt: str
    use_adapter: bool = True
    max_new_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    system_prompt: Optional[str] = None


class GenerateResponse(BaseModel):
    text: str
    used_adapter: bool
    tokens_generated: int
    elapsed_ms: float


def _do_generate(prompt: str, use_adapter: bool, max_new_tokens: int,
                 temperature: float, system_prompt: Optional[str]) -> GenerateResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    if use_adapter and not _adapter_loaded:
        raise HTTPException(status_code=400,
                            detail="服务器启动时未加载 adapter，无法使用 use_adapter=true")

    # 构建 messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    result = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # transformers 5.x returns BatchEncoding; 4.x returns plain tensor
    if isinstance(result, torch.Tensor):
        input_ids = result.to(_model.device)
    else:
        input_ids = result["input_ids"].to(_model.device)
    n_input = input_ids.shape[-1]
    attention_mask = torch.ones_like(input_ids)

    gen_kwargs: dict = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        pad_token_id=_tokenizer.eos_token_id,
    )
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature

    lock_wait_start = time.perf_counter()
    with _generate_lock:
        lock_wait_ms = (time.perf_counter() - lock_wait_start) * 1000
        if lock_wait_ms >= 1:
            log.info(
                "等待推理锁 %.1f ms use_adapter=%s max_new_tokens=%s",
                lock_wait_ms,
                use_adapter,
                max_new_tokens,
            )

        t0 = time.perf_counter()
        with torch.no_grad():
            if _adapter_loaded and not use_adapter:
                with _model.disable_adapter():
                    output_ids = _model.generate(**gen_kwargs)
            else:
                output_ids = _model.generate(**gen_kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    new_tokens = output_ids[0][n_input:]
    tokens_generated = len(new_tokens)
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    log.info(
        "生成完成 input_tokens=%s output_tokens=%s elapsed_ms=%.1f "
        "lock_wait_ms=%.1f use_adapter=%s max_new_tokens=%s",
        n_input,
        tokens_generated,
        elapsed_ms,
        lock_wait_ms,
        use_adapter,
        max_new_tokens,
    )

    return GenerateResponse(
        text=text,
        used_adapter=use_adapter and _adapter_loaded,
        tokens_generated=tokens_generated,
        elapsed_ms=round(elapsed_ms, 1),
    )


def _jobs_compatible(left: _GenerateJob, right: _GenerateJob) -> bool:
    return (
        left.use_adapter == right.use_adapter
        and left.max_new_tokens == right.max_new_tokens
        and left.temperature == right.temperature
        and left.system_prompt == right.system_prompt
    )


def _do_generate_batch(jobs: list[_GenerateJob]) -> list[GenerateResponse]:
    if len(jobs) == 1:
        job = jobs[0]
        return [
            _do_generate(
                job.prompt,
                job.use_adapter,
                job.max_new_tokens,
                job.temperature,
                job.system_prompt,
            )
        ]

    first = jobs[0]
    if first.use_adapter and not _adapter_loaded:
        raise HTTPException(status_code=400,
                            detail="服务器启动时未加载 adapter，无法使用 use_adapter=true")

    rendered_prompts = []
    for job in jobs:
        messages = []
        if job.system_prompt:
            messages.append({"role": "system", "content": job.system_prompt})
        messages.append({"role": "user", "content": job.prompt})
        rendered_prompts.append(
            _tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        )

    encoded = _tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
    )
    input_ids = encoded["input_ids"].to(_model.device)
    attention_mask = encoded["attention_mask"].to(_model.device)
    n_input = input_ids.shape[-1]

    gen_kwargs: dict = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=first.max_new_tokens,
        do_sample=(first.temperature > 0.0),
        pad_token_id=_tokenizer.eos_token_id,
    )
    if first.temperature > 0.0:
        gen_kwargs["temperature"] = first.temperature

    lock_wait_start = time.perf_counter()
    with _generate_lock:
        lock_wait_ms = (time.perf_counter() - lock_wait_start) * 1000
        if lock_wait_ms >= 1:
            log.info(
                "等待推理锁 %.1f ms batch_size=%s use_adapter=%s max_new_tokens=%s",
                lock_wait_ms,
                len(jobs),
                first.use_adapter,
                first.max_new_tokens,
            )

        t0 = time.perf_counter()
        with torch.no_grad():
            if _adapter_loaded and not first.use_adapter:
                with _model.disable_adapter():
                    output_ids = _model.generate(**gen_kwargs)
            else:
                output_ids = _model.generate(**gen_kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    responses = []
    output_tokens = []
    for idx, job in enumerate(jobs):
        new_tokens = output_ids[idx][n_input:]
        tokens_generated = len(new_tokens)
        output_tokens.append(tokens_generated)
        responses.append(
            GenerateResponse(
                text=_tokenizer.decode(new_tokens, skip_special_tokens=True),
                used_adapter=job.use_adapter and _adapter_loaded,
                tokens_generated=tokens_generated,
                elapsed_ms=round(elapsed_ms, 1),
            )
        )

    log.info(
        "批量生成完成 batch_size=%s input_tokens=%s output_tokens=%s elapsed_ms=%.1f "
        "lock_wait_ms=%.1f use_adapter=%s max_new_tokens=%s",
        len(jobs),
        n_input,
        output_tokens,
        elapsed_ms,
        lock_wait_ms,
        first.use_adapter,
        first.max_new_tokens,
    )
    return responses


def _run_generate_jobs(jobs: list[_GenerateJob]) -> list[GenerateResponse]:
    if len(jobs) <= 1:
        return _do_generate_batch(jobs)

    try:
        return _do_generate_batch(jobs)
    except Exception:
        log.exception("批量生成失败，回退到逐条生成 batch_size=%s", len(jobs))
        return _do_generate_batch(jobs[:1]) + _run_generate_jobs(jobs[1:])


def _collect_batch(first_job: _GenerateJob) -> list[_GenerateJob]:
    if _max_batch_size <= 1 or _batch_wait_seconds <= 0:
        return [first_job]

    batch = [first_job]
    deferred = []
    deadline = time.perf_counter() + _batch_wait_seconds

    while len(batch) < _max_batch_size:
        timeout = max(0.0, deadline - time.perf_counter())
        if timeout == 0:
            break
        try:
            job = _generate_queue.get(timeout=timeout)
        except queue.Empty:
            break

        if _jobs_compatible(first_job, job):
            batch.append(job)
        else:
            deferred.append(job)

    for job in deferred:
        _generate_queue.put(job)
        _generate_queue.task_done()

    return batch


def _generate_worker_loop() -> None:
    while True:
        first_job = _generate_queue.get()
        batch = _collect_batch(first_job)
        running_jobs = []
        try:
            for job in batch:
                if job.future.set_running_or_notify_cancel():
                    running_jobs.append(job)
                else:
                    _generate_queue.task_done()

            if not running_jobs:
                continue

            queue_wait_ms = (time.perf_counter() - running_jobs[0].queued_at) * 1000
            if queue_wait_ms >= 1:
                log.info(
                    "等待推理队列 %.1f ms pending=%s batch_size=%s use_adapter=%s max_new_tokens=%s",
                    queue_wait_ms,
                    _generate_queue.qsize(),
                    len(running_jobs),
                    running_jobs[0].use_adapter,
                    running_jobs[0].max_new_tokens,
                )

            results = _run_generate_jobs(running_jobs)
            for job, result in zip(running_jobs, results):
                job.future.set_result(result)
        except Exception as exc:
            for job in running_jobs:
                job.future.set_exception(exc)
        finally:
            for _ in running_jobs:
                _generate_queue.task_done()


def _ensure_generate_worker_started() -> None:
    global _generate_worker_thread

    if _generate_worker_thread is not None and _generate_worker_thread.is_alive():
        return

    with _generate_worker_lock:
        if _generate_worker_thread is not None and _generate_worker_thread.is_alive():
            return
        _generate_worker_thread = threading.Thread(
            target=_generate_worker_loop,
            name="llm-generate-worker",
            daemon=True,
        )
        _generate_worker_thread.start()


def _submit_generate(
    prompt: str,
    use_adapter: bool,
    max_new_tokens: int,
    temperature: float,
    system_prompt: Optional[str],
) -> GenerateResponse:
    _ensure_generate_worker_started()
    future: Future = Future()
    _generate_queue.put(
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
    return future.result()


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    return _submit_generate(req.prompt, req.use_adapter, req.max_new_tokens,
                            req.temperature, req.system_prompt)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/info")
def info():
    vram_mb = None
    if torch.cuda.is_available():
        vram_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)
    return {
        "model": _model_name,
        "adapter": _adapter_path if _adapter_loaded else None,
        "adapter_loaded": _adapter_loaded,
        "device": str(next(_model.parameters()).device) if _model else None,
        "vram_allocated_mb": vram_mb,
    }


# ── 入口 ──────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="本地 LLM HTTP 服务器（/generate 单端点切换 base / adapter）")
    p.add_argument("--model", default="unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
                   help="base 模型名称或本地路径")
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter 目录（不传则只提供 base 推理）")
    p.add_argument("--port", type=int, default=8788)
    p.add_argument("--host", default="0.0.0.0")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _load_model(args.model, args.adapter)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
