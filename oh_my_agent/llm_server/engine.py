"""ModelEngine：封装 LLM 模型加载、prompt 渲染、批量生成。"""

import logging
import sys
import time
from typing import Optional

import torch

from .config import ServerConfig

log = logging.getLogger(__name__)


class ModelEngine:
    """封装模型/tokenizer/adapter 状态，提供统一的批量生成接口。"""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._adapter_loaded: bool = False
        self._model_name: str = ""
        self._adapter_path: str = ""

    # ── 模型加载 ──────────────────────────────────────────────────────────────

    def load(self, cfg: ServerConfig) -> None:
        """加载 base 模型与可选 LoRA adapter。"""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            sys.exit("[Error] unsloth 未安装。pip install unsloth")

        log.info("加载 base 模型: %s", cfg.model_name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            dtype=None,
            load_in_4bit=True,
            local_files_only=True,
        )

        if cfg.adapter_path:
            log.info("加载 LoRA adapter: %s", cfg.adapter_path)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, cfg.adapter_path)
            self._adapter_loaded = True
            self._adapter_path = cfg.adapter_path
        else:
            self._adapter_loaded = False

        FastLanguageModel.for_inference(model)
        model.eval()

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._model = model
        self._tokenizer = tokenizer
        self._model_name = cfg.model_name
        log.info("模型加载完成。adapter_loaded=%s", self._adapter_loaded)

    # ── Prompt 渲染 ───────────────────────────────────────────────────────────

    def render(self, prompt: str, system_prompt: Optional[str]) -> str:
        """将 prompt + system_prompt 渲染为 chat template 文本（不 tokenize）。

        使用 tokenize=False 返回字符串，避免 transformers 4.x vs 5.x 兼容问题。
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    # ── 批量生成 ──────────────────────────────────────────────────────────────

    def generate_batch(
        self,
        prompts: list[str],
        use_adapter: bool,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[list[str], list[int]]:
        """批量生成。返回 (texts, token_counts)。

        - 用 self._tokenizer(prompts, return_tensors="pt", padding=True) 编码
        - 根据 use_adapter 决定是否 disable_adapter
        - 无 _generate_lock（调用方 BatchScheduler 已保证单线程）
        - 记录 input_tokens / output_tokens / elapsed_ms / use_adapter / batch_size
        """
        if self._model is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        if use_adapter and not self._adapter_loaded:
            raise ValueError("服务器启动时未加载 adapter，无法使用 use_adapter=True")

        encoded = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        input_ids = encoded["input_ids"].to(self._model.device)
        attention_mask = encoded["attention_mask"].to(self._model.device)
        # 批量左 padding 后，n_input 是 padding 后的统一序列长度
        n_input = input_ids.shape[-1]

        gen_kwargs: dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        t0 = time.perf_counter()
        with torch.no_grad():
            if self._adapter_loaded and not use_adapter:
                with self._model.disable_adapter():
                    output_ids = self._model.generate(**gen_kwargs)
            else:
                output_ids = self._model.generate(**gen_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        texts: list[str] = []
        token_counts: list[int] = []
        for i in range(len(prompts)):
            new_tokens = output_ids[i][n_input:]
            token_counts.append(len(new_tokens))
            texts.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True))

        log.info(
            "批量生成完成 batch_size=%s input_tokens=%s output_tokens=%s "
            "elapsed_ms=%.1f use_adapter=%s max_new_tokens=%s",
            len(prompts),
            n_input,
            token_counts,
            elapsed_ms,
            use_adapter,
            max_new_tokens,
        )

        return texts, token_counts

    # ── 状态信息 ──────────────────────────────────────────────────────────────

    def info(self) -> dict:
        """返回模型状态信息（供 /info 端点使用）。"""
        vram_allocated_mb = None
        if torch.cuda.is_available():
            vram_allocated_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)

        device = None
        if self._model is not None:
            device = str(next(self._model.parameters()).device)

        return {
            "model": self._model_name,
            "adapter": self._adapter_path if self._adapter_loaded else None,
            "adapter_loaded": self._adapter_loaded,
            "device": device,
            "vram_allocated_mb": vram_allocated_mb,
        }
