"""使用原生 PFIT 提示、单 BOS 与常驻 HTTP 服务执行固定模型推理。"""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from threading import Lock

from transformers import AutoTokenizer

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.interfaces import Answerer
from kgqa.agent.workflow.models import AnswerResult
from kgqa.agent.workflow.persistence import ArtifactStore
from kgqa.pfit.eval import is_rejection_response, parse_output
from kgqa.pfit.formats import build_user_content, select_format_prompt
from kgqa.runtime import file_fingerprint
from kgqa.serving.llm.client import LLMClient


class PfitPromptRenderer:
    def __init__(self, config: WorkflowConfig, tokenizer: AutoTokenizer) -> None:
        self.config, self.tokenizer = config, tokenizer
        self.system_prompt = select_format_prompt("v2", True, reject_prompt=False)

    def render(self, question: str, paths: tuple[dict, ...]) -> tuple[str, int, tuple[dict, ...]]:
        selected = paths[:self.config.max_paths]
        while True:
            metadata = [(item.get("path", []), item.get("log_score", 0.0), i + 1)
                        for i, item in enumerate(selected)]
            prompt = build_user_content(metadata, question, show_score=False, path_format="chain")
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            if len(ids) <= self.config.max_input_tokens:
                return prompt, len(ids), selected
            if not selected:
                raise ValueError("问题本身超过输入 token 预算")
            selected = selected[:-1]


class PfitAnswerer(Answerer):
    def __init__(self, config: WorkflowConfig, model: str, adapter: str, cache_dir: Path,
                 url: str = "http://127.0.0.1:8788") -> None:
        self.config = config
        self.model, self.adapter = str(Path(model).resolve()), str(Path(adapter).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.client = LLMClient(url, timeout=300)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = select_format_prompt("v2", True, reject_prompt=False)
        self.renderer = PfitPromptRenderer(config, self.tokenizer)
        self.adapter_sha256 = file_fingerprint(Path(self.adapter) / "adapter_model.safetensors")["sha256"]
        self.lock = Lock()
        self.actual_calls = 0
        self.info = self.client.info()
        served = self.info.get("adapter", self.info.get("adapter_path", ""))
        if str(Path(served or "").resolve()) != self.adapter:
            raise ValueError(f"服务 adapter 与配置不一致: {served}")
        served_model = self.info.get("model", self.info.get("model_path", ""))
        if str(Path(served_model or "").resolve()) != self.model:
            raise ValueError(f"服务 model 与配置不一致: {served_model}")

    def render(self, question: str, paths: tuple[dict, ...]) -> tuple[str, int, tuple[dict, ...]]:
        return self.renderer.render(question, paths)

    def answer(self, question: str, paths: tuple[dict, ...]) -> AnswerResult:
        prompt, tokens, selected = self.render(question, paths)
        target = self.cache_path(prompt)
        with ArtifactStore.cache_lock(target.with_suffix(".lock")):
            return self._answer_locked(prompt, tokens, selected, target)

    def _answer_locked(self, prompt: str, tokens: int, selected: tuple[dict, ...], target: Path) -> AnswerResult:
        if target.exists():
            raw = json.loads(target.read_text())
            raw["answers"], raw["cited_indices"] = tuple(raw["answers"]), tuple(raw["cited_indices"])
            return replace(AnswerResult(**raw), cache_hit=True, used_path_count=len(selected))
        response = self.client.generate(
            prompt, use_adapter=True, max_new_tokens=self.config.max_new_tokens,
            temperature=0.0, system_prompt=self.system_prompt,
        )
        if not response.used_adapter:
            raise ValueError("服务未使用固定 adapter，禁止混用基座输出")
        parsed = parse_output(response.text, "v2")
        rejected = is_rejection_response(parsed)
        answers = [] if rejected else parsed["answers"]
        display_map = {display: canonical for path in selected
                       for display, canonical in path.get("display_to_canonical", {}).items()}
        answers = [display_map.get(answer, answer) for answer in answers]
        result = AnswerResult(tuple(answers), rejected, parsed["format_ok"],
                              tuple(sorted(parsed["cited_indices"])), response.text,
                              tokens, response.tokens_generated, response.elapsed_ms, len(selected))
        ArtifactStore.atomic_text(target, json.dumps(asdict(result), ensure_ascii=False))
        with self.lock:
            self.actual_calls += 1
        return result

    def cache_path(self, prompt: str) -> Path:
        signature = json.dumps(["native_pfit_v2_single_bos", self.model, self.adapter, self.adapter_sha256,
                                self.system_prompt, prompt, self.config.max_new_tokens])
        return self.cache_dir / (hashlib.sha256(signature.encode()).hexdigest() + ".json")
