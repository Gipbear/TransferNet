"""从本次真实生成缓存补齐早期轨迹的路径与调用成本元数据。"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from kgqa.agent.workflow.persistence import ArtifactStore
from kgqa.agent.workflow.pfit import PfitPromptRenderer


class CachedTraceReader:
    def __init__(self, renderer: PfitPromptRenderer, model: str, adapter: str,
                 max_new_tokens: int, cache_dir: Path, adapter_sha256: str = "") -> None:
        self.renderer, self.model, self.adapter = renderer, model, adapter
        self.max_new_tokens, self.cache_dir = max_new_tokens, Path(cache_dir)
        self.adapter_sha256 = adapter_sha256

    def enrich(self, record: dict, import_cache: bool = False) -> dict:
        row = deepcopy(record)
        initial, initial_paths = self._cached(row["question"], row["question"]["paths"], import_cache)
        self._verify(row["initial"], initial)
        initial["cache_hit"] = True
        row["initial"], row["initial_paths"] = initial, initial_paths
        calls = [initial]
        if row["pfit_calls"] > 1:
            followup, paths = self._cached(row["question"], row["tool"]["paths"], import_cache)
            calls.append(followup)
            if row["source"] != "baseline":
                self._verify(row["final"], followup)
                row["final"], row["final_paths"] = followup, paths
        if row["source"] == "baseline":
            row["final"], row["final_paths"] = deepcopy(initial), initial_paths
        elif row["source"] == "comparison":
            row["final"].update(prompt_tokens=0, completion_tokens=0, elapsed_ms=0, used_path_count=0)
            row["final_paths"] = initial_paths
        row["call_results"] = calls
        return row

    def _cached(self, question: dict, paths: list, import_cache: bool) -> tuple[dict, list]:
        prompt, tokens, selected = self.renderer.render(question["question"], tuple(paths))
        signature = [self.model, self.adapter, self.renderer.system_prompt, prompt, self.max_new_tokens]
        key = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        result = json.loads((self.cache_dir / (key + ".json")).read_text())
        if result["prompt_tokens"] != tokens:
            raise ValueError("缓存提示 token 数不一致，禁止补造元数据")
        result.update(used_path_count=len(selected), cache_hit=False, generation_error="")
        if import_cache:
            if not self.adapter_sha256:
                raise ValueError("迁移缓存必须先验证 adapter 权重哈希")
            current = ["native_pfit_v2_single_bos", self.model, self.adapter, self.adapter_sha256,
                       self.renderer.system_prompt, prompt, self.max_new_tokens]
            target = self.cache_dir / (hashlib.sha256(json.dumps(current).encode()).hexdigest() + ".json")
            ArtifactStore.atomic_text(target, json.dumps(result, ensure_ascii=False))
        return result, list(selected)

    @staticmethod
    def _verify(recorded: dict, cached: dict) -> None:
        fields = ("answers", "is_rejection", "format_ok", "cited_indices", "raw_output")
        if any(recorded.get(key) != cached.get(key) for key in fields):
            raise ValueError("轨迹与真实生成缓存不一致，禁止改写预测")
