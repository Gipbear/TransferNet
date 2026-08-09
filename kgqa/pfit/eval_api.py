"""OpenAI 兼容 API 版路径监督评测。

与 kgqa.pfit.eval 的差异只有"谁来生成":提示词构造、输出解析、逐样本指标和汇总
全部复用同一批函数,产物 schema 也一致,可直接与本地评测结果同口径比较。

密钥只从环境变量读取(默认 SILICONFLOW_API_KEY),不接受命令行传入,避免落进
进程列表和 shell 历史。

    python -m kgqa.pfit.eval_api --dataset cwq --input <JSONL> --exp_dir <DIR> \
        --model Qwen/Qwen3.5-9B --system_prompt_file <TXT>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

from kgqa.pfit.eval import (
    compute_answer_metrics,
    compute_faithfulness,
    get_all_path_entities,
    is_rejection_response,
    label_golden_indices,
    parse_output,
    summarize,
    truncate_paths_by_score,
)
from kgqa.pfit.formats import build_user_content
from kgqa.pfit.specs import get_pfit_spec
from kgqa.runtime import (
    add_runtime_arguments,
    configure_runtime,
    emit_event,
    update_progress,
)

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"


def call_chat(base_url: str, api_key: str, model: str, messages: list,
              *, max_tokens: int, temperature: float, enable_thinking: bool,
              timeout: int, max_retries: int) -> tuple[str, dict]:
    """返回 (正文, usage)。失败重试用指数退避 + 抖动。"""
    import requests

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(f"{base_url}/chat/completions", json=payload,
                              headers=headers, timeout=timeout)
            if r.status_code == 200:
                d = r.json()
                msg = d["choices"][0]["message"]
                return (msg.get("content") or ""), d.get("usage", {})
            # 429/5xx 才值得重试
            if r.status_code not in (408, 429, 500, 502, 503, 504):
                return "", {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
            last_err = f"HTTP {r.status_code}"
        except Exception as exc:                      # noqa: BLE001 - 网络异常一律重试
            last_err = repr(exc)
        if attempt < max_retries:
            time.sleep(min(2 ** attempt, 20) + random.random())
    return "", {"error": f"重试 {max_retries} 次仍失败: {last_err}"}


def run_eval_api(*, dataset: str, input_path: str, exp_dir: str, model: str,
                 system_prompt_file: str, fmt: str = "v2",
                 path_format: str = "chain", entity_repr: str | None = None,
                 base_url: str = DEFAULT_BASE_URL, api_key_env: str = "SILICONFLOW_API_KEY",
                 limit: int = 0, max_paths: int = 0,
                 concurrency: int = 8, max_tokens: int = 512,
                 temperature: float = 0.0, enable_thinking: bool = False,
                 timeout: int = 120, max_retries: int = 4,
                 run_dir: str | None = None) -> dict:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"环境变量 {api_key_env} 未设置")

    spec = get_pfit_spec(dataset)
    entity_repr = entity_repr or spec.default_entity_repr
    if spec.entity_map_path:
        raise ValueError(
            f"{dataset} 需要 MID→Name 映射,本 API 评测器暂不支持;"
            "请用 kgqa.pfit.eval 本地评测"
        )

    with open(system_prompt_file, encoding="utf-8") as f:
        system_prompt = f.read().strip()
    if not system_prompt:
        raise ValueError(f"system prompt 文件为空: {system_prompt_file}")

    with open(input_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if limit > 0:
        samples = samples[:limit]

    eval_dir = os.path.join(exp_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    def build_messages(sample):
        question = spec.clean_question(sample.get("question", ""), sample.get("topics", []))
        mmr_paths = truncate_paths_by_score(
            sample.get("mmr_reason_paths", []), max_paths)
        paths_with_meta = [
            (p.get("path", []), p.get("log_score", 0.0), i + 1)
            for i, p in enumerate(mmr_paths)
        ]
        user_content = build_user_content(paths_with_meta, question,
                                          show_score=False, path_format=path_format)
        return ([{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_content}], mmr_paths)

    done = [0]
    usages = []

    def work(sample):
        messages, mmr_paths = build_messages(sample)
        raw, usage = call_chat(base_url, api_key, model, messages,
                               max_tokens=max_tokens, temperature=temperature,
                               enable_thinking=enable_thinking,
                               timeout=timeout, max_retries=max_retries)
        golden = sample.get("golden", [])
        parsed = parse_output(raw, fmt)
        golden_indices = label_golden_indices(mmr_paths, golden)
        path_entities = get_all_path_entities(mmr_paths)
        answer_m = compute_answer_metrics(parsed["answers"], golden)
        faith_m = compute_faithfulness(parsed["cited_indices"], golden_indices,
                                       parsed["answers"], path_entities)
        done[0] += 1
        usages.append(usage)
        if done[0] % 25 == 0 or done[0] == len(samples):
            update_progress(run_dir, completed=done[0], total=len(samples),
                            status="running", phase="API 路径监督评测")
        return {
            "sample_index":        sample.get("sample_index", -1),
            "question":            sample.get("question", ""),
            "hop":                 sample.get("hop"),
            "golden":              golden,
            "mmr_answer_path_hit": bool(golden_indices),
            "llm_raw_output":      raw,
            "llm_pred":            parsed["answers"],
            "is_rejection":        is_rejection_response(parsed),
            "llm_pred_expanded_mids": None,
            "llm_pred_disambiguated_mids": None,
            "cited_indices":       sorted(parsed["cited_indices"]),
            "golden_path_indices": sorted(golden_indices),
            "format_ok":           parsed["format_ok"],
            "hit1":                answer_m["hit1"],
            "hit_any":             answer_m["hit_any"],
            "precision":           round(answer_m["precision"], 4),
            "recall":              round(answer_m["recall"], 4),
            "f1":                  round(answer_m["f1"], 4),
            "exact_match":         answer_m["exact_match"],
            "tp":                  answer_m["tp"],
            "pred_n":              answer_m["pred_n"],
            "gold_n":              answer_m["gold_n"],
            "citation_accuracy":   faith_m["citation_accuracy"],
            "citation_recall":     faith_m["citation_recall"],
            "hallucination_rate":  faith_m["hallucination_rate"],
            "hallucinated_entities": faith_m["hallucinated_entities"],
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(work, samples))

    with open(os.path.join(eval_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = summarize(results, group_by_hop=spec.group_by_hop)
    n_err = sum(1 for u in usages if "error" in u)
    summary["api"] = {
        "model": model, "base_url": base_url, "enable_thinking": enable_thinking,
        "temperature": temperature, "max_tokens": max_tokens,
        "system_prompt_file": os.path.abspath(system_prompt_file),
        "failed_calls": n_err,
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usages),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usages),
    }
    with open(os.path.join(eval_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def build_parser():
    p = argparse.ArgumentParser(description="API 版路径监督评测(与本地 eval 同口径)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--system_prompt_file", required=True)
    p.add_argument("--format", default="v2", dest="fmt")
    p.add_argument("--path_format", default="chain")
    p.add_argument("--entity_repr", default=None)
    p.add_argument("--base_url", default=DEFAULT_BASE_URL)
    p.add_argument("--api_key_env", default="SILICONFLOW_API_KEY",
                   help="读取密钥的环境变量名;密钥不接受命令行传入")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_paths", type=int, default=0,
                   help="按检索得分保留前 N 条路径(≤0=不截断)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--enable_thinking", action="store_true")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max_retries", type=int, default=4)
    add_runtime_arguments(p)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    run_dir = a.run_dir or a.exp_dir
    configure_runtime(a, command="API 路径监督评测",
                      manifest={"dataset": a.dataset, "input": a.input, "model": a.model})
    summary = run_eval_api(
        dataset=a.dataset, input_path=a.input, exp_dir=a.exp_dir, model=a.model,
        system_prompt_file=a.system_prompt_file, fmt=a.fmt, path_format=a.path_format,
        entity_repr=a.entity_repr, base_url=a.base_url, api_key_env=a.api_key_env,
        limit=a.limit, max_paths=a.max_paths,
        concurrency=a.concurrency, max_tokens=a.max_tokens,
        temperature=a.temperature, enable_thinking=a.enable_thinking,
        timeout=a.timeout, max_retries=a.max_retries, run_dir=str(run_dir))
    update_progress(run_dir, completed=1, total=1, status="completed",
                    phase="API 路径监督评测")
    emit_event(run_dir, "phase_end", phase="API 路径监督评测")
    print(json.dumps(summary.get("overall", summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
