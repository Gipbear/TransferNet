"""
答案校验脚本：逐条读取 JSONL，用 LLM 判断预测答案是否正确，结果保存为 JSON。

前提：服务器已启动
    conda run -n py312_t271_cuda python -m oh_my_agent.llm_server.server \
        --adapter models/webqsp/ablation/groupJ_schema_name --port 8788
"""

import json
from pathlib import Path

from oh_my_agent.llm_server.client import LLMClient
from oh_my_agent.tools import AnswerCheckTool
from oh_my_agent.tools.answer_check import build_paths_text

# ── 配置 ──────────────────────────────────────────────────────────────────────
JSONL_PATH = "data/output/WebQSP/simple_agent_eval_debug.jsonl"
OUTPUT_PATH = "data/output/WebQSP/llm_answer_check.json"
SERVER_URL = "http://localhost:8788"


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    client = LLMClient(SERVER_URL)
    checker = AnswerCheckTool(
        client=client,
        mode="verify",
        default_use_adapter=False,
        default_max_new_tokens=256,
    )
    print("health:", client.health())
    print("info  :", client.info())
    print()

    records = load_records(JSONL_PATH)
    print(f"共 {len(records)} 条记录，开始逐条校验...\n")

    results = []
    llm_correct = 0

    for i, rec in enumerate(records):
        question = rec.get("question", "")
        pred_answers = rec.get("pred_answer_names", [])
        result = checker(
            question,
            pred_answers,
            build_paths_text(rec),
        )

        is_correct = result.verdict == "CORRECT"
        if is_correct:
            llm_correct += 1

        results.append({
            "sample_index": rec.get("sample_index", i),
            "question": question,
            "pred_answers": pred_answers,
            "gold_mids": rec.get("gold_mids", []),
            "user_prompt": result.prompt,
            "llm_raw_output": result.raw_output,
            "verdict": result.verdict,
            "path_verdicts": result.path_verdicts,
            "path_reasons": result.path_reasons,
            "any_valid_path": result.any_valid_path,
            "match": result.match,
            "match_detail": result.match_detail,
            "hit1": rec.get("hit1"),
            "hit_any": rec.get("hit_any"),
            "f1": rec.get("f1"),
            "tokens_generated": result.tokens_generated,
            "elapsed_ms": result.elapsed_ms,
        })

        marker = "✓" if is_correct else "✗"
        print(
            f"[{i+1:3d}/{len(records)}] {marker} hit1={rec.get('hit1')}  "
            f"verdict={result.verdict:10s}  {question[:60]}"
        )

    total = len(results)
    hit1_sum = sum(1 for r in results if r.get("hit1") == 1)
    agree = sum(
        1 for r in results
        if (r["verdict"] == "CORRECT") == (r.get("hit1") == 1)
    )

    summary = {
        "total": total,
        "llm_correct": llm_correct,
        "llm_correct_rate": round(llm_correct / total, 4) if total else 0,
        "hit1_correct": hit1_sum,
        "hit1_rate": round(hit1_sum / total, 4) if total else 0,
        "llm_hit1_agree": agree,
        "llm_hit1_agree_rate": round(agree / total, 4) if total else 0,
    }

    output = {"summary": summary, "results": results}
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'─'*60}")
    print(f"LLM 判断正确  : {llm_correct}/{total}  ({summary['llm_correct_rate']:.1%})")
    print(f"hit1 正确     : {hit1_sum}/{total}  ({summary['hit1_rate']:.1%})")
    print(f"LLM 与 hit1 一致率: {agree}/{total}  ({summary['llm_hit1_agree_rate']:.1%})")
    print(f"\n结果已保存至 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
