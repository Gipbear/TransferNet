"""
多次调用大模型，每次只判断一个路径是否正确，正确回答Y错误回答N，最终将所有正确的路径尾实体作为答案。

前提：服务器已启动
    conda run -n py312_t271_cuda python -m oh_my_agent.llm_server.server \
        --adapter models/webqsp/ablation/groupJ_schema_name --port 8788
"""

import json
from pathlib import Path
from tqdm import tqdm

from oh_my_agent.llm_server.client import LLMClient
from oh_my_agent.common.prompting import _format_schema_chain

# ── 配置 ──────────────────────────────────────────────────────────────────────
JSONL_PATH = "data/output/WebQSP/simple_agent_eval_debug.jsonl"
OUTPUT_PATH = "data/output/WebQSP/llm_path_by_path_eval.json"
SERVER_URL = "http://localhost:8788"

SYSTEM_PROMPT = """You are a KGQA evaluator. You will be given a question and a single reasoning path from a knowledge graph.
Determine if the relation in this path matches the intent of the question.

Use a loose standard: KG relation names are often broader or more general than the exact phrasing of the question. If the relation broadly or approximately covers the concept the question is asking about, judge Y. Focus on whether the tail entity would be a reasonable answer to the question, not whether the relation name is an exact synonym of the question's wording. Only judge N when the relation clearly targets a different concept entirely.

Answer ONLY 'Y' if the path correctly answers the question, or 'N' if it does not.

Examples:
Q: What is Obama's father's name?
Path: Barack Obama - [people.person.parents] -> Barack Obama Sr.
Output: Y

Q: Who directed Inception?
Path: Inception - [film.film.starring] -> Leonardo DiCaprio
Output: N"""


def build_prompt(question: str, path_edges: list) -> str:
    path_text = _format_schema_chain(path_edges)
    return f"Q: {question}\nPath: {path_text}\nOutput:"


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
    print("health:", client.health())
    print("info  :", client.info())
    print()

    records = load_records(JSONL_PATH)
    # 取少量测试还是全量？由于需要多次调用大模型，全量可能会很慢。默认全量，可以通过切片来快速测试。
    # records = records[:10]  # Uncomment to test on first 10 records
    print(f"共 {len(records)} 条记录，开始逐条路径校验...\n")

    results = []
    total_f1 = 0.0
    total_hit1 = 0.0
    total_hit_any = 0.0

    for i, rec in enumerate(tqdm(records, desc="Evaluating")):
        question = rec.get("question", "")
        named_paths = rec.get("named_mmr_reason_paths", [])
        raw_paths = rec.get("raw_mmr_reason_paths", [])
        gold_mids = rec.get("gold_mids", [])

        # 只评估 cited_path_indices 指向的路径（1-based，过滤越界和 0）
        cited_indices = sorted(
            idx for idx in rec.get("cited_path_indices", [])
            if 0 < idx <= len(named_paths)
        )

        predicted_answers = []
        path_evaluations = []

        for orig_idx in cited_indices:
            path_dict = named_paths[orig_idx - 1]
            path_edges = path_dict.get("path", [])
            if not path_edges:
                continue

            tail_entity = path_edges[-1][-1]
            prompt = build_prompt(question, path_edges)

            response = client.generate(
                prompt,
                use_adapter=False,
                max_new_tokens=2,
                temperature=0.0,
                system_prompt=SYSTEM_PROMPT,
            )

            output = response.text.strip().upper()
            is_correct = output == "Y"

            path_evaluations.append({
                "path_index": orig_idx,
                "path_text": _format_schema_chain(path_edges),
                "llm_output": output,
                "is_correct": is_correct,
                "tail_entity": tail_entity,
            })

            if is_correct and tail_entity not in predicted_answers:
                predicted_answers.append(tail_entity)

        # 从判为正确的路径中，取对应 raw_path 的尾 MID
        predicted_mids = set()
        for pe in path_evaluations:
            if pe["is_correct"]:
                raw_idx = pe["path_index"] - 1
                if raw_idx < len(raw_paths):
                    raw_edges = raw_paths[raw_idx].get("path", [])
                    if raw_edges:
                        predicted_mids.add(raw_edges[-1][-1])
        
        # 如果大模型选出的实体中包含了至少一个金标准实体，则认为本题判断正确（Hit@Any）
        gold_mids_set = set(gold_mids)
        is_final_correct = bool(predicted_mids & gold_mids_set)
        if is_final_correct:
            total_hit_any += 1

        # 计算新 F1（基于 predicted_mids vs gold_mids）
        if predicted_mids and gold_mids_set:
            inter = len(predicted_mids & gold_mids_set)
            prec = inter / len(predicted_mids)
            rec_ = inter / len(gold_mids_set)
            new_f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0.0
        elif not predicted_mids and not gold_mids_set:
            new_f1 = 1.0
        else:
            new_f1 = 0.0
        total_f1 += new_f1

        topic_entity = ", ".join(rec.get("named_topics", []))
        gold_names = rec.get("gold_answer_names", gold_mids)
        orig_hit_any = rec.get("hit_any", "N/A")
        orig_f1 = rec.get("f1", rec.get("macro_f1", "N/A"))
        correct_count = sum(1 for pe in path_evaluations if pe["is_correct"])
        total_count = len(path_evaluations)

        # 打印当前记录结果
        print(f"\n[{i+1}/{len(records)}] 最终判断: {'✓ 正确' if is_final_correct else '✗ 错误'}")
        print(f"问题    : {question}")
        print(f"主题词  : {topic_entity}")
        print(f"Gold 答案: {', '.join(str(x) for x in gold_names) if gold_names else '无'}")
        print(f"预测答案: {', '.join(predicted_answers) if predicted_answers else '无'}")
        print(f"路径判断: {correct_count}/{total_count} 条判为正确")
        print(f"原始指标: hit_any={orig_hit_any}  f1={orig_f1}")
        print(f"新指标  : hit_any={int(is_final_correct)}  f1={new_f1:.4f}")
        print(f"路径评估:")
        for pe in path_evaluations:
            print(f"  [{pe['llm_output']}] {pe['path_text']}")
        print("-" * 60)
        
        results.append({
            "sample_index": rec.get("sample_index", i),
            "question": question,
            "topic_entity": topic_entity,
            "original_pred_answers": rec.get("pred_answer_names", []),
            "new_pred_answers": predicted_answers,
            "is_final_correct": is_final_correct,
            "path_evaluations": path_evaluations
        })

    n = len(records)
    output_data = {
        "summary": {
            "total": n,
            "total_hit_any": total_hit_any,
            "hit_any_rate": total_hit_any / n if n else 0,
            "avg_f1": total_f1 / n if n else 0,
        },
        "results": results,
    }
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n=== 总结 ===")
    print(f"样本数  : {n}")
    print(f"Hit@Any : {total_hit_any}/{n} = {total_hit_any/n:.4f}" if n else "Hit@Any : 0/0")
    print(f"Avg F1  : {total_f1/n:.4f}" if n else "Avg F1  : 0.0000")
    print(f"结果已保存至 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
