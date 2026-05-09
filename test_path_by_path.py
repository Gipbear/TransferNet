"""
多次调用大模型，每次只判断一个路径是否正确，正确回答Y错误回答N，最终从所有正确路径中抽取答案实体。

前提：服务器已启动
    conda run -n py312_t271_cuda python -m oh_my_agent.llm_server.server \
        --adapter models/webqsp/ablation/groupJ_schema_name --port 8788
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from oh_my_agent.llm_server.client import LLMClient
from oh_my_agent.common.prompting import _format_schema_chain
from oh_my_agent.tools.cited_path_check import CITED_PATH_CHECK_SYSTEM

# ── 配置 ──────────────────────────────────────────────────────────────────────
JSONL_PATH = "data/output/WebQSP/simple_agent_eval_debug.jsonl"
OUTPUT_PATH = "data/output/WebQSP/llm_path_by_path_eval.json"
SERVER_URL = "http://localhost:8788"


def build_prompt(question: str, path_edges: list) -> str:
    path_text = _format_schema_chain(path_edges)
    return f"Q: {question}\nPath: {path_text}\nOutput:"


def is_reverse_relation(relation: str) -> bool:
    return str(relation).endswith("_reverse")


def answer_entity_from_path(
    path_edges: list,
    *,
    reference_edges: list | None = None,
    gold_entities: set[str] | list[str] | None = None,
) -> str:
    """Extract answer entity from paths judged correct.

    For diagnostics with gold labels, if a 2-hop chain is rendered as
    ``E0 -> E1 <- E2`` and E1 is gold while E2 is not, use E1 instead of E2.
    """
    if not path_edges:
        return ""
    reference_edges = reference_edges or path_edges
    gold_set = set(gold_entities or [])
    if (
        gold_set
        and len(reference_edges) == 2
        and len(reference_edges[0]) >= 3
        and len(reference_edges[1]) >= 3
        and not is_reverse_relation(reference_edges[0][1])
        and is_reverse_relation(reference_edges[1][1])
        and reference_edges[0][2] == reference_edges[1][0]
        and reference_edges[0][2] in gold_set
        and reference_edges[-1][-1] not in gold_set
    ):
        return path_edges[0][2]
    return path_edges[-1][-1]


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Path-by-path LLM checker evaluation")
    parser.add_argument("--input", default=JSONL_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--server-url", default=SERVER_URL)
    parser.add_argument("--limit", type=int, default=0, help="0 means all records")
    parser.add_argument("--quiet", action="store_true", help="Only print final summary")
    return parser.parse_args()


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    client = LLMClient(args.server_url)
    system_prompt = CITED_PATH_CHECK_SYSTEM
    print("health:", client.health())
    print("info  :", client.info())
    print("prompt:", f"({len(system_prompt)} chars)")
    print()

    records = load_records(args.input)
    if args.limit > 0:
        records = records[: args.limit]
    # 取少量测试还是全量？由于需要多次调用大模型，全量可能会很慢。默认全量，可以通过切片来快速测试。
    # records = records[:10]  # Uncomment to test on first 10 records
    print(f"共 {len(records)} 条记录，开始逐条路径校验...\n")

    results = []
    total_f1 = 0.0
    total_hit1 = 0.0
    total_hit_any = 0.0
    total_path_count = 0
    llm_yes_count = 0
    vetoed_path_count = 0

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

            raw_edges = []
            raw_idx = orig_idx - 1
            if 0 <= raw_idx < len(raw_paths):
                raw_edges = raw_paths[raw_idx].get("path", [])
            answer_entity = answer_entity_from_path(
                path_edges,
                reference_edges=raw_edges,
                gold_entities=gold_mids,
            )
            prompt = build_prompt(question, path_edges)

            response = client.generate(
                prompt,
                use_adapter=False,
                max_new_tokens=2,
                temperature=0.0,
                system_prompt=system_prompt,
            )

            output = response.text.strip().upper()
            llm_is_correct = output.startswith("Y")
            if llm_is_correct:
                llm_yes_count += 1
            is_correct = llm_is_correct

            path_evaluations.append({
                "path_index": orig_idx,
                "path_text": _format_schema_chain(path_edges),
                "llm_output": output,
                "llm_is_correct": llm_is_correct,
                "is_correct": is_correct,
                "answer_entity": answer_entity,
            })

            if is_correct and answer_entity not in predicted_answers:
                predicted_answers.append(answer_entity)

        # 从判为正确的路径中，取对应 raw_path 的答案 MID
        predicted_mids = set()
        for pe in path_evaluations:
            if pe["is_correct"]:
                raw_idx = pe["path_index"] - 1
                if raw_idx < len(raw_paths):
                    raw_edges = raw_paths[raw_idx].get("path", [])
                    if raw_edges:
                        predicted_mids.add(
                            answer_entity_from_path(raw_edges, gold_entities=gold_mids)
                        )
        
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
        total_path_count += total_count

        if not args.quiet:
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
                print(f"  [{pe['llm_output']} -> {'Y' if pe['is_correct'] else 'N'}] {pe['path_text']}")
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
            "prompt_chars": len(system_prompt),
            "total_paths": total_path_count,
            "llm_yes_paths": llm_yes_count,
            "vetoed_paths": vetoed_path_count,
            "final_yes_paths": llm_yes_count - vetoed_path_count,
        },
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n=== 总结 ===")
    print(f"样本数  : {n}")
    print(f"Hit@Any : {total_hit_any}/{n} = {total_hit_any/n:.4f}" if n else "Hit@Any : 0/0")
    print(f"Avg F1  : {total_f1/n:.4f}" if n else "Avg F1  : 0.0000")
    print(f"Prompt   : ({len(system_prompt)} chars)")
    print(f"Paths    : total={total_path_count} llm_yes={llm_yes_count} vetoed={vetoed_path_count} final_yes={llm_yes_count - vetoed_path_count}")
    print(f"结果已保存至 {args.output}")

if __name__ == "__main__":
    main()
