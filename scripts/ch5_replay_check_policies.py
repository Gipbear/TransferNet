"""补采完成后回放校验策略对照，分别固定批数与使用原生停止条件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kgqa.agent.common import load_entity_map
from kgqa.agent.replay import _ReplaySession
from scripts.ch5_collect_alternate_checks import AlternateChecks
from scripts.ch5_followup_replay import FollowupReplay
from scripts.replay_ch5_ablation import _load_jsonl, replay_config


class CheckPolicyReplay:
    @staticmethod
    def backfill(output: Path, archive: Path) -> None:
        marker = f"<!-- ch5-check-policies:{output.resolve()} -->"
        if marker in archive.read_text(encoding="utf-8"):
            return
        rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
        lines = [marker, "", "### 同轨迹checker策略对照完成", "",
                 f"- 输出：`{output}`；10个配置，每配置N=1581。",
                 "- fixed2固定最多两批并关闭早停；native_stop使用各策略原生停止条件。",
                 "- no_check只接受生成模型实际引用的路径，无新增模型判断；补全后二次推理仍未包含。",
                 "- 耗时继承原轨迹及补采调用，不是独占环境的端到端测速。", "",
                 "- 本表兼容现役parser；格式不完整输出的实际数量见每行summary的invalid_output_batches，不能当作完整有效判断。", "",
                 "| 配置 | Hit | Hit@1 | Macro-F1 | Micro-F1 | EM | 平均批数 |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for name, row in rows.items():
            if row["n"] != 1581:
                raise ValueError("不能回填非完整样本的正式对照表")
            metrics = " | ".join(f"{row[key] * 100:.2f}%" for key in
                                 ("hit_any", "hit1", "macro_f1", "micro_f1", "exact_match"))
            lines.append(f"| {name} | {metrics} | {row['avg_batches_used']:.4f} |")
        with archive.open("a", encoding="utf-8") as handle:
            handle.write("\n\n" + "\n".join(lines) + "\n")

    @staticmethod
    def unchecked(check: dict) -> dict:
        candidates = check["candidate_answers"]
        return check | {
            "check_mode": "disabled", "raw_output": "", "rejected_answer_indices": [],
            "accepted_path_indices": list(check["cited_path_indices"]),
            "predicted_answer_names": [c["name"] for c in candidates if c["name"]],
            "predicted_mids": [c["mid"] for c in candidates if c["mid"]],
            "path_evaluations": [p | {"is_correct": True, "raw_output": ""} for p in check["path_evaluations"]],
            "total_tokens_generated": 0, "total_elapsed_ms": 0.0,
            "any_accepted_path": bool(check["cited_path_indices"]),
        }

    @classmethod
    def transform(cls, record: dict, cache: dict[tuple[int, int], dict], mode: str) -> dict:
        iterations = []
        for iteration in record["iterations"]:
            first = iteration["batch_index"] == 1
            check = iteration["path_check"]
            if mode == "no_check" or (mode == "skip_first_strict" and first):
                check = cls.unchecked(check)
            elif (mode == "all_strict" and first) or (mode == "all_loose" and not first):
                check = cache[(record["sample_index"], iteration["batch_index"])]["path_check"]
            iterations.append(iteration | {"path_check": check})
        return record | {"iterations": iterations}

    def run(self, source: Path, checks: Path, output: Path) -> None:
        state = json.loads((checks / "progress.json").read_text(encoding="utf-8"))
        if state["status"] != "completed" or state["completed_batches"] != state["total_batches"]:
            raise ValueError("checker补采尚未完成，拒绝输出全量对照")
        records = _load_jsonl(source / "checked_batch_eval.jsonl")
        requests = [AlternateChecks.request(r, it) for r in records for it in r["iterations"]]
        collector = AlternateChecks(source, checks, "http://127.0.0.1:8788")
        cache = collector.cached(requests)
        if len(cache) != len(requests):
            raise ValueError("补采缓存缺失，不能回放所有策略")
        output.mkdir(parents=True, exist_ok=True)
        runner = FollowupReplay(source, output, Path("data/resources/WebQSP/fbwq_full/mapped_entities.txt"))
        runner.write_json("manifest.json", {
            "source": str(source.resolve()), "checks": str(checks.resolve()),
            "source_manifest": json.loads((checks / "manifest.json").read_text(encoding="utf-8")),
            "scope": "同轨迹checker策略对照；固定批数与各模式原生早停分开报告；无新的答题生成",
        })
        session = _ReplaySession(load_entity_map(str(runner.entity_map)), hybrid_check=True)
        runner.verify(records, session)
        rows = {}
        for mode in ("hybrid", "all_loose", "all_strict", "skip_first_strict", "no_check"):
            selected = [self.transform(record, cache, mode) for record in records]
            hybrid = mode in {"hybrid", "skip_first_strict"}
            session.agent.check_tool_after_first = session.agent.check_tool if hybrid else None
            for stop in (False, True):
                flags = runner.base | {"no_early_stop": not stop, "no_all_wrong_after_answer_stop": not stop}
                name = f"{mode}_{'native_stop' if stop else 'fixed2'}"
                result, summary = replay_config(selected, session=session, allow_prefix=True, **flags)
                summary["invalid_output_batches"] = sum(
                    not AlternateChecks.valid_result(it["path_check"])
                    for row in result for it in row["iterations"] if it["path_check"]["check_mode"] != "disabled"
                )
                runner.save_variant(name, result, summary, flags | {"hybrid_check": hybrid})
                rows[name] = summary
                runner.write_json("results.json", rows)
                runner.write_json("progress.json", {"status": "running", "completed": len(rows), "total": 10})
                print(f"[DONE] {name}: Hit={summary['hit_any']} Hit@1={summary['hit1']} "
                      f"Macro-F1={summary['macro_f1']} Micro-F1={summary['micro_f1']} EM={summary['exact_match']}",
                      flush=True)
        runner.write_json("progress.json", {"status": "completed", "completed": len(rows), "total": 10})


def main() -> None:
    parser = argparse.ArgumentParser(description="补采checker完成后的同轨迹策略对照")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    CheckPolicyReplay().run(args.source, args.checks, args.output)
    if args.archive:
        CheckPolicyReplay.backfill(args.output, args.archive)


if __name__ == "__main__":
    main()
