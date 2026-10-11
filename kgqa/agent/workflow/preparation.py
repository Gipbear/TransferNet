"""对齐既有 RoG/MID 子图，只导出运行时白名单字段。"""

import json
from collections import defaultdict
from pathlib import Path

from kgqa.agent.workflow.models import RuntimeQuestion
from kgqa.retrieve.cwq_comparison import align_sample_entity_ids


class InputPreparation:
    def __init__(self, rog_dir: str | Path, mid_dir: str | Path) -> None:
        self.rog_dir, self.mid_dir = Path(rog_dir), Path(mid_dir)
        self.names = (self.rog_dir / "entities.txt").read_text().splitlines()
        self.mids = (self.mid_dir / "entities.txt").read_text().splitlines()

    def prepare(self, retrieval: str | Path, output: str | Path, split: str = "test") -> dict:
        records = {row["sample_index"]: row for row in self._rows(Path(retrieval))}
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        count = unresolved = 0
        with (self.rog_dir / f"{split}_simple.json").open() as left, \
                (self.mid_dir / f"{split}_simple.json").open() as right, target.open("w") as handle:
            for index, pair in enumerate(zip(left, right, strict=True)):
                if index not in records:
                    continue
                rog, identified = [json.loads(line) for line in pair]
                if rog["id"] != identified["id"] or rog["question"] != records[index]["question"]:
                    raise ValueError(f"样本 {index} ID 或问题不一致")
                question = self._question(index, rog, identified, records[index])
                handle.write(json.dumps(question.to_dict(), ensure_ascii=False) + "\n")
                count += 1
                unresolved += sum(not item["mids"] for item in question.candidates)
        if count != len(records):
            raise ValueError("输入样本数量不一致，禁止丢弃未对齐样本")
        return {"n": count, "unresolved_candidate_n": unresolved, "uses_gold": False}

    def _question(self, index: int, rog: dict, identified: dict, retrieved: dict) -> RuntimeQuestion:
        mapping = align_sample_entity_ids(rog["subgraph"]["tuples"], identified["subgraph"]["tuples"], index)
        name_to_mids = defaultdict(set)
        for local_id, mid_ids in mapping.items():
            for mid_id in mid_ids:
                name_to_mids[self.names[local_id]].add(self.mids[mid_id].lstrip(":"))
        topics = tuple(dict.fromkeys(self.mids[int(i)].lstrip(":") for i in identified["entities"]))
        paths, candidates, seen = [], [], set()
        for item in retrieved.get("mmr_reason_paths", []):
            path = {"path": item["path"], "log_score": item.get("log_score", 0.0)}
            if item["path"]:
                tail = item["path"][-1][2]
                mids = sorted(name_to_mids[tail])
                if len(mids) == 1:
                    path["tail_mid"] = mids[0]
                if tail not in seen:
                    candidates.append({"name": tail, "mids": mids,
                                       "resolution": "resolved" if len(mids) == 1 else "ambiguous"})
                    seen.add(tail)
            paths.append(path)
        return RuntimeQuestion(index, rog["id"], rog["question"], topics, tuple(paths), tuple(candidates))

    @staticmethod
    def _rows(path: Path) -> list[dict]:
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
