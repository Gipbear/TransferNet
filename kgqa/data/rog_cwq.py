"""把 HuggingFace ``rmanluo/RoG-cwq`` 转成本项目的 NSM 格式（CWQ *_simple.json）。

转换后的目录可直接喂给 ``CompWebQ.train`` 和 ``kgqa`` 全链路，无需改动下游代码。

与现有 ``data/input/CWQ`` 的关键差别是**实体口径**：这里以实体名（name）建全局词表，
而原始 NSM 数据以 Freebase MID 建表。两者互为补充：

- MID 口径偏严，同一实体的重复 MID 条目会被判为不同实体（假阴性）
- name 口径偏松，同名不同实体会被合并（假阳性）

RoG / GNN-RAG 等工作按 name 评测，采用本转换结果才能与它们同口径对比。

直接读 parquet 而不走 ``datasets.load_dataset``：后者会在 HF_DATASETS_CACHE
下解包出约 37GB 的 arrow 缓存，在小盘机器上会写爆磁盘。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Iterator

# HF 上的 split 名 -> 本项目文件名前缀（NSM 用 dev 而非 validation）
SPLITS = {"train": "train", "validation": "dev", "test": "test"}
REPO_ID = "rmanluo/RoG-cwq"


def _norm(name: str) -> str:
    """规范化实体/关系名，使其对下游的 ``line.strip()`` 读法幂等。

    RoG-cwq 的名字里混有前导空格（``' Frank Harris'``）。若原样写进 entities.txt，
    ``CompWebQ/data.py`` 建表时的 ``ent2id[line.strip()]`` 会把它和 ``'Frank Harris'``
    合并，于是 ent2id 比文件行数短——而 ``*_simple.json`` 里的 id 是按写出时的行号，
    从第一个冲突处起全体错位。这个错位不报错，只让指标莫名偏低。在建表阶段就归一，
    写出与读回才对得上。顺带压掉内部换行，避免一个实体被写成两行。
    """
    return " ".join(name.split())


def _parquet_files(repo_dir: str, split: str) -> list[str]:
    data_dir = os.path.join(repo_dir, "data")
    files = [f for f in os.listdir(data_dir) if f.startswith(f"{split}-") and f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"{data_dir} 下没有 {split} 的 parquet 分片")
    return [os.path.join(data_dir, f) for f in sorted(files)]


def iter_split(repo_dir: str, split: str, batch_size: int = 64) -> Iterator[dict]:
    """逐条产出样本，内存占用只与单个 record batch 相关。"""
    import pyarrow.parquet as pq

    for path in _parquet_files(repo_dir, split):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield row


def build_vocab(repo_dir: str, splits=tuple(SPLITS)) -> tuple[dict[str, int], dict[str, int]]:
    """扫一遍全量数据建立 name->id 与 relation->id。

    实体词表须同时覆盖子图实体、topic 实体和答案实体：答案不在子图里是常态
    （CWQ 约有两成样本如此），但下游 ``ent2id[kb_id]`` 仍要能查到它。
    """
    ent2id: dict[str, int] = {}
    rel2id: dict[str, int] = {}
    for split in splits:
        for n, row in enumerate(iter_split(repo_dir, split), 1):
            for h, r, t in row["graph"]:
                for e in (_norm(h), _norm(t)):
                    if e not in ent2id:
                        ent2id[e] = len(ent2id)
                r = _norm(r)
                if r not in rel2id:
                    rel2id[r] = len(rel2id)
            for name in list(row["q_entity"]) + list(row["answer"]):
                name = _norm(name or "")
                if name and name not in ent2id:
                    ent2id[name] = len(ent2id)
            if n % 5000 == 0:
                print(f"  [vocab] {split} {n} 条, 实体 {len(ent2id):,} 关系 {len(rel2id):,}",
                      flush=True)
    return ent2id, rel2id


def convert_split(repo_dir: str, split: str, out_path: str,
                  ent2id: dict[str, int], rel2id: dict[str, int]) -> dict:
    stats = {"total": 0, "written": 0, "empty_graph": 0,
             "no_topic": 0, "answer_out_of_graph": 0}
    with open(out_path, "w", encoding="utf-8") as out:
        for row in iter_split(repo_dir, split):
            stats["total"] += 1
            graph = row["graph"]
            if not graph:
                stats["empty_graph"] += 1
                continue  # 与 CompWebQ DataLoader 跳过空子图的规则对齐

            tuples = [[ent2id[_norm(h)], rel2id[_norm(r)], ent2id[_norm(t)]]
                      for h, r, t in graph]
            sub_ents = sorted({e for tp in tuples for e in (tp[0], tp[2])})
            topics = [ent2id[_norm(e)] for e in row["q_entity"] if _norm(e) in ent2id]
            answers = [{"kb_id": _norm(a), "text": _norm(a)} for a in row["answer"] if _norm(a or "")]
            if not topics:
                stats["no_topic"] += 1
            in_graph = set(sub_ents)
            if not any(ent2id[a["kb_id"]] in in_graph for a in answers):
                stats["answer_out_of_graph"] += 1

            out.write(json.dumps({
                "id": row["id"],
                "question": row["question"].strip(),
                "entities": topics,
                "answers": answers,
                "subgraph": {"tuples": tuples, "entities": sub_ents},
            }, ensure_ascii=False) + "\n")
            stats["written"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="RoG-cwq -> NSM 格式转换")
    parser.add_argument("--output_dir", required=True, help="输出目录，如 data/input/RoG-CWQ")
    parser.add_argument("--repo_dir", default="",
                        help="RoG-cwq 快照目录；留空则自动 snapshot_download")
    args = parser.parse_args()

    repo_dir = args.repo_dir
    if not repo_dir:
        from huggingface_hub import snapshot_download
        repo_dir = snapshot_download(repo_id=REPO_ID, repo_type="dataset")
    print(f"[INFO] 数据快照: {repo_dir}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] 第一遍：建立全局 name/relation 词表 ...", flush=True)
    ent2id, rel2id = build_vocab(repo_dir)
    print(f"[INFO] 实体 {len(ent2id):,}  关系 {len(rel2id):,}", flush=True)

    for name, vocab in (("entities.txt", ent2id), ("relations.txt", rel2id)):
        # 下游按行号建表，所以「写出 N 行」必须能读回 N 个互不相同的 key，
        # 否则 *_simple.json 里的 id 会整体错位且不报错。
        assert all(k == k.strip() and "\n" not in k for k in vocab), f"{name} 词表未规范化"
        with open(os.path.join(args.output_dir, name), "w", encoding="utf-8") as fh:
            for key, _ in sorted(vocab.items(), key=lambda kv: kv[1]):
                fh.write(key + "\n")

    print("[INFO] 第二遍：写出 *_simple.json ...", flush=True)
    for split, prefix in SPLITS.items():
        out_path = os.path.join(args.output_dir, f"{prefix}_simple.json")
        st = convert_split(repo_dir, split, out_path, ent2id, rel2id)
        print(f"  {prefix}: 写出 {st['written']}/{st['total']} 条"
              f"（空子图 {st['empty_graph']}，无 topic {st['no_topic']}，"
              f"答案不在子图内 {st['answer_out_of_graph']}）", flush=True)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
