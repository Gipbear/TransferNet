"""
计算指定目录下三个评测 JSONL 文件的 hit1/hit_any/P/R/F1 宏平均指标。
用法:
    python scripts/calc_metrics.py <目录>
"""
import json
import sys
from pathlib import Path


# 每个文件使用的字段映射: (hit1, hit_any, precision, recall, f1)
FIELD_MAP = {
    "initial_retrieval.jsonl": ("mmr_top1_hit", "mmr_answer_path_hit", "mmr_precision", "mmr_answer_recall", "mmr_f1"),
    "initial_answer.jsonl":    ("hit1", "hit_any", "precision", "recall", "f1"),
    "checked_batch_eval.jsonl": ("hit1", "hit_any", "precision", "recall", "f1"),
}


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def calc_file(path: Path, fields: tuple[str, ...]) -> dict:
    f_hit1, f_hit_any, f_p, f_r, f_f1 = fields
    hit1s, hit_anys, ps, rs, f1s = [], [], [], [], []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            hit1s.append(float(d.get(f_hit1, 0)))
            hit_anys.append(float(d.get(f_hit_any, 0)))
            ps.append(float(d.get(f_p, 0)))
            rs.append(float(d.get(f_r, 0)))
            f1s.append(float(d.get(f_f1, 0)))
    return {
        "n": len(hit1s),
        "hit1":    mean(hit1s),
        "hit_any": mean(hit_anys),
        "P":       mean(ps),
        "R":       mean(rs),
        "F1":      mean(f1s),
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/calc_metrics.py <目录>")
        sys.exit(1)

    dir_path = Path(sys.argv[1])
    if not dir_path.is_dir():
        print(f"目录不存在: {dir_path}")
        sys.exit(1)

    header = f"{'文件':<30}  {'n':>5}  {'hit1':>7}  {'hit_any':>7}  {'P':>7}  {'R':>7}  {'F1':>7}"
    print(header)
    print("-" * len(header))

    for fname, fields in FIELD_MAP.items():
        fpath = dir_path / fname
        if not fpath.exists():
            print(f"{fname:<30}  (文件不存在)")
            continue
        m = calc_file(fpath, fields)
        print(
            f"{fname:<30}  {m['n']:>5}"
            f"  {m['hit1']:>7.4f}  {m['hit_any']:>7.4f}"
            f"  {m['P']:>7.4f}  {m['R']:>7.4f}  {m['F1']:>7.4f}"
        )


if __name__ == "__main__":
    main()
