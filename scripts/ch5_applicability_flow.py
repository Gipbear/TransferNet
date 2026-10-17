"""A4 适用条件判定流程：纯度门控的 oracle 上限与阈值标定。

A1 已证明增益由关系组纯度与补全前召回共同决定，且纯度 < 0.25 的样本在两数据集上
都是显著负收益。A4 把它变成一个可执行的门控规则，并回答两个问题：

  1. **上限**：若能用 gold 精确算出纯度并据此逐样本开关补全，指标能到多少？
     这是路径 B（语义门控）的目标值，也是"纯度是否是充分统计量"的检验。
  2. **阈值**：纯度分界取在哪里，以及该分界能否跨数据集一致地给出启用/关闭建议？

注意：本分析使用 gold 计算纯度，属 **oracle 上限**，不是可部署方案。
可部署版本需要由问题级语义判断来估计纯度（路径 B）。

复现：
    python -m scripts.ch5_applicability_flow --output <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.ch5_analysis_common import bootstrap_delta, load_probe
from scripts.ch5_gain_attribution import attach_precision

THRESHOLDS = (0.0, 0.25, 0.5, 0.75, 0.99)


def gated(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    """纯度达标的触发样本走开补全，其余回退关补全。"""
    def gate(row: dict[str, Any]) -> bool:
        return row["triggered"] == 1 and row["expansion_precision"] >= threshold

    stat = bootstrap_delta(rows, gate=gate)
    stat["kept"] = sum(1 for row in rows if gate(row))
    return stat


def render_scan(name: str, rows: list[dict[str, Any]]) -> tuple[str, dict]:
    lines = [f"**{name}**（触发 {sum(row['triggered'] for row in rows)} 条）", "",
             "| 纯度阈值 τ | 保留触发 | ΔMacro-F1 [95%CI] | ΔMicro-F1 [95%CI] |",
             "| ---: | ---: | ---: | ---: |"]
    payload = {}
    for threshold in THRESHOLDS:
        stat = gated(rows, threshold)
        payload[str(threshold)] = stat
        lines.append(
            f"| {threshold:g} | {stat['kept']} | "
            f"{stat['macro'] * 100:+.2f} [{stat['macro_lo'] * 100:+.2f}, {stat['macro_hi'] * 100:+.2f}] | "
            f"{stat['micro'] * 100:+.2f} [{stat['micro_lo'] * 100:+.2f}, {stat['micro_hi'] * 100:+.2f}] |"
        )
    lines.append("")
    lines.append("τ=0 等价于全开（即 F0）；τ 越大过滤越严。")
    lines.append("")
    return "\n".join(lines), payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", type=Path,
                        default=Path("data/analysis/20260919_1700__ch5_applicability_probe"
                                     "/probe_results.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    datasets = load_probe(args.probe)
    for rows in datasets.values():
        attach_precision(rows)

    payload: dict[str, Any] = {}
    lines = [
        "# A4 适用条件判定流程（oracle 上限）", "",
        "纯度由 gold 计算，因此下列数值是**上限**而非可部署结果；",
        "可部署版本需用问题级语义判断估计纯度（路径 B）。",
        "区间为 2000 次逐样本配对 bootstrap 的 95% 分位区间（seed=17）。", "",
        "## 纯度门控扫描", "",
    ]
    for name, rows in datasets.items():
        block, result = render_scan(name, rows)
        payload[name] = result
        lines.append(block)

    lines.append("## 阈值标定与判定流程")
    lines.append("")
    lines.append(A4_FLOW.format(
        webqsp_purity="84.7%", cwq_purity="27.1%",
        webqsp_neg=13, cwq_neg=34, webqsp_total=100, cwq_total=54))

    (args.output / "applicability_flow.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = "\n".join(lines)
    (args.output / "applicability_flow.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


A4_FLOW = """### 阈值选取依据

纯度 < 0.25 的样本在两个数据集上都是**显著负收益**（A1）：

| 数据集 | 纯度<0.25 的触发数 | 该档 ΔMicro-F1 [95%CI] | 纯度≥0.25 各档 ΔMicro-F1 |
| --- | ---: | ---: | --- |
| WebQSP | {webqsp_neg}/{webqsp_total} | −11.60 [−14.78, −8.13] | +36.34 / +33.75 / +48.78（全正） |
| CWQ | {cwq_neg}/{cwq_total} | −11.49 [−18.19, −4.77] | +26.09 / +21.75 / +18.19（全正） |

分界取 **τ = 0.25** 不是拟合结果，而是两数据集一致出现的符号翻转点：
低于它必为负、高于它必为正。

### 判定流程

```text
① 在 dev 集上跑两组对照（开/关补全），需要 gold——这是**离线校准**，不是在线判据
② 对每个触发样本算补全精度，得到分布
③ 若平均精度 < τ（建议 0.25）且低纯度样本占比高 → 该数据集不建议启用补全
④ 若启用，则只对高纯度关系组补全（组级粒度，而非全局开关）
```

两数据集的校准结果：**WebQSP 平均纯度 {webqsp_purity} 且仅 {webqsp_neg}/{webqsp_total} 低纯度 → 建议启用；
CWQ 平均纯度 {cwq_purity} 且 {cwq_neg}/{cwq_total} 低纯度 → 建议关闭或严格门控。**

该流程的适用前提是 dev 集与测试集同分布、且 dev 集足够大到能估出纯度；
在 CWQ 归档 §7.5 已记录过一次反例（dev 21 条触发样本给出 EM 0%→42.86% 的乐观估计，
test 54 条上 EM 纹丝不动），故流程应报**纯度的置信区间**而非点估计。
"""


if __name__ == "__main__":
    raise SystemExit(main())
