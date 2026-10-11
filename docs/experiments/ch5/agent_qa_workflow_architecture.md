# 第五章 协同问答工作流智能体：完整逻辑与系统设计规范（收紧合规版）

第 1—5 节保留初版设计与历史审计，第 6 节为 v1 验收范围。当前修复版本的策略与资源契约见第 7 节。

> **修订说明（2026-10-09 第三轮审查严格收紧）**：
> 1. **两份文档全面同步**：彻底废止所有“Oracle Bound”“理论上限”等表述，定性为**gold 辅助下的历史跨模型探索性诊断参考**；RSDPR 归因客观定性为各因素贡献尚未严格受控隔离；
> 2. **严格还原属性标注历史规则**：明确触发必须满足**属性值严格匹配问句数值且候选命中数为 1—3 个**（$1 \le |matched| \le 3$），未满足坚决不改，严密对齐历史 +49 题收益口径；
> 3. **补齐工作流时序与接口缺口**：
>    - 确立统一前置初答时序：全量题目先生成基座初答 `base_answers` 与 `is_rejection`，作为比较分支回退/截断基准与多跳门控判据；
>    - 补齐深捞关系选择机制：通过问句 TF-IDF 相似度从训练集预测关系链模板，协同 `topic_mids` 下发查询；
>    - 补齐深捞状态机（`ok` / `truncated` / `empty` / `failed`）与路径为空回退契约；
>    - 明确 120 条三元组上限与 5 秒超时为“待验证初始配置”。

---

## 1. 核心设计思想与分流治理逻辑

面对复杂知识图谱问答（CWQ 全量 3,531 题），不同题型的失败机制存在异质性，系统设计核心是**分流治理、对症下药、鲁棒防御**：

以下 387/1544/1600 是问句预测路由规模，不是官方题型计数。原始官方 test 标签为比较及极值
410（comparative 213、superlative 197）、conjunction 1575、composition 1546。运行决策只使用问句预测。

1. **比较/极值题（Comparative / Superlative，全量 387 题）**：
   - **痛点**：局部子图缺乏连续数值与时间字面量属性，模型仅凭拓扑连接易发生盲猜或极性颠倒；
   - **治理**：去知识库反查真实数值/日期，依据极性排序或过滤；
   - **防御契约**：候选属性链匹配分差（`chain_margin`）必须满足门控阈值，且返回有效数值；任何超时、异常、低置信或并列无法区分时，**平滑回退至前置生成的基座初答 `base_answers`**。
2. **合取题（Conjunction，全量 1,544 题）**：
   - **痛点**：拓扑连通但存在大量同构候选实体，缺少细粒度判别约束；
   - **治理**：在输入路径尾部挂载真实数值标签（模式二自然富化）；
   - **严格还原的历史触发契约**：**必须同时满足（1）问句提取数值常数；（2）QLever 反查候选尾实体的属性值与该数值严格匹配；（3）匹配的候选实体数量严格为 1 到 3 个（$1 \le |matched| \le 3$）**。任何未满足项坚决不改，直接沿用前置初答 `base_answers`。
3. **多跳链式题（Composition，全量 1,600 题）**：
   - **痛点**：证据链不足（包括子图无金实体、检索漏选、约束未闭合等）导致微调模型判定拒答；
   - **治理**：仅在模型判定拒答时触发外部知识图谱深捞扩展与路径重选；
   - **防御契约**：基座初答有效（`not is_rejection`）时直接输出 `base_answers`，严禁触发扩展；深捞扩展**严格无 Gold 参与**，通过问句语义预测关系模板执行搜索；若深捞失败或路径为空，平滑回退 `base_answers`。

---

## 2. 端到端系统工作流规范流程图

```mermaid
flowchart TD
    Start(["输入问题 Q, 主题实体, 局部子图"]) --> Base_Retrieve["【阶段 0: 基础路径检索】<br/>TransferNet + MMR 选出基础 Top-20 路径"]
    Base_Retrieve --> Base_Infer["【阶段 1: 统一基座前置初答】<br/>全量微调底座统一推理，产出 base_answers 与 is_rejection"]
    Base_Infer --> Router{"【阶段 2: 题型路由器】<br/>分析问句结构与语义特征"}

    %% 模块1：数值与极值处理器
    Router -->|比较 / 极值题 (387 题)| Module1["【模块 1: 数值与极值处理器】"]
    Module1 --> M1_1["1. 抽取候选实体，查询属性链与数值"]
    M1_Gate{"2. 属性链分差 chain_margin ≥ 阈值<br/>且成功查得有效非空数值？"}
    M1_1 --> M1_Gate
    M1_Gate -->|否 / 异常 / 超时 / 并列无法决策| M1_Fallback["安全回退：平滑返回前置生成的 base_answers<br/>(零误伤防线)"]
    M1_Gate -->|是 (触发题数待受控评测)| M1_Type{"3. 判定细分子类型"}
    M1_Type -->|极值题 (Superlative)| M1_Top1["按极性排序取 Top-1<br/>(若存在数值并列 Ties，保留并列候选)"]
    M1_Type -->|比较过滤题 (Comparative)| M1_Filter["按不等式过滤，返回合规实体列表<br/>(以 max(len(base_answers), 1) 作为截断保护上限)"]
    M1_Top1 --> M1_Out["输出重排/过滤答案"]
    M1_Filter --> M1_Out

    %% 模块2：合取属性标注器
    Router -->|合取题 (1,544 题)| Module2["【模块 2: 属性标注器】"]
    M2_Check{"【历史严格触发契约】<br/>1. 问句命中 CVT 模板并提取数值<br/>2. 候选尾实体属性值严格匹配问句数值<br/>3. 匹配实体数满足 1 ≤ |matched| ≤ 3？"}
    Module2 --> M2_Check
    M2_Check -->|否 / 未匹配 / 查空 / 候选 > 3| M2_Keep["保持原始输入：直接返回前置 base_answers<br/>(坚决不做泛化标注，规避反噬)"]
    M2_Check -->|是 (严格受控子集)| M2_Enrich["在路径尾部自然追加属性标签:<br/>如: 莫桑比克 (童工比例: 1.8)"]
    M2_Enrich --> M2_LLM["输入统一微调模型重新推理"]
    M2_LLM --> M2_Out["输出合取题预测答案"]

    %% 模块3：多跳深捞救援器
    Router -->|多跳链式题 (1,600 题)| Module3["【模块 3: 多跳深捞救援器】"]
    M3_Gate{"检查阶段 1 解析器 is_rejection<br/>(输出答案文本为 None)？"}
    Module3 --> M3_Gate
    M3_Gate -->|未拒答 (模型给出有效答案)| M3_KeepAns["直接输出前置 base_answers<br/>(严格不触发扩展，防止外部噪声干扰)"]
    M3_Gate -->|明确判定拒答| M3_Rescue["【启动外部救援流水线】"]
    M3_Rescue --> M3_Rel["1. 问句 TF-IDF 相似度预测候选关系链模板<br/>(从训练集提取 1-2 跳短链与 3-4 跳长链)"]
    M3_Rel --> M3_QLever["2. 外部 QLever 盲跑深捞扩展子图<br/>(输入: topic_mids + 预测关系链; 预算 ≤120 条, 超时 5s)"]
    M3_CheckState{"3. 检查查询状态机<br/>status == ok 且 edges 非空？"}
    M3_QLever --> M3_CheckState
    M3_CheckState -->|异常 / 超时 / 截断 / 空结果| M3_FailFallback["安全回退：返回前置 base_answers (None)"]
    M3_CheckState -->|成功获得扩展子图| M3_RSDPR["4. RSDPR 路径检索与冗余抑制<br/>(TransferNet 打分 + MMR 选出 Top-20)"]
    M3_CheckPaths{"5. 选出的救援路径集是否非空？"}
    M3_RSDPR --> M3_CheckPaths
    M3_CheckPaths -->|路径集为空| M3_FailFallback
    M3_CheckPaths -->|成功生成有效路径| M3_LLM2["6. 同一微调模型执行二次应急问答"]
    M3_LLM2 --> M3_ResOut["输出救援后预测答案"]

    %% 终态汇总
    M1_Fallback --> Final_Output["【终态答案结算与痕迹记录】"]
    M1_Out --> Final_Output
    M2_Keep --> Final_Output
    M2_Out --> Final_Output
    M3_KeepAns --> Final_Output
    M3_FailFallback --> Final_Output
    M3_ResOut --> Final_Output
```

---

## 3. 历史实验审计与科学归因定调

### 3.1 救援结果必须杜绝“标签参与检索”（Gold 泄漏）
- **事实确认**：历史脚本（`run_k32_p10_expansion.py:217`）读取了 `r["golden"]`，仅对未覆盖金答案的样本发起深捞。
- **定调结论**：该结果属于**gold 辅助下的历史跨模型探索性诊断参考**，不可作为无监督系统的实际表现，更不能称为“性能上限”。
- **工程红线**：生产代码和后续评测中，触发检索的唯一依据是微调模型的 `is_rejection`，严格禁止在检索或路由阶段读取任何 ground truth 标签。

---

### 3.2 跨模型相加不可构成大盘指标
- **事实确认**：
  - 基线 2,184 题来自 `全量拒答_1ep`（在 Composition 上拒答 605 题，全量拒答 691 题）；
  - 历史救援实验的 287 题运行于旧版 `本域微调_拒答`（在 Composition 上拒答 573 题）。
- **定调结论**：跨不同模型 Adapter 的算术相加属于**历史探索性诊断**，不能构成系统整体指标，更不能称为“理论上限”。正式指标必须在统一的 `全量拒答_1ep` 权重下端到端串联重测。

---

### 3.3 RSDPR 与扩展检索的归因隔离说明
- **事实确认**（`second_wave_expansion_summary.json`）：
  - 第一轮（基础候选池）：纯 QLever 朴素选路命中 258 题；第一轮 RSDPR 命中 249 题；
  - 第二轮（放宽至 $K=32, P=10$）：综合配置命中 287 题。
- **定调结论**：第二轮综合配置命中 287 题，但**检索范围、选路算法和 gold 辅助调度共同发生了变化，各因素的独立贡献尚未严格受控隔离**。不能断言扩大检索是主导因素，更不可宣称 RSDPR 单独多救回 29 题。

---

### 3.4 门控定义、触发规模与题型差异
- **事实确认**：
  - `chain_margin >= 0.10` 是候选**属性谓词链的语义匹配置信度分差**，非实体物理数值差；
  - 157 题属于原规则覆盖量；历史 `margin_only_0_10` 实际应用 114 题（Hit@1 41 救回、0 误伤）；43 救回来自另一套复合策略（`precision_first_candidate`）。因此修改后门控的**实际触发题数必须待受控评测后测定**；
  - 极值题（Superlative）取 Top-1（处理并列值 Ties）；比较过滤题（Comparative）为多实体过滤，须保留全部合规实体并根据初答规模截断保护。任何不确信情况平滑回退初答。

---

### 3.5 拒答成因分层与误伤对照说明
- **事实确认**（全量拒答 691 题审计）：
  - 571 题（82.63%）：原名称子图无标准答案名称；
  - 100 题（14.47%）：两跳内可达，但检索未覆盖答案尾；
  - 20 题（2.89%）：**已有答案尾仍拒答，需进一步检查证据**（其中 3 题经人工审阅证据较充分，不可统一归因于约束缺失）。
- **定调结论**：
  - 不能概括为“原图无答案节点”；
  - 历史实验中“拦截 60 题误伤”**特指在旧 adapter 下的历史对照结果**，说明拒答门控能防止已知正确解被外部噪声冲刷。

---

### 3.6 属性标注模块的受控边界
- **事实确认**（`full_mode2_enrichment_results.json` 与 `multi_entity_conjunction_experiment.py:75-88`）：
  - 全量合取题若泛化启用两阶段标注（含模糊多实体关系），大盘呈现 72 救回、119 误伤，**净亏损 -47 题**；
  - 严格限定在历史受控条件（问句提取数值 + 真实数值严格匹配 + 匹配候选数为 1—3 个）时，才呈现 52 救回、3 误伤、净增 +49 题。
- **定调结论**：
  - 生产代码必须严格还原该历史判定条件，任何放宽或修改必须重新验证，不得直接沿用 +49 收益；
  - 其余未匹配样本坚决保持原始路径不变。

---

## 4. 生产级系统实现规范与鲁棒契约（Python 契约接口）

代码实现必须完全符合以下接口契约：

```python
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class QLeverQueryResult:
    status: str  # "ok", "failed", "empty", "truncated"
    edges: list[list[str]]
    error_msg: Optional[str] = None
    seconds: float = 0.0


class ComparativeWingProcessor:
    """模块 1：比较与极值处理器"""

    def __init__(self, chain_margin_threshold: float = 0.10, timeout_sec: float = 5.0):
        self.chain_margin_threshold = chain_margin_threshold
        self.timeout_sec = timeout_sec

    def process(
        self,
        sample: dict[str, Any],
        base_answers: list[str],  # 必传：前置生成的基座初答，作为回退与截断基准
    ) -> list[str]:
        # 1. 尝试查询属性谓词链与实体数值
        res = self._query_attributes_and_values(sample, timeout=self.timeout_sec)
        
        # 门控校验：失败、超时、分差不足、或无候选实体时，平滑回退基座初答
        if not res.success or res.chain_margin < self.chain_margin_threshold or not res.entities_with_values:
            return base_answers

        sub_type = sample.get("sub_type", "comparative")
        
        # 2. 极值题处理：排序后取 Top-1，兼顾数值并列 (Ties)
        if sub_type == "superlative":
            sorted_items = sorted(res.entities_with_values, key=lambda x: x[1], reverse=sample.get("desc", True))
            top_val = sorted_items[0][1]
            top_entities = [name for name, val in sorted_items if abs(val - top_val) < 1e-6]
            return top_entities

        # 3. 比较过滤题处理：按条件阈值过滤，并按初答长度截断保护
        filtered = self._filter_by_condition(res.entities_with_values, sample.get("condition"))
        if not filtered:
            return base_answers
        cap_len = max(len(base_answers), 1)
        return filtered[:cap_len]


class ConjunctionWingProcessor:
    """模块 2：合取属性标注器 (严格还原历史触发规则)"""

    def __init__(self, num_kb: Any):
        self.num_kb = num_kb

    def process_paths(
        self,
        sample: dict[str, Any],
        base_paths: list[dict[str, Any]],
        candidate_mids: list[str],
        cand_map: dict[str, str],
    ) -> tuple[list[dict[str, Any]], bool]:
        # 契约 A: 问句数值提取
        numbers = self._extract_numbers(sample["question"])
        if not numbers or not candidate_mids:
            return base_paths, False

        # 契约 B: 预测前 3 条候选属性链并执行严格数值校验
        predicted_chains = self.num_kb.predict_top_k(sample["question"], top_k=3)
        for num in numbers:
            for chain in predicted_chains:
                matched_mids = self._verify_candidates_with_qlever(candidate_mids, chain, num)
                # 契约 C: 核心限制——匹配实体数必须严格在 1 到 3 之间
                if 0 < len(matched_mids) <= 3:
                    matched_names = [cand_map[m] for m in matched_mids if m in cand_map]
                    if matched_names:
                        # 契约 D: 执行路径尾实体自然属性文本挂载
                        enriched_paths = self._tag_tail_attributes(
                            base_paths, matched_names, chain, num
                        )
                        return enriched_paths, True

        # 未完全满足上述严格条件时，坚决保持原始路径不变
        return base_paths, False


class MultihopRescueProcessor:
    """模块 3：多跳拒答救援器 (零 Gold 参与)"""

    def __init__(
        self,
        pattern_kb: Any,            # 关系模板抽取器 (TF-IDF 相似度匹配)
        triple_budget: int = 120,   # 待验证初始配置: 扩展子图三元组数量上限 (LIMIT 120)
        query_timeout_sec: float = 5.0,  # 待验证初始配置: SPARQL 查询超时时间
        max_query_retries: int = 2,
    ):
        self.pattern_kb = pattern_kb
        self.triple_budget = triple_budget
        self.query_timeout_sec = query_timeout_sec
        self.max_query_retries = max_query_retries

    def process(
        self,
        sample: dict[str, Any],
        base_answers: list[str],
        is_rejection: bool,  # 使用现有解析器判定，实际输出为 None
    ) -> list[str]:
        # 契约 1: 严格遵守拒答触发，未拒答直接返回初答
        if not is_rejection:
            return base_answers

        # 契约 2: 依据问句语义预测关系链模板 (零 Gold 参与)
        predicted_patterns = self.pattern_kb.extract_patterns(sample["question"], k_neighbors=32, p_limit=10)
        if not predicted_patterns:
            return base_answers

        # 契约 3: QLever 盲跑深捞扩展子图，显式检查状态机
        query_res: QLeverQueryResult = self._qlever_expand_blind(
            topic_mids=sample["topic_mids"],
            patterns=predicted_patterns,
            budget_limit=self.triple_budget,
            timeout=self.query_timeout_sec,
            max_retries=self.max_query_retries,
        )
        # 异常、超时、或空结果时，平滑回退初答
        if query_res.status != "ok" or not query_res.edges:
            return base_answers

        # 契约 4: RSDPR 路径检索；若路径为空，平滑回退初答
        rescue_paths = self._rsdpr_select(query_res.edges, sample["question"], top_k=20)
        if not rescue_paths:
            return base_answers

        # 契约 5: 二次应急推理
        return self._llm_infer(sample["question"], rescue_paths)
```

---

## 5. 待执行路线与“策略冻结”实验规划

为杜绝测试集过拟合（Test Set Overfitting）与数据泄露隐患，系统验证严格按照如下策略分阶段实施：

1. **策略与超参冻结（Strategy Freezing）**：
   - 在 **CWQ 验证集（dev）** 或未参与前期探索的保留集上，冻结路由分类规则、比较门控阈值（`chain_margin`）、71 条数值模板以及深捞查询预算（120 条三元组、5 秒超时）；
   - 一旦在 dev 上固化策略，代码与超参进入只读冻结状态。
2. **统一底座端到端重跑**：
   - 锁定现役统一模型底座 `全量拒答_1ep`；
   - 调度全量测试集（N=3,531）进行无 gold 盲跑，评测各分支协同后的实际产盘。
3. **评测定性说明**：
   - 在已探索过的 CWQ test 集上重跑，**其学术定位是验证统一系统工程实现的复现性与闭环有效性**，不得声称为完全隔离的独立盲测。

## 6. 已实现代码与本次验收范围（2026-10-09）

v1 代码快照归档于验收目录的 `source_snapshot_final/`，配置为 `experiments/configs/ch5/cwq_workflow_v1.json`。
第 4 节代码用于解释设计；实际数据契约以 `models.py` 的 dataclass 和 `interfaces.py` 的 ABC 为准。

| 责任 | 实现 |
|---|---|
| 无标签输入、MID 对齐 | `preparation.py`、`models.py` |
| 训练集路由和关系/属性模板 | `templates.py`、`router.py` |
| 初答、分流、两次 PFIT 上限和异常回退 | `orchestrator.py` |
| 数值比较、严格数值标注、拒答救援 | `tools/comparison.py`、`tools/conjunction.py`、`tools/rescue.py` |
| 查询状态、缓存、超时与来源 | `qlever.py`、`persistence.py` |
| 原生 PFIT 提示、单 BOS、1792 输入 token 预算 | `pfit.py` |
| 新外部图 TransferNet 评分与原生 RSDPR | `graph_score.py`、`kgqa/serving/graph_server.py` |
| 预测落盘、恢复、独立评分与模块关闭回放 | `run.py`、`scoring.py`、`ablations.py` |

本次输入保留全量 3531 题；运行输入不含 gold、官方题型或测试 SPARQL。运行路由是训练得到的预测，
前文 387/1544/1600 与本轮问句预测路由数量一致，不能当作官方题型分布。模板训练使用官方 train 的 27639 题，
与 adapter 构建后实际 SFT 的 27631 条分别记录。
数值模板索引的最低相似度固定为 0.12；配置修改会明确报错，改变该值须重建索引并重新验证。

本次沿用设计默认阈值，冻结后仅在 dev 验收路由，准确率为 92.44%（N=3519）；未完成独立 dev QA
选参实验，因此 test 全量结果仍属于探索性实现验证。历史数值标注的“问句数值严格匹配 + 1—3 MID”
触发契约保留；本实现按候选链优先顺序查询，不能据此宣称历史 +49 题收益得到逐题复现。

深捞模板可以包含 3—4 跳，现役 TransferNet checkpoint 的神经推理步数为 2。两者不等价；
新图阶段使用该固定 checkpoint 评分，并记录 `num_steps` 和未知关系，不能声称新增了四跳神经模型。
`max_queries=60` 约束关系扩展查询，名称解析可另发一次查询；独立评分统计两者的实际总量。
v1 截断状态保留原因并整体回退；5 秒是单次 HTTP 请求超时，重试最多 2 次，不是整题时延上限。

模型统一为 `全量拒答_1ep/seed_17`，原生 `v2/name/chain` 提示沿用该模型实际评测配置的
`reject_prompt=false`，不额外添加拒答提示。全量初答和后续生成均通过同一个常驻服务完成。
回退时保持实际初答，成功改写仍可能误伤，最终必须报告配对 rescue/harm/net。

完整验收证据和实测结果归档于 `data/analysis/20261009_0131__cwq_workflow_implementation/README.md`。
本轮全量实际初答 Hit=61.57%，完整工作流 Hit=63.72%，87 救回、11 误伤、净增 76 题；
分支关闭回放、完整指标与负结果见 `experiment_results.md` 第 10 节。

## 7. 修复版本与显式局部救援策略（v2）

当前代码位于 `kgqa/agent/workflow/`，新配置为 `experiments/configs/ch5/cwq_workflow_v2.json`。
比较过滤优先保留仍满足数值条件的初答，按不同实体计数后执行答案规模上限。

| 契约 | v2 行为 |
|---|---|
| 名称资源 | `entity_names_file` 显式指定；对解析的同一份字节计算 SHA-256，并纳入 manifest |
| 名称失败 | 不将 MID 当作成功名称写入缓存；后续允许重试或用已知候选名称补全 |
| 答案候选 | 路径尾必须是已解析名称；中间 CVT/MID 可以保留以维持路径结构 |
| 局部成功 | `allow_partial_rescue=true` 保留已完成且在预算内的完整路径，并记录 `partial`、`stop_reasons` |
| 严格模式 | `allow_partial_rescue=false` 遇查询失败、截断或预算停止时回退初答 |
| 模板兜底 | `allow_template_fallback=true` 仅在 GNN 路径为空时使用模板选路，标记 `score_source=template_fallback` |
| 成本计数 | 邻接请求数、成功请求数与停止原因分别记录；内部截断不额外计为一次查询 |

两个救援开关默认关闭，v2 实验配置明确开启；模板兜底与 GNN 选路必须分开统计。
这是新的策略版本，v1 的 63.72% 不作为 v2 指标。v2 沿用同一 adapter、原生提示、Top-20 和 token 预算，
复用 v1 已实际生成的初答进行逐题版本对照；新增提示仍通过常驻模型服务实际生成。
新实验继续属于已探索 test 上的实现验证，不据 test 收益调参。全量实测与三组实际开关对照已完成，
证据归档于同一验收目录的 `revision2/`。v2 Hit=66.81%，相对同题初答 192 救回、7 误伤、净增 185 题；
相对 v1 为 110 救回、1 误伤、净增 109 题。详细结果见 `experiment_results.md` 第 11 节。
