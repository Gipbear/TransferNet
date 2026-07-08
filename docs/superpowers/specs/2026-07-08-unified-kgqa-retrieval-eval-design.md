# 统一 KGQA 框架 · Stage 1：评测 + 路径检索层 设计文档

- 日期：2026-07-08
- 范围：新建 greenfield 包 `kgqa/` 的第一层——**TransferNet 模型接口 + 路径检索双后端 + 评测 + 三数据集（WebQSP / MetaQA / CWQ）适配器**
- 不含（后续横切）：SFT（数据集构建/评测/网格）、PV-GAC（原 checked-batch agent）、TransferNet 训练循环统一

## 1. 背景与目标

现有代码（`oh_my_agent/`、`llm_infer/`、`scripts/`）围绕 WebQSP 单数据集硬编码，检索逻辑散落在 `scripts/offline_path_search.py`、`oh_my_agent/path_retrieve_server/`、`oh_my_agent/tools/path_retrieve.py`。三个 TransferNet 数据集模块（`MetaQA_KB/`、`WebQSP/`、`CompWebQ/`）已存在并各有 checkpoint，但下游检索/评测只对 WebQSP 打通。

**首要目标**：统一可复现的工程框架，把三数据集收敛到一套带数据集适配器的代码；实验数字作为框架的验证副产物。**切分方式**：横切、按阶段逐层打通，本 spec 为第一层。**架构决策**（已确认）：

- 检索走**双后端**（在线实时前向 + 离线得分缓存），同一接口、同一 engine。
- 代码 **greenfield 新建 `kgqa/` 包**；4 个 TransferNet 模块保留作 model + 训练。
- 采用**方案 C**：共享引擎 + 两个可插拔策略口（KG 边来源、得分 dump 格式）+ 薄数据适配器。
- TransferNet 训练循环**暂不并入**（标记待定），仅定义 `ScoreProducer` 模型接口供 online 后端与 dump 使用。

## 2. 包结构

`kgqa/` 是统领整条链路的**总包**（放法一）：数据集适配器、`types`、`eval` 三阶段共享；本 spec 落 `retrieve/`，后续横切阶段以**平级子包**接入 `sft/`（微调：建集/评测/网格）与 `agent/`（PV-GAC）。下方结构中 `sft/`、`agent/` 为预留位，stage 1 不实现。

```
kgqa/
├── types.py               # QASample / ScoreCache / ReasonPath / RetrieveResult / MetricSpec
├── datasets/              # 薄适配器层（数据提供者）
│   ├── base.py            #   DatasetAdapter(ABC)
│   ├── webqsp.py  metaqa.py  cwq.py
│   └── registry.py        #   name → adapter
├── models/                # ScoreProducer 模型接口（薄封装现有 model.py，不重写训练）
│   ├── base.py            #   ScoreProducer(ABC)
│   ├── webqsp.py  metaqa.py  cwq.py
├── kg/                    # 【策略口 1】KG 边来源
│   ├── base.py            #   KGEdgeSource(ABC)
│   ├── global_kg.py       #   WebQSP / MetaQA 全局邻接表
│   └── subgraph_kg.py     #   CWQ 逐样本子图
├── scores/                # 【策略口 2】得分 dump / load 格式
│   ├── base.py            #   ScoreDumper / ScoreLoader(ABC)
│   ├── webqsp.py  metaqa.py  cwq.py
├── retrieve/             # 检索引擎（数据集无关，共享）
│   ├── engine.py          #   reconstruct → 候选搜索 → MMR
│   ├── mmr.py             #   复用 utils/path_utils
│   └── backends/
│       ├── base.py        #   RetrieveBackend(ABC)
│       ├── offline.py     #   读 .pt 缓存
│       └── online.py      #   ScoreProducer 实时前向
├── eval/                 # 评测（共享）
│   ├── answer_eval.py     #   答案级指标
│   ├── path_eval.py       #   路径级 + 多样性指标
│   └── metrics.py
├── config/               # 每数据集配置（hop 上限 / 阈值 / 指标口径 / 路径）
│   └── webqsp.yaml  metaqa.yaml  cwq.yaml
├── server/               # 常驻检索服务（薄壳，持有一个 RetrieveBackend）
│   └── path_retrieve_server.py
├── cli/
│   ├── dump_scores.py  retrieve.py  eval.py
├── sft/                  # 【预留，stage 2】微调：建集 / 评测 / 网格
└── agent/                # 【预留，stage 3】PV-GAC
```

**保留不动**：`MetaQA_KB/ WebQSP/ CompWebQ/`（model + 训练循环）、`utils/path_utils.py`（被 `retrieve/mmr.py` 复用）。
**迁移后退役**：`scripts/offline_path_search.py` 检索逻辑、`oh_my_agent/path_retrieve_server/`、`oh_my_agent/tools/path_retrieve.py`。迁移完成并通过回归测试后再删除旧代码。

## 3. 核心接口（方案 C 的心脏）

### 3.1 DatasetAdapter（薄数据提供者，每数据集 ~60–100 行）

```python
class DatasetAdapter(ABC):
    name: str                         # "webqsp" | "metaqa" | "cwq"
    max_hop: int                      # 2 / 3 / 4
    def load_qa(path, limit) -> list[QASample]     # 统一 QASample
    def entity_name(entity_id) -> str              # MID→name；MetaQA 恒等
    def score_producer() -> ScoreProducer          # §3.4
    def kg_edge_source(sample) -> KGEdgeSource      # 策略口 1
    def score_io() -> tuple[ScoreDumper, ScoreLoader]  # 策略口 2
    def metric_spec() -> MetricSpec                 # §5 指标口径
```

### 3.2 KGEdgeSource（策略口 1：真正发散点之一）

```python
class KGEdgeSource(ABC):
    def neighbors(node_id, rel_id) -> list[int]
    def all_edges() -> Iterable[tuple[int, int, int]]
# global_kg.py  : WebQSP / MetaQA 全局邻接表（进程内加载一次，跨样本复用）
# subgraph_kg.py: CWQ 每样本独立子图（随 sample 走）
```

### 3.3 ScoreDumper / ScoreLoader（策略口 2：发散点之二）

```python
class ScoreLoader(ABC):
    def load(cache_path) -> ScoreCache   # rel_probs / ent_scores / topic_ids / hop_attn / e_score
class ScoreDumper(ABC):
    def dump(samples, producer, out_path) -> None
# 复用现有 offline_path_search 的 reconstruct_* 逻辑，按数据集包一层
```

### 3.4 ScoreProducer（模型接口，训练循环不并入）

```python
class ScoreProducer(ABC):
    def load_checkpoint(ckpt_path) -> None
    def forward(sample) -> ScoreCache    # 加载 ckpt → 前向 → 中间得分
# 内部直接调用现有 WebQSP/model.py 等；训练循环仍留原模块
```

### 3.5 共享引擎

`retrieve/engine.py` 仅依赖 `DatasetAdapter` / `KGEdgeSource` / `ScoreLoader`（或 `ScoreCache`），执行 reconstruct → 候选搜索 → MMR，**不认识任何具体数据集**。这是「90% 逻辑共享」的落点。

统一数据结构定义在 `kgqa/types.py`：`QASample(question, topic_ids, gold_ids, sample_index)`、`ScoreCache`、`ReasonPath`、`RetrieveResult`、`MetricSpec`。

## 4. 检索双后端与数据流

**离线路径**（全量评测 / 参数 sweep / 消融回放）：
```
dump(一次):  ScoreProducer.forward(全 test) → ScoreDumper → xxx_test.pt
检索(多次):  ScoreLoader.load(cache) → engine(reconstruct→候选→MMR) → RetrieveResult
```

**在线路径**（小样本 / demo / 调参迭代）：
```
ScoreProducer.load_ckpt() 常驻 → forward(sample) → engine(同一条) → RetrieveResult
```

**一致性保证**：两后端唯一区别是 `ScoreCache` 来源（读缓存 vs 实时前向），engine 之后完全同一路径。由 `test_backend_parity` 锁死：同一 ckpt 下 online 与 offline 对同批样本产出的路径逐条一致（同时验证 dump 无损）。

**统一 CLI**（`--dataset` 选适配器，`--backend` 切后端）：
```bash
python -m kgqa.cli.dump_scores --dataset webqsp --ckpt ... --split test
python -m kgqa.cli.retrieve    --dataset metaqa --backend online  --ckpt ...
python -m kgqa.cli.eval        --dataset cwq    --backend offline --cache ...
```

**常驻服务** `server/path_retrieve_server.py`：薄壳，启动时按参数持有一个 `RetrieveBackend`（online 或 offline），HTTP 层只解析请求 + 调 engine；三数据集共用同一服务实现，靠 `--dataset` 加载对应适配器。

## 5. 评测层与指标（尽量给全）

指标计算全部落 `kgqa/eval/`，口径由适配器 `MetricSpec` 决定（用哪些指标、gold 按 MID 还是 name 比、hop 上限），**计算逻辑三数据集共享**。输出统一 `summary.json` + 逐样本 `jsonl`，字段跨数据集对齐便于横向比。

### 5.1 答案级指标（TransferNet 预测质量）

三数据集用**同一套完整指标**（计算逻辑共享，`answer_eval.py` 单一实现）：

**统一指标集**：hits@1、hits@any、macro-P/R/F1、micro-P/R/F1、exact_match。

| 数据集 | gold 比对 | 报告视图 | 备注 |
|---|---|---|---|
| WebQSP | MID | overall | macro-F1 为论文对齐主指标 |
| CWQ | MID | overall | macro-F1 为 CWQ 官方主指标 |
| MetaQA | 实体名 | **overall + 分 hop（1/2/3-hop）双视图** | 每条样本带 `hop` 字段（已确认），答案为集合、P/R/F1 有意义 |

说明：三数据集沿用各自 `predict.py` 现有 gold 比对口径，收敛到 `answer_eval.py` 单一实现，保证同一套计算。MetaQA `data/input/MetaQA_KB/test.json` 每条含 `hop∈{1,2,3}`（test 39093 条），因此**除整体外，额外按 hop 分组各报同一套完整指标**——`summary.json` 结构为 `{"overall": {...}, "by_hop": {"1": {...}, "2": {...}, "3": {...}}}`。WebQSP/CWQ 无自然 hop 分区，只报 overall（`by_hop` 缺省为空）。`MetricSpec` 用 `group_by`(可选 `"hop"`) 字段驱动分组，逻辑仍共享。

### 5.2 路径级指标（检索出的推理路径质量，三数据集共享）

- **命中类**：path-hit（任一检索路径尾 ∈ gold）、path-recall（gold 被检索尾覆盖比例）、path-precision（检索尾中命中 gold 比例）、path-F1。
- **多样性**：Jaccard 相似度、尾实体唯一率、答案覆盖率（复用 `utils/path_utils.py`）。
- **检索统计**：平均路径数、平均 hop、平均 elapsed_ms。

路径级指标同样支持 `group_by="hop"`：MetaQA 额外输出 `by_hop` 分跳视图（overall + 1/2/3-hop），与答案级共用同一分组机制。

## 6. 数据准备缺口与风险（stage 1 必须显式补齐）

| 数据集 | 缺口 | stage 1 需做 |
|---|---|---|
| WebQSP | 无（缓存 / ckpt / 输入齐全） | 迁移 + parity/回归验证既有结果 |
| MetaQA | 无 `dump_scores`、实体名非 MID、3-hop | 写 MetaQA `ScoreDumper` + 适配器；`global_kg` 复用 KB；分 hop 评测 |
| CWQ | `data/input/CWQ` 未落地、逐样本子图、ckpt 仅 0.42 偏弱 | 从 `data/resources/CWQ` 构建输入；`subgraph_kg` 策略；ckpt 质量风险如实记录，不追新训 |

**风险声明**：CWQ 检索质量受限于弱 ckpt（0.42）。stage 1 目标是「管线跑通 + 产出可复现指标」，CWQ 数字偏低是**已知项、非 bug**；是否重训 CWQ 留到训练层再议（训练待定）。

## 7. 测试与迁移验证

- `test_backend_parity`：online vs offline 路径逐条一致（锁一致性 + dump 无损）。
- `test_webqsp_regression`：新包对 WebQSP test 1581 的检索指标 = 现有 `full_20260613_1722` 口径，迁移不掉点。
- 每个适配器单测：QA 解析、`entity_name`、KG 边来源（小样本 fixture）。
- 三数据集各跑 `--limit 20` smoke，确保端到端不炸。

## 8. 交付物

1. `kgqa/` 包（types / datasets / models / kg / scores / retrieve / eval / config / server / cli）。
2. 三数据集适配器 + 各自 `ScoreDumper`/`ScoreLoader`/`ScoreProducer`/`KGEdgeSource`。
3. CWQ 输入构建脚本 + MetaQA dump 能力。
4. 三数据集的答案级 + 路径级评测结果（`summary.json`）。
5. 测试套件（parity / regression / adapter 单测 / smoke）。
6. 旧检索代码迁移退役（回归通过后删除）。

## 9. 明确不做（YAGNI）

- 不统一 TransferNet 训练循环（仅留 `ScoreProducer` 挂载点）。
- 不重训任何 ckpt（含弱 CWQ）。
- 不触碰 SFT 与 PV-GAC（后续横切阶段）。
- 不做超出三数据集的通用化。
