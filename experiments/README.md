# KGQA 可复现实验入口

本目录是第三、四、五章的现役实验编排入口。历史保留脚本仍在 `scripts/`，仅用于
论文复现核对；新实验不得再向旧目录写入产物。本文档按章节补充运行方式；当前先给出
第三章。

## 脚本结构

```text
experiments/
├── common.py                    # 跨章节公共运行工具
├── configs/                     # 按章节划分的版本化配置
├── ch3/                         # 第三章：检索、路径与下游 QA
│   ├── run.py                   # python -m experiments.ch3.run
│   └── run_downstream_qa.py     # python -m experiments.ch3.run_downstream_qa
├── ch4/
│   └── run.py                   # python -m experiments.ch4.run
└── ch5/
    └── run.py                   # python -m experiments.ch5.run
```

## 术语

- 参数扫描：在固定得分缓存下比较检索参数。
- 已确认检索配置：第三章完成参数扫描后人工确认的配置；第四、五章只能引用此类配置。
- 基准正式评测：完整测试集评测，不等同于冒烟验证。
- 回放消融：读取基准正式评测记录进行确定性后处理，不重复调用语言模型。
- 运行清单：每次运行目录中的 `run_manifest.json`，记录命令、版本、配置和输入来源。

路径和命令行参数使用英文标识，但所有配置说明、注释和结果报告均使用中文。

## 第三章：检索实验

### 实验设计

- 每个数据集先生成 `topk={100,250,500,1000}` 的 train/test score 缓存，用于观察
  top-k 饱和性。默认下游缓存为 `topk=500`，但应根据饱和性结果人工确认是否调整。
- 固定 `threshold=0.01`。WebQSP 与 RoG-CWQ 使用控制变量法分析 `beam_size`、`lambda_val` 与终点
  实体融合权重 `eta`，其余参数固定为正式配置；MetaQA 的历史配置仍可使用三维网格。
  `lambda_val=0.0` 是无多样性惩罚对照；其余值控制 MMR 的关系集合重叠惩罚。
- `parameter_scan.items` 可显式填写待运行参数组；三个取值列表则表示展开三维网格。编排器
  均生成稳定候选编号，例如 `beam=50，λ=0.2，η=1.0` 为 `beam50_lambda02_eta1`。
- `retrieve.eta` 是 top-k 饱和性评测使用的基准值；参数扫描中每个候选的 `eta` 会覆盖该值。
- `eta` 是论文中的终点实体分数融合权重。`alpha_final` 已废弃，现役命令、配置、接口和
  输出均不接受该字段。
- `step_score_mode` 明确逐跳路径排序使用的分数：`joint` 为关系与实体对数分数之和，
  `relation_only` 仅保留关系分数，`entity_only` 仅保留实体分数。三种模式均固定要求关系和
  尾实体同时通过阈值，因此该实验只比较排序分数，不改变候选空间。
- `relation_only` 与 `entity_only` 必须设置 `eta=0`，避免终点实体分数重新参与排序。运行
  清单以 `score_scheme` 记录固定交集候选规则、逐跳分数模式和终点融合权重。
- `score_component_ablation` 使用显式实验项，不并入 `parameter_scan` 的笛卡尔积。WebQSP
  当前包含联合基线、移除终点融合、仅关系和仅实体四组；它们固定使用已确认的
  `beam=20，λ=0.2，threshold=0.01`。
- `shortest_path_baseline` 是**基于 TransferNet 候选答案的最短路径后处理基线**：只取最终实体
  分数 Top-20 候选，在现有知识图谱中枚举不超过可用推理步数的有向最短路径。它只借鉴
  GNN-RAG 的路径构造环节，不是完整 GNN-RAG 或 ReaRev 复现；当前在 WebQSP 与 RoG-CWQ 启用。
- 检索汇总的 `backbone` 是 TransferNet 等基础检索模型的原始实体预测指标；其
  `prediction` 不受路径重建、重排序或分数消融影响，不能用于比较第三章方法。
  `path` 才是路径尾实体作为预测集合得到的路径答案指标（含 `answer_hit`、`top1_hit`、
  Precision、Recall 和 F1）。后续固定大模型基于路径上下文得到的问答结果应单独记为
  第三章 QA 指标，不能与前两者混写。
- 不根据测试集指标自动选择配置。完成扫描后，由人工在配置中填写确认理由和
  `selected_candidate`，再发布给第四、五章使用。

### 运行步骤

#### WebQSP 正式复现顺序（`transfernet_v2`）

以下顺序以当前 `webqsp_transfernet_v2.json` 为准：`topk=500`、已确认参数组 `beam20_lambda02_eta1`（beam=20，λ=0.2，η=1.0），关系得分采用全局归一化，并在候选选择前完成有效边过滤。`transfernet_v1` 仅保留为历史结果，不再作为正式数值来源。耗时来自本机 `py312_t271_cuda` 环境的实际运行记录；checkpoint、GPU、磁盘缓存和数据加载状态不同会使时间波动，建议将其作为容量规划参考而非严格上限。

| 步骤 | 命令或操作 | 本次实测 / 预计耗时 | 关键产物与完成判据 |
| --- | --- | --- | --- |
| 0. 核对配置 | 阅读配置、确认 checkpoint 与扫描范围 | < 1 分钟 | `experiments/configs/ch3/webqsp_transfernet_v2.json`；当前已固定为 `confirmed`，不再根据测试集重新选参。 |
| 1. 演练 | `python -m experiments.ch3.run --dataset webqsp --dry_run` | < 1 分钟 | 仅打印将执行的 score、top-k 评测和参数扫描命令；不加载模型。 |
| 2. score 与 top-k 饱和性 | `python -m experiments.ch3.run --dataset webqsp --config experiments/configs/ch3/webqsp_transfernet_v2.json --phase scores` | 已有共享缓存时仅重新评测；重新前向约 26 分钟 | 复用或生成共享 score 缓存，并写出 `topk_saturation/transfernet_v2/topk{100,250,500,1000}_{train,test}/*_summary.json`。 |
| 3. 检索参数扫描 | `python -m experiments.ch3.run --dataset webqsp --config experiments/configs/ch3/webqsp_transfernet_v2.json --phase scan` | 本次 25 分 22 秒 | 16 组单因素敏感性：分别改变 `beam_size`、`lambda_val`、`eta`，其余参数固定为正式配置；每组输出 1,581 条测试结果与汇总至 `confirmed_profiles/transfernet_v2/candidates/<参数组>/test.{jsonl,summary.json}`。 |
| 4. 配置冻结 | 核对正式参数、归一化模式与候选过滤顺序 | 约 5 分钟 | 保持 `status=confirmed`、`selected_candidate=beam20_lambda02_eta1`；扫描只解释趋势，不用于测试集重新选参。 |
| 5. 发布正式上游产物 | `python -m experiments.ch3.run --dataset webqsp --config experiments/configs/ch3/webqsp_transfernet_v2.json --phase publish` | 测试集复用候选仅需数秒；本次 train 约 2 分 56 秒 | 正式产物为 `confirmed_profiles/transfernet_v2/{train,test}.jsonl` 与 `confirmed_config.json`；发布目录 `publish/progress.json` 应为 `completed`。第四、五章后续实验应引用这些正式文件。 |
| 6. 排序分数消融 | `python -m experiments.ch3.run --dataset webqsp --config experiments/configs/ch3/webqsp_transfernet_v2.json --phase score_ablation` | 约 8–15 分钟 | 固定候选空间，比较 `joint_eta1`、`joint_eta0`、`relation_only`、`entity_only` 四组，输出至 `score_component_ablations/transfernet_v2/<实验项>/test_summary.json`。 |
| 7. 候选答案最短路径基线 | 复用 `transfernet_v1` 的既有 SP 结果 | 无需重跑 | SP 不使用关系/实体局部归一化，也不经过本次有效边预过滤逻辑；因此继续引用 `shortest_path_baselines/transfernet_v1/top20_hop_available/test_summary.json`。 |

若希望连续运行第 2、3、6、7 步，可使用下列快捷命令；它**不会**替代第 4 步人工确认，也不会自动执行发布：

```bash
python -m experiments.ch3.run --dataset webqsp --phase all
```

在进入下一步前，可用以下检查确认上一步完成：

```bash
# 第 2 步：查看 top-k 饱和性汇总
cat data/output/kgqa/ch3_retrieval/webqsp/transfernet/topk_saturation/transfernet_v2/topk500_test/test_summary.json

# 第 3 步：查看已确认候选的扫描汇总（人工确认前仍位于 candidates/）
cat data/output/kgqa/ch3_retrieval/webqsp/transfernet/confirmed_profiles/transfernet_v2/candidates/beam20_lambda02_eta1/test_summary.json

# 第 5 步：确认发布状态和正式测试集
cat data/output/kgqa/ch3_retrieval/webqsp/transfernet/confirmed_profiles/transfernet_v2/publish/progress.json
wc -l data/output/kgqa/ch3_retrieval/webqsp/transfernet/confirmed_profiles/transfernet_v2/test.jsonl
```

#### RoG-CWQ 运行记录（`rog_transfernet_v1`，2026-08-07）

上游 ckpt `data/ckpt/RoG_CWQ_20260805_bge_rev_ep50_lr3e4/model-37-0.4803.pt`（bge-base-en-v1.5 + `--rev`），
全量 test 3531 条，`topk=500`。配置为 `experiments/configs/ch3/cwq_rog_transfernet_v1.json`，
已于 2026-08-08 人工确认（`status=confirmed`）并 `--phase publish` 发布。

| 步骤 | 命令 | 实测耗时 | 关键产物 |
| --- | --- | --- | --- |
| score 与 top-k 饱和性（test） | `--phase scores` | dump 302 秒（11.69 题/s）+ 评测 235 秒 | `topk_saturation/rog_transfernet_v1/topk500_test/` |
| 检索参数扫描 | `--phase scan` | 18 组共约 68 分钟（每组约 225 秒，共享一次缓存加载） | `confirmed_profiles/rog_transfernet_v1/candidates/<参数组>/test.{jsonl,summary.json}` |
| 最短路径基线 | `--phase shortest_path` | 21.9 秒（161.55 题/s） | `shortest_path_baselines/rog_transfernet_v1/top20_hop_available/` |
| score 与检索（train 子集） | `--phase scores` | dump 460 秒 + 检索 386 秒 | `topk500_train/train.jsonl`，6000 条 |

**`--rev` 是必需的。** CWQ 的关系词表在 `--rev` 下由 6649 翻倍为 13298（`rev_id = fwd_id + 6649`）。
`kgqa/backbone/cwq.py` 现在会核对 ckpt 的关系分类器输出维度与词表是否一致并在不符时抛错——
此前 `load_state_dict(strict=False)` 会静默跳过 shape 不匹配的关系分类器、让它保持随机初始化，
得分全是噪声却不报错。同理 `--bert_name` 必须与训练一致（本 ckpt 为 bge），
`experiments/ch3/run.py` 已从 `score_source` 透传这两个字段。

**train 只能用子集。** score cache 在内存中全量累积后才落盘，全量 train 27631 条时
`anon-rss` 涨到 22.3 GB 被 OOM killer 杀死（本机 23 GB）。现用随机抽样的
`data/input/RoG-CWQ/train_subset6000.jsonl`（seed=17），分布与全量一致
（三元组均值 3954 vs 3922，答案数 2.19 vs 2.30）。若需全量，须先让 `dump_scores` 支持分块落盘。

扫描结论：**beam 是唯一实质杠杆**（PathHit@K 从 beam3 的 56.05 升到 beam100 的 76.52，
路径精确率同时从 41.32 崩到 8.96，边际递减明显）；**λ=0.2 是拐点**（0→0.2 关系多样性 +5.98pt
而 PathHit 仅 −0.05pt，越过后转为纯代价）；**η=1.0 是 PathHit 峰值**且对精确率调节幅度仅 +3pt。
后两者与 WebQSP 冻结值相同，属独立验证而非迁移。完整数值与取值依据见
`data/analysis/20260807_2050__cwq_param_scan/`。

**两档 beam 并存。** `cwq_rog_transfernet_v1`（beam=20）保留用于与 WebQSP 主实验同参可比；
`cwq_rog_transfernet_beam30.json`（beam=30）另立配置，供第四章冲指标主实验使用——
PathHit@20 从 73.61 抬到 74.48（端到端天花板 +0.87pt），路径块约 782 token，
在 `max_seq_len=1280` 内（建 SFT 集时实测截断率仅 4.5%）；beam50 的 1153 token 会大面积触发截断。
两者共用同一份 score 缓存（`paths.score_dir` 只按 `topk{N}_{split}` 命名，不含 `config_id`），
所以 beam30 的 `--phase publish` 只需补跑检索，不重新 dump 得分。

> ⚠️ `paths.score_dir` 不含 `config_id`：换 ckpt 或换 `--rev` 设置后跑同名 `topk{N}_{split}`
> 会**静默覆盖**已有缓存。CWQ 上旧 `cwq_transfernet_v1.json`（6649 词表）与新配置（13298 词表）
> 就有这个冲突，切换配置前先确认缓存归属。

#### 通用命令参考

```bash
# 0. 先核对 WebQSP 配置：在以下文件填写或确认 checkpoint；其余数据集也在 configs/ch3/。
#    如需改默认 top-k 或扫描范围，直接编辑 retrieve 与 parameter_scan 字段。
sed -n '1,180p' experiments/configs/ch3/webqsp_transfernet_v2.json

# 1. 演练：只展示 score 缓存、top-k 评测和“参数组数×数据划分数”的参数扫描任务。
python -m experiments.ch3.run --dataset webqsp \
  --config experiments/configs/ch3/webqsp_transfernet_v2.json --dry_run

# 2. 实际运行：先生成并评测 top-k 饱和性缓存，再运行 beam/λ/eta 完整对比、排序分数消融和最短路径基线。
python -m experiments.ch3.run --dataset webqsp \
  --config experiments/configs/ch3/webqsp_transfernet_v2.json --phase all

# 3. 审核每组 train/test 汇总指标与日志（示例为 beam=50、λ=0.2、eta=1.0）。
cat data/output/kgqa/ch3_retrieval/webqsp/transfernet/confirmed_profiles/transfernet_v2/\
candidates/beam50_lambda02_eta1/test_summary.json

# 4. 人工确认：在 JSON 中填写 confirmation_reason，设 status 为 confirmed，并令
#    selected_candidate 为某个参数组 ID（如 beam50_lambda02_eta1）。随后发布正式检索结果。
python -m experiments.ch3.run --dataset webqsp \
  --config experiments/configs/ch3/webqsp_transfernet_v2.json --phase publish

# 5. 固定已确认的 beam/λ 后，执行四组逐跳分数消融；不会重新生成 score 缓存。
python -m experiments.ch3.run --dataset webqsp \
  --config experiments/configs/ch3/webqsp_transfernet_v2.json --phase score_ablation

# 6. 仅使用已存在的 topk500_test score 缓存与知识图谱邻接表；不加载 TransferNet checkpoint。
python -m experiments.ch3.run --dataset webqsp --phase shortest_path
```

WebQSP 的单因素敏感性使用显式参数组，避免运行无须解释的参数交互组合：

```json
"parameter_scan": {
  "items": [
    {"beam_size": 20, "lambda_val": 0.2, "eta": 1.0},
    {"beam_size": 50, "lambda_val": 0.2, "eta": 1.0},
    {"beam_size": 20, "lambda_val": 0.5, "eta": 1.0},
    {"beam_size": 20, "lambda_val": 0.2, "eta": 1.5}
  ]
}
```

需要完整交互网格时，仍可填写 `beam_size`、`lambda_val`、`eta` 三个列表，总组数为三者
长度的乘积。论文正式 WebQSP 实验只使用上述控制变量口径，不据测试集敏感性结果重新选参。

可单独执行 `--phase scores`、`--phase scan`、`--phase score_ablation`、`--phase shortest_path` 或 `--phase publish`，便于中断后按阶段恢复。
每个任务目录都有 `run_manifest.json`、`progress.json`、`logs/run.log`、
`logs/events.jsonl` 和 `logs/console.log`。第三章产物如下：

运行时会显示三类进度：第三章任务总数、score 前向样本数和逐题路径检索数；进度文件默认每
50 题原子更新一次。终端不需要进度条时，可传 `--no_progress`；可用
`--progress_interval 100` 调整进度文件更新间隔。

```text
data/output/kgqa/
├── shared/webqsp/backbones/transfernet/scores/
│   └── topk{100,250,500,1000}_{train,test}/
└── ch3_retrieval/webqsp/transfernet/
    ├── topk_saturation/transfernet_v2/
    │   ├── topk{100,250,500,1000}_{train,test}/{score,evaluation}/
    │   └── parameter_scan/<参数组>/<split>/
    ├── score_component_ablations/transfernet_v2/
    │   ├── <实验项>/test/{run_manifest.json,progress.json,logs/}
    │   ├── <实验项>/test.jsonl
    │   ├── <实验项>/test_summary.json
    │   └── batch/                         # 四组任务共享的离线缓存批处理日志
    ├── shortest_path_baselines/transfernet_v1/
    │   └── top20_hop_available/
    │       ├── test/{run_manifest.json,progress.json,logs/}
    │       ├── test.jsonl
    │       └── test_summary.json
    └── confirmed_profiles/transfernet_v2/
        ├── candidates/<参数组>/{train,test}.jsonl
        ├── candidates/<参数组>/{train,test}_summary.json
        ├── {train,test}.jsonl                 # 仅人工确认并发布后产生
        └── confirmed_config.json               # 同上
```

### 第三章：多检索路径下游大模型 QA

该对照评测“相同大模型面对不同检索上下文”的影响，不是第四章的训练源消融。六组固定为：无路径、
最短路径、普通 Score-Beam（`beam=20，λ=0，η=0`）、终点感知 Score-Beam
（`beam=20，λ=0，η=1.0`）、固定加性惩罚（`beam=20，λ=0.2，η=1.0，penalty_mode=fixed`）
和 TARRS（`beam=20，λ=0.2，η=1.0，penalty_mode=adaptive`）。普通 Score-Beam 的
`η` 必须为 0。

先进行不加载模型的演练。它会校验六份 JSONL 的题目和 golden 完全对齐，并展示单次模型加载后
依次评测所选条件的批处理命令。`transfernet_v2` 中无路径条件的题目输入与 SP 路径均未受归一化
和预过滤修改影响，正式重跑只包括普通 Score-Beam、终点感知、固定惩罚和 TARRS 四组：

```bash
python -m experiments.ch3.run_downstream_qa \
  --dataset webqsp \
  --config experiments/configs/ch3/webqsp_transfernet_v2_downstream_qa.json \
  --condition score_beam,terminal_score_beam,fixed,tarrs \
  --layer base_zeroshot --phase eval --smoke_size 20 --dry_run --no_progress
```

实际冒烟会从共同的 WebQSP 测试集按 hop 分层抽样，避免 `--limit` 只取文件开头；模型和 adapter
仅加载一次。冒烟通过后去掉 `--smoke_size 20` 即运行全量 1,581 条：

```bash
# 四个受影响条件的 20 条分层冒烟
python -m experiments.ch3.run_downstream_qa \
  --dataset webqsp \
  --config experiments/configs/ch3/webqsp_transfernet_v2_downstream_qa.json \
  --condition score_beam,terminal_score_beam,fixed,tarrs \
  --layer base_zeroshot --phase eval --smoke_size 20 --no_progress

# 四个受影响条件的全量 1,581 条评测
python -m experiments.ch3.run_downstream_qa \
  --dataset webqsp \
  --config experiments/configs/ch3/webqsp_transfernet_v2_downstream_qa.json \
  --condition score_beam,terminal_score_beam,fixed,tarrs \
  --layer base_zeroshot --phase eval --no_progress

# 四组完成后，与未受影响的 no_path、shortest_path 结果汇总六组报告
python -m experiments.ch3.run_downstream_qa \
  --dataset webqsp \
  --config experiments/configs/ch3/webqsp_transfernet_v2_downstream_qa.json \
  --condition all --layer base_zeroshot --phase report --no_progress
```

新结果输出位于 `ch3_retrieval/webqsp/transfernet/downstream_qa/transfernet_v2/`：每组有独立的
`run_manifest.json`、`progress.json`、`eval/predictions.jsonl` 和 `eval/summary.json`；共享模型
批处理的完整控制台输出位于对应 `batch*/logs/console.log`。报告写到
`reports/<层次>/{smoke_<n>,full}/`。`fixed_pfit_adapter` 层只接受来自
`ch4_pfit/.../adapter/` 且训练清单指向已确认 `train.jsonl` 的 adapter；训练源消融需要新建训练集
和训练多个 LoRA，不由此命令执行。

MetaQA P5 只评测 P4 已冻结的 3-hop 子集。`smoke_30` 保留五组共同的前 30 条样本用于链路演练；正式结果只运行 TARRS 完整方法的 14,274 条下游 QA，不再做条件对比或消融。中断后重跑会复用已完成条件：

```bash
# P5：30 条五条件冒烟
python -m experiments.ch3.run_downstream_qa \
  --dataset metaqa --condition all --layer base_zeroshot \
  --phase eval --smoke_size 30 --no_progress

# P5：TARRS 完整方法全量评测与单条件报告
python -m experiments.ch3.run_downstream_qa \
  --dataset metaqa --condition tarrs --layer base_zeroshot \
  --phase eval --progress_interval 100
python -m experiments.ch3.run_downstream_qa \
  --dataset metaqa --condition tarrs --layer base_zeroshot --phase report
```

MetaQA P5 输出位于
`ch3_retrieval/metaqa/transfernet/downstream_qa/transfernet_v1_3hop/`，目录结构与 WebQSP 相同。

#### 云端或新环境运行的前置产物

第三章检索结果目录被 gitignore。若在云端或新环境运行上述全量下游 QA，须先从已完成的本地
实验同步以下 WebQSP 输入；不要复制整个 `data/output/`：

```text
data/output/kgqa/ch3_retrieval/webqsp/transfernet/
├── confirmed_profiles/transfernet_v2/
│   ├── confirmed_config.json
│   └── test.jsonl
├── confirmed_profiles/transfernet_v2/candidates/
│   └── beam20_lambda0_eta0/test.jsonl
├── penalty_ablations/transfernet_v2/
│   ├── none/test.jsonl
│   └── fixed/test.jsonl
└── shortest_path_baselines/transfernet_v1/top20_hop_available/test.jsonl
```

这些文件分别对应正式 TARRS、普通/终点感知 Score-Beam、固定惩罚与 SP 条件；同步后先执行
`--dry_run` 核对题目与 golden 对齐，再运行实际评测。

## 第四章：本地 QLoRA 训练的两个环境前提

`experiments.ch4.run --phase train` 在本机（RTX 4060 Ti 16 GB，WSL2）跑通需要两项运行时设置。
两者都只在命令行临时覆盖，**不要写进项目脚本或 shell 配置**。

```bash
NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m experiments.ch4.run --dataset cwq \
  --config experiments/configs/ch4/cwq_target_v1.json \
  --profile experiments/configs/ch3/cwq_rog_transfernet_beam30.json --phase train
```

1. **`NO_PROXY` 不能含 IPv6 方括号写法。** 本机默认 `NO_PROXY=[::1],127.*,localhost`，
   httpx 解析 `[::1]` 会抛 `InvalidURL: Invalid port: ':1]'`。unsloth 的
   `FastLanguageModel.from_pretrained` 内部调 `HfFileSystem.glob` 必然触发，训练启动即退出；
   而评测走 transformers 的 `AutoModel.from_pretrained`，不触发。
   症状是**评测正常、训练秒退**，很容易误判成显存或数据问题。
2. **必须开 `expandable_segments`。** 默认分配器在第 7 步就把保留显存推到 15.9 GB / 16.38 GB（97%），
   此时才采样了 2.5% 的 micro-batch，后续遇到长序列批次有 OOM 风险；
   开启后同步数只占 9.8 GB，速度不变（约 36 s/step）。
