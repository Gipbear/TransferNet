# 统一 KGQA 框架 · Stage 2:pfit(Path-Faithful Instruction Tuning)设计文档

- 日期:2026-07-11
- 范围:新建 `kgqa/pfit/` 子包(落地 stage1 spec 预留的 `sft/` 位,更名 pfit)——**SFT 建集 / QLoRA 训练 / 推理评测三步流水 + WebQSP / MetaQA 双数据集 + 实验编排脚本**
- 前置:stage1(检索+评测)三数据集已合并(PR #2/#3/#4);阶段一范围 = MetaQA + WebQSP 最终态,CWQ 暂缓

## 1. 目标与已拍板决策

**首要目标**:把 Ch4 的 `llm_infer/`(WebQSP 专用)重构为数据集无关的 `kgqa/pfit/`,统一吃 `kgqa.cli.retrieve` 的输出,WebQSP 先做(回归锚)、MetaQA 再适配(新增量),**不写两套内容**。

用户已确认的决策:

1. 包位置 = `kgqa/pfit/`(享受 `kgqa/datasets/` adapter 注册表,不再重复解决实体名/gold 口径问题)。
2. WebQSP 只做 **parity + 抽查**,不重跑 A-J 全消融矩阵(Ch4 结论已定稿)。
3. **schema / schema_gloss 全局废弃**:pfit 不实现,相关内容不迁移(路径格式只保留 arrow/tuple/chain/nl);relation gloss 加载随之不迁。
4. **跨数据集迁移 / 混合数据集训练不做**。
5. MetaQA **混合跳数单模型**:1/2/3-hop 分层采样合训一个 adapter,评测按 by_hop 分跳报告;不做 per-hop 多模型。
6. **新输出统一放 `data/output/kgqa/<ds>/`**,与旧实验结果隔离。
7. **执行模式**:功能全备、实验选择性跑——本期只做 ~100 样本量级 smoke 验证 + 备齐实验脚本;**合入 PR 前由用户选一个配置跑完整训练+推理看指标**,其余实验用户空闲时自跑。
8. MetaQA train **dump 20K**(分层)→ SFT **采样 5K**(暂定,后看)。

## 2. 数据流与文件组织

**pfit 唯一上游输入 = `kgqa.cli.retrieve` 输出 JSONL**。现输出键 `question/topics/hop/mmr_reason_paths/prediction/sample_index`,与老 `llm_infer/build_kgcot_dataset.py` 的输入天然兼容,**缺 `golden`(gold 答案)字段——本 spec 需给 retrieve 输出补上**(name 口径,经 adapter id2ent 还原;向后兼容:pfit build 读不到 `golden` 时报错并提示重跑 retrieve)。

```
data/output/kgqa/<ds>/                  # ds = webqsp | metaqa
├── scores/                             # dump 缓存(如 train_20k.pt)
├── retrieve/<tag>.jsonl                # 检索输出 = pfit 上游(如 train_20k.jsonl / test.jsonl)
└── pfit/<exp_id>/                      # 一目录一实验,自描述
    ├── manifest.json                   # 配置快照:检索参数、格式、增强、训练超参、上游文件指纹
    ├── sft_train.jsonl                 # 建集产物
    ├── adapter/                        # LoRA adapter
    └── eval/
        ├── predictions.jsonl
        └── summary.json                # EM/F1/hit/hallucination/citation(MetaQA 加 by_hop)
```

`manifest.json` 是相对老流程的新增物:老 run_ablation 配置散在脚本变量里,manifest 让实验目录自描述;断点续跑判定从「文件存在」升级为「文件存在且 manifest 一致」。

## 3. 包结构与复用/差异切分

```
kgqa/pfit/
├── __init__.py
├── formats.py      # 路径格式 arrow/tuple/chain/nl × 实体表示 mid/name + 输出格式 v1-v4 提示词
│                   #   (迁自 llm_infer/kg_format.py,删 schema/schema_gloss/gloss 加载)
├── build.py        # 建集:格式化 + 增强(shuffle/score 保留/干扰比例/拒答样本)+ 按 hop 分层采样 + manifest
├── train.py        # Unsloth QLoRA(迁自 train_sft.py:智能截断保金路径、prompt masking)
├── eval.py         # 推理 + 忠实度评测(迁自 eval_faithfulness.py,指标加 group_by=hop)
├── manifest.py     # 配置快照与断点续跑判定
└── specs.py        # PfitDatasetSpec 注册表:每数据集差异钩子
```

均以 `python -m kgqa.pfit.build|train|eval` 运行(argparse main,同 kgqa.cli 惯例)。

**完全复用(数据集无关,一行不分叉)**:formats、增强、采样、train、eval 指标、manifest、编排。

**PfitDatasetSpec(每数据集一个薄 spec,只放差异)**:

| 钩子 | webqsp | metaqa |
|---|---|---|
| entity_repr 可选值 | mid / name(mid2name 映射文件) | 仅 name(天然名字,免映射) |
| 问题清洗 | BERT wordpiece / 特殊 token 去除 | `[brackets]` topic 标注处理 |
| hop | 恒 2(无分层需要) | 1/2/3,分层采样 + by_hop 评测 |
| 拒答样本构造 | 支持(Hit@K=0) | 不适用(检索天花板 0.99+,不启用) |
| 默认检索参数 | 沿用 Ch4 终版 | 沿用 stage1/网格结果 |

## 4. 实验矩阵(功能全备,选择性跑)

编排脚本 `scripts/run_pfit.sh`:实验注册表 + 断点续跑(承 run_ablation.sh 风格,但配置进 manifest)。

**注册的实验**(本期均不实际全量跑,仅 smoke 验证通路):

| exp_id | 数据集 | 类型 | 内容 |
|---|---|---|---|
| `webqsp_main` | webqsp | 训练 | chain+name+v2(= Ch4 最优 groupAname_v2),**parity 锚点** |
| `webqsp_spot_nl` | webqsp | 训练 | nl+name+v2,消融通路抽查(对照 Ch4 groupD) |
| `webqsp_base_zeroshot` | webqsp | eval-only | base model 零样本 chain+name × v1/v2(对照 Ch4 groupE) |
| `webqsp_nopaths` | webqsp | eval-only | base 无路径 + 微调模型无路径(对照 Ch4 groupH) |
| `metaqa_main` | metaqa | 训练 | chain+name+v2,5K 分层混合跳数,**核心新数字** |
| `metaqa_spot_nl` | metaqa | 训练 | nl+name+v2,验证 chain 最优在短关系名 KG 成立 |
| `metaqa_base_zeroshot` | metaqa | eval-only | base 零样本 chain+name × v1/v2 |
| `metaqa_nopaths` | metaqa | eval-only | 无路径基线 |

**明确不做**:A/B/C/G 组重跑(结论沿用 Ch4)、F 组 MetaQA(不适用)、schema 全系(J/CJ/CJT,已废弃)、跨数据集迁移与混训。

## 5. 数据准备

- **WebQSP**:train(~3K)与 test(1581)用 stage1 离线后端重跑 retrieve 至 `data/output/kgqa/webqsp/retrieve/`(offline 免 ckpt,成本低;train split 需先 dump train 缓存)。
- **MetaQA**:train 329K 全量 dump 不可行(test 39K 已 718MB)→ **先分层采样索引再 dump**:新增小工具生成按 hop 分层的 20K 索引文件,`kgqa.cli.dump_scores` 扩展 `--indices_file`(接受样本索引列表);retrieve 后 build 阶段再分层采到 5K。

## 6. Parity 与验收门槛

1. **建集 parity(免 GPU 硬门槛,类比 stage1 回归保真锁)**:同一检索输入 + 同配置 + 固定 seed,`kgqa.pfit.build` 产物与 `llm_infer/build_kgcot_dataset.py` 产物**逐条文本一致**(prompt/completion 级),覆盖 arrow/tuple/chain/nl × v1-v4(schema 除外)。
2. **smoke(~100 样本)**:WebQSP 与 MetaQA 各走通 build → train → eval 三步,产出指标与 manifest,断点续跑生效。
3. **全量训练 + 指标**:合入 PR 前用户选一个配置(预期 `webqsp_main` 或 `metaqa_main`)跑完整训练+推理,人工核指标;其余实验后置。

训练本身有随机性,数字级 parity 只做量级核对(vs Ch4 已有数字),不设逐位门槛。

## 7. 老代码去留

- `llm_infer/` 与 `scripts/run_ablation.sh` **迁移期保留不动**(Ch4 论文数字复现凭证,且建集 parity 测试要引用它);pfit 验证稳定后整体退役,schema 相关代码随退役一并物理删除,本期不单独动它。
- 不触碰 `kgqa/retrieve/engine.py` 数值内核、`kgqa/eval/` 既有指标行为(只加 pfit 侧复用)。

## 8. 风险

| 风险 | 应对 |
|---|---|
| retrieve 输出补 `golden` 需动 stage1 已合并代码 | 纯增量字段,旧消费方不读该键;补回归测试 |
| MetaQA 关系短名下 v1-v4 提示词/引用格式可能有隐性 WebQSP 假设 | smoke 阶段人工抽看样本;formats parity 测试兜住格式层 |
| Unsloth 训练依赖 GPU,CI/单测无法覆盖 train 主体 | train 单测只测数据整形/截断/masking 纯函数;端到端靠 smoke |
| MetaQA train dump 索引采样是 dump_scores 新参数 | 默认不传时行为不变,加回归测试 |
