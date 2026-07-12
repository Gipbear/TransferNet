# kgqa Stage2 — pfit 迁移与 smoke 验证记录（2026-07-11）

## 概要

Ch4 `llm_infer/` 重构为数据集无关的 `kgqa/pfit/`（Path-Faithful Instruction Tuning）流水：
build（建集+manifest）→ train（QLoRA）→ eval（推理+忠实度,支持 by_hop）。
本期按「功能全备、实验选择性跑」执行：全部通路 ~100 样本 smoke 验证,全量实验留待用户按需跑。
schema/schema_gloss 全局废弃(路径格式只剩 arrow/tuple/chain/nl);`llm_infer/` 只读保留作 parity 参照。

- spec: `docs/superpowers/specs/2026-07-11-kgqa-stage2-pfit-design.md`
- plan: `docs/superpowers/plans/2026-07-11-kgqa-stage2-pfit.md`（11 任务,checkbox 已回填）

## 目录约定

```
data/output/kgqa/<ds>/            # ds = webqsp | metaqa,与旧实验结果隔离
├── subsets/                      # 分层子集 qa 文件(仅 metaqa,subset_qa 产出)
├── scores/                       # dump_scores 缓存(train.pt / train_20k.pt)
├── retrieve/                     # kgqa.cli.retrieve 输出(train*.jsonl / test.jsonl,含 golden)
└── pfit/<exp_id>[_变体后缀]/      # 实验目录
    ├── manifest.json             # 配置快照+上游指纹,三阶段断点续跑依据
    ├── sft_train.jsonl           # build 产物
    ├── adapter/                  # train 产物(LoRA)
    └── eval/{predictions.jsonl,summary.json}
```

实验目录后缀自描述：`FMT=v1` → `_v1`；`ADAPTER=...` → `_ft`；`LIMIT=100` → `_smoke100`。

## 数据准备(已完成,缓存/检索输出均 gitignored)

| 数据集 | qa 文件 | ckpt | 检索参数 | 产出 |
|---|---|---|---|---|
| WebQSP train | `qa_train_webqsp_fixed.txt`(2996) | `WebQSP_run_20260518_2241/model-49-0.7154.pt`(bge) | beam20 λ0.2 tail_blend | `scores/train.pt` + `retrieve/train.jsonl` |
| WebQSP test | `qa_test_webqsp_fixed_1581.txt`(1581) | 复用既有缓存 `webqsp_test_1581.pt` | 同上 | `retrieve/test.jsonl` |
| MetaQA train | `train.pt` 分层 20K 子集(5837/7227/6936) | `MetaQA_KB/model_epoch-6_acc-0.9937.pt` | 同上 | `subsets/train_20k.pt` + `scores/train_20k.pt` + `retrieve/train_20k.jsonl` |
| MetaQA test | `test.pt` 全量(39093) | 复用既有缓存 `metaqa_test_full.pt` | 同上 | `retrieve/test.jsonl` |

- 检索参数选择：beam20 对齐 Ch4 训练数据路径条数(predict_train.jsonl 为 20 条/样本),λ0.2 用 Ch5 修正后的官方值;MetaQA 网格显示 beam20/50、各 λ 指标近乎并列(answer_path_hit 0.9961 vs 0.9995)。score 缓存与 beam/λ 无关,换参数只需秒级重跑 retrieve。
- `golden` 字段(Task 1,id2ent 同空间口径:WebQSP=MID,MetaQA=name)已全量核验:train/test 无缺失无空值,抽样与 qa 文件 gold 一致。

## smoke 结果(LIMIT=100,非正式指标,只验通路)

| 实验 | train_loss | hit1 | hit_any | macro_f1 | EM | halluc | fmt |
|---|---|---|---|---|---|---|---|
| webqsp_main_smoke100 | 0.136 | 0.74 | 0.87 | 0.62 | 0.38 | 0.011 | 1.0 |
| metaqa_main_smoke100 | 0.089 | 0.96 | 0.96 | 0.91 | 0.82 | 0.0 | 1.0 |

- 两条流水 build→train→eval 全通,重跑同命令三阶段均 manifest skip(断点续跑生效)。
- 建集 parity(Task 4B)在真实 WebQSP train 检索输出上复核:chain+name+v2 与 arrow/tuple/nl 共 4 配置 × 50 条,pfit 与 legacy `build_kgcot_dataset` messages 逐字符一致。
- MetaQA 补充分层 eval(各 hop 10 条,借 smoke adapter):1/2/3-hop hit1 全 1.0、幻觉 0,by_hop 分组与 3-hop 长 prompt 通路验证通过。
- **注意**:`eval --limit N` 取 test 前缀,而 MetaQA test.jsonl 按 hop 分块有序,smoke 的 by_hop 仅含 hop1;全量跑不受影响。

## 全量结果(2026-07-12,PR 前置门槛)

`webqsp_main` 全量跑通(build 2948 条 → train 2 epochs → eval test 1581,单 run),输出 `data/output/kgqa/webqsp/pfit/webqsp_main/`:

| 指标 | webqsp_main(单 run) | 论文表4-9 PFIT+Cite(3-run mean) | 差值 |
|---|---|---|---|
| Hit@1 | **85.83** | 85.17 | +0.66 |
| Hits | **89.44** | 89.02 | +0.42 |
| Macro F1 | **77.91** | 77.22 | +0.69 |
| EM | **63.63** | 61.55 | +2.08 |
| Cit-P | **83.35** | 81.83 | +1.52 |
| Cit-R | **86.50** | 86.30 | +0.20 |
| HalRate | 0.14 | 0.04 | +0.10 |

- 对照口径:**论文终稿 `docs/chapter5-writing` 分支 `chapter4_new.md` 表 4-9 PFIT+Cite 行**(全量 test、3 次推理均值,配置同构 chain+Cite+K=20)。全指标 parity 且略高(+0.4~+2.1pt),幻觉同为近零量级 → **PR 前置门槛通过**。
- 勿用 `data/analysis/chapter5_metrics.json` 里的 `ch4_finetuned_single_batch`(hit1 0.7884)对照,那是论文 Ch4 重跑前的过时旧基线(论文现行数字归档见 `data/analysis/20260614_1800__chapter34_word_latest`)。
- rejection 段全 miss(100 条 unanswerable 全部作答)符合预期:main 配置 `include_rejection=false`。
- format_compliance 0.9994;train 数据 hop 分布 1-hop 1756 / 2-hop 1192,skip 48。

## smoke 期间发现并修复的问题

1. `subset_qa` 仅支持 JSON,而 MetaQA dump 实际输入是四段 pickle 的预处理 `.pt` → 支持双格式,`.pt` 按 hops 分层采索引、四数组同索引切片(996d7d3)。
2. MetaQA `e_s` 占位符回填失效:spec 钩子只替换大写 `E_S`,vocab 解码后真实问题为小写 → 大小写不敏感正则回填(2177bfc)。人工抽看 SFT 样本(plan Task 10 检查项)抓到。
3. pfit eval 迁移漏了 legacy 的 transformers FutureWarning 过滤(cdb7d88);WebQSP 系默认 `bert_name` 对齐现役 bge ckpt(27fad78)。

## 8 个注册实验与运行方式

```bash
bash scripts/run_pfit.sh --exp <exp_id> [--phase build|train|eval|all]
# 变体环境变量:LIMIT(smoke)、FMT(输出格式覆盖)、ADAPTER(外部 adapter)、EPOCHS
```

| exp_id | 类型 | 状态 |
|---|---|---|
| webqsp_main | 训练(chain+name+v2,Ch4 groupAname_v2 parity 锚) | smoke ✅,**全量 ✅(2026-07-12,见「全量结果」节,PR 门槛通过)** |
| webqsp_spot_nl | 训练(nl+name+v2,对照 Ch4 groupD) | 待用户跑 |
| webqsp_base_zeroshot | eval-only(base 零样本;`FMT=v1` 换格式) | 待用户跑 |
| webqsp_nopaths | eval-only(无路径;`ADAPTER=<dir>` 得微调变体) | 待用户跑 |
| metaqa_main | 训练(chain+v2,5K 分层混合跳数,核心新数字) | smoke ✅,**全量待用户跑(PR 前置门槛候选)** |
| metaqa_spot_nl | 训练(nl+v2) | 待用户跑 |
| metaqa_base_zeroshot | eval-only | 待用户跑 |
| metaqa_nopaths | eval-only | 待用户跑 |

全量跑法:去掉 `LIMIT` 即可(数据准备已就绪,断点续跑安全重入)。
WebQSP 全量口径对照 Ch4 groupAname_v2;MetaQA 关注 summary 的 overall+by_hop。

## 回归

- 全套单测:`python -m unittest discover -s tests -t . -p 'test*.py'` — 261 tests OK
  (必须带 `-t .`,否则 `tests/kgqa` 遮蔽项目 `kgqa` 包,72 个用例 import error)
- bash:`tests/run_pfit_lib_test.sh`、`tests/run_ablation_lib_test.sh` 均 PASS
