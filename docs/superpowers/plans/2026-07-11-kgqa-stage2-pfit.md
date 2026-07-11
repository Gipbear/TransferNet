# kgqa Stage2 — pfit(Path-Faithful Instruction Tuning)Implementation Plan

> **For agentic workers:** 按用户既定偏好,本 plan 在当前会话直接逐任务执行(不派 subagent)。Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 把 `llm_infer/`(WebQSP 专用 Ch4 流水)重构为数据集无关的 `kgqa/pfit/`,统一吃 `kgqa.cli.retrieve` 输出;WebQSP 建集 parity 锁行为,MetaQA 走通混合跳数新流水;实验脚本全备、本期仅 smoke 验证,全量实验由用户选择性跑。

**Architecture:** `kgqa/pfit/{formats,build,train,eval,manifest,specs}.py`;差异进 `PfitDatasetSpec` 薄注册表,格式/增强/训练/评测代码不分叉;新输出统一 `data/output/kgqa/<ds>/`。schema/schema_gloss 全局废弃不迁移。

**Tech Stack:** Unsloth QLoRA(`unsloth/meta-llama-3.1-8b-instruct-bnb-4bit`)、transformers、unittest;`llm_infer/` 只读保留作 parity 参照与 Ch4 复现凭证。

**Spec:** `docs/superpowers/specs/2026-07-11-kgqa-stage2-pfit-design.md`

## Global Constraints

- 本地执行统一 conda 环境 `py312_t271_cuda`(先激活,不用 `conda run`)。
- 不改 `kgqa/retrieve/engine.py` 数值内核;不改 `kgqa/eval/` 既有指标行为;`llm_infer/` 与 `scripts/run_ablation.sh` 本期只读不动。
- pfit 路径格式仅 arrow/tuple/chain/nl;**不实现 schema/schema_gloss/gloss 加载**。
- 所有新输出写 `data/output/kgqa/<ds>/`,不落旧目录。
- 训练/推理类任务仅做 ~100 样本 smoke;全量训练由用户在 PR 合入前选一个配置执行。
- 提交按文件名暂存(禁 `git add -A`),中文 Conventional Commits + HEREDOC,Co-Authored-By 现场读 `git config user.name/user.email` + 当前会话模型。
- 每个 Task 结束跑 `python -m unittest discover -s tests -p 'test*.py'`(至少 tests/kgqa + 本期新增)确认零回归。

---

### Task 1: retrieve 输出补 `golden` 字段(name 口径)

**Files:**
- Modify: `kgqa/cli/retrieve.py`(输出 dict 加 `golden`)
- Modify(如需): `kgqa/retrieve/backends/{offline,online}.py` / `kgqa/types.py:RetrieveResult`(携带 gold)
- Test: `tests/kgqa/test_retrieve_golden.py`

- [ ] 失败测试:retrieve 输出 JSONL 每行含 `golden`,为 gold_ids 经 adapter id2ent 还原后的 name 列表(MetaQA 天然 name;WebQSP 为 MID→name,无映射时回退 MID 并保留原样)
- [ ] 实现:`RetrieveResult` 加 `golden: list[str]`(默认空,旧消费方不受影响),两 backend 填充,retrieve.py 写出
- [ ] 回归:tests/kgqa 全绿;抽 3 条 webqsp offline 检索确认 `golden` 与 qa 文件 gold 一致

### Task 2: `kgqa/pfit/formats.py` 迁移 + 与 llm_infer 文本级 parity

**Files:**
- Create: `kgqa/pfit/__init__.py`、`kgqa/pfit/formats.py`
- Test: `tests/kgqa/test_pfit_formats.py`

- [ ] 失败测试(parity):对同一组假路径/问题,`kgqa.pfit.formats` 与 `llm_infer.kg_format` 在 arrow/tuple/chain/nl × mid/name × v1-v4 下产出**逐字符一致**的 prompt/completion(schema 系不在断言范围)
- [ ] 实现:从 `llm_infer/kg_format.py` 迁移,删除 schema/schema_gloss 分支与 gloss 加载,问题清洗抽为可注入钩子(默认不清洗)
- [ ] 回归全绿

### Task 3: `kgqa/pfit/specs.py` — PfitDatasetSpec 注册表

**Files:**
- Create: `kgqa/pfit/specs.py`
- Test: `tests/kgqa/test_pfit_specs.py`

- [ ] 失败测试:`get_pfit_spec("webqsp")` 提供 entity_repr ∈ {mid,name}(name 需 mid2name)、BERT wordpiece 问题清洗、拒答开关可用、hop 恒 2;`get_pfit_spec("metaqa")` 仅 name、`[brackets]` 清洗、hop ∈ {1,2,3} 且分层键可用、拒答不可用(启用即抛错)
- [ ] 实现:dataclass + 注册表,钩子为纯函数,复用 `kgqa/datasets/` 已有实体映射加载逻辑
- [ ] 回归全绿

### Task 4: `kgqa/pfit/build.py` + `manifest.py` — 建集与配置快照

**Files:**
- Create: `kgqa/pfit/build.py`、`kgqa/pfit/manifest.py`
- Test: `tests/kgqa/test_pfit_build.py`

- [ ] 失败测试 A(输入契约):读 kgqa retrieve JSONL;缺 `golden` 键时报错并提示重跑 retrieve
- [ ] 失败测试 B(建集 parity,免 GPU 硬门槛):同一 WebQSP 检索输入 + 同配置 + 固定 seed,`kgqa.pfit.build` 与 `llm_infer/build_kgcot_dataset.py` 产物逐条 messages 文本一致(chain+name+v2 至少 20 条;再抽 arrow/tuple/nl 各 1 配置)
- [ ] 失败测试 C(MetaQA 分层):hop 分层采样 N 条后各跳占比符合预期;混合跳数写入 `_meta.hop`
- [ ] 失败测试 D(manifest):产物目录含 manifest.json(配置+上游文件指纹);同配置重跑跳过,改配置重跑报不一致
- [ ] 实现:格式化走 formats.py,增强(shuffle/score/distractor/拒答)迁自老 build,采样支持按 hop 分层;`python -m kgqa.pfit.build` argparse 入口
- [ ] 回归全绿

### Task 5: `kgqa/pfit/train.py` — QLoRA 训练迁移

**Files:**
- Create: `kgqa/pfit/train.py`
- Test: `tests/kgqa/test_pfit_train_prep.py`

- [ ] 失败测试(纯函数,免 GPU):智能截断保金路径、prompt masking、序列整形与 `llm_infer/train_sft.py` 对应函数行为一致(直接对拍)
- [ ] 实现:迁移 train_sft.py,入参改造为 manifest/目录约定(`--exp_dir` 读 sft_train.jsonl,adapter 写 `exp_dir/adapter/`);Unsloth 加载等 GPU 主体不在单测覆盖
- [ ] 回归全绿(GPU 端到端留给 Task 9/10 smoke)

### Task 6: `kgqa/pfit/eval.py` — 推理 + 忠实度评测(含 by_hop)

**Files:**
- Create: `kgqa/pfit/eval.py`
- Test: `tests/kgqa/test_pfit_eval_metrics.py`

- [ ] 失败测试(指标纯函数):EM/F1/hit/hallucination/citation 计算与 `llm_infer/eval_faithfulness.py` 对拍一致;`group_by=hop` 时 summary 含 overall + by_hop 分组
- [ ] 实现:迁移 eval_faithfulness,推理层支持 adapter / base 零样本(无 adapter)/ `--no_paths` 三形态;输出 `eval/{predictions.jsonl,summary.json}`
- [ ] 回归全绿

### Task 7: MetaQA 分层 dump — `dump_scores --indices_file` + 索引工具

**Files:**
- Modify: `kgqa/cli/dump_scores.py`(新参数,默认不传行为不变)
- Create: `kgqa/pfit/sample_indices.py`(按 hop 分层生成索引文件)
- Test: `tests/kgqa/test_dump_indices.py`

- [ ] 失败测试:索引工具对带 hop 标签的 qa 数据分层采样 N 条,各跳占比正确、可复现(seed);dump_scores 传 `--indices_file` 时仅 dump 指定索引且缓存 meta 记录来源,不传时行为与现状逐字节兼容
- [ ] 实现 + 回归全绿

### Task 8: `scripts/run_pfit.sh` — 实验注册表与编排

**Files:**
- Create: `scripts/run_pfit.sh`(可配套 `scripts/run_pfit_lib.sh`)
- Test: `tests/run_pfit_lib_test.sh`(bash 函数级,承 run_ablation_lib 测试风格)

- [ ] 注册 spec §4 的 8 个实验(webqsp_main / webqsp_spot_nl / webqsp_base_zeroshot / webqsp_nopaths / metaqa_main / metaqa_spot_nl / metaqa_base_zeroshot / metaqa_nopaths)
- [ ] 三步流水 build→train→eval,断点续跑基于 manifest 一致性;`--exp <id> --phase build|train|eval|all`;`LIMIT=100` 环境变量支持 smoke
- [ ] bash 测试通过(dry-run 校验命令拼装,不真跑训练)

### Task 9: WebQSP 数据准备 + smoke(~100 样本,GPU)

**Files:**
- 产出: `data/output/kgqa/webqsp/scores/train.pt`、`retrieve/{train,test}.jsonl`、`pfit/webqsp_main_smoke100/`

- [ ] dump WebQSP train split 缓存 + offline retrieve train/test 至新目录(test 可复用既有缓存重跑 retrieve,确认 `golden` 字段在位)
- [ ] `LIMIT=100 bash scripts/run_pfit.sh --exp webqsp_main --phase all`:三步走通,eval summary 产出且 hit>0,manifest/断点续跑生效
- [ ] 建集 parity 测试(Task 4B)在真实 train 检索输出上复核一次

### Task 10: MetaQA 数据准备 + smoke(~100 样本,GPU)

**Files:**
- 产出: `data/output/kgqa/metaqa/scores/train_20k.pt`、`retrieve/{train_20k,test}.jsonl`、`pfit/metaqa_main_smoke100/`

- [ ] 分层索引 20K → dump → retrieve(train_20k);test 检索输出同步落新目录
- [ ] `LIMIT=100 bash scripts/run_pfit.sh --exp metaqa_main --phase all`:三步走通,summary 含 by_hop 分组且 hit>0
- [ ] 人工抽看 ≥5 条 MetaQA SFT 样本:提示词/引用格式对短关系名无 WebQSP 隐性假设(spec 风险项)

### Task 11: 文档与收尾

**Files:**
- Create: `docs/experiments/experiments_kgqa_stage2_pfit_<date>.md`
- Modify: `AGENTS.md`(pfit 命令与目录约定,替换 llm_infer 段落表述为「legacy,复现凭证」)

- [ ] 实验记录:smoke 结果、目录约定、8 个实验的运行方式与「待用户跑」清单
- [ ] AGENTS.md 更新;plan/spec checkbox 回填
- [ ] 全套测试最终回归

---

## PR 合入前置门槛(用户执行)

- [ ] 用户选定一个配置(预期 `webqsp_main` 或 `metaqa_main`)跑完整 build→train→eval,人工核指标(WebQSP 对照 Ch4 groupAname_v2 量级);其余实验后置自跑
