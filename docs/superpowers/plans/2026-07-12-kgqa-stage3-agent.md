# kgqa Stage3 — agent(PV-GAC checked-batch)Implementation Plan

> **For agentic workers:** 按用户既定偏好,本 plan 在当前会话直接逐任务执行(不派 subagent)。Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 把 Ch5 `oh_my_agent/`(WebQSP 专用 checked-batch/PV-GAC 流水)重构为数据集无关的 `kgqa/agent/` + `kgqa/llm_server/`,检索服务功能上移 `kgqa/server/`;WebQSP 以 gatev2 全量回放逐位 parity 锁行为,MetaQA 留 spec 钩子 + base 零样本 smoke;本期不产新官方数字。

**Architecture:** `kgqa/agent/{checked_batch,replay,specs}.py` + `tools/` + `common/` + `cli/` + `demo_page/`;`kgqa/llm_server/`(整包平移);`kgqa/server/`(薄壳补齐 group_tails/θ 参数化/drop_loopback/question+topics 检索 + client)。差异进 `AgentDatasetSpec` 薄注册表,主逻辑/工具/指标不分叉。

**Tech Stack:** FastAPI/uvicorn 常驻服务、transformers+PEFT(llm_server)、unittest;`oh_my_agent/` 迁移期只读保留作 parity 对拍参照。

**Spec:** `docs/superpowers/specs/2026-07-12-kgqa-stage3-agent-design.md`

## Global Constraints

- 本地执行统一 conda 环境 `py312_t271_cuda`(先激活,不用 `conda run`)。
- **行为逐位不变是本期第一原则**:主逻辑/指标/回放只换包路径与命名,不做逻辑"顺手优化";发现疑似 bug 先记录,经用户确认再单独修。
- 不改 `kgqa/retrieve/engine.py` 数值内核;不动 `kgqa/eval/`、`kgqa/pfit/` 既有行为;`oh_my_agent/` 本期只读不动(parity 对拍要用)。
- 新评测输出统一 `data/output/kgqa/<ds>/agent/<run_id>/`;旧 `data/output/WebQSP/checked_batch_agent/` 只读(论文数字凭证)。
- GPU 任务仅 smoke 量级(≤50 样本);不重跑 Ch5 全量。
- 提交按文件名暂存(禁 `git add -A`),中文 Conventional Commits + HEREDOC,Co-Authored-By 现场读 `git config user.name/user.email` + 当前会话模型。
- 每个 Task 结束跑 `python -m unittest discover -s tests -t . -p 'test*.py'` 确认零回归(`-t .` 必须带)。

---

### Task 1: `kgqa/llm_server/` 整包平移

**Files:**
- Create: `kgqa/llm_server/`(app/client/config/constraints/engine/scheduler/server,迁自 `oh_my_agent/llm_server/`)
- Modify: `tests/test_llm_server_server.py` 等引用该包的测试(import 改写)
- Modify: `scripts/llm_server.sh`(模块入口 `python -m kgqa.llm_server.server`)

- [x] 平移 + import 改写,逻辑零改动;测试迁 import 后全绿
- [x] `./scripts/llm_server.sh start|status` 冒烟:服务能起、health 正常(与 Task 9 合并验证)
- [x] 回归全绿

### Task 2: `kgqa/server/` 检索服务补齐 + 服务 parity

**Files:**
- Modify: `kgqa/server/path_retrieve_server.py`(薄壳 → 全功能)
- Create: `kgqa/server/client.py`(迁自 `oh_my_agent/path_retrieve_server/client.py`,schema 兼容)
- Test: `tests/kgqa/test_server_full.py`(新)、`tests/test_path_retrieve_client.py`(迁 import)
- Modify: `scripts/path_retrieve_server.sh`(入口切换)

- [x] 失败测试:按 question / topic_entities 定位样本;group_tails 在线构建;θ 启动参数(默认 0.9 行为不变);`PATH_DROP_LOOPBACK` 开关;响应 schema 含 legacy 全字段
- [x] **服务 parity(免 GPU 硬门槛)**:同 `webqsp_test_1581.pt` 缓存同参数,新 service 与 legacy `CachedPathRetriever` 抽样逐样本一致(三元组序列/prediction/group_tails 逐位,log_score 容差 1e-6)
- [x] 回归全绿

### Task 3: `kgqa/agent/common/` + `specs.py` — 共享件与数据集钩子

**Files:**
- Create: `kgqa/agent/common/`(metrics/eval_records/output_parser/prompting/paths/entity_mapping,原样迁)
- Create: `kgqa/agent/specs.py`(AgentDatasetSpec 注册表)
- Test: `tests/kgqa/test_agent_specs.py`(新)、`tests/test_simple_agent_common.py` 等(迁 import)

- [x] 失败测试:`get_agent_spec("webqsp")` 提供 tab+`[MID]` QA 加载、mapped_entities MID→name 映射(惰性一次构建)、wordpiece 清洗、hop=2、检索/adapter 默认参数;`get_agent_spec("metaqa")` 提供 `.pt`/文本加载、恒等实体映射、brackets 清洗、hop∈{1,2,3}
- [x] 实现:common 原样迁;qa_data/entity_mapping 的 WebQSP 硬编码抽为 spec 钩子(默认行为与现状逐位一致)
- [x] 回归全绿

### Task 4: `kgqa/agent/tools/` — 三工具迁移

**Files:**
- Create: `kgqa/agent/tools/{path_retrieve,answer_with_paths,cited_path_check}.py`
- Test: `tests/test_simple_agent_tools.py` 等(迁 import)

- [x] 迁移:client 引用改 `kgqa.server.client` / `kgqa.llm_server.client`;实体映射来源改走 spec(构造参数保留,默认从 spec 取)
- [x] 工具结果 dataclass 字段与序列化(to_dict)逐位不变(回放与 JSONL 记录依赖)
- [x] 回归全绿

### Task 5: `kgqa/agent/{checked_batch,replay}.py` — 主逻辑与回放迁移

**Files:**
- Create: `kgqa/agent/checked_batch.py`(`CheckedBatchWebQAgent` → `CheckedBatchAgent`,别名保留兼容)、`kgqa/agent/replay.py`
- Test: `tests/test_checked_batch_replay.py`、`test_hop_filter.py`、`test_large_answer_expansion.py`、`test_group_tails_online.py`、`test_stop_policy_sweep.py` 等(迁 import)

- [x] 迁移:仅改包路径/命名/依赖注入,run-flags 默认值与后处理逻辑逐位不变
- [x] 既有行为测试(hop_filter/expansion/replay 对齐护栏)迁 import 后全绿
- [x] 回归全绿

### Task 6: gatev2 全量回放 parity(免 GPU 硬门槛)

**Files:**
- Create: `tests/kgqa/test_agent_gatev2_parity.py`(抽样常驻)+ 一次性全量核验脚本(scratch,不入库)

- [x] 用 `ch5_full_rerun_20260627_2306/full_trace/checked_batch_eval.jsonl` 全量 1581 条回放:`kgqa.agent.replay` 产出的逐样本 hit/EM/引用与 summary 指标**逐位一致**于 Ch5 终版 gatev2(`score2_hopoff_top3_max2_gatev2`)
- [x] 常驻测试固化抽样(≥50 条)parity,防未来回归
- [x] 回归全绿

### Task 7: `kgqa/agent/cli/` + 脚本入口 + tests 收口

**Files:**
- Create: `kgqa/agent/cli/{eval_checked_batch,run_checked_batch}.py`(输出目录改 `data/output/kgqa/<ds>/agent/`,加 `--dataset`)
- Modify: `scripts/run_checked_batch_agent_eval.sh`(入口与默认输出切换)
- Modify: 全仓剩余 oh_my_agent 引用测试迁移完毕

- [x] CLI 迁移:参数面保持,新增 `--dataset`(默认 webqsp 行为不变);JSONL/summary 字段结构不变
- [x] `grep -rn "oh_my_agent" tests/ scripts/ kgqa/` 仅剩 legacy 对拍测试的显式引用(白名单注释标明)
- [x] 回归全绿

### Task 8: demo_page 迁移保可用

**Files:**
- Create: `kgqa/agent/demo_page/`(平移 + import 改写)

- [x] 平移后实际启动 demo 服务,页面加载、回放/检索点检通过(`tests/test_demo_page_server.py` 迁 import 全绿)
- [x] 回归全绿

### Task 9: WebQSP 在线端到端 smoke(GPU)

**Files:**
- 产出: `data/output/kgqa/webqsp/agent/smoke_50/`

- [x] 新入口起两服务(`scripts/path_retrieve_server.sh` + `scripts/llm_server.sh`,均已指向 kgqa)
- [x] `python -m kgqa.agent.cli.eval_checked_batch --limit 50 --check_mode hybrid-reject-list ...`:端到端跑通,指标与既有 quick_50 量级一致(在线生成随机性,不设逐位门槛),stop_reason/citation 记录完整
- [x] 断点续跑/输出目录约定生效

### Task 10: MetaQA 通路 base 零样本 smoke(GPU)

**Files:**
- 产出: `data/output/kgqa/metaqa/agent/smoke_base/`

- [x] MetaQA test 缓存(复用既有 `data/output/MetaQA_KB/score_cache/metaqa_test_full.pt`,同 stage2 惯例)起检索服务,base 零样本答题,≥30 条且**显式覆盖 3-hop**(注意 test 按 hop 分块有序,取样须跨块)
- [x] 验证:恒等实体映射、brackets 清洗、group_tails 在 MetaQA 邻接表上构建正常;hop_filter/expansion 等 run-flag 无 2-hop 隐性假设(发现即记录,经确认再修)
- [x] 指标/JSONL 记录结构完整(数字仅通路验证,非正式)

### Task 11: 文档与收尾

**Files:**
- Create: `docs/experiments/experiments_kgqa_stage3_agent_<date>.md`
- Modify: `AGENTS.md`(Ch5 命令/目录约定更新;oh_my_agent 标 legacy 只读,同 llm_infer 待遇)

- [x] 实验记录:parity 结果、smoke 结果、新旧入口对照、「等 metaqa_main 后补」清单
- [x] AGENTS.md 更新;plan checkbox 回填
- [ ] 记忆更新（仅在用户显式要求时执行）
- [x] 全套测试最终回归

---

## PR 合入前置门槛

- [x] gatev2 全量回放 parity 逐位一致(Task 6,免 GPU)+ 服务 parity(Task 2)双门槛通过;WebQSP 在线 smoke 量级正常(Task 9)。本期无全量训练/评测,不需要用户跑大实验
