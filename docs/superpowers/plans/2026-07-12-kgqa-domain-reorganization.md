# kgqa 能力域目录重组 Implementation Plan

**Goal:** 将 `kgqa/` 从“技术层目录与论文阶段目录混用”的结构，渐进重组为以 `backbone`、`retrieve`、`pfit`、`agent` 为主的能力域结构；消除检索层对 Ch5 `agent` 的反向依赖，同时保持 Ch3/Ch4/Ch5 的运行结果、命令入口和产物契约不变。

**Architecture:** 采用“稳定契约 + 新路径实现 + 旧路径兼容 shim”的两阶段迁移。`core` 保存无业务归属的数据契约和跨域纯工具；`backbone` 只封装三个原始 TransferNet 实现的在线得分生产器；`retrieve` 拥有检索所需的数据集适配、图、缓存、内核、评测与 HTTP API；`pfit` 维持独立的 Ch4 JSONL 消费者；`agent` 维持 Ch5 编排；LLM HTTP 服务归为 `serving`，而不是隐式归入其中任一论文阶段。

**Tech Stack:** Python 3.12、PyTorch、FastAPI、unittest；不新增依赖。

## 非目标

- 不移动或修改 `WebQSP/`、`MetaQA_KB/`、`CompWebQ/` 原始实现；它们继续作为 parity 参照。
- 不改变 `kgqa/retrieve/engine.py` 的检索公式、排序、MMR、阈值语义或浮点口径。
- 不合并 `kgqa/pfit/eval.py` 与 agent 指标逻辑；二者受不同 legacy parity 约束，去重另立任务。
- 不更改 score cache 格式、retrieve JSONL 格式、路径检索 HTTP schema、默认端口或实验输出目录。
- 本计划不包含提交；只有用户明确要求时才创建 commit。

## 目标目录与依赖方向

```text
kgqa/
├── core/                         # 无业务归属的契约、文本/实体映射、通用答案指标
│   ├── contracts.py
│   ├── entity_map.py
│   ├── qa_formats.py
│   └── answer_metrics.py
├── backbone/                     # TransferNet -> ScoreBundle 在线前向适配
│   ├── base.py
│   ├── webqsp.py
│   ├── metaqa.py
│   └── cwq.py
├── retrieve/                     # Ch3 检索能力域
│   ├── cache/
│   ├── datasets/
│   ├── graph/
│   ├── backends/
│   ├── engine.py
│   ├── eval/
│   ├── api/
│   └── cli/
├── pfit/                         # Ch4，继续以 retrieve JSONL 为输入
├── agent/                        # Ch5
│   ├── tools/
│   ├── web/
│   ├── replay.py
│   └── cli/
└── serving/
    └── llm/                      # 本地/远程 LLM HTTP 服务与 client
```

允许的依赖方向：`core <- backbone`、`core <- retrieve <- agent`；`serving` 被 `agent` 使用；`pfit` 只消费稳定 JSONL 契约，除必要的 `core` 纯工具外不依赖检索运行时。禁止 `core/retrieve/backbone` import `agent`，禁止 `core` import 任一能力域。

## 迁移红线与兼容策略

1. 所有旧模块路径在迁移期保持可 import；旧模块仅重导出新实现，禁止复制两份业务逻辑。
2. 保持以下命令仍可运行：
   - `python -m kgqa.cli.dump_scores`
   - `python -m kgqa.cli.retrieve`
   - `python -m kgqa.cli.eval`
   - `python -m kgqa.server.path_retrieve_server`
   - `python -m kgqa.llm_server.server`
   - `python -m kgqa.pfit.{build,train,eval}`
   - `python -m kgqa.agent.cli.eval_checked_batch`
3. score cache 版本、retrieve JSONL 字段、`PathRetrieveClient` 请求/响应字段、`LLMClient` 请求/响应字段保持字节级或字段级兼容；新增字段须可选。
4. 每一个“移动”先由新模块承载实现，再把旧路径改为 shim；禁止在同一任务里同时改变算法和模块位置。
5. 迁移完成前不改已有实验脚本的 import/命令。兼容层稳定后，单独批次迁移脚本和文档；旧路径删除需另一个明确批准的 breaking-change 任务。

## Task 0: 建立基线与架构护栏

**Files:**

- Create: `tests/kgqa/test_package_boundaries.py`
- Modify: none

**Steps:**

- [x] 运行基线：`python -m unittest discover -s tests -t . -p 'test*.py' -v`。
- [x] 新增静态边界测试，扫描 `kgqa/core/`、`kgqa/backbone/`、`kgqa/retrieve/` 的 Python import：断言它们不 import `kgqa.agent`；断言 `kgqa.core` 不 import `kgqa.{backbone,retrieve,pfit,agent,serving}`。
- [ ] 在测试中白名单 legacy 兼容模块，避免把过渡 shim 误判为业务依赖；业务实现文件必须满足新边界。

**Verify:** 基线与新增边界测试均通过；真实 checkpoint/cache 测试仅在 `RUN_KGQA_ARTIFACT_TESTS=1` 时执行。2026-07-12 默认测试基线通过，artifact 测试按预期跳过。

## Task 1: 提取 `core`，先消除现有反向依赖

**Files:**

- Create: `kgqa/core/{__init__,contracts,entity_map,qa_formats,answer_metrics}.py`
- Modify: `kgqa/types.py`、`kgqa/models/base.py`、`kgqa/scores/base.py`
- Modify: `kgqa/agent/common/{entity_mapping,qa_data,metrics}.py`
- Modify: `kgqa/datasets/webqsp.py`、`kgqa/eval/answer_eval.py` 及直接消费者
- Test: `tests/kgqa/test_core_contracts.py`、`tests/kgqa/test_package_boundaries.py`

**Interfaces:**

- `core.contracts` 成为 `QASample`、`ReasonPath`、`RetrieveResult`、`MetricSpec`、`SampleScore`、`CacheMeta`、`ScoreBundle`、`ScoreLoader`、`ScoreProducer` 的唯一实现位置。
- `core.qa_formats` 提供 WebQSP 行解析；`core.entity_map` 提供 MID/name 映射；`core.answer_metrics` 提供被检索评测和 agent 共用的答案指标。
- 原 `kgqa.types`、`kgqa.models.base` 及 `agent/common` 对应模块保留同名导出，确保现有 tests/scripts 可不改路径运行。

**Steps:**

- [x] 先为 core 契约、WebQSP 解析、实体映射和答案指标写等价性测试；覆盖原有 public symbol、返回值和异常类型。
- [x] 将实现迁入 `core`，旧模块缩为显式 `from ... import ...` 的兼容 shim。
- [x] 将 `datasets/webqsp.py` 和 `eval/answer_eval.py` 的业务 import 改为 `core`，以移除当前 `datasets -> agent`、`eval -> agent` 的反向边。
- [x] 更新边界测试，确认检索域不再 import `kgqa.agent`。

**Verify:**

- `python -m unittest tests.kgqa.test_dataset_webqsp tests.kgqa.test_answer_eval tests.test_simple_agent_common -v`
- `python -m unittest discover -s tests/kgqa -t . -p 'test*.py' -v`

## Task 2: 建立 `backbone`，隔离 TransferNet 在线适配

**Files:**

- Create: `kgqa/backbone/{__init__,base,webqsp,metaqa,cwq}.py`
- Modify: `kgqa/models/{__init__,base,webqsp,metaqa,cwq}.py`
- Modify: `kgqa/cli/{dump_scores,retrieve}.py`
- Test: `tests/kgqa/test_backbone_factory.py`

**Interfaces:**

- `kgqa.backbone.make_score_producer()` 保持现有数据集分发、默认 BERT 名称和参数语义。
- `ScoreProducer` 只依赖 `core.contracts` 与原始 TransferNet 模块；不得依赖 `retrieve`、`pfit` 或 `agent`。
- `kgqa.models.*` 变为兼容 shim，保留现有外部 import 与测试路径。

**Steps:**

- [x] 写工厂参数透传和三个 producer import 的测试，不加载 checkpoint。
- [x] 移动实现到 `backbone`，旧 `models` 模块仅重导出。
- [x] 让新 `retrieve/cli`（尚未移动时为旧 `cli`）直接依赖 `backbone`，不再依赖 `models` 兼容层。

**Verify:**

- `python -m unittest tests.kgqa.test_cli_dispatch tests.kgqa.test_models_webqsp tests.kgqa.test_models_metaqa tests.kgqa.test_models_cwq -v`
- `python -m unittest discover -s tests/kgqa -t . -p 'test*.py' -v`

## Task 3: 将 Ch3 完整收敛至 `retrieve`

**Files:**

- Create: `kgqa/retrieve/{cache,datasets,graph,eval,api,cli}/...`
- Modify to shims: `kgqa/{scores,datasets,kg,eval,server,cli}/...`
- Modify: 内部业务 import 与 `scripts/path_retrieve_server.sh`
- Test: `tests/kgqa/test_retrieve_import_compat.py`、既有检索/服务测试

**Interfaces:**

- `retrieve/cache` 接管现 `scores` 的加载与缓存序列化；缓存格式不变。
- `retrieve/datasets` 接管 registry/adapter；`retrieve/graph` 接管 `GlobalKG` 与 `KGEdgeSource`。
- `retrieve/api` 接管 `PathRetrieveService`、FastAPI schema/client/server；其中 Ch5 的 `prediction_threshold` 与 `engine.build_prediction()` 的不同语义必须保留。
- `retrieve/cli` 接管 dump/retrieve/eval 的实现；旧 `kgqa.cli.*` 和 `kgqa.server.*` 必须是可执行的 `python -m` 转发模块，而非仅 import 导出。

**Steps:**

- [ ] 先迁移 cache、graph、datasets，逐包改内部 import；保留 `scores`、`kg`、`datasets` shim。
- [ ] 迁移 `engine.py`、`backends/`、路径/答案评测到 `retrieve`；不改任何检索计算语句。
- [ ] 迁移 server 到 `retrieve/api`，先用 HTTP client contract 测试锁定 `/health`、`/info`、`/retrieve` 行为，再替换旧 `server` 为 shim。
- [ ] 迁移 CLI 实现并保留旧模块的 `main()` 转发；仅在所有命令冒烟通过后更新项目内脚本到新路径。

**Verify:**

- `python -m unittest tests.kgqa.test_engine tests.kgqa.test_backend_parity tests.kgqa.test_server tests.kgqa.test_server_full -v`
- `python -m kgqa.cli.dump_scores --help && python -m kgqa.cli.retrieve --help && python -m kgqa.cli.eval --help`
- `python -m kgqa.server.path_retrieve_server --help`
- 具备本地 artifact 时：设置 `RUN_KGQA_ARTIFACT_TESTS=1`，运行 WebQSP、MetaQA、CWQ 的 online/offline parity 测试。

## Task 4: 归位 `serving` 与 Ch5 的 Web 界面

**Files:**

- Create: `kgqa/serving/llm/...`
- Create: `kgqa/agent/web/...`
- Modify to shims: `kgqa/llm_server/...`、`kgqa/agent/demo_page/...`
- Modify: `kgqa/agent/{tools,cli,checked_batch,replay}.py`
- Test: `tests/test_llm_*`、`tests/test_checked_batch_*`、`tests/test_demo_page_*`

**Interfaces:**

- `serving.llm` 保持 `LLMClient`、`GenerateResponse`、本地/硅基流后端、端口 8788 与 `/generate` schema；不引入 agent 业务逻辑。
- `agent.web` 只迁移 demo/replay 展示层；`CheckedBatchAgent`、tool 输出和 JSONL trace 字段不变。
- `kgqa.llm_server.*`、`kgqa.agent.demo_page.*` 保留兼容导出和 module entrypoint。

**Steps:**

- [ ] 先以 client/server contract 测试锁定 LLM API，随后移动实现与创建 shim。
- [ ] 再移动 demo 页面；静态资源路径、FastAPI 路由和回放 JSON schema 必须保持。
- [ ] 最后将 agent 的业务 import 指向 `core`、`retrieve.api.client`、`serving.llm.client`，不得经由旧兼容层。

**Verify:**

- `python -m unittest tests.test_llm_client tests.test_llm_server_server tests.test_checked_batch_agent tests.test_demo_page_server -v`
- `python -m kgqa.llm_server.server --help`
- `python -m kgqa.agent.cli.eval_checked_batch --help`

## Task 5: Pfit 隔离确认与迁移收尾

**Files:**

- Modify: `kgqa/pfit/*`（仅在需要改为 `core` 纯工具时）
- Modify: `scripts/*.py`、`scripts/*.sh`、项目文档中的 kgqa import/entrypoint
- Test: `tests/kgqa/test_pfit_*`、`tests/run_pfit_lib_test.sh`

**Steps:**

- [ ] 检查 pfit 仅消费 retrieve JSONL；若使用 core 工具，先补逐字符 parity 测试再替换 import。
- [ ] 禁止改动 pfit 的 format、prompt、解析、训练与评测实现，仅消除已经过时的路径引用。
- [ ] 在兼容 shim 已通过全量测试后，更新 scripts/docs 至新 canonical 路径；保留旧路径以支持历史实验命令。
- [ ] 生成一份 `data/analysis/YYYYMMDD_HHMM__kgqa-domain-reorg/README.md`，记录迁移前后目录、兼容模块、验证命令与尚未移除的 shim。

**Verify:**

- `bash tests/run_pfit_lib_test.sh`
- `python -m unittest discover -s tests -t . -p 'test*.py' -v`
- `bash tests/run_ablation_lib_test.sh`
- `git diff --check`

## 完成条件

- 新目录中不存在 `retrieve -> agent` 或 `core -> domain` 的业务 import。
- 所有旧 `python -m kgqa.*` 入口和现有 scripts 继续可用。
- 单元测试全绿；需要真实模型/缓存的 parity 测试在本地 artifact 可用时通过。
- Ch3 检索结果、Ch4 pfit 输入输出、Ch5 HTTP/trace 契约未发生未记录的变化。
- 旧路径兼容层有明确清单；删除 shim 仅在后续 breaking-change 计划中进行。
