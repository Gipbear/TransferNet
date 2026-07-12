# 统一 KGQA 框架 · Stage 3:agent(PV-GAC checked-batch)设计文档

- 日期:2026-07-12
- 范围:落地 stage1 spec 预留的 `kgqa/agent/` 位——**Ch5 checked-batch(PV-GAC)流水 + llm_server 迁入总包 + 检索服务功能上移 + WebQSP / MetaQA 双数据集**;横切三层「检索(stage1)→ SFT(stage2 pfit)→ PV-GAC(stage3)」的最后一层
- 前置:stage1/stage2 已合并(PR #2-#5);Ch5 论文数字已终版(gatev2),本期是工程迁移不产新官方数字

## 1. 目标与已拍板决策

**首要目标**:把 Ch5 `oh_my_agent/`(WebQSP 专用,~5100 行)重构为数据集无关的 `kgqa/agent/` + `kgqa/llm_server/`,检索服务统一到 `kgqa/server/`;WebQSP 以 **gatev2 离线回放逐项 parity** 锁行为,MetaQA 留 spec 钩子 + base 零样本 smoke。

用户已确认的决策(2026-07-12,勿重新讨论):

1. **llm_server 迁 `kgqa/llm_server/`**(总包收编,与检索服务对称;现被 Ch5 agent 与 demo_page 使用,pfit eval 自行加载模型不依赖它)。
2. **demo_page 跟随迁移保持可用**(答辩演示要用),随 agent 落 `kgqa/agent/demo_page/`。
3. **MetaQA 范围 = 数据集无关化 + base 零样本 smoke**;正式数字等 `metaqa_main` adapter 训完再跑,不阻塞本期。
4. **旧 `oh_my_agent/` 只读保留、后置物理删除**(同 `llm_infer/` 策略):gatev2 回放 parity 通过后标记 legacy,等 Ch5 新数字全部落地再删。
5. `kgqa/server/` 现为薄壳,**须补齐 legacy 服务功能后才退役** `oh_my_agent/path_retrieve_server/`;`scripts/offline_path_search.py` 的检索逻辑随之退役(stage1 spec 既定)。

## 2. 现状与迁移映射

| 现模块 | 行数 | 去向 | 处理 |
|---|---|---|---|
| `oh_my_agent/agent/` | ~970 | `kgqa/agent/{checked_batch,replay}.py` | 迁移;`CheckedBatchWebQAgent` 更名 `CheckedBatchAgent`(仅命名,行为逐位不变) |
| `oh_my_agent/cli/` | ~560 | `kgqa/agent/cli/` | 迁移;输出目录改 `data/output/kgqa/<ds>/agent/<run_id>/` |
| `oh_my_agent/tools/` | ~540 | `kgqa/agent/tools/` | 迁移;实体映射从构造参数改走 AgentDatasetSpec |
| `oh_my_agent/common/` | ~750 | `kgqa/agent/common/` + `specs.py` | metrics/eval_records/output_parser/prompting 原样迁;qa_data/entity_mapping 的数据集耦合抽进 spec 钩子 |
| `oh_my_agent/path_retrieve_server/` | ~470 | 功能上移 `kgqa/server/` | service 的 group_tails/θ/drop_loopback/按 question+topics 检索补进薄壳;client 迁 `kgqa/server/client.py`,HTTP schema 保持兼容 |
| `oh_my_agent/llm_server/` | ~1040 | `kgqa/llm_server/` | 整包平移 + import 改写,不动逻辑 |
| `oh_my_agent/demo_page/` | ~750 | `kgqa/agent/demo_page/` | 平移 + import 改写,启动验证 |

**外围牵连**:`tests/` 下 10+ 文件引用 oh_my_agent(逐个迁 import);`scripts/{path_retrieve_server,llm_server,run_checked_batch_agent_eval}.sh` 换模块入口;AGENTS.md 命令段更新。

## 3. 数据集差异切分(AgentDatasetSpec)

类比 `kgqa/pfit/specs.py`,数据集差异收敛到 `kgqa/agent/specs.py` 薄注册表,主逻辑/工具/指标一行不分叉:

| 钩子 | webqsp | metaqa |
|---|---|---|
| QA 加载 | tab 分隔 + `question [MID]` 标注(现 qa_data.py) | 预处理 `.pt` / 文本,`[brackets]` topic 标注(复用 pfit spec 经验) |
| 实体映射 | MID→name(mapped_entities.txt,~400 万条,反向映射惰性一次构建) | 恒等映射(天然 name,免文件) |
| 问题清洗 | BERT wordpiece / `[CLS]``[SEP]` 去除 | brackets 处理 |
| hop | 恒 2 | 1/2/3(hop_filter 等 run-flag 对 3-hop 的适用性在 smoke 验证) |
| 检索服务参数 | dataset=webqsp + test 1581 缓存 | dataset=metaqa + stage2 已有 test 缓存 |
| LLM adapter | Ch5 现役 adapter | `metaqa_main` 产物(未训,base 零样本兜底) |

## 4. 检索服务上移(kgqa/server 补齐)

现薄壳只支持按 `sample_index` 查离线缓存;legacy `CachedPathRetriever`(service.py)多出的能力全部上移:

1. **按 question / topic_entities 定位样本**(agent 线上入口)。
2. **group_tails 在线构建**(邻接表实时算,承 2026-06-13 在线化结论,29536 组 100% 一致的实现直接迁)。
3. **θ(PREDICTION_SCORE_THRESHOLD=0.9)从模块常量提升为服务启动参数**,默认 0.9 行为不变——顺带为 Ch5 遗留的 θ sweep(5.6.1 注明"需重跑检索")铺路。
4. **drop_loopback 开关**(env `PATH_DROP_LOOPBACK`,默认开)。
5. HTTP schema 与 legacy `RetrieveRequest/Response` 兼容(group_tails 字段在位),agent 侧 client 无感切换。

**服务 parity(免 GPU)**:同一 `webqsp_test_1581.pt` 缓存、同参数,新 service 与 legacy service 逐样本响应一致(三元组序列 / prediction / group_tails 逐位,log_score 容差 1e-6)。

## 5. Parity 与验收门槛

1. **gatev2 回放 parity(免 GPU 硬门槛,类比 stage2 建集 parity)**:用 `data/output/WebQSP/checked_batch_agent/ch5_full_rerun_20260627_2306` 的 `full_trace/checked_batch_eval.jsonl` 录制,迁移后的 `kgqa.agent.replay` 全量 1581 条回放,逐样本 hit/EM/引用与 summary 指标**逐位一致**于 Ch5 终版 `score2_hopoff_top3_max2_gatev2`(当前工作区代码已可精确复现,迁移不得破坏)。
2. **检索服务 parity**:见 §4。
3. **WebQSP 在线 smoke(GPU)**:新入口起两服务,`--limit 50` 端到端跑通,指标与既有 quick_50 量级一致(在线生成有随机性,不设逐位门槛)。
4. **MetaQA 通路 smoke(GPU)**:MetaQA 缓存起检索服务 + base 零样本答题,少量样本(含 3-hop)跑通,引用/指标/JSONL 记录结构完整。
5. tests 迁移后全量 `unittest discover -s tests -t .` 零回归。

## 6. 输出与命名约定

- 新评测输出统一 `data/output/kgqa/<ds>/agent/<run_id>/`(checked_batch_eval.jsonl / summary / initial_*),与旧 `data/output/WebQSP/checked_batch_agent/` 隔离;旧目录只读(论文数字凭证)。
- 指标实现**原样迁移**,本期不与 `kgqa/pfit/eval.py` 做指标库合并重构(两者口径有差异:agent 是批停/引用/幻觉的 Ch5 口径;合并留待后续,避免动 parity 基线)。

## 7. 老代码去留

- `oh_my_agent/` 整包 + `scripts/offline_path_search.py`:迁移期保留不动(parity 测试要引用 legacy service/agent 对拍);**全部门槛通过后标 legacy 只读**(AGENTS.md 注明,同 `llm_infer/` 待遇),物理删除后置到 Ch5 新数字(含 MetaQA)全部落地。
- 不动 `kgqa/retrieve/engine.py` 数值内核与 `kgqa/eval/`、`kgqa/pfit/` 既有行为。

## 8. 风险

| 风险 | 应对 |
|---|---|
| 回放对 batch_size / check_mode(hybrid)/ run-flags 敏感,配置漂移即批次对不齐 | 回放参数固定取 canonical 配置;`_assert_batch_alignment` 已有护栏,parity 测试断言 summary 逐位 |
| 反向 entity_map(400 万条)构建成本高 | 保持 `_ReplaySession` 单次构建摊销的现有设计;spec 钩子只换加载来源不换缓存策略 |
| kgqa/server 补 group_tails 需数据集邻接表 | 复用 adapter/GlobalKG 已有结构;MetaQA 侧 smoke 验证 |
| agent 主逻辑可能有隐性 2-hop 假设(hop_filter/expansion) | MetaQA 3-hop smoke 显式覆盖;发现即修并补测试 |
| demo_page 前端静态资源路径随包移动失效 | 迁移后实际起服务点检页面 |
