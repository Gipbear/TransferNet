# AGENTS.md

TransferNet(EMNLP 2021)多跳 KGQA 实现,扩展为三章实验:

- **Ch3**: TransferNet + MMR 多样性 beam search 检索推理路径
- **Ch4**: 用 TransferNet 推理路径对 LLaMA 3.1 8B 做 QLoRA SFT(现役 `kgqa/pfit/`;`llm_infer/` 只读保留作 parity 参照与 Ch4 复现凭证)
- **Ch5**: `kgqa/agent/` — checked-batch(PV-GAC)推理流水线(分批答题 + LLM reject 检查,首批 loose、后续批 strict);`oh_my_agent/` 只读保留作 WebQSP parity 与论文复现凭证

## 环境与语言

- 本地 Conda 环境 `py312_t271_cuda` 用于运行 Python、测试和实验命令;除非用户明确指定其他环境,先激活该环境再直接运行命令,不要反复询问环境选择。
- 不要把 `conda run -n py312_t271_cuda ...` 作为默认执行方式(长任务和服务的终端输出会延迟)。
- 以上只约束本机执行;生成或修改项目脚本、配置、代码时,不要写入 Conda 环境切换命令,除非用户明确要求。
- 默认使用中文与用户沟通(需求澄清、plan、进度更新、评审意见、最终总结均适用);代码、命令、日志、报错、配置键名、API 名称、文件路径保持原文,必要时补充中文解释。

## 协作约束

- 不使用 Codex subagent 或其他子代理;所有分析、实现、测试、评审与验证均由当前主代理直接完成。

## 常用命令

### TransferNet 训练与预测

各数据集模块以 Python module 方式运行(`python -m`),不要以脚本方式运行:

```bash
python -m MetaQA_KB.preprocess --input_dir <METAQA_DIR> --output_dir <PROCESSED_DIR>  # 仅 MetaQA 需要
python -m MetaQA_KB.train --glove_pt <GLOVE_PT> --input_dir <PROCESSED_DIR> --save_dir <CKPT_DIR>
python -m WebQSP.train --input_dir <DATA_DIR> --save_dir <CKPT_DIR>
python -m CompWebQ.train --input_dir <DATA_DIR> --save_dir <CKPT_DIR>
python -m WebQSP.predict --input_dir <DATA_DIR> --ckpt <CKPT_PATH> --mode test
```

### kgqa 统一检索框架

`kgqa/` 是 stage1 重构出的统一 KGQA 检索框架,通过 dataset adapter 注册表分发 webqsp / metaqa / cwq:

```bash
python -m kgqa.retrieve.cli.dump_scores --dataset <DS> ...   # 生成 score 缓存
python -m kgqa.retrieve.cli.retrieve --dataset <DS> --backend offline|online --input_dir <DIR> ...  # offline=score 缓存,online=ckpt 实时
python -m kgqa.retrieve.cli.eval ...
```

### kgqa/pfit 训练流水(Ch4 现役,stage2)

数据集无关的 build→train→eval 流水,输出统一 `data/output/kgqa/<ds>/`,三阶段断点续跑基于实验目录 manifest.json:

```bash
python -m kgqa.pfit.subset_qa --input <QA> --output <SUB> --n 20000  # 按 hop 分层子集(JSON 或 MetaQA 预处理 .pt)
python -m kgqa.pfit.build --dataset <DS> --input <RETRIEVE_JSONL> --exp_dir <DIR> ...  # 建 SFT 集(输入须含 golden)
python -m kgqa.pfit.train --exp_dir <DIR>            # QLoRA,adapter 写 <DIR>/adapter/
python -m kgqa.pfit.eval --dataset <DS> --input <TEST_JSONL> --exp_dir <DIR> [--adapter <DIR>/adapter]
bash scripts/run_pfit.sh --exp <exp_id> [--phase build|train|eval|all]  # 实验注册表编排(8 实验;LIMIT/FMT/ADAPTER 变体)
```

数据集差异集中在 `kgqa/pfit/specs.py`(entity_repr、问题清洗、hop 分层、拒答开关);路径格式只剩 arrow/tuple/chain/nl(schema 系已废弃)。实验清单与目录约定见 `docs/experiments/experiments_kgqa_stage2_pfit_20260711.md`。

### LLM SFT(Ch4 legacy,只读)

`llm_infer/` 与 `scripts/run_ablation.sh` 为 Ch4 原始实现,只读保留(pfit 建集 parity 的对拍参照);新实验一律走 `kgqa/pfit/`。消融分组:A 输出格式 / B 训练数据 / C 检索参数 / D 路径输入格式;基座 `unsloth/meta-llama-3.1-8b-instruct-bnb-4bit`。

### 实验脚本

```bash
bash scripts/run_grid.sh webqsp|metaqa|cwq [ckpt]  # MMR beam/lambda 网格搜索
bash scripts/run_pfit.sh --exp <exp_id> --phase all # Ch4 pfit 实验编排(smoke 加 LIMIT=100)
bash scripts/run_checked_batch_agent_eval.sh        # Ch5 checked-batch 评测(自动确保服务在线)
python scripts/collect_ablation_results.py          # 消融日志汇总为 CSV
```

脚本均支持断点续跑(输出已存在则跳过)和环境变量覆盖;默认参数以各脚本头部为准,不要依赖文档里的历史值。

### 测试

```bash
python -m unittest discover -s tests -t . -p 'test*.py' -v
bash tests/run_ablation_lib_test.sh  # ablation 库函数测试
bash tests/run_pfit_lib_test.sh     # run_pfit.sh 命令拼装 dry-run 测试
```

`-t .` 必须带:缺省顶层目录时 `tests/kgqa` 会遮蔽项目 `kgqa` 包,72 个用例报 import error。
默认 discovery 不运行依赖 gitignored checkpoint、score cache 或真实数据的测试；显式执行这些测试时设置 `RUN_KGQA_ARTIFACT_TESTS=1`，并确保模型/tokenizer 已可离线加载。

## 常驻服务与 Ch5 评测

两个常驻 HTTP 服务,一次加载多次复用:

- `path_retrieve_server`(默认 `http://localhost:8789`):score 缓存检索,由 `kgqa.retrieve.api.path_retrieve_server` 提供;通过 `DATASET`/`INPUT_DIR`/`CACHE` 覆盖可服务 WebQSP 或 MetaQA
- `llm_server`(默认 `http://localhost:8788`):base model + LoRA adapter 生成

```bash
./scripts/path_retrieve_server.sh start|status
./scripts/llm_server.sh start|status
PORT_BUSY_ACTION=kill ./scripts/llm_server.sh start  # 端口被旧进程占用且确认要替换时
```

- 服务已启动时,一律通过 HTTP client 调用(`kgqa.retrieve.api.client`、`kgqa.serving.llm.client.LLMClient`),不要在测试或对比脚本里重新加载 base model / adapter / 检索器。
- 做路径检索一致性(parity)检查时,把已有 JSONL 里的 `topics`/`hop`/`beam_size`/`lambda_val` 原样传给服务;`log_score` 允许 `1e-6` 量级浮点差,重点比对三元组序列和 prediction 是否一致。

评测入口是批量 CLI(依赖上述两个服务):

```bash
python -m kgqa.agent.cli.eval_checked_batch \
    --dataset webqsp \
    --input data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt \
    --output data/output/kgqa/webqsp/agent/quick_50 \
    --limit 50 \
    --check_mode hybrid-reject-list \
    --path_retrieve_url http://localhost:8789 \
    --llm_server_url http://localhost:8788
```

去掉 `--limit` 即全量 1581 条。输出写入 `data/output/kgqa/<ds>/agent/<run_id>/`（`--output` 可显式指定）:`checked_batch_eval.jsonl`(逐样本记录)、`checked_batch_eval_summary.json`(hit1/hit_any/macro_f1/exact_match/citation_accuracy/stop_reason_counts)、`initial_retrieval.jsonl` / `initial_answer.jsonl`(初始检索与首批答题)。同目录重跑会复用已完成样本；旧 `data/output/WebQSP/checked_batch_agent/` 只读保留。

## 架构

### 数据集模块(4 套并行实现)

`MetaQA_KB/`、`MetaQA-Text/`、`WebQSP/`、`CompWebQ/` 各自有 `model.py`/`train.py`/`predict.py`/`data.py`,都定义 `TransferNet(nn.Module)`:

| 模块 | 问题编码 | KG 表示 | 关键差异 |
|------|---------|---------|---------|
| MetaQA_KB | BiGRU + GloVe | 全局稀疏矩阵(`Knowledge_graph.py`) | 3-hop,防环 |
| MetaQA-Text | BiGRU + GloVe | 文本关系(`desc_encoder` BiGRU) | 按分数裁剪活跃实体 |
| WebQSP | BERT/RoBERTa | 内联稀疏矩阵 | 2-hop,sigmoid 关系分布,`entity_range` 掩码 |
| CompWebQ | BERT | 逐样本三元组(`index_add`) | 多路推理(way 乘积) |

核心推理机制 `follow(e, r) = Mobj^T @ (Msubj @ e^T * Mrel @ r^T)`:可微稀疏矩阵乘做 KG 遍历;每 hop 依次为 step encoder → 问题注意力 → 关系分类 → `follow()`。

### 其他模块

- `kgqa/`:统一检索框架。`cli/` 三个入口、`datasets/`(adapter 注册表)、`scores/`(逐数据集 ScoreProducer)、`retrieve/backends/`(offline score 缓存 / online ckpt 实时)、`eval/`、`server/`(路径检索服务)、`llm_server/`、`agent/`(Ch5 checked-batch、replay、tools、demo_page)、`pfit/`(Ch4 现役 SFT 流水:formats/specs/build/train/eval/manifest/subset_qa)
- `utils/`:BiGRU 编码器、RAdam(`misc.py`)、MMR beam search 与路径/多样性指标(`path_utils.py`)、多阈值评测统计(`eval_utils.py`)
- `oh_my_agent/`(Ch5 legacy,只读):原始 checked-batch、服务与 demo_page 实现;仅用于与 `kgqa/agent/`、`kgqa/server/`、`kgqa/llm_server/` 做 WebQSP parity 及历史论文复现
- `llm_infer/`(Ch4 legacy,只读):`kg_format.py`、`train_sft.py`、`build_kgcot_dataset.py`、`eval_faithfulness.py`;已迁移至 `kgqa/pfit/`,保留作 parity 对拍参照

## 关键约定

- Loss 用加权 MSE,正样本重加权(MetaQA `answers*9+1`,WebQSP `answers*99+1`)
- 每 hop 后分数 clamp(>1 时可微缩放);默认优化器 RAdam(`utils/misc.py`);梯度裁剪 value=0.5、norm=2
- `data/` 与 `models/` 已 gitignore;GloVe 需先 `python pickle_glove.py` 预处理为 pickle
- Docker 使用国内镜像源(清华 PyPI、中科大 APT)

## 分析归档

- 探索阶段的最终产物(分析结论、核对报告、阶段性 README、误差分析摘要)归档到 `data/analysis/` 下,不要散落在临时脚本目录或 `data/output/` 根目录。
- 目录名含时间戳时统一用分钟级格式 `YYYYMMDD_HHMM__slug`,不用秒级。
- 同一会话产生的分析内容默认收敛到同一个归档目录、单一 README;发现多份高度重叠的归档时合并为单一入口,保留信息更完整的版本,删除重复与空目录。

## Git 提交规范

### 提交前三步(并行执行)

```bash
git status             # 不要加 -uall,大仓库会有内存问题
git diff
git log --oneline -10
```

### Commit Message 格式

遵循 Conventional Commits,消息中文(type/scope 保持英文):

```
type(scope): 中文简述(≤50 字)

- 变更项一(文件/模块:做了什么)
- ...

Co-Authored-By: <git config user.name> <<git config user.email>>
Co-Authored-By: 当前协作模型/助手名称 <对应 noreply 邮箱>
```

正文用 `-` 列表逐项简述;仅 1 项变更时可省略正文。**Co-Authored-By 的用户信息必须现场执行 `git config user.name` / `git config user.email` 读取,禁止使用记忆、对话历史或硬编码值**;协作模型行按当前会话实际使用的模型填写(如 `Claude Opus 4.8 <noreply@anthropic.com>`、`Codex <noreply@openai.com>`)。

type 精确选词,不混用:`feat` 全新功能或文件 / `fix` 修 bug / `refactor` 重构 / `test` 测试 / `docs` 仅文档 / `chore` 构建、依赖、配置 / `perf` 性能。scope 取模块简称,如 `agent`、`eval`、`llm-server`、`llm-infer`、`kgqa`。

### 暂存与提交

1. 按文件名暂存,不用 `git add -A` / `git add .`,避免把 `.env`、大二进制文件意外纳入。
2. 用 HEREDOC 传 commit message(`git commit -m "$(cat <<'EOF' ... EOF)"`),防止引号和换行出错。
3. 提交后运行 `git status` 确认无残留变更。

### 安全红线

- 只有用户明确要求时才创建提交,不主动提交。
- 不跳过 hook(`--no-verify`);hook 报错时定位根因修复,新建 commit 重新提交,不用 `--amend`。
- 不提交含密钥的文件(`.env`、凭据 JSON 等),发现时主动告知用户。
- 不对 `main` / `master` 执行 force push;`--amend` 仅在用户明确要求时使用。
