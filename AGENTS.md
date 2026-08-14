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

## Code Style

- 单行不超过 120 字符（函数签名、函数体、调用均适用）。
- 函数/方法签名、两三个参数的函数调用不超过 120 字符时不换行。
- 函数/方法的有效代码实现不超过 50 行；超过则拆分为子函数。
- if/else/for/while 展开写；仅极简单行（如 `if not x: return`）可写一行。
- 列表推导式仅用于简单过滤/映射，逻辑复杂时用普通 for 循环。
- 单个文件长度建议在 500 行左右；若文件内逻辑紧密、拆分会破坏可读性，可不拆分；否则应重构或拆分为多个模块。

### Import 规范

- import 移到文件开头；函数体内的延迟导入仅用于刻意规避循环依赖。
- 无效导入及时删除；改动后不再使用的导入必须清理。
- 单条 import 不换行：`from X import (Y,)` → `from X import Y`。
- 短 import 单行：两个符号且总长 ≤120 字符时 `from X import A, B`，不拆多行。
- 同组 import 间不留空行：stdlib / third-party / local 三组之间各一个空行，组内不拆空行。
- 无命名冲突时不重命名：`as Foo` 仅在有同名符号冲突时使用。

## 重构流程

- 不直接动手。用户提供大致重构方向 → 双方讨论确定架构方案 → 才开始实施。
- 若用户未给方向，主动询问或发起讨论，不要自行假设。
- 开始前必须先做概况，包括：模块的文件架构、依赖关系、代码量、重构策略。

## 代码组织

- 避免扁平化散落函数，倾向模块化。
- 不是为复杂而定义复杂类，而是把相关功能聚合到一个类里实现（如坐标转换封装成 `CoordTransform` 类，对外提供方法）。
- 工具类（坐标转换、IO、配置等）统一封装成类供其他模块调用，提升管理、调用、可读性。
- 用抽象基类（ABC）定义接口契约，多实现走策略模式或模板方法模式。
- 数据类用 `@dataclass`：配置类、数据模型都适用。
- 配置项较多时倾向 YAML 驱动（dataclass schema 对应 YAML 结构），运行时保留配置副本便于复现。
- 函数签名注解类型，现代风格 `list[dict]`、`str | None`；不强制注解每个局部变量。

## 写文件与注释

- 预期很长的文件不一次性写完；先创建文件搭好骨架（类/函数定义、注解、注释），功能体用 `...` 或简单占位实现，后续逐步填充。
- 函数/类注释一句话概括即可，除非重要或易混淆才多写；注释写得很复杂往往说明责任边界不清晰，应反思设计而非堆注释。
- 算法类函数可以复杂说明细节，放在函数、类或 docstring 中。
- 每个模块文件开头写 docstring，一句话概括职责，必要时列出提供的内容。
- 业务逻辑注释和 docstring 以中文为主。

## 测试规范

- 修复 bug 或新增功能前，先写失败测试复现问题/定义预期行为（红 → 绿）。
- 重构时先确保现有测试通过，重构后再次验证。
- 测试框架用 `unittest`，测试文件放 `tests/` 目录，命名 `test_<module>.py`。
- 核心库代码强制 TDD；`scripts/` 入口脚本可豁免。

## 交互规则

- 每次开口（无论提问、汇报、总结）都以「报告大王！」开头。
- 每次任务完成后必须给总结：常规任务简要总结，特别重要的任务详细总结。

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

`kgqa/` 是统一 KGQA 检索框架,通过 dataset adapter 注册表分发各数据集;具体支持范围以 CLI 帮助和 `experiments/README.md` 为准:

```bash
python -m kgqa.retrieve.cli.dump_scores --dataset <DS> ...   # 生成 score 缓存
python -m kgqa.retrieve.cli.retrieve --dataset <DS> --backend offline|online --input_dir <DIR> ...  # offline=score 缓存,online=ckpt 实时
python -m kgqa.retrieve.cli.eval ...
```

### kgqa/pfit 训练流水(Ch4)

`kgqa/pfit/` 提供数据集无关的 build→train→eval 流水;模块级运行可通过实验目录中的 `manifest.json` 断点续跑:

```bash
python -m kgqa.pfit.subset_qa --input <QA> --output <SUB> --n 20000  # 按 hop 分层子集(JSON 或 MetaQA 预处理 .pt)
python -m kgqa.pfit.build --dataset <DS> --input <RETRIEVE_JSONL> --exp_dir <DIR> ...  # 建 SFT 集(输入须含 golden)
python -m kgqa.pfit.train --exp_dir <DIR>            # QLoRA,adapter 写 <DIR>/adapter/
python -m kgqa.pfit.eval --dataset <DS> --input <TEST_JSONL> --exp_dir <DIR> [--adapter <DIR>/adapter]
```

正式第四章实验使用 `python -m experiments.ch4.run`; `scripts/run_pfit.sh` 仅作为历史兼容入口,不得作为新实验的默认编排器。路径格式由 pfit CLI 管理,当前为 `arrow`/`nl`/`tuple`/`chain`,`schema` 系已废弃。现役实验清单与目录约定见 `experiments/README.md` 和 `docs/experiments/experiments_kgqa_reproducible_layout.md`；历史迁移验证记录见 `docs/experiments/experiments_kgqa_stage_history.md`。

### LLM SFT(Ch4 legacy,只读)

`llm_infer/` 与 `scripts/run_ablation.sh` 为 Ch4 原始实现,只读保留(pfit 建集 parity 的对拍参照);新实验通过 `experiments.ch4.run` 编排并调用 `kgqa/pfit/`。消融分组和基座模型以实验配置与 `experiments/README.md` 为准。

### 实验入口

```bash
python -m experiments.ch3.run --dataset <DS> --phase <PHASE>  # Ch3 检索实验
python -m experiments.ch4.run --dataset <DS> --config <CONFIG> --profile <PROFILE>  # Ch4 pfit 实验
python -m experiments.ch5.run --dataset <DS> --config <CONFIG> --profile <PROFILE>  # Ch5 PV-GAC 实验
```

正式参数、阶段和输出目录以 `experiments/README.md` 及对应配置为准。`scripts/` 下的 `run_grid.sh`、`run_pfit.sh`、`run_checked_batch_agent_eval.sh` 和 `collect_ablation_results.py` 仅用于历史复现、兼容性核对或旧结果整理。

### 测试

```bash
python -m unittest discover -s tests -t . -p 'test*.py' -v
bash tests/run_ablation_lib_test.sh  # ablation 库函数测试
bash tests/run_pfit_lib_test.sh     # run_pfit.sh 命令拼装 dry-run 测试
```

`-t .` 必须带:缺省顶层目录时 `tests/kgqa` 会遮蔽项目 `kgqa` 包,导致测试导入错误。
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
- 做路径检索一致性(parity)检查时,应复用已有 JSONL 中的样本与检索参数(包括适用时的 `topics`、`hop`、`beam_size`、`lambda_val`、`eta`、`penalty_mode`);数值容差以对应 schema/测试为准,重点比对三元组序列和 prediction 是否一致。

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

去掉 `--limit` 即全量运行。直接调用 CLI 时,`--output` 可指定运行目录;正式 Ch5 编排由 `experiments.ch5.run` 负责写入 `data/output/kgqa/ch5_pv_gac/<ds>/<config_id>/`。输出包括 `checked_batch_eval.jsonl`、`checked_batch_eval_summary.json`、`initial_retrieval.jsonl` 和 `initial_answer.jsonl`;同目录重跑会复用已完成样本,旧 `data/output/WebQSP/checked_batch_agent/` 只读保留。

## 代码边界

- `kgqa/` 是当前统一框架;`retrieve/`、`pfit/`、`agent/`、`serving/` 和 `experiments/` 分别承担检索、路径监督微调、checked-batch、服务和实验编排。
- `oh_my_agent/` 与 `llm_infer/` 是只读 legacy/parity 参考;新功能和新实验不要写入这两个目录。
- 具体模块关系、模型实现和数据集差异以源代码、测试及 `experiments/README.md` 为准,不要把实现细节复制到本文件。

## 分析归档

- 探索阶段的最终产物(分析结论、核对报告、阶段性 README、误差分析摘要)归档到 `data/analysis/` 下,不要散落在临时脚本目录或 `data/output/` 根目录。
- 目录名含时间戳时统一用分钟级格式 `YYYYMMDD_HHMM__slug`,不用秒级。
- 同一会话产生的分析内容默认收敛到同一个归档目录、单一 README;发现高度重叠的归档时先报告并确认,不要自动删除。

## Git 提交规范

### 提交前检查

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

```

正文用 `-` 列表逐项简述;仅 1 项变更时可省略正文。

type 精确选词,不混用:`feat` 全新功能或文件 / `fix` 修 bug / `refactor` 重构 / `test` 测试 / `docs` 仅文档 / `chore` 构建、依赖、配置 / `perf` 性能。scope 取模块简称,如 `agent`、`eval`、`llm-server`、`llm-infer`、`kgqa`。

### 暂存与提交

1. 按文件名暂存,不用 `git add -A` / `git add .`,避免把 `.env`、大二进制文件意外纳入。
2. 提交后运行 `git status` 确认无残留变更。

### 安全红线

- 只有用户明确要求时才创建提交,不主动提交。
- 不跳过 hook(`--no-verify`);hook 报错时定位根因修复。
- 不提交含密钥的文件(`.env`、凭据 JSON 等),发现时主动告知用户。
- 不对 `main` / `master` 执行 force push。
