# AGENTS.md

## Environment Constraint

- 默认使用 Conda 环境 `py312_t271_cuda` 运行本项目的 Python、测试和实验命令；除非用户明确指定其他环境，不要反复询问环境选择。

## Response Language

- 默认使用中文与用户沟通，除非用户明确要求英文或其他语言。
- 该要求适用于完整交互过程，包括需求澄清、plan、进度更新、测试反馈、代码评审意见和最终总结。
- 代码、命令、报错信息、日志片段、配置键名、API 名称和文件路径保持原文，必要时在中文语境中补充解释。

## Analysis Archiving

- 探索阶段产生的最终产物需要归档到 `data/analysis/` 下，例如分析结论、核对报告、阶段性 README 和误差分析摘要。
- 归档结果应按任务语义或时间戳组织目录，避免把这类最终结果散落在临时脚本目录或 `data/output/` 根目录。
- 若目录名包含时间戳，统一使用分钟级格式 `YYYYMMDD_HHMM__slug`，不要使用秒级时间戳。
- 同一个会话产生的分析/归档内容默认不要拆成多个目录或多个并列 README；优先在同一个归档目录中连续整理实验过程、阶段结果和最终结论。
- 如果已经出现多份内容高度重叠的分析 README 或归档目录，默认合并为单一入口，保留信息更完整的版本，并删除重复归档与空目录。

## Local HTTP Services

- 如果本地 LLM server 已经启动，调用大模型时优先使用 `oh_my_agent.llm_server.client.LLMClient` 访问 HTTP 接口，不要在测试或对比脚本中重新加载 base model / adapter。
  - 默认地址：`http://localhost:8788`
  - 推荐启动方式：
    ```bash
    conda run -n py312_t271_cuda ./scripts/llm_server.sh start
    ```
  - 状态检查：
    ```bash
    ./scripts/llm_server.sh status
    ```
  - 直接模块启动示例：
    ```bash
    conda run -n py312_t271_cuda python -m oh_my_agent.llm_server.server \
      --adapter models/webqsp/ablation/groupJ_schema_name \
      --port 8788
    ```
  - 客户端示例：
    ```python
    from oh_my_agent.llm_server.client import LLMClient

    client = LLMClient("http://localhost:8788")
    resp = client.generate("...", use_adapter=True)
    ```

- 如果本地 TransferNet path server 已经启动，检索 MMR 路径时优先使用 `oh_my_agent.path_server.client.PathRetrievalClient` 访问 HTTP 接口，不要为了抽样验证、接口测试或 JSONL 对比而重新实例化 `TransferNetPathRetriever`。
  - 默认地址：`http://localhost:8787`
  - 推荐启动方式：
    ```bash
    conda run -n py312_t271_cuda ./scripts/path_server.sh start
    ```
  - 状态检查：
    ```bash
    ./scripts/path_server.sh status
    ```
  - 直接模块启动示例：
    ```bash
    conda run -n py312_t271_cuda python -m oh_my_agent.path_server.server \
      --dataset webqsp \
      --input_dir data/input/WebQSP \
      --ckpt data/ckpt/WebQSP/model-29-0.6411.pt \
      --port 8787
    ```
  - 客户端示例：
    ```python
    from oh_my_agent.path_server.client import PathRetrievalClient

    client = PathRetrievalClient("http://localhost:8787")
    resp = client.retrieve(
        "who was vice president after kennedy died",
        topic_entities=["m.0d3k14"],
        hop=2,
        beam_size=20,
        lambda_val=0.2,
    )
    ```

- 做路径检索一致性检查时，直接调用 path server 接口，并把 `data/output/.../beam*.jsonl` 中的 `topics`、`hop`、`beam_size`、`lambda_val` 原样传入。比较结果时允许 `log_score` 存在 `1e-6` 量级浮点差异，重点检查路径三元组序列和 prediction 是否一致。

## oh_my_agent Evaluation

- 当前 `oh_my_agent` 的评测不是单独的 HTTP eval service，而是 `python -m oh_my_agent.cli.eval_webqsp` 批量调用两个常驻服务：
  - `path_server`：`http://localhost:8787`，负责 TransferNet MMR 路径检索。
  - `llm_server`：`http://localhost:8788`，负责 base model + LoRA adapter 生成答案。
- 评测前优先确认两个服务已启动：
  ```bash
  ./scripts/path_server.sh status
  ./scripts/llm_server.sh status
  ```
- 如果需要从当前默认配置启动两个服务：
  ```bash
  conda run -n py312_t271_cuda ./scripts/path_server.sh start
  conda run -n py312_t271_cuda ./scripts/llm_server.sh start
  ```
- 如果端口被旧进程占用且明确要替换旧服务，可使用：
  ```bash
  PORT_BUSY_ACTION=kill conda run -n py312_t271_cuda ./scripts/path_server.sh start
  PORT_BUSY_ACTION=kill conda run -n py312_t271_cuda ./scripts/llm_server.sh start
  ```
- 快速小样本评测：
  ```bash
  conda run -n py312_t271_cuda python -m oh_my_agent.cli.eval_webqsp \
    --input data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt \
    --output data/output/WebQSP/simple_agent_eval_20.jsonl \
    --limit 20 \
    --beam_size 20 \
    --lambda_val 0.2 \
    --path_server_url http://localhost:8787 \
    --llm_server_url http://localhost:8788
  ```
- 完整评测：
  ```bash
  conda run -n py312_t271_cuda python -m oh_my_agent.cli.eval_webqsp \
    --input data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt \
    --output data/output/WebQSP/simple_agent_eval_full.jsonl \
    --beam_size 20 \
    --lambda_val 0.2 \
    --path_server_url http://localhost:8787 \
    --llm_server_url http://localhost:8788
  ```
- 输出包括逐样本 JSONL 和同名前缀的 summary，例如 `simple_agent_eval_full.jsonl` 与 `simple_agent_eval_full_summary.json`。

## Git Commit Conventions

### 提交前三步（并行执行）

提交前必须同时运行以下命令，再起草 commit message：

```bash
git status          # 查看未追踪文件（不要加 -uall，会导致大仓库内存问题）
git diff            # 查看已暂存 + 未暂存的变更
git log --oneline -10  # 参考本仓库的提交风格
```

### Commit Message 格式

遵循 Conventional Commits，消息使用中文（scope 和 type 保持英文）：

```
type(scope): 中文简述（≤50 字）

- 变更项一（文件/模块：做了什么）
- 变更项二
- ...

Co-Authored-By: <git config user.name> <<git config user.email>>
Co-Authored-By: 当前协作模型/助手名称 <对应 noreply 邮箱>
```

正文使用 `-` 列表，每项一行，简述该文件/模块发生了什么变更。变更项较少（1 项）时可省略正文。

**`Co-Authored-By` 用户信息必须执行 `git config user.name` / `git config user.email` 读取，禁止使用 memory、对话历史或硬编码值。** 可写多行，每行一个协作者。完整示例（Codex 会话）：

```
Co-Authored-By: jsh-smi-wsl <1099048889@qq.com>
Co-Authored-By: Codex <noreply@openai.com>
```

**type 选词规则（精确，不要混用）：**

| type | 含义 |
|------|------|
| `feat` | 完全新增的功能或文件 |
| `fix` | 修复已有功能的 bug |
| `refactor` | 重构，不新增功能也不修 bug |
| `test` | 新增或修改测试 |
| `docs` | 仅文档变更 |
| `chore` | 构建脚本、依赖、配置等维护性变更 |
| `perf` | 性能优化 |

**scope** 取模块简称，例如 `path-server`、`llm-server`、`eval`、`llm-infer`、`agent`。

### 暂存与提交操作规范

1. **按文件名暂存**，不要用 `git add -A` 或 `git add .`，避免把 `.env`、大二进制文件等意外纳入。
2. **用 HEREDOC 传入消息**，防止引号和换行格式出错：
   ```bash
   git commit -m "$(cat <<'EOF'
   feat(eval): 新增 citation accuracy 指标计算

   - eval_faithfulness.py: 新增 citation_accuracy / hallucination_rate 指标
   - tests/test_eval.py: 补充对应单测

   Co-Authored-By: jsh-smi-wsl <1099048889@qq.com>
   Co-Authored-By: Codex <noreply@openai.com>
   EOF
   )"
   ```
3. **提交后运行 `git status`** 确认成功，无残留变更。

### 安全红线

- **只有用户明确要求时才创建提交**，不主动提交。
- 不跳过 hook（`--no-verify`）；hook 报错时定位根因并修复，再重新暂存提交（新建 commit，不用 `--amend`）。
- 不提交含密钥的文件（`.env`、凭据 JSON 等），发现时主动告知用户。
- 不对 `main` / `master` 执行 `push --force`；有此需求时先确认。
- `--amend` 仅在用户明确要求时使用；hook 失败后的修复一律新建 commit，防止覆盖历史。
