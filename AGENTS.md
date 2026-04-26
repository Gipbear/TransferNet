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

## Local HTTP Services

- 如果本地 LLM server 已经启动，调用大模型时优先使用 `oh_my_agent.llm_server.client.LLMClient` 访问 HTTP 接口，不要在测试或对比脚本中重新加载 base model / adapter。
  - 默认地址：`http://localhost:8788`
  - 启动示例：
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
  - 启动示例：
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
