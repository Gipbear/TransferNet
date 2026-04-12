# PathfinderAgent 设计文档

## 1. 概述
**PathfinderAgent** 是一个多阶段的知识图谱问答（KGQA）增强推理系统。它融合了启发式符号推理引擎（TransferNet）的路径检索能力与大型语言模型（LLaMA 3.1 8B）的语义理解、改写和反思校验能力。其核心目标是提升在 WebQSP 等多跳问答数据集上的召回率（Hits）和准确度（Hit@1 & F1）。

## 2. 系统架构与核心模块

系统位于 `pathfinder_agent/` 目录下，由主控 Agent 和 5 个核心工具（Tools）模块构成：

### 2.1 主控模块 (`agent.py`)
- **`PathfinderAgent`**: 管理大模型、LoRA Adapter 及 TransferNet 包装器的初始化与状态。
- **5步流水线控制**: 调度多视角改写、主路检索、推理提取、校验拦截（可能触发降级二次检索）、聚合输出。

### 2.2 多路语义重写 (`tools/query_rewriter.py`)
- **目的**: 解决原始用户问题在 TransferNet 中因单一问法导致零检索或检索质量差的问题。
- **机制**:
  - LLM 生成 2 个语义相近但不同视角的衍生问句（总共 3路 并行）。
  - **强制约束**: 必须完整保留 `[Topic Entity]`（带方括号形式），以确保与后续的检索管道（字典映射）100% 兼容。

### 2.3 动态路径检索 (`tools/dynamic_retriever.py`)
- **目的**: 提供灵活的知识图谱路径下钻能力，摒弃前期静态 JSONL 文件读取。
- **机制**:
  - 包装了原有的 `WebQSP/predict.py` MMR 束搜索逻辑（`TransferNetWrapper.retrieve`）。
  - 支持 **主路检索**（`beam=20, lambda=0.2`）与 **降级Fallback检索**（`beam=50, lambda=1.0`）。
  - **启发式剪枝**: 在检索出路径后，强行执行 `_filter_by_tail_node`，保证同一个尾节点最多只保留 2 条支撑路径，总数截断为 30，增加给大模型的信息多样度并防止上下文溢出。

### 2.4 推理与依据提取 (`tools/llm_reasoner.py`)
- **目的**: 基于输入提示词和证据路径进行问答推理。
- **机制**: 采用已在 `groupD_chain_name` 上微调的 LoRA 取代基座，利用特定 Prompt 让模型严格按照 **V2 输出格式** 执行：
  ```
  Supporting Paths: 1, 3
  Answer: entity_name
  ```

### 2.5 答案校验与自我反思 (`tools/answer_verifier.py`)
- **目的**: 对抗大模型的幻觉及低级语法切分错误，提供反思依据。
- **机制**: 另一套英文 Prompt 让大模型对刚才输出的候选答案及依赖路径做检查。
  - 检测 **槽位不匹配（Slot Mismatch）**：如问“父亲”但回答了“配偶”。
  - 检测 **超分幻觉（Hallucination）**：答案在路径中并未出现。
  - 检测 **异常切分（Surface Split）**：如包含逗号的单个实体被强行切碎。
  - 如果判定为 `INVALID`，反馈至 Agent，触发降级二次检索（使用 fallback 配置），极大程度挽救易错样本。

### 2.6 多答案路口聚合 (`tools/answer_aggregator.py`)
- **目的**: 处理来自 3 个独立问题重写路径的所有候选答案。
- **机制**:
  - 第一步先进行扁平化并集（Flat Union，去除大小写一致性重复）。
  - 若答案实体过多，再次调用 LLM Prompt，对该并集答案做过滤和逻辑纠偏，去除明显与问句核心类型不符的杂项。

## 3. 工作流运行示例 (Data Flow)
每处理一条样本（如 `run_agent_eval.py` 中），流程流动如下：

1. `PathfinderAgent.run(question, topic_entity)` 被调用。
2. 调用 `query_rewriter`，生成变体 `Q1` (原始), `Q2`, `Q3`。
3. **并发/循环** 处理每一路 `Q_i`：
   - a. TransferNet 检索获取最高可信度的 top-K 主路路径组 `PrimaryPaths_i`。
   - b. 将 `Q_i` 与 `PrimaryPaths_i` 输入 `llm_reasoner` 得到 `Candidate_i`。
   - c. 送入 `answer_verifier`：
      - 若返回 `VALID`，跳到下一点。
      - 若返回 `INVALID: <reason>`，则触发降低筛选阈值的 Fallback Retrievel 获得 `FallbackPaths_i`，再送入 `llm_reasoner`，更新候选 `Candidate_i`。
4. 将所有合法产生的候选答案合集发给 `answer_aggregator` 进行精装合并。
5. 返回并进入标准的 `Hit@1 / F1` 指标评估。

## 4. 技术栈与环境依赖
- **大模型底座**: `unsloth/meta-llama-3.1-8b-instruct-bnb-4bit`
- **加速与微调**: Unsloth 的 `FastLanguageModel` 推理接口（4-bit 量化），挂载 PEFT 的 LoRA Adapter。
- **预处理与Tokenizer**: Transformers（`apply_chat_template` 等需要注意规避 BatchEncoding 和 Tensor 冲突问题，统一使用 `tokenizer.pad()` 构造）。
- **指标统计**: 完全复用原 `llm_infer/eval_faithfulness.py` 的 F1 与 Hits 计算逻辑。
