# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TransferNet implementation (EMNLP 2021) for multi-hop KGQA, extended with:
- **Ch3**: TransferNet + MMR diversity beam search for reasoning path retrieval
- **Ch4**: QLoRA SFT of LLaMA 3.1 8B using TransferNet reasoning paths
- **Ch5**: PathfinderAgent — multi-stage reasoning pipeline (query rewrite → MMR retrieval → LLM reason → verify → aggregate)

## Common Commands

### Environment Setup
Default Conda environment: `py312_t271_cuda`. Use this environment for Python, tests, and experiment commands unless the user explicitly requests another environment; do not ask repeatedly which environment to use.

### Response Language
Default to Chinese for user-facing communication unless the user explicitly asks for another language. This applies to planning, progress updates, review feedback, test summaries, and final responses. Keep code, commands, logs, error messages, config keys, API names, and file paths in their original form, with Chinese explanation when helpful.

```bash
conda env create -f environment.yml    # Full environment (Python 3.10, PyTorch 2.2.2, CUDA 12.1)
pip install -r requirements.txt        # Minimal dependencies
docker build -t transfernet .          # Docker alternative
```

### TransferNet Training & Prediction
```bash
# Preprocess (MetaQA only)
python -m MetaQA_KB.preprocess --input_dir <METAQA_DIR> --output_dir <PROCESSED_DIR>

# Train (run as module, not script)
python -m MetaQA_KB.train --glove_pt <GLOVE_PT> --input_dir <PROCESSED_DIR> --save_dir <CKPT_DIR>
python -m WebQSP.train --input_dir <DATA_DIR> --save_dir <CKPT_DIR>
python -m CompWebQ.train --input_dir <DATA_DIR> --save_dir <CKPT_DIR>

# Predict
python -m MetaQA_KB.predict --input_dir <PROCESSED_DIR> --ckpt <CKPT_PATH> --mode test
python -m WebQSP.predict --input_dir <DATA_DIR> --ckpt <CKPT_PATH> --mode test
```

### LLM SFT (Chapter 4)
```bash
# Build text dataset → Train → Evaluate (full pipeline)
bash scripts/run_ablation.sh --dataset webqsp --group A --phase all

# Individual steps
python -m llm_infer.build_kgcot_dataset ...
python -m llm_infer.train_sft --train <JSONL> --output_dir <DIR> --epochs 3
python -m llm_infer.eval_faithfulness ...
```

### Experiment Scripts
```bash
bash scripts/run_grid.sh webqsp|metaqa|cwq [ckpt_path]     # MMR beam/lambda grid search
bash scripts/run_ablation.sh --dataset X --group A|B|C|D    # Ch4 ablation experiments
bash scripts/run_agent_experiments.sh                        # Ch5 agent evaluation
python scripts/collect_ablation_results.py                   # Parse ablation logs to CSV
```

### Tests
```bash
conda run -n py312_t271_cuda python -m unittest discover -s tests -p 'test*.py' -v
bash tests/run_ablation_lib_test.sh    # Tests for ablation library functions
conda run -n py312_t271_cuda python -m unittest tests/test_pathfinder_agent.py -v
conda run -n py312_t271_cuda python -m unittest tests/test_pathfinder_replay.py -v
```

### PathfinderAgent Evaluation (Chapter 5)
```bash
# 启动模型服务器（避免每次重新加载，推荐）
conda run -n py312_t271_cuda python scripts/model_server.py \
    --ckpt data/ckpt/WebQSP/model-29-0.6411.pt \
    --input_dir data/input/WebQSP --port 8787

# 客户端模式评测（连接已启动的服务器）
python run_agent_eval.py --server-url http://localhost:8787 \
    --input <EVAL_JSONL> --output <OUTPUT_JSONL>

# 独立模式（单次使用，含 3-4 分钟模型加载开销）
python run_agent_eval.py --ckpt <CKPT> --input_dir <DATA_DIR> \
    --input <EVAL_JSONL> --output <OUTPUT_JSONL>
```

### oh_my_agent Services & Evaluation
当前 `oh_my_agent` 的评测入口是 `python -m oh_my_agent.cli.eval_webqsp`。它不是单独的 HTTP eval service，而是一个批量 CLI：先调用常驻 `path_server` 检索 TransferNet MMR 路径，再调用常驻 `llm_server` 用本地 LLM + LoRA adapter 生成答案，最后写出逐样本 JSONL 和 summary。

```bash
# 推荐：启动两个常驻服务
conda run -n py312_t271_cuda ./scripts/path_server.sh start
conda run -n py312_t271_cuda ./scripts/llm_server.sh start

# 检查服务状态
./scripts/path_server.sh status
./scripts/llm_server.sh status

# 如果端口被旧服务占用且明确要替换
PORT_BUSY_ACTION=kill conda run -n py312_t271_cuda ./scripts/path_server.sh start
PORT_BUSY_ACTION=kill conda run -n py312_t271_cuda ./scripts/llm_server.sh start
```

默认服务地址：
- `path_server`: `http://localhost:8787`
- `llm_server`: `http://localhost:8788`

快速小样本评测：
```bash
conda run -n py312_t271_cuda python -m oh_my_agent.cli.eval_webqsp \
    --input data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1580.txt \
    --output data/output/WebQSP/simple_agent_eval_20.jsonl \
    --limit 20 \
    --beam_size 20 \
    --lambda_val 0.2 \
    --path_server_url http://localhost:8787 \
    --llm_server_url http://localhost:8788
```

完整评测：
```bash
conda run -n py312_t271_cuda python -m oh_my_agent.cli.eval_webqsp \
    --input data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1580.txt \
    --output data/output/WebQSP/simple_agent_eval_full.jsonl \
    --beam_size 20 \
    --lambda_val 0.2 \
    --path_server_url http://localhost:8787 \
    --llm_server_url http://localhost:8788
```

输出文件：
- `*.jsonl`: 逐样本记录，包含检索路径、LLM 输出、答案解析、citation 与 answer metrics。
- `*_summary.json`: 聚合指标，如 `hit1`、`hit_any`、`macro_f1`、`exact_match`、`citation_accuracy`、`hallucination_rate`、`format_compliance`。

## Architecture

### Dataset Modules (4 parallel implementations)
Each module (`MetaQA_KB/`, `MetaQA-Text/`, `WebQSP/`, `CompWebQ/`) has its own `model.py`, `train.py`, `predict.py`, `data.py`. All define a `TransferNet(nn.Module)` class but differ in:

| Module | Question Encoder | KG Representation | Key Difference |
|--------|-----------------|-------------------|----------------|
| MetaQA_KB | BiGRU + GloVe | Global sparse matrices via `Knowledge_graph.py` | 3-hop, cycle prevention |
| MetaQA-Text | BiGRU + GloVe | Text-form relations via `desc_encoder` BiGRU | Score-based active entity pruning |
| WebQSP | BERT/RoBERTa | Inline sparse matrices | 2-hop, sigmoid relation dist, `entity_range` masking |
| CompWebQ | BERT | Per-sample triples via `index_add` | Multi-way reasoning (product of ways) |

### Core Reasoning Mechanism
The `follow(e, r)` operation: `Mobj^T @ (Msubj @ e^T * Mrel @ r^T)` — differentiable sparse matrix multiplication for KG traversal. Each hop: step encoder → attention over question → relation classifier → `follow()`.

### Shared Utilities (`utils/`)
- `BiGRU.py`: GRU and BiGRU encoders with packed sequence handling
- `misc.py`: RAdam optimizer, GloVe loading, MetricLogger
- `path_utils.py`: MMR diversity beam search, path metrics (hit/recall/precision/F1), diversity metrics (Jaccard, tail uniqueness, coverage)
- `eval_utils.py`: Multi-threshold evaluation statistics
- `lr_scheduler.py`: Multiple LR scheduler implementations

### PathfinderAgent (`pathfinder_agent/`)
- `agent.py`: 多阶段推理入口 — query rewrite → MMR retrieval → LLM reason → verify → aggregate
- `config.py`: 路径/beam/lambda 配置；默认 adapter: `models/webqsp/ablation/groupAname_v2`
- `tools/query_rewriter.py`: 问题改写，normalize + expand
- `tools/dynamic_retriever.py`: 两阶段 MMR 检索（primary beam20/λ0.2，fallback beam50/λ1.0）
- `tools/llm_reasoner.py`: LoRA adapter 推理，输出 reasoning + answer
- `tools/answer_verifier.py`: 验证候选答案是否出现在路径中
- `tools/answer_aggregator.py`: 多路径答案投票聚合
- `scripts/model_server.py`: HTTP 模型服务器，一次加载多次复用（避免重复启动开销）

### LLM Layer (`llm_infer/`)
- `kg_format.py`: Path formatting (arrow/tuple/chain × MID/name) and system prompts for output formats (V1-V5)
- `train_sft.py`: QLoRA via Unsloth + HuggingFace SFTTrainer. Smart truncation preserves golden paths. Prompt masking (loss only on assistant replies)
- `build_kgcot_dataset.py`: Builds training JSONL from TransferNet MMR paths
- `eval_faithfulness.py`: Citation accuracy, hallucination rate, F1 evaluation
- `agent/react_loop.py`: `KGReActAgent` — Thought/Action/Observation loop
- `agent/tools.py`: `ToolRegistry` with RetrievePaths, ReasonAndCite, VerifyCitation, DecomposeQuestion, Finish tools
- `agent/agent_config.py`: Agent configuration dataclass

### Experiment Orchestration (`scripts/`)
- `run_ablation.sh` sources `run_ablation_lib.sh` for dataset context/adapter resolution
- All scripts support resumability (skip if output exists) and environment variable overrides
- Ablation groups: A (output format), B (training data), C (retrieval params), D (path input format)
- Base LLM: `unsloth/meta-llama-3.1-8b-instruct-bnb-4bit`

## Key Conventions

- All dataset modules run as Python modules (`python -m MetaQA_KB.train`), not scripts
- Loss uses weighted MSE with heavy positive weighting (e.g., `answers * 9 + 1` for MetaQA, `answers * 99 + 1` for WebQSP)
- Score clamping after each hop: differentiable rescaling for values > 1
- Default optimizer: RAdam (custom implementation in `utils/misc.py`)
- Gradient clipping: value=0.5, norm=2
- Data directory (`data/`) and model checkpoints (`models/`) are gitignored
- GloVe embeddings must be pre-pickled via `python pickle_glove.py`
- The project uses Chinese mirror sources in Docker (Tsinghua PyPI, USTC APT)
