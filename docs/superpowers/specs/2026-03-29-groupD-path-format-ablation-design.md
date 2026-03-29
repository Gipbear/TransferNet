# GroupD 路径输入格式消融实验设计

## 背景与动机

GroupA 消融实验中，路径数据占用过多 token，导致 ~35% 的训练样本被截断（max_seq_len=1280）。
V5 自然语言格式将截断率降至 0.7%，证明路径格式对 token 效率有显著影响。

GroupD 独立消融两个正交维度：
1. **路径结构格式**：arrow / tuple / chain
2. **实体表示方式**：MID / 实体名称（name）

## 实验矩阵

3 格式 × 2 实体表示 = **6 组实验**，输出格式统一固定 v2。

| 实验 ID | 格式 | 实体 | 路径示例（2跳） |
|---------|------|------|----------------|
| D-arrow-mid | arrow | MID | `Path 1: (m.0a) -[r1]-> (m.0b) (m.0b) -[r2]-> (m.0c)` |
| D-arrow-name | arrow | name | `Path 1: (Jamaica) -[r1]-> (English) (English) -[r2]-> (UK)` |
| D-tuple-mid | tuple | MID | `1: (m.0a, r1, m.0b), (m.0b, r2, m.0c)` |
| D-tuple-name | tuple | name | `1: (Jamaica, r1, English), (English, r2, UK)` |
| D-chain-mid | chain | MID | `1: m.0a -> r1 -> m.0b -> r2 -> m.0c` |
| D-chain-name | chain | name | `1: Jamaica -> r1 -> English -> r2 -> UK` |

### 关系名处理

关系名保持原始 Freebase 格式（如 `location.country.official_language`），
**不做** `_rel_to_text()` 转换。原因：arrow baseline 不做转换，
为控制变量仅改结构格式，关系名表示应保持一致。
（nl/v5 格式的关系名转换属于 GroupA 实验范围，不在 GroupD 中重复测试。）

### 实体名称处理

- 映射源：`data/resources/WebQSP/fbwq_full/mapped_entities.txt`（~96万条，51% 覆盖率）
- 未映射的 MID 保持原样
- 输入路径和输出 Answer 均使用名称（`-name` 实验）
- 评估时通过 name→MID 反向映射与 golden 比对

## 控制变量

与 GroupA/B/C 保持一致：
- beam=20, lambda=0.2
- shuffle=on, show_score=off
- 5 epochs QLoRA (rank=16)
- LLaMA 3.1 8B (Unsloth)
- 3 runs per config (NUM_RUNS=3)

## 格式化函数定义

### format_path_str_tuple

```python
def format_path_str_tuple(path_edges, log_score, idx, show_score=False):
    """三元组格式：1: (s, r, o), (s, r, o)"""
    triples = ", ".join(f"({e[0]}, {e[1]}, {e[2]})" for e in path_edges)
    if show_score:
        return f"{idx} [score={log_score:.4f}]: {triples}"
    return f"{idx}: {triples}"
```

### format_path_str_chain

```python
def format_path_str_chain(path_edges, log_score, idx, show_score=False):
    """连续链式格式（去重中间实体）：1: e0 -> r1 -> e1 -> r2 -> e2"""
    if not path_edges:
        return f"{idx}:"
    parts = [path_edges[0][0]]  # 起始实体
    for e in path_edges:
        parts.extend([e[1], e[2]])
    chain = " -> ".join(parts)
    if show_score:
        return f"{idx} [score={log_score:.4f}]: {chain}"
    return f"{idx}: {chain}"
```


## 实体名称映射方案

### 加载映射

```python
def load_entity_map(path: str) -> dict[str, str]:
    """加载 MID→Name 映射表（tab-separated）"""
    mid2name = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                mid2name[parts[0]] = parts[1]
    return mid2name
```

### 应用映射

在 `build_user_content` 和输出构造（`output_v2` 等）中，
将 path_edges 中的实体 MID 替换为名称（未映射的保持原样）。

### 评估反向映射

eval 时需要 name→MID 映射。由于名称可能不唯一（多个 MID 对应同一名称），
采用策略：
1. 构建 `name2mids: dict[str, list[str]]`（一对多）
2. 模型输出的 answer entity 先在 name2mids 中查找，匹配到的任何 MID 命中 golden 即算正确
3. 若 answer 本身已是 MID 格式，直接比对

## 需要修改的文件

### 1. `llm_infer/kg_format.py`
- 新增 `format_path_str_tuple`、`format_path_str_chain`
- 新增 `load_entity_map(path) -> dict`
- 新增 `apply_entity_map(path_edges, mid2name) -> path_edges`（替换实体）
- 扩展 `build_user_content` 的 `path_format` 参数支持 `tuple`/`chain`

### 2. `llm_infer/build_kgcot_dataset.py`
- `--path_format` 扩展 choices：`["arrow", "nl", "tuple", "chain"]`
- 新增 `--entity_map` 参数（可选，指向 mapped_entities.txt）
- `make_sample` 中：若有 entity_map，替换路径实体和答案实体
- 关系名保持原始 Freebase 格式，不做 `_rel_to_text` 转换

### 3. `llm_infer/eval_faithfulness.py`
- `--path_format` 同步扩展
- 新增 `--entity_map` 参数
- 评估匹配逻辑：当使用 entity_map 时，构建反向映射 name→MIDs，
  模型输出的 answer entity 通过反向映射转回 MID 再与 golden 比对

### 4. `scripts/run_ablation.sh`
- 新增 `# ── Group D: Path Input Format Ablation ──` 配置块
- 8 组实验配置，每组调用 build → train → eval 三步流水线
- D-arrow-mid 可复用 GroupA v2 baseline adapter（跳过训练）

## System Prompt 处理

v2 prompt 中写的是 "using entity IDs from the paths"，name 变体中路径和答案均为名称。
**决策**：name 变体新增 `SYSTEM_PROMPT_V2_NAME`，将 "entity IDs" 替换为 "entity names"，
通过 `build_kgcot_dataset.py` 和 `eval_faithfulness.py` 中根据是否提供 `--entity_map` 自动选择。

## 验证计划

1. **单元验证**：对每种格式函数写简单断言，确认单跳/多跳输出正确
2. **数据构建验证**：用 `--sample 10` 构建少量样本，人工检查格式正确性和 token 统计
3. **名称映射验证**：检查映射覆盖率，确认未映射 MID 保持不变
4. **端到端验证**：挑 1 组配置完整跑 build → train → eval，确认指标输出正常
5. **反向映射验证**：手动构造含名称的模型输出，验证 name→MID 匹配逻辑正确
