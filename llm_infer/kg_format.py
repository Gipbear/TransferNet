"""
KG 路径格式化工具函数（共享模块）

供 llm_infer/build_kgcot_dataset.py 和 llm_infer/eval_faithfulness.py 共同使用，
确保训练与评估时路径字符串格式、System Prompt 完全一致。
"""


# ─── System Prompts ───────────────────────────────────────────────────────────

SYSTEM_PROMPT_ANSWER_ONLY = (
    "You are a KGQA assistant. "
    "Given reasoning paths from a knowledge graph and a question, "
    "answer the question using entity IDs from the paths.\n"
    "Rules:\n"
    "- Only output entity IDs that appear in the provided paths.\n"
    "- Do not generate or fabricate new entity IDs.\n"
    "Output format:\nAnswer: <entity_id> | <entity_id>"
)

SYSTEM_PROMPT_V2 = (
    "You are a KGQA assistant. "
    "Given reasoning paths from a knowledge graph and a question, "
    "identify which paths support the answer, then extract the answer "
    "from the tail entities of those supporting paths.\n"
    "Rules:\n"
    "- Only output entity IDs that appear in the provided paths.\n"
    "- Do not generate or fabricate new entity IDs.\n"
    "Output format:\n"
    "Supporting Paths: <path numbers>\n"
    "Answer: <entity_id> | <entity_id>"
)

SYSTEM_PROMPT_V3 = (
    "You are a KGQA assistant. "
    "Given reasoning paths from a knowledge graph and a question, "
    "output a JSON object with the supporting path indices and the answer entity IDs.\n"
    "Rules:\n"
    "- Only output entity IDs that appear in the provided paths.\n"
    "- Do not generate or fabricate new entity IDs.\n"
    'Output format: {"reasoning": ["Path 1", "Path 3"], "answer": ["<entity_id>", "<entity_id>"]}'
)

# V4: Compact CoT —— 一句话推理 + citation + 答案
SYSTEM_PROMPT_V4 = (
    "You are a KGQA assistant. "
    "Given reasoning paths from a knowledge graph and a question, "
    "briefly explain which paths answer the question, then output the answer.\n"
    "Rules:\n"
    "- Only output entity IDs that appear in the provided paths.\n"
    "- Do not generate or fabricate new entity IDs.\n"
    "- Keep reasoning to 1-2 sentences.\n"
    "Output format:\n"
    "Reasoning: <brief explanation>\n"
    "Supporting Paths: <path numbers>\n"
    "Answer: <entity_id> | <entity_id>"
)

# V5: Natural Language Path Input —— 路径用自然语言表示（输出格式与 V2 相同）
# System Prompt 与 V2 完全相同，变化在 build_user_content 的路径格式化方式
SYSTEM_PROMPT_V5 = SYSTEM_PROMPT_V2

# V11: Full CoT（备用，暂不纳入主消融）—— 带 [Reasoning]/[Answer] 双段结构
SYSTEM_PROMPT_V11 = (
    "You are a KGQA assistant. "
    "Given reasoning paths from a knowledge graph and a question, "
    "reason step by step about which paths support the answer, then output the answer entity IDs.\n"
    "Rules:\n"
    "- Only output entity IDs that appear in the provided paths.\n"
    "- Do not generate or fabricate new entity IDs.\n"
    "Output format:\n[Reasoning]\nPath 1 → <tail_entity> via [rel1] -> [rel2]\n...\n[Answer]\nSupporting Paths: 1, 3\nAnswer: <entity_id>"
)

# V1 与 V0 零样本共用同一 prompt（仅输出答案）
SYSTEM_PROMPT_V1 = SYSTEM_PROMPT_ANSWER_ONLY

FORMAT_PROMPTS = {
    "v0":  SYSTEM_PROMPT_ANSWER_ONLY,
    "v1":  SYSTEM_PROMPT_ANSWER_ONLY,
    "v2":  SYSTEM_PROMPT_V2,
    "v3":  SYSTEM_PROMPT_V3,
    "v4":  SYSTEM_PROMPT_V4,   # Compact CoT
    "v5":  SYSTEM_PROMPT_V5,   # Natural Language Path（同 V2 prompt，输入格式不同）
    "v11": SYSTEM_PROMPT_V11,  # Full CoT（备用）
}


# ─── 路径格式化 ───────────────────────────────────────────────────────────────

def format_path_str(path_edges: list, log_score: float, idx: int,
                    show_score: bool = True) -> str:
    """将路径序列化为符号表示字符串（默认格式）。

    show_score=True（默认）:  'Path N [score=S]: (e0) -[r]-> (e1) ...'
    show_score=False:          'Path N: (e0) -[r]-> (e1) ...'
    """
    chain = " ".join(f"({e[0]}) -[{e[1]}]-> ({e[2]})" for e in path_edges)
    if show_score:
        return f"Path {idx} [score={log_score:.4f}]: {chain}"
    return f"Path {idx}: {chain}"


def _rel_to_text(rel: str) -> str:
    """将关系名轻量转换为可读文本（替换 . 和 _ 为空格）。"""
    return rel.replace(".", " ").replace("_", " ")


def format_path_str_nl(path_edges: list, log_score: float, idx: int,
                       show_score: bool = True) -> str:
    """将路径序列化为自然语言句子（供 V5 使用）。

    单跳: 'Path N [score=S]: subject rel object'
    多跳: 'Path N [score=S]: s1 r1 o1; s2 r2 o2; ...'
    """
    parts = [f"{e[0]} {_rel_to_text(e[1])} {e[2]}" for e in path_edges]
    chain = "; ".join(parts)
    if show_score:
        return f"Path {idx} [score={log_score:.4f}]: {chain}"
    return f"Path {idx}: {chain}"


def build_user_content(paths_with_meta: list, question: str,
                       show_score: bool = True,
                       path_format: str = "arrow") -> str:
    """构建 User 消息：问题前置，路径列表随后。

    paths_with_meta: [(path_edges, log_score, display_idx), ...]
    path_format: 'arrow'（默认符号格式）或 'nl'（自然语言格式，供 V5 使用）
    """
    fmt_fn = format_path_str_nl if path_format == "nl" else format_path_str
    lines = [f"Question: {question}", "", "Reasoning Paths:"]
    for path_edges, log_score, display_idx in paths_with_meta:
        lines.append(fmt_fn(path_edges, log_score, display_idx, show_score))
    return "\n".join(lines)
