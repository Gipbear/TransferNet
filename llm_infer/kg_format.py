"""
KG 路径格式化工具函数（共享模块）

供 llm_infer/build_kgcot_dataset.py 和 llm_infer/eval_faithfulness.py 共同使用，
确保训练与评估时路径字符串格式、System Prompt 完全一致。
"""

import re


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
    'Output format: {"reasoning": ["1", "3"], "answer": ["<entity_id>", "<entity_id>"]}'
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
    "Output format:\n[Reasoning]\n1 → <tail_entity> via [rel1] -> [rel2]\n...\n[Answer]\nSupporting Paths: 1, 3\nAnswer: <entity_id>"
)

# V1 与 V0 零样本共用同一 prompt（仅输出答案）
SYSTEM_PROMPT_V1 = SYSTEM_PROMPT_ANSWER_ONLY

# NO_PATHS: 无检索路径输入，直接基于参数化知识回答（Group H）
SYSTEM_PROMPT_NO_PATHS = (
    "You are a KGQA assistant. "
    "Answer the following question based on your knowledge.\n"
    "Output format:\nAnswer: <entity_id> | <entity_id>"
)

# V2_NAME: 与 V2 相同，但用于实体名称（-name 变体），区别在于措辞
SYSTEM_PROMPT_V2_NAME = (
    "You are a KGQA assistant. "
    "Given reasoning paths from a knowledge graph and a question, "
    "identify which paths support the answer, then extract the answer "
    "from the tail entities of those supporting paths.\n"
    "Rules:\n"
    "- Only output entity names that appear in the provided paths.\n"
    "- Do not generate or fabricate new entity names.\n"
    "Output format:\n"
    "Supporting Paths: <path numbers>\n"
    "Answer: <entity_name> | <entity_name>"
)

# V2_REJECT: 与 V2 相同，但新增拒答规则（Group F）
SYSTEM_PROMPT_V2_REJECT = (
    "You are a KGQA assistant. "
    "Given reasoning paths from a knowledge graph and a question, "
    "identify which paths support the answer, then extract the answer "
    "from the tail entities of those supporting paths.\n"
    "Rules:\n"
    "- Only output entity IDs that appear in the provided paths.\n"
    "- Do not generate or fabricate new entity IDs.\n"
    "- If none of the path tail entities could reasonably answer the question, output:\n"
    "  Supporting Paths: (none)\n"
    "  Answer: (none)\n"
    "Output format:\n"
    "Supporting Paths: <path numbers>\n"
    "Answer: <entity_id> | <entity_id>"
)

# V2_NAME_REJECT: 与 V2_NAME 相同，但新增拒答规则（Group F）
SYSTEM_PROMPT_V2_NAME_REJECT = (
    "You are a KGQA assistant. "
    "Given reasoning paths from a knowledge graph and a question, "
    "identify which paths support the answer, then extract the answer "
    "from the tail entities of those supporting paths.\n"
    "Rules:\n"
    "- Only output entity names that appear in the provided paths.\n"
    "- Do not generate or fabricate new entity names.\n"
    "- If none of the path tail entities could reasonably answer the question, output:\n"
    "  Supporting Paths: (none)\n"
    "  Answer: (none)\n"
    "Output format:\n"
    "Supporting Paths: <path numbers>\n"
    "Answer: <entity_name> | <entity_name>"
)

FORMAT_PROMPTS = {
    "v0":           SYSTEM_PROMPT_ANSWER_ONLY,
    "v1":           SYSTEM_PROMPT_ANSWER_ONLY,
    "v2":           SYSTEM_PROMPT_V2,
    "v3":           SYSTEM_PROMPT_V3,
    "v4":           SYSTEM_PROMPT_V4,          # Compact CoT
    "v5":           SYSTEM_PROMPT_V5,          # Natural Language Path（同 V2 prompt，输入格式不同）
    "v11":          SYSTEM_PROMPT_V11,         # Full CoT（备用）
    "v2_name":      SYSTEM_PROMPT_V2_NAME,     # Entity-name variant
    "v2_reject":      SYSTEM_PROMPT_V2_REJECT,      # Rejection-aware MID variant（Group F）
    "v2_name_reject": SYSTEM_PROMPT_V2_NAME_REJECT, # Rejection-aware name variant（Group F）
    "no_paths":     SYSTEM_PROMPT_NO_PATHS,    # 无路径输入（Group H）
}


# ─── 路径格式化 ───────────────────────────────────────────────────────────────

def format_path_str(path_edges: list, log_score: float, idx: int,
                    show_score: bool = False) -> str:
    """将路径序列化为符号表示字符串（默认格式）。

    show_score=False（默认）: 'N: (e0) -[r]-> (e1) ...'
    show_score=True:           'N [score=S]: (e0) -[r]-> (e1) ...'
    """
    chain = " ".join(f"({e[0]}) -[{e[1]}]-> ({e[2]})" for e in path_edges)
    if show_score:
        return f"{idx} [score={log_score:.4f}]: {chain}"
    return f"{idx}: {chain}"


def _rel_to_text(rel: str) -> str:
    """将关系名轻量转换为可读文本（替换 . 和 _ 为空格）。"""
    return rel.replace(".", " ").replace("_", " ")


_REL_LABEL_OVERRIDES = {
    "containedby": "contained by",
}


def _split_reverse_relation(rel: str) -> tuple[str, bool]:
    """返回去掉 _reverse 的基础关系名，以及该边是否为反向边。"""
    if rel.endswith("_reverse"):
        return rel[: -len("_reverse")], True
    return rel, False


def _rel_to_schema_label(rel: str) -> str:
    """将 Freebase 关系转成 schema 格式使用的短标签。

    关系的语义由 subject -> object 定义，因此 schema 格式只展示属性名
    的最后一段，并由箭头方向表达是否使用了反向边。
    """
    base_rel, _ = _split_reverse_relation(rel)
    leaf = base_rel.rsplit(".", 1)[-1]
    return _REL_LABEL_OVERRIDES.get(leaf, leaf.replace("_", " "))


def format_path_str_nl(path_edges: list, log_score: float, idx: int,
                       show_score: bool = False) -> str:
    """将路径序列化为自然语言句子（供 V5 使用）。

    单跳: 'N: subject rel object'
    多跳: 'N: s1 r1 o1; s2 r2 o2; ...'
    """
    parts = [f"{e[0]} {_rel_to_text(e[1])} {e[2]}" for e in path_edges]
    chain = "; ".join(parts)
    if show_score:
        return f"{idx} [score={log_score:.4f}]: {chain}"
    return f"{idx}: {chain}"


def format_path_str_tuple(path_edges: list, log_score: float, idx: int,
                          show_score: bool = False) -> str:
    """将路径序列化为三元组格式。
    'N: (s, r, o), (s, r, o)'
    """
    if not path_edges:
        return f"{idx}:"
    triples = ", ".join(f"({e[0]}, {e[1]}, {e[2]})" for e in path_edges)
    if show_score:
        return f"{idx} [score={log_score:.4f}]: {triples}"
    return f"{idx}: {triples}"


def format_path_str_chain(path_edges: list, log_score: float, idx: int,
                          show_score: bool = False) -> str:
    """将路径序列化为连续链式格式（去重中间实体）。
    单跳: 'N: e0 -> r -> e1'
    多跳: 'N: e0 -> r1 -> e1 -> r2 -> e2'
    """
    if not path_edges:
        return f"{idx}:"
    parts = [path_edges[0][0]]
    for e in path_edges:
        parts.extend([e[1], e[2]])
    chain = " -> ".join(parts)
    if show_score:
        return f"{idx} [score={log_score:.4f}]: {chain}"
    return f"{idx}: {chain}"


def format_path_str_schema(path_edges: list, log_score: float, idx: int,
                           show_score: bool = False) -> str:
    """将路径序列化为 schema-aware 连续链式格式。

    关系字典中的关系语义按 subject -> object 定义。普通边显示为
    's - [relation] -> o'；反向边 '*_reverse' 保持路径遍历顺序，
    但用 '<- [relation] -' 表明基础关系的语义方向与遍历方向相反。

    单跳: 'N: Person A - [place of birth] -> City'
    多跳: 'N: Person A - [place of birth] -> City <- [contained by] - Country'
    """
    if not path_edges:
        return f"{idx}:"

    parts = [path_edges[0][0]]
    for head, rel, tail in path_edges:
        current_head = head
        if parts[-1] != current_head:
            parts.append(current_head)
        label = _rel_to_schema_label(rel)
        _, is_reverse = _split_reverse_relation(rel)
        arrow = f"<- [{label}] -" if is_reverse else f"- [{label}] ->"
        parts.extend([arrow, tail])

    chain = " ".join(parts)
    if show_score:
        return f"{idx} [score={log_score:.4f}]: {chain}"
    return f"{idx}: {chain}"


# ─── 实体映射工具 ─────────────────────────────────────────────────────────────

def load_entity_map(path: str) -> dict:
    """从 tab-separated 文件加载 MID→Name 映射表。"""
    emap = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" not in line:
                continue
            mid, name = line.split("\t", 1)
            emap[mid] = name
    return emap


def apply_entity_map(path_edges: list, emap: dict) -> list:
    """将路径中实体 MID 替换为名称，未映射的 MID 保持原样。
    返回新列表，不修改原始数据。
    """
    return [
        [emap.get(e[0], e[0]), e[1], emap.get(e[2], e[2])]
        for e in path_edges
    ]


def map_answers(answers: list, emap: dict) -> list:
    """将答案实体 MID 替换为名称，未映射保持原样。"""
    return [emap.get(a, a) for a in answers]


def build_reverse_entity_map(emap: dict) -> dict:
    """构建 name→set(MIDs) 反向映射（供 eval 时 name→MID 匹配使用）。
    一个名称可能对应多个 MID（一对多）。
    """
    rev = {}
    for mid, name in emap.items():
        key = name.lower().strip()
        rev.setdefault(key, set()).add(mid)
    return rev


# ─── User 消息构建 ────────────────────────────────────────────────────────────

_QUESTION_BOUNDARY_TOKEN_RE = re.compile(r"\s*(?:\[CLS\]|\[SEP\])\s*")
_WORDPIECE_MARKER_RE = re.compile(r"\s*##\s*")


def clean_question_text(question: str) -> str:
    """去掉 WebQSP 问题中的 BERT 特殊 token 和 wordpiece 标记。"""
    question = _QUESTION_BOUNDARY_TOKEN_RE.sub(" ", question or "")
    question = _WORDPIECE_MARKER_RE.sub("", question)
    return " ".join(question.split()).strip()


_FORMAT_FN_MAP = {
    "arrow":  format_path_str,
    "nl":     format_path_str_nl,
    "tuple":  format_path_str_tuple,
    "chain":  format_path_str_chain,
    "schema": format_path_str_schema,
}


def build_user_content(paths_with_meta: list, question: str,
                       show_score: bool = False,
                       path_format: str = "arrow",
                       entity_map: dict = None) -> str:
    """构建 User 消息：问题前置，路径列表随后。

    paths_with_meta: [(path_edges, log_score, display_idx), ...]
    path_format: 'arrow'（默认）/ 'nl'（自然语言）/ 'tuple'（三元组）/ 'chain'（连续链式）/ 'schema'（语义方向感知）
    entity_map: MID→Name 映射表（可选），提供时对路径实体做替换
    """
    fmt_fn = _FORMAT_FN_MAP.get(path_format)
    if fmt_fn is None:
        raise ValueError(f"未知 path_format {path_format!r}，有效值：{list(_FORMAT_FN_MAP)}")
    lines = [f"Question: {question}", "", "Reasoning Paths:"]
    for path_edges, log_score, display_idx in paths_with_meta:
        edges = apply_entity_map(path_edges, entity_map) if entity_map else path_edges
        lines.append(fmt_fn(edges, log_score, display_idx, show_score))
    return "\n".join(lines)


def build_user_content_no_paths(question: str) -> str:
    """构建无路径 User 消息（Group H）：仅包含问题本身，不附带检索路径。"""
    return f"Question: {question}"
