"""PfitDatasetSpec:pfit 的数据集差异钩子注册表。

格式化 / 增强 / 训练 / 评测代码全部数据集无关,差异只进这里:
实体表示、问题清洗、拒答支持、hop 分层。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from kgqa.pfit import formats

_PROJECT_DIR = Path(__file__).resolve().parents[2]

# WebQSP 问题清洗:BERT 特殊 token 与 wordpiece 标记(与 llm_infer.kg_format 行为一致)
_QUESTION_BOUNDARY_TOKEN_RE = re.compile(r"\s*(?:\[CLS\]|\[SEP\])\s*")
_WORDPIECE_MARKER_RE = re.compile(r"\s*##\s*")
_ES_PLACEHOLDER_RE = re.compile(r"\bE_S\b", re.IGNORECASE)


def _clean_question_webqsp(question: str, topics: list[str]) -> str:
    question = _QUESTION_BOUNDARY_TOKEN_RE.sub(" ", question or "")
    question = _WORDPIECE_MARKER_RE.sub("", question)
    return " ".join(question.split()).strip()


def _clean_question_metaqa(question: str, topics: list[str]) -> str:
    """MetaQA 问题以 E_S 占位 topic 实体,回填为实体名。

    原始 qa 文件为大写 E_S;经 vocab 解码(score 缓存→retrieve)后为小写 e_s,
    两种形态都要回填。
    """
    if topics:
        return _ES_PLACEHOLDER_RE.sub(topics[0], question or "")
    return question


@dataclass(frozen=True)
class PfitDatasetSpec:
    name: str
    entity_reprs: tuple[str, ...]                     # 支持的实体表示("mid"/"name")
    default_entity_repr: str
    clean_question: Callable[[str, list[str]], str]   # (question, topics) -> str
    supports_rejection: bool                          # 拒答样本构造是否可用
    group_by_hop: bool                                # 分层采样与 by_hop 评测
    hops: tuple[int, ...]
    entity_map_path: Optional[str] = None             # MID→Name 映射文件(仅需映射的数据集)

    def load_entity_map(self) -> dict:
        """entity_repr="name" 时的实体映射;天然 name 的数据集返回空表(恒等映射)。"""
        if self.entity_map_path is None:
            return {}
        return formats.load_entity_map(self.entity_map_path)


_SPECS: dict[str, PfitDatasetSpec] = {
    "webqsp": PfitDatasetSpec(
        name="webqsp",
        entity_reprs=("mid", "name"),
        default_entity_repr="name",
        clean_question=_clean_question_webqsp,
        supports_rejection=True,
        group_by_hop=False,
        hops=(1, 2),
        entity_map_path=str(_PROJECT_DIR / "data/resources/WebQSP/fbwq_full/mapped_entities.txt"),
    ),
    "metaqa": PfitDatasetSpec(
        name="metaqa",
        entity_reprs=("name",),
        default_entity_repr="name",
        clean_question=_clean_question_metaqa,
        supports_rejection=False,
        group_by_hop=True,
        hops=(1, 2, 3),
    ),
}


def get_pfit_spec(dataset: str) -> PfitDatasetSpec:
    if dataset not in _SPECS:
        raise KeyError(f"pfit 未注册数据集 {dataset!r},可用:{sorted(_SPECS)}")
    return _SPECS[dataset]
