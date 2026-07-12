"""AgentDatasetSpec:Ch5 checked-batch agent 的数据集差异钩子注册表。

主逻辑 / 工具 / 指标 / 记录全部数据集无关,差异只进这里:
QA 样本加载、实体映射(MID→name 或恒等)、hop 范围、by_hop 汇总开关。

QA 源形态:
- webqsp:原始 QA 文本(tab 分隔,问句尾 ``[MID]`` 标注 topic),问题经 BERT
  特殊 token 清洗后同时用于检索服务定位(服务端 normalize)与 LLM 提示词。
- metaqa:``kgqa.cli.retrieve`` 输出 JSONL(question 为 vocab 解码的 ``e_s``
  占位形态,topics/golden 天然实体名)。检索定位走 ``sample_index``,
  LLM 提示词用 ``e_s`` 回填 topic 后的展示问题。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from kgqa.core.entity_map import load_entity_map
from kgqa.core.qa_formats import load_webqsp_qa_samples

_PROJECT_DIR = Path(__file__).resolve().parents[2]

_ES_PLACEHOLDER_RE = re.compile(r"\be_s\b", re.IGNORECASE)


@dataclass(frozen=True)
class AgentQASample:
    """数据集无关的 agent 评测样本。

    question 供 agent 全程使用(LLM 提示词;webqsp 同时用于检索服务 question
    定位);sample_index 非 None 时检索服务改按索引定位(metaqa,展示问题与
    缓存问题形态不同)。topic_id/gold_ids 与检索服务实体空间一致
    (webqsp=MID,metaqa=name)。
    """

    question_raw: str
    question: str
    topic_id: str
    gold_ids: list[str]
    sample_index: Optional[int] = None


def _load_qa_webqsp(path: str, limit: int = 0) -> list[AgentQASample]:
    return [
        AgentQASample(
            question_raw=s.question_raw,
            question=s.question,
            topic_id=s.topic_mid,
            gold_ids=list(s.gold_mids),
        )
        for s in load_webqsp_qa_samples(path, limit=limit)
    ]


def _load_qa_metaqa(path: str, limit: int = 0) -> list[AgentQASample]:
    samples: list[AgentQASample] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            question_raw = rec.get("question", "")
            topics = rec.get("topics") or []
            topic = topics[0] if topics else ""
            question = (_ES_PLACEHOLDER_RE.sub(topic, question_raw)
                        if topic else question_raw)
            samples.append(AgentQASample(
                question_raw=question_raw,
                question=question,
                topic_id=topic,
                gold_ids=list(rec.get("golden") or []),
                sample_index=rec.get("sample_index"),
            ))
            if limit > 0 and len(samples) >= limit:
                break
    return samples


@dataclass(frozen=True)
class AgentDatasetSpec:
    name: str
    hops: tuple[int, ...]
    load_qa: Callable[..., list[AgentQASample]]
    group_by_hop: bool                      # 汇总是否按 hop 分组
    entity_map_path: Optional[str] = None   # MID→name 映射文件(仅需映射的数据集)

    def load_entity_map(self) -> dict[str, str]:
        """需要 MID→name 的数据集读映射文件;天然 name 的返回空表(恒等映射)。"""
        if self.entity_map_path is None:
            return {}
        return load_entity_map(self.entity_map_path)


_SPECS: dict[str, AgentDatasetSpec] = {
    "webqsp": AgentDatasetSpec(
        name="webqsp",
        hops=(1, 2),
        load_qa=_load_qa_webqsp,
        group_by_hop=False,
        entity_map_path=str(_PROJECT_DIR / "data/resources/WebQSP/fbwq_full/mapped_entities.txt"),
    ),
    "metaqa": AgentDatasetSpec(
        name="metaqa",
        hops=(1, 2, 3),
        load_qa=_load_qa_metaqa,
        group_by_hop=True,
        entity_map_path=None,
    ),
}


def get_agent_spec(dataset: str) -> AgentDatasetSpec:
    if dataset not in _SPECS:
        raise KeyError(f"agent 未注册数据集 {dataset!r},可用:{sorted(_SPECS)}")
    return _SPECS[dataset]
