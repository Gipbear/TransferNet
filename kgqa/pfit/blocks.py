"""分块构样：把同一题的长候选池按排名切块，每块单独成一个训练样本。

P2 分段拒答实验专用（beam40 池拆成 1–20 / 21–40 两块）：块内有 gold 尾匹配的块
给「引用 + 答案」，整块无匹配的块给「Supporting Paths: None / Answer: None」。
同题各块共享 group 键（`sample_index`），train/val 按 group 切分，同题两块不得跨集。

每条样本的随机性由 `(seed, group, block)` 派生，与遍历顺序无关——这样
「只用第一块」的基线（A 组）必定是「全部块」（B 组）的逐条子集，两组只差多出来的块。

三组对照（见 experiments/configs/ch4 的 beam40 配置）：
  A 自然拒答：`build_natural`，第一块 + 天然无命中拒答；
  B 排名分块：`build_blocks`，A + 第二块样本（块内无命中即为拒答监督）；
  C 合成拒答：`build_synthetic_group`，A + 与 B 等量同比例的额外样本，
    但额外样本从第一块构造（有正例的块删掉正例当拒答、答案从前块重采样），
    用来区分「拒答监督本身有效」与「多出来的排名后段路径有效」。
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from kgqa.pfit.build import augment, label_paths, output_v2, output_v2_reject
from kgqa.pfit.formats import build_user_content, map_answers, select_format_prompt

DEFAULT_SEED = 17


@dataclass(frozen=True)
class BlockSpec:
    """分块规格：每块路径数与最多块数。"""

    block_size: int = 20
    max_blocks: int = 2


BLOCK_GROUPS = ("natural", "blocks", "synthetic")


def build_group_samples(builder: BlockBuilder, records: list[dict], group: str) -> list[dict]:
    """按组别建集；synthetic 的额外样本从第一块构造，数量与回答/拒答占比对齐 blocks 组。"""
    if group == "natural":
        return builder.build_natural(records)
    if group == "blocks":
        return builder.build_blocks(records)
    if group == "synthetic":
        extras = [s for s in builder.build_blocks(records) if s["_meta"]["block"] > 1]
        return builder.build_synthetic_group(records, extras)
    raise ValueError(f"未知分块组别 {group!r},可用:{BLOCK_GROUPS}")


def block_stats(samples: list[dict]) -> dict:
    """分块建集的样本构成（进 manifest，供事后核对三组是否等量同比例）。"""
    rejections = sum(1 for s in samples if s["_meta"]["is_rejection"])
    return {
        "n_answer": len(samples) - rejections,
        "n_rejection": rejections,
        "by_block": dict(sorted(Counter(s["_meta"]["block"] for s in samples).items())),
        "by_origin": dict(sorted(Counter(s["_meta"]["origin"] for s in samples).items())),
    }


def grouped_split_indices(groups: list, val_ratio: float,
                          seed: int = DEFAULT_SEED) -> tuple[list[int], list[int]]:
    """按 group 切 train/val：整组进同一侧，返回 (train_idx, val_idx)。

    验证集大小尽量贴近 `val_ratio`，但只以整组为单位逼近；组数太少时验证集可能为空。
    """
    if not groups:
        return [], []
    if val_ratio <= 0:
        return list(range(len(groups))), []

    unique = list(dict.fromkeys(groups))
    if len(unique) < 2:
        return list(range(len(groups))), []

    order = list(unique)
    random.Random(seed).shuffle(order)
    target = round(len(groups) * val_ratio)
    by_group: dict = {}
    for idx, group in enumerate(groups):
        by_group.setdefault(group, []).append(idx)

    val_groups: list = []
    n_val = 0
    for group in order:
        if n_val >= target:
            break
        val_groups.append(group)
        n_val += len(by_group[group])

    val_set = set(val_groups)
    val_idx = [i for i, g in enumerate(groups) if g in val_set]
    train_idx = [i for i, g in enumerate(groups) if g not in val_set]
    return train_idx, val_idx


class BlockBuilder:
    """按排名分块构样，产物格式与 kgqa.pfit.build 的 messages JSONL 一致。"""

    def __init__(self, spec: BlockSpec, *, fmt: str = "v2", shuffle: bool = True,
                 distractor_ratio: float | None = None, show_score: bool = False,
                 path_format: str = "chain", use_entity_names: bool = True,
                 entity_map: dict | None = None, clean_question: Callable | None = None,
                 system_prompt: str | None = None, seed: int = DEFAULT_SEED) -> None:
        self.spec = spec
        self.fmt = fmt
        self.shuffle = shuffle
        self.distractor_ratio = distractor_ratio
        self.show_score = show_score
        self.path_format = path_format
        self.entity_map = entity_map
        self.clean_question = clean_question or (lambda question, topics: question)
        # 三组都含拒答样本，统一用拒答版 system prompt，保证组间唯一的变量是建集方式
        self.system_prompt = system_prompt or select_format_prompt(
            "v2", use_entity_names, reject_prompt=True)
        self.seed = seed

    # ── 切块 ────────────────────────────────────────────────────────────────

    def split(self, record: dict) -> list[tuple[int, list[dict]]]:
        """按排名把候选池切成块，返回 [(1-based 块号, 该块路径), ...]（空块不返回）。"""
        paths = record.get("mmr_reason_paths") or []
        blocks = []
        for index in range(self.spec.max_blocks):
            start = index * self.spec.block_size
            chunk = paths[start:start + self.spec.block_size]
            if chunk:
                blocks.append((index + 1, chunk))
        return blocks

    # ── 单块构样 ────────────────────────────────────────────────────────────

    def build_block(self, record: dict, block_index: int, paths: list[dict],
                    origin: str = "block") -> dict | None:
        """把一块路径渲染成一条样本；无问题、无 golden 或块为空时返回 None。"""
        question = self.clean_question(record.get("question", ""), record.get("topics", []))
        golden = record.get("golden") or []
        if not question or not golden or not paths:
            return None

        labeled = label_paths(paths, golden)
        has_golden = any(is_g for _, _, is_g in labeled)
        rng = random.Random(f"{self.seed}:{self.group_of(record)}:{block_index}:{origin}")
        if has_golden:
            return self._answer_sample(record, block_index, labeled, question, origin, rng)
        return self._rejection_sample(record, block_index, labeled, question, origin, rng)

    def group_of(self, record: dict) -> object:
        """样本的 group 键：同题各块必须共享它。"""
        group = record.get("sample_index")
        return group if group is not None else record.get("question", "")

    def _lexical_select(self, labeled: list, rng: random.Random) -> list:
        """按本块内的策略选路径并重新编号（1-based）。"""
        return augment(labeled, self.shuffle, self.distractor_ratio, rng)

    def _render(self, record: dict, block_index: int, question: str, labeled: list,
                assistant: str, origin: str, *, is_rejection: bool,
                answers: list, golden_indices: list) -> dict:
        paths_with_meta = [(edges, score, i + 1) for i, (edges, score, _) in enumerate(labeled)]
        user_content = build_user_content(paths_with_meta, question,
                                          show_score=self.show_score,
                                          path_format=self.path_format,
                                          entity_map=self.entity_map)
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant},
            ],
            "_meta": {
                "question": question,
                "golden": record.get("golden") or [],
                "path_answers": answers,
                "golden_path_indices": golden_indices,
                "n_golden": len(golden_indices),
                "n_distractor": len(labeled) - len(golden_indices),
                "format": self.fmt,
                "show_score": self.show_score,
                "path_format": self.path_format,
                "entity_map_used": bool(self.entity_map),
                "hop": record.get("hop"),
                "group": self.group_of(record),
                "block": block_index,
                "origin": origin,
                "is_rejection": is_rejection,
            },
        }

    def _answer_sample(self, record: dict, block_index: int, labeled: list, question: str,
                       origin: str, rng: random.Random) -> dict:
        labeled = self._lexical_select(labeled, rng)
        golden_indices = [i + 1 for i, (_, _, is_g) in enumerate(labeled) if is_g]
        answers = list(dict.fromkeys(edges[-1][2] for edges, _, is_g in labeled if is_g and edges))
        if self.entity_map:
            answers = map_answers(answers, self.entity_map)
        return self._render(record, block_index, question, labeled,
                            output_v2(golden_indices, answers), origin,
                            is_rejection=False, answers=answers, golden_indices=golden_indices)

    def _rejection_sample(self, record: dict, block_index: int, labeled: list, question: str,
                          origin: str, rng: random.Random) -> dict:
        distractors = self._lexical_select([p for p in labeled if not p[2]], rng)
        return self._render(record, block_index, question, distractors,
                            output_v2_reject(), origin,
                            is_rejection=True, answers=[], golden_indices=[])

    # ── 三组建集 ────────────────────────────────────────────────────────────

    def build_natural(self, records: list[dict]) -> list[dict]:
        """A 组：只用第一块，天然无命中即拒答。"""
        return self._collect(records, max_blocks=1)

    def build_blocks(self, records: list[dict]) -> list[dict]:
        """B 组：全部块，每块一条样本。"""
        return self._collect(records, max_blocks=self.spec.max_blocks)

    def _collect(self, records: list[dict], max_blocks: int) -> list[dict]:
        samples = []
        for record in records:
            for block_index, paths in self.split(record)[:max_blocks]:
                sample = self.build_block(record, block_index, paths)
                if sample is not None:
                    samples.append(sample)
        return samples

    def build_synthetic_extras(self, records: list[dict], extras: list[dict]) -> list[dict]:
        """C 组的额外样本：数量与回答/拒答占比逐项对齐 B 组的 `extras`。"""
        n_reject = sum(1 for sample in extras if sample["_meta"]["is_rejection"])
        n_answer = len(extras) - n_reject
        resampled = self._rounds(records, n_answer, self._resample_answer)
        synthetic = self._rounds(records, n_reject, self._synthetic_rejection)
        return resampled + synthetic

    def build_synthetic_group(self, records: list[dict], extras: list[dict]) -> list[dict]:
        """C 组：A 组 + 合成额外样本。"""
        return self.build_natural(records) + self.build_synthetic_extras(records, extras)

    def _rounds(self, records: list[dict], need: int,
                make_one: Callable[[dict], dict | None]) -> list[dict]:
        """按轮次遍历（每轮重新打乱）重复取样，直到凑够 need 条。"""
        if need <= 0 or not records:
            return []
        rng = random.Random(f"{self.seed}:{make_one.__name__}")
        out: list[dict] = []
        while len(out) < need:
            batch = list(records)
            rng.shuffle(batch)
            produced = 0
            for record in batch:
                if len(out) >= need:
                    break
                sample = make_one(record)
                if sample is None:
                    continue
                out.append(sample)
                produced += 1
            if produced == 0:
                break
        return out

    def _resample_answer(self, record: dict) -> dict | None:
        """从前块重采样一条答题样本（允许与 A 组重复，用来配平答案占比）。"""
        blocks = self.split(record)
        if not blocks:
            return None
        block_index, paths = blocks[0]
        sample = self.build_block(record, block_index, paths, origin="resample")
        if sample is None or sample["_meta"]["is_rejection"]:
            return None
        return sample

    def _synthetic_rejection(self, record: dict) -> dict | None:
        """删掉前块的正例路径，把同一个块当成拒答样本。"""
        blocks = self.split(record)
        if not blocks:
            return None
        golden = {g.lower().strip() for g in record.get("golden") or []}
        block_index, paths = blocks[0]
        kept = []
        for path in paths:
            edges = path.get("path") or []
            tail = edges[-1][2].lower().strip() if edges else None
            if tail is None or tail not in golden:
                kept.append(path)
        if not kept:
            return None
        sample = self.build_block(record, block_index, kept, origin="synthetic_rejection")
        if sample is None or not sample["_meta"]["is_rejection"]:
            return None
        return sample
