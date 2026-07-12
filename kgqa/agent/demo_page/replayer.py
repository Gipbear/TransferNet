"""离线重放单例：一次加载 entity_map/轨迹，多次复用。"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from kgqa.agent.replay import _ReplaySession
from kgqa.agent.common import load_entity_map
from .data import load_trace_index
from .kg_paths import KGPathResolver
from .service import FINAL_CHECK_FLAGS, paths_to_graph, shape_replay_result


class DemoReplayer:
    def __init__(self, *, entity_map_path: str, trace_path: str,
                 kg_dir: str | None = None) -> None:
        self._entity_map_path = entity_map_path
        self._trace_path = trace_path
        self._kg_dir = kg_dir
        self._session: Optional[_ReplaySession] = None
        self._trace: Optional[dict[int, dict[str, Any]]] = None
        self._kg_resolver: Optional[KGPathResolver] = None
        self.entity_map: dict[str, str] = {}
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            entity_map = load_entity_map(self._entity_map_path)
            trace = load_trace_index(self._trace_path)
            session = _ReplaySession(entity_map, hybrid_check=True)
            kg_dir = Path(self._kg_dir) if self._kg_dir else Path(self._entity_map_path).parent
            kg_resolver = KGPathResolver(kg_dir)
            # 全部就绪后再发布，避免其他线程读到半初始化状态
            self.entity_map = entity_map
            self._trace = trace
            self._session = session
            self._kg_resolver = kg_resolver if kg_resolver.available() else None

    def replay(self, sample_index: int, *, score_margin: float = 2.0,
               enable_relation_expansion: bool = True,
               expansion_min_answers: int = 8, expansion_top_groups: int = 3,
               eval_view: bool = False) -> dict[str, Any]:
        self._ensure()
        record = self._trace.get(sample_index)
        if record is None:
            raise KeyError(f"sample_index {sample_index} 不在回放轨迹中")
        flags = dict(FINAL_CHECK_FLAGS)
        flags.update(score_margin=score_margin,
                     enable_relation_expansion=enable_relation_expansion,
                     expansion_min_answers=expansion_min_answers,
                     expansion_top_groups=expansion_top_groups)
        batch_size = flags.pop("batch_size")
        ema = flags.pop("expansion_min_answers")
        etg = flags.pop("expansion_top_groups")
        result = self._session.replay(
            record, allow_prefix=True, batch_size=batch_size,
            expansion_min_answers=ema, expansion_top_groups=etg, **flags)
        shaped = shape_replay_result(
            result.to_dict(), self.entity_map, kg_path_resolver=self._kg_resolver)
        shaped["graph"] = paths_to_graph(
            record.get("named_mmr_reason_paths", []),
            record.get("named_topics", []),
            shaped.get("kg_completion_paths", []))
        if eval_view:
            from replay_ch5_ablation import _record_for_result, _sample_from_record
            rec = _record_for_result(sample_index, _sample_from_record(record), result)
            shaped["eval"] = {
                # 展示层按名字去重（不同 MID 可能同名），指标仍按 MID 计算
                "gold_names": list(dict.fromkeys(
                    self.entity_map.get(m, m)
                    for m in record.get("gold_mids", []))),
                "hit1": rec["hit1"], "f1": rec["f1"],
                "exact_match": rec["exact_match"],
                "citation_accuracy": rec["citation_accuracy"],
            }
        return shaped
