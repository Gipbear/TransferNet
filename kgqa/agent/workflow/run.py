"""运行单模型工作流，持久化无标签预测并验证断点输入一致。"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import local
from time import perf_counter, sleep

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.models import RuntimeQuestion
from kgqa.agent.workflow.orchestrator import MasterQAWorkflow
from kgqa.agent.workflow.persistence import ArtifactStore
from kgqa.agent.workflow.pfit import PfitAnswerer
from kgqa.agent.workflow.qlever import QLeverClient
from kgqa.agent.workflow.router import QuestionRouter
from kgqa.agent.workflow.templates import TemplateLibrary
from kgqa.agent.workflow.tools.comparison import NumericReranker
from kgqa.agent.workflow.tools.conjunction import AttributeTagger
from kgqa.agent.workflow.tools.rescue import GraphRescuer
from kgqa.runtime import update_progress
from kgqa.serving.graph_server import GraphScoringClient


class WorkflowRunner:
    def __init__(self, config: WorkflowConfig, library: TemplateLibrary, answerer: PfitAnswerer,
                 output: Path, cache_dir: Path, graph_url: str,
                 wait_initial: bool = False, names: dict | None = None) -> None:
        self.config, self.library, self.answerer = config, library, answerer
        self.output, self.cache_dir, self.graph_url = Path(output), Path(cache_dir), graph_url
        self.output.mkdir(parents=True, exist_ok=True)
        self.thread_state = local()
        self.wait_initial = wait_initial
        self.names = dict(names or {})

    def _workflow(self) -> tuple[MasterQAWorkflow, QLeverClient]:
        if hasattr(self.thread_state, "workflow"):
            return self.thread_state.workflow, self.thread_state.kg
        kg = QLeverClient(self.config, self.cache_dir / "qlever")
        comparison = NumericReranker(self.config, self.library, kg)
        conjunction = AttributeTagger(self.config, self.library, kg)
        scorer = GraphScoringClient(self.graph_url) if self.config.rescue_scorer == "transfernet" else None
        rescue = GraphRescuer(self.config, self.library, kg, scorer, names=dict(self.names))
        workflow = MasterQAWorkflow(self.config, self.answerer, QuestionRouter(self.library),
                                    comparison, conjunction, rescue)
        self.thread_state.workflow, self.thread_state.kg = workflow, kg
        return workflow, kg

    def _one(self, question: RuntimeQuestion) -> dict:
        if self.wait_initial:
            prompt, _, _ = self.answerer.render(question.question, question.paths)
            target = self.answerer.cache_path(prompt)
            started = perf_counter()
            while not target.exists():
                if perf_counter() - started > 1200:
                    raise TimeoutError("等待本次实际基座生成缓存超过 1200 秒")
                sleep(0.25)
        workflow, kg = self._workflow()
        start = len(kg.events)
        record = workflow.answer(question)
        events = kg.events[start:]
        stats = {"query_n": len(events), "network_query_n": sum(not e.cache_hit for e in events),
                 "query_ms": sum(e.elapsed_ms for e in events),
                 "queries": [{"hash": e.query_hash, "status": e.status, "cache_hit": e.cache_hit,
                              "elapsed_ms": e.elapsed_ms, "error": e.error} for e in events]}
        if record.tool:
            record = replace(record, tool=replace(record.tool, stats={**record.tool.stats, **stats}))
        row = record.to_dict()
        row["runtime_sha256"] = self._question_hash(question)
        return row

    def run(self, questions: list[RuntimeQuestion], workers: int = 4) -> dict:
        if workers < 1:
            raise ValueError("workers 必须为正数")
        with ArtifactStore.exclusive(self.output / ".predictions.lock"):
            try:
                return self._run_locked(questions, workers)
            except Exception:
                update_progress(self.output, status="failed", phase="CWQ工作流")
                raise

    def _run_locked(self, questions: list[RuntimeQuestion], workers: int) -> dict:
        target = self.output / "predictions_no_gold.jsonl"
        done = self._completed(target, questions)
        pending = [q for q in questions if q.sample_index not in done]
        started = perf_counter()
        count = len(done)
        update_progress(self.output, completed=count, total=len(questions), status="running", phase="CWQ工作流")
        with ThreadPoolExecutor(max_workers=workers) as pool, target.open("a") as handle:
            for row in pool.map(self._one, pending):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                count += 1
                if count % 20 == 0 or count == len(questions):
                    update_progress(self.output, completed=count, total=len(questions),
                                    status="running", phase="CWQ工作流")
                    print(json.dumps({"completed": count, "total": len(questions),
                                      "wall_seconds": round(perf_counter() - started, 1)}), flush=True)
        update_progress(self.output, completed=count, total=len(questions), status="completed", phase="CWQ工作流")
        return {"n": count, "session_wall_seconds": perf_counter() - started,
                "actual_pfit_generations": self.answerer.actual_calls}

    def _completed(self, target: Path, questions: list[RuntimeQuestion]) -> dict:
        expected = {q.sample_index: self._question_hash(q) for q in questions}
        completed = {}
        if not target.exists():
            return completed
        with target.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    if not line.endswith(b"\n") and not handle.read():
                        ArtifactStore.recover_tail(target, offset, line)
                        break
                    raise ValueError("输出中间记录损坏，禁止跳过") from exc
                index = row["question"]["sample_index"]
                if index in completed or row["runtime_sha256"] != expected.get(index):
                    raise ValueError("断点样本重复或输入变化，禁止混用旧结果")
                completed[index] = row
                if not line.endswith(b"\n"):
                    with target.open("ab") as writer:
                        writer.write(b"\n")
        return completed

    @staticmethod
    def _question_hash(question: RuntimeQuestion) -> str:
        serialized = json.dumps(question.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
