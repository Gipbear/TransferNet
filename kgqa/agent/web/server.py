"""展示页 HTTP 服务：题库搜索 / 检索代理 / 离线重放 + 静态前端。"""
from __future__ import annotations

import argparse
import os
import types
from pathlib import Path
from typing import Any, Callable

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from kgqa.agent.common import apply_entity_map, map_entities
from .data import QuestionIndex
from .schema import ReplayIn, RetrieveIn
from .service import FINAL_RETRIEVAL, paths_to_graph

STATIC_DIR = Path(__file__).parent / "static"

DEFAULT_QA = "data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt"
DEFAULT_TRACE = ("data/output/WebQSP/checked_batch_agent/"
                 "ch5_full_rerun_20260627_2306/full_trace/checked_batch_eval.jsonl")
DEFAULT_ENTITY_MAP = "data/resources/WebQSP/fbwq_full/mapped_entities.txt"
DEFAULT_KG_DIR = "data/resources/WebQSP/fbwq_full"


def _namedify(raw_response: Any, entity_map: dict[str, str]) -> Any:
    """把 PathRetrieveClient 返回的 raw MID 版 RetrieveResponse 名字化。

    复用 kgqa/agent/tools/path_retrieve.py PathRetrieveTool 同款逻辑：
    - apply_entity_map: 路径 head/tail MID → name（关系保留原值）
    - map_entities: topics MID 列表 → name 列表
    - entity_map.get(mid, mid): prediction key 名字化
    返回带三个命名属性 + elapsed_ms 的 SimpleNamespace，保持与 stub 同样接口。
    """
    named_paths = [
        {
            "path": apply_entity_map(p.get("path", []), entity_map),
            "log_score": p.get("log_score", 0.0),
        }
        for p in raw_response.mmr_reason_paths
    ]
    named_topics = map_entities(raw_response.topics, entity_map)
    named_prediction = {
        entity_map.get(mid, mid): score
        for mid, score in raw_response.prediction.items()
    }
    return types.SimpleNamespace(
        named_mmr_reason_paths=named_paths,
        named_topics=named_topics,
        named_prediction=named_prediction,
        elapsed_ms=raw_response.elapsed_ms,
    )


def create_app(*, questions: QuestionIndex,
               retrieve_fn: Callable[..., Any], replayer: Any) -> FastAPI:
    app = FastAPI(title="KG-LLM 多跳问答演示页", version="1.0")

    @app.get("/api/questions")
    def api_questions(q: str = "", limit: int = 20):
        return [{"sample_index": e.sample_index, "question": e.question}
                for e in questions.search(q, limit=limit)]

    @app.post("/api/retrieve")
    def api_retrieve(req: RetrieveIn):
        try:
            questions.get(req.sample_index)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            resp = retrieve_fn(sample_index=req.sample_index,
                               beam_size=req.beam_size, lambda_val=req.lambda_val,
                               eta=req.eta)
        except requests.exceptions.RequestException as exc:
            # 注意：requests 的连接异常不是内建 ConnectionError 子类，必须按 requests 体系捕获
            raise HTTPException(status_code=502, detail=(
                "path_retrieve_server 未启动"
                "（./scripts/path_retrieve_server.sh start）")) from exc
        return {
            "graph": paths_to_graph(resp.named_mmr_reason_paths, resp.named_topics),
            "prediction": resp.named_prediction,
            "is_final_config": (
                req.beam_size == FINAL_RETRIEVAL["beam_size"]
                and req.lambda_val == FINAL_RETRIEVAL["lambda_val"]
                and req.eta == FINAL_RETRIEVAL["eta"]),
            "elapsed_ms": resp.elapsed_ms,
        }

    @app.post("/api/replay")
    def api_replay(req: ReplayIn):
        try:
            return replayer.replay(
                req.sample_index, score_margin=req.score_margin,
                enable_relation_expansion=req.enable_relation_expansion,
                expansion_min_answers=req.expansion_min_answers,
                expansion_top_groups=req.expansion_top_groups,
                eval_view=req.eval_view)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        def index():
            return FileResponse(STATIC_DIR / "index.html")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("DEMO_PAGE_PORT", "8790")))
    parser.add_argument("--qa_file",
                        default=os.environ.get("DEMO_QA_FILE", DEFAULT_QA))
    parser.add_argument("--trace_file",
                        default=os.environ.get("DEMO_TRACE_FILE", DEFAULT_TRACE))
    parser.add_argument("--entity_map",
                        default=os.environ.get("DEMO_ENTITY_MAP", DEFAULT_ENTITY_MAP))
    parser.add_argument("--kg_dir",
                        default=os.environ.get("DEMO_KG_DIR", DEFAULT_KG_DIR))
    parser.add_argument("--path_retrieve_url",
                        default=os.environ.get("PATH_RETRIEVE_URL",
                                               "http://localhost:8789"))
    args = parser.parse_args()

    from kgqa.retrieve.api.client import PathRetrieveClient
    from .replayer import DemoReplayer

    replayer = DemoReplayer(entity_map_path=args.entity_map,
                            trace_path=args.trace_file,
                            kg_dir=args.kg_dir)
    client = PathRetrieveClient(args.path_retrieve_url)

    def retrieve_named(**kw):
        raw = client.retrieve(**kw)
        # RetrieveResponse 是 raw MID 版；借 replayer 的 entity_map 名字化。
        replayer._ensure()  # noqa: SLF001
        return _namedify(raw, replayer.entity_map)

    app = create_app(questions=QuestionIndex.from_file(args.qa_file),
                     retrieve_fn=retrieve_named, replayer=replayer)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
