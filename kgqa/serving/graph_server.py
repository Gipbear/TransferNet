"""提供常驻 TransferNet 外部子图评分 HTTP 服务。"""

import argparse

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.graph_score import TransferNetGraphScorer
from kgqa.agent.workflow.interfaces import GraphScorer


class GraphRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    topics: list[str]
    edges: list[list[str]]
    workflow: dict


class GraphScoringClient(GraphScorer):
    def __init__(self, url: str = "http://127.0.0.1:8791") -> None:
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False

    def score(self, question: str, topics: tuple, edges: list, config: WorkflowConfig) -> dict:
        scoring_fields = ("max_paths", "triple_budget", "lambda_val", "eta", "penalty_mode")
        workflow = {name: getattr(config, name) for name in scoring_fields}
        response = self.session.post(self.url + "/score", timeout=120, json={
            "question": question, "topics": list(topics), "edges": edges, "workflow": workflow,
        })
        response.raise_for_status()
        return response.json()


def create_app(scorer: TransferNetGraphScorer) -> FastAPI:
    app = FastAPI(title="CWQ TransferNet Graph Scoring")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "num_steps": scorer.num_steps}

    @app.post("/score")
    def score(request: GraphRequest) -> dict:
        config = WorkflowConfig.from_dict(request.workflow)
        if len(request.edges) > config.triple_budget:
            return {"paths": [], "reason": "triple_budget_exceeded"}
        return scorer.score(request.question, tuple(request.topics), request.edges, config)

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TransferNet 外部子图常驻评分")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--relations", required=True)
    parser.add_argument("--bert_name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args(argv)
    scorer = TransferNetGraphScorer(args.ckpt, args.relations, args.bert_name, args.device)
    uvicorn.run(create_app(scorer), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
