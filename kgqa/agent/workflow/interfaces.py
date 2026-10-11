"""声明可替换的模型调用、路由、知识源与图评分接口。"""

from abc import ABC, abstractmethod

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.models import AnswerResult, QueryResult, RouteDecision, RuntimeQuestion, ToolOutcome


class Answerer(ABC):
    @abstractmethod
    def answer(self, question: str, paths: tuple[dict, ...]) -> AnswerResult:
        """返回固定提示与权重下的模型输出。"""


class Router(ABC):
    @abstractmethod
    def classify(self, question: str) -> RouteDecision:
        """仅按问题文本预测路由。"""


class WorkflowTool(ABC):
    @abstractmethod
    def run(self, question: RuntimeQuestion, initial: AnswerResult) -> ToolOutcome:
        """根据可观察输入生成工具决策。"""


class KnowledgeSource(ABC):
    @abstractmethod
    def attributes(self, mids: list[str], chain: tuple[str, ...] | list[str]) -> QueryResult:
        """查询指定实体的属性事实。"""

    @abstractmethod
    def intersect_topic_attribute(self, topics: list[str] | tuple[str, ...],
                                  chain: tuple[str, ...] | list[str],
                                  target_value: str) -> QueryResult:
        """根据主题词与属性数值约束逆向检索相交实体。"""

    @abstractmethod
    def adjacency(self, frontier: list[str], step: dict) -> QueryResult:
        """查询指定关系与方向的邻接。"""

    @abstractmethod
    def labels(self, mids: list[str]) -> QueryResult:
        """查询实体英文展示名称。"""


class GraphScorer(ABC):
    @abstractmethod
    def score(self, question: str, topics: tuple[str, ...], edges: list[list[str]], config: WorkflowConfig) -> dict:
        """对外部子图评分并返回带来源的路径。"""
