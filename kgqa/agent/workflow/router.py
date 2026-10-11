"""封装冻结的训练集文本分类器，保持运行时接口单一。"""

from kgqa.agent.workflow.interfaces import Router
from kgqa.agent.workflow.models import RouteDecision
from kgqa.agent.workflow.templates import TemplateLibrary


class QuestionRouter(Router):
    def __init__(self, library: TemplateLibrary) -> None:
        self.library = library

    def classify(self, question: str) -> RouteDecision:
        return self.library.classify(question)
