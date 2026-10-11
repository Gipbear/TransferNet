"""按冻结路由执行初答、符号处理、补证据重答与预算回退。"""

from dataclasses import replace
from time import perf_counter

import requests

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.interfaces import Answerer, Router, WorkflowTool
from kgqa.agent.workflow.models import AnswerResult, RouteDecision, RuntimeQuestion, ToolOutcome, WorkflowRecord


class MasterQAWorkflow:
    def __init__(self, config: WorkflowConfig, answerer: Answerer, router: Router,
                 comparison: WorkflowTool, conjunction: WorkflowTool, rescue: WorkflowTool) -> None:
        self.config = config
        self.answerer = answerer
        self.router = router
        self.tools = {"comparison": comparison, "conjunction": conjunction, "rescue": rescue}

    def answer(self, question: RuntimeQuestion) -> WorkflowRecord:
        started = perf_counter()
        initial = self.answerer.answer(question.question, question.paths)
        route = self.router.classify(question.question)
        initial_paths = self._used_paths(question.paths, initial)
        final, paths, source, reason, calls, tool = initial, initial_paths, "baseline", "kept_initial", 1, None
        call_results = [initial]
        branch = self._branch(route, initial)
        if branch:
            tool = self._run_tool(branch, question, initial)
            reason = tool.reason
            if (not tool.applied and initial.is_rejection
                    and self.config.allow_cascade_rescue and self.config.rescue_enabled
                    and branch != "rescue" and self.config.max_pfit_calls > 1):
                branch = "rescue"
                tool = self._run_tool(branch, question, initial)
                reason = tool.reason
            if tool.applied and tool.answers is not None:
                final = replace(initial, answers=tool.answers, is_rejection=False, cited_indices=(), raw_output="",
                                prompt_tokens=0, completion_tokens=0, elapsed_ms=0, used_path_count=0)
                source = branch
            elif tool.applied and tool.paths:
                if calls >= self.config.max_pfit_calls:
                    reason = "pfit_budget_exhausted"
                else:
                    followup = self._followup(question.question, tool.paths)
                    calls += 1
                    call_results.append(followup)
                    if followup.format_ok:
                        final, paths, source = followup, self._used_paths(tool.paths, followup), branch
                    else:
                        reason = "followup_transport_error" if followup.generation_error else "followup_format_error"
        return WorkflowRecord(
            question, route, initial, final, paths, source, reason, calls,
            (perf_counter() - started) * 1000, tool, initial_paths, tuple(call_results),
        )

    @staticmethod
    def _used_paths(paths: tuple[dict, ...], answer: AnswerResult) -> tuple[dict, ...]:
        return paths if answer.used_path_count is None else paths[:answer.used_path_count]

    def _run_tool(self, branch: str, question: RuntimeQuestion, initial: AnswerResult) -> ToolOutcome:
        try:
            return self.tools[branch].run(question, initial)
        except (requests.RequestException, TimeoutError) as exc:
            return ToolOutcome(False, "tool_transport_error", stats={"error": f"{type(exc).__name__}: {exc}"[:400]})

    def _followup(self, question: str, paths: tuple[dict, ...]) -> AnswerResult:
        started = perf_counter()
        try:
            return self.answerer.answer(question, paths)
        except (requests.RequestException, TimeoutError) as exc:
            return AnswerResult((), False, False, elapsed_ms=(perf_counter() - started) * 1000,
                                generation_error=f"{type(exc).__name__}: {exc}"[:400])

    def _branch(self, route: RouteDecision, initial: AnswerResult) -> str | None:
        if not initial.format_ok:
            return None
        if route.confidence >= self.config.type_confidence:
            if route.kind in ("comparative", "superlative") and self.config.comparison_enabled:
                return "comparison"
            if route.kind == "conjunction" and self.config.conjunction_enabled and self.config.max_pfit_calls > 1:
                return "conjunction"
        if initial.is_rejection and self.config.rescue_enabled and self.config.max_pfit_calls > 1:
            return "rescue"
        return None
