"""Tool registry and tool implementations for the KG-ReAct Agent.

Classes
-------
AgentContext     -- shared mutable context passed to all tools
BaseTool         -- abstract base class every tool must subclass
ToolRegistry     -- dispatches tool_name -> BaseTool.call()
RetrievePathsTool    -- re-runs MMR diversity selection on pre-computed paths
ReasonAndCiteTool    -- SFT model inference in V2 format
VerifyCitationTool   -- pure-logic citation consistency check (no LLM)
DecomposeQuestionTool -- LLM-based question decomposition
FinishTool           -- sentinel tool; signals loop termination
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    """Shared mutable state passed to all tools at init time.

    The agent loop sets ``question`` and ``current_record`` before each
    :meth:`KGReActAgent.run` call; tools update ``current_paths`` and
    ``last_reasoning`` during execution.

    Attributes
    ----------
    question:
        The current natural-language question being answered.
    current_paths:
        List of path dicts retrieved by the most recent RetrievePathsTool call.
        Each dict has at least a ``"path"`` key (list of (subj, rel, obj) triples)
        and a ``"log_score"`` key.
    last_reasoning:
        Dict returned by ``eval_faithfulness.parse_output`` from the most recent
        ReasonAndCiteTool call (keys: ``answers``, ``cited_indices``, ``format_ok``).
    current_record:
        The raw dataset record for the current sample, as produced by
        ``WebQSP/predict.py``.  Expected keys:
          ``mmr_reason_paths``  -- list of pre-computed path dicts
          ``e_score``           -- entity score tensor (optional, not used here)
          ``hop_attn``          -- hop attention tensor (optional, not used here)
    """

    question: str = ""
    current_paths: list = field(default_factory=list)
    last_reasoning: dict = field(default_factory=dict)
    current_record: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    """Abstract base class for all agent tools.

    Subclasses must set :attr:`name` and :attr:`description` as class-level
    attributes and implement :meth:`call`.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def call(self, args_string: str) -> str:
        """Execute the tool and return an observation string.

        Parameters
        ----------
        args_string:
            Raw string from inside the LLM's ``Action: tool_name(...)`` call —
            not yet parsed into Python objects.  Each tool is responsible for
            its own parsing.

        Returns
        -------
        str
            Observation text to be fed back into the agent conversation.
            Errors are returned as ``"[Error] ..."`` strings rather than raised.
        """


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Dispatches tool calls by name.

    Usage::

        registry = ToolRegistry()
        registry.register(RetrievePathsTool(...))
        observation = registry.call("retrieve_paths", "question, K=5")
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register *tool* under its :attr:`~BaseTool.name`.

        Parameters
        ----------
        tool:
            A :class:`BaseTool` instance to register.
        """
        if not tool.name:
            raise ValueError(f"Tool {type(tool).__name__} has no name set.")
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def call(self, tool_name: str, args_string: str) -> str:
        """Dispatch *tool_name* and return the observation string.

        Parameters
        ----------
        tool_name:
            Name of the registered tool to call.
        args_string:
            Raw argument string (content between outermost parentheses).

        Returns
        -------
        str
            The tool's observation.

        Raises
        ------
        KeyError
            If *tool_name* is not registered.
        """
        if tool_name not in self._tools:
            available = ", ".join(sorted(self._tools))
            raise KeyError(
                f"Unknown tool '{tool_name}'. Available tools: {available}"
            )
        return self._tools[tool_name].call(args_string)

    def list_tools(self) -> list[str]:
        """Return a sorted list of registered tool names."""
        return sorted(self._tools)


# ---------------------------------------------------------------------------
# Helper: MMR diversity re-selection
# ---------------------------------------------------------------------------

def _mmr_reselect(
    candidate_paths: list,
    k: int,
    lambda_val: float,
) -> list:
    """Re-run MMR diversity selection on *candidate_paths*.

    Operates on ``log_score`` values stored in each path dict (key
    ``"log_score"``).  Uses a simple greedy MMR over scalar scores:

      score(p) = lambda_val * relevance(p)
                 - (1 - lambda_val) * max_similarity_to_selected(p)

    where relevance is the normalised log_score and similarity between two
    scalar scores is ``1 - |s_i - s_j| / score_range``.

    Parameters
    ----------
    candidate_paths:
        Full list of path dicts with ``"log_score"`` keys.
    k:
        Maximum number of paths to return.
    lambda_val:
        Trade-off between relevance and diversity (0 = most diverse, 1 = most
        relevant).

    Returns
    -------
    list
        Selected path dicts, at most *k* items, in order of selection.
    """
    if not candidate_paths:
        return []

    k = min(k, len(candidate_paths))

    scores = [p.get("log_score", 0.0) for p in candidate_paths]
    min_s = min(scores)
    max_s = max(scores)
    score_range = max_s - min_s if max_s != min_s else 1.0

    # Normalise to [0, 1]
    norm_scores = [(s - min_s) / score_range for s in scores]

    selected_indices: list[int] = []
    remaining = list(range(len(candidate_paths)))

    for _ in range(k):
        if not remaining:
            break

        best_idx = None
        best_mmr = float("-inf")

        for i in remaining:
            relevance = norm_scores[i]
            if selected_indices:
                # Similarity to the most similar already-selected path
                max_sim = max(
                    1.0 - abs(norm_scores[i] - norm_scores[j])
                    for j in selected_indices
                )
            else:
                max_sim = 0.0

            mmr = lambda_val * relevance - (1.0 - lambda_val) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)
        else:
            break  # no valid candidate found

    return [candidate_paths[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# Tool 1: RetrievePathsTool
# ---------------------------------------------------------------------------

class RetrievePathsTool(BaseTool):
    """Retrieve top-K diverse reasoning paths via MMR beam search.

    Uses pre-computed ``mmr_reason_paths`` from the current dataset record
    (set via the shared :class:`AgentContext`) and re-runs MMR diversity
    selection with caller-supplied K and lambda_val.

    Args string examples
    --------------------
    - ``'question, K=10, lambda_val=0.5'``
    - ``'"Who directed Inception?", K=15'``
    - ``'K=5'``
    """

    name = "retrieve_paths"
    description = (
        "Retrieve the top-K diverse reasoning paths from the knowledge graph. "
        "Args: question (ignored; uses pre-loaded sample), K=10, lambda_val=0.5"
    )

    def __init__(self, context: AgentContext) -> None:
        self._ctx = context

    # -- arg parsing helpers -------------------------------------------------

    @staticmethod
    def _parse_k(args_string: str, default: int = 10) -> int:
        m = re.search(r"\bK\s*=\s*(\d+)", args_string, re.IGNORECASE)
        if m:
            return max(1, int(m.group(1)))
        return default

    @staticmethod
    def _parse_lambda(args_string: str, default: float = 0.5) -> float:
        m = re.search(r"\blambda_val\s*=\s*([0-9]*\.?[0-9]+)", args_string, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
        return default

    # -- main call -----------------------------------------------------------

    def call(self, args_string: str) -> str:
        try:
            k = self._parse_k(args_string)
            lambda_val = self._parse_lambda(args_string)

            record = self._ctx.current_record
            if not record:
                return "[Error] No current record loaded. Set context.current_record before calling retrieve_paths."

            candidates = record.get("mmr_reason_paths", [])
            if not candidates:
                return "[Error] current_record has no 'mmr_reason_paths' field or it is empty."

            selected = _mmr_reselect(candidates, k=k, lambda_val=lambda_val)

            # Store in context for downstream tools
            self._ctx.current_paths = selected

            # Format output
            lines = [f"Retrieved {len(selected)} paths:"]
            for idx, path_dict in enumerate(selected, start=1):
                edges = path_dict.get("path", [])
                log_score = path_dict.get("log_score", 0.0)
                # format_path_str signature: (path_edges, log_score, idx)
                chain = " ".join(
                    f"({e[0]}) -[{e[1]}]-> ({e[2]})" for e in edges
                )
                lines.append(f"Path {idx}: {chain}")

            logger.debug(
                "retrieve_paths: K=%d lambda=%.2f -> selected %d/%d paths",
                k, lambda_val, len(selected), len(candidates),
            )
            return "\n".join(lines)

        except Exception as exc:
            logger.warning("retrieve_paths error: %s", exc, exc_info=True)
            return f"[Error] retrieve_paths: {exc}"


# ---------------------------------------------------------------------------
# Tool 2: ReasonAndCiteTool
# ---------------------------------------------------------------------------

class ReasonAndCiteTool(BaseTool):
    """Run SFT model inference in V2 format and return a structured observation.

    Uses the question and current paths from :class:`AgentContext`.
    The parsed result is stored back in ``context.last_reasoning`` for use by
    :class:`VerifyCitationTool`.

    Args string
    -----------
    Ignored — context is taken from the shared :class:`AgentContext`.
    """

    name = "reason_and_cite"
    description = (
        "Run the fine-tuned reasoning model on the current question and paths. "
        "Returns predicted answer entities and cited path indices. Args: ignored."
    )

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        context: AgentContext,
        device: str = "cuda",
        max_new_tokens: int = 256,
        output_format: str = "v2",
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._ctx = context
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._output_format = output_format

    def call(self, args_string: str) -> str:  # noqa: ARG002
        try:
            # Import here to avoid circular/conditional import issues
            import sys
            import os
            # Ensure llm_infer is importable
            _llm_infer_dir = os.path.join(os.path.dirname(__file__), "..")
            if _llm_infer_dir not in sys.path:
                sys.path.insert(0, _llm_infer_dir)

            from kg_format import FORMAT_PROMPTS, build_user_content
            from eval_faithfulness import parse_output

            question = self._ctx.question
            paths = self._ctx.current_paths

            if not question:
                return "[Error] reason_and_cite: context.question is empty."
            if not paths:
                return "[Error] reason_and_cite: context.current_paths is empty. Call retrieve_paths first."

            # Build paths_with_meta: [(path_edges, log_score, display_idx), ...]
            paths_with_meta = [
                (p.get("path", []), p.get("log_score", 0.0), i + 1)
                for i, p in enumerate(paths)
            ]

            system_prompt = FORMAT_PROMPTS[self._output_format]
            user_content = build_user_content(
                paths_with_meta, question, path_format="arrow"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            enc = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self._device)
            attn_mask = enc.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(self._device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0][input_ids.shape[-1]:]
            raw_output = self._tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            parsed = parse_output(raw_output, self._output_format)

            # Store result in shared context
            self._ctx.last_reasoning = parsed

            answers = parsed.get("answers", [])
            cited = sorted(parsed.get("cited_indices", set()))
            format_ok = parsed.get("format_ok", False)

            logger.debug(
                "reason_and_cite: answers=%s cited=%s format_ok=%s",
                answers, cited, format_ok,
            )

            answer_str = " | ".join(answers) if answers else "(none)"
            cited_str = ", ".join(str(i) for i in cited) if cited else "(none)"
            return (
                f"Answer: {answer_str}\n"
                f"Supporting Paths: {cited_str}\n"
                f"Format OK: {format_ok}"
            )

        except Exception as exc:
            logger.warning("reason_and_cite error: %s", exc, exc_info=True)
            return f"[Error] reason_and_cite: {exc}"


# ---------------------------------------------------------------------------
# Tool 3: VerifyCitationTool
# ---------------------------------------------------------------------------

class VerifyCitationTool(BaseTool):
    """Check consistency between cited paths and predicted answer (no LLM call).

    Reads ``context.last_reasoning`` (set by :class:`ReasonAndCiteTool`) and
    ``context.current_paths``.  Returns a JSON string that the agent loop
    deserialises into ``AgentResult.final_verification``.

    Three checks
    ------------
    1. **tail_match**: For each cited path index, does the tail entity appear
       among the predicted answer entities?
    2. **hallucination_detected**: Are any predicted answer entities absent from
       *all* entities in *all* current paths?
    3. **score_confidence**: Is the top path score above the threshold (high) or
       below (low)?

    Args string
    -----------
    Ignored — all data is taken from shared :class:`AgentContext`.
    """

    name = "verify_citation"
    description = (
        "Verify that cited paths are consistent with the predicted answer. "
        "No LLM call; uses output from reason_and_cite. Args: ignored."
    )

    def __init__(self, context: AgentContext, path_score_threshold: float = -3.0) -> None:
        self._ctx = context
        self._threshold = path_score_threshold

    def call(self, args_string: str) -> str:  # noqa: ARG002
        try:
            import sys
            import os
            _llm_infer_dir = os.path.join(os.path.dirname(__file__), "..")
            if _llm_infer_dir not in sys.path:
                sys.path.insert(0, _llm_infer_dir)

            from eval_faithfulness import get_all_path_entities

            reasoning = self._ctx.last_reasoning
            paths = self._ctx.current_paths

            if not reasoning or (not reasoning.get("answers") and not reasoning.get("cited_indices")):
                result = {
                    "consistent": False,
                    "issues": ["reason_and_cite has not been called yet"],
                    "details": {
                        "tail_match": False,
                        "hallucination_detected": False,
                        "score_confidence": "unknown",
                    },
                }
                return json.dumps(result, ensure_ascii=False)

            answers: list[str] = reasoning.get("answers", [])
            cited_indices: set = {int(i) for i in reasoning.get("cited_indices", set())}

            answer_set = {a.lower().strip() for a in answers if a.strip()}

            # -- Check 1: tail match -----------------------------------------
            tail_match = True
            tail_issues: list[str] = []
            for idx in sorted(cited_indices):
                path_idx = idx - 1  # convert 1-based to 0-based
                if path_idx < 0 or path_idx >= len(paths):
                    tail_issues.append(f"Cited path {idx} is out of range (only {len(paths)} paths available)")
                    tail_match = False
                    continue
                edges = paths[path_idx].get("path", [])
                if not edges:
                    tail_issues.append(f"Cited path {idx} has no edges")
                    tail_match = False
                    continue
                tail_entity = edges[-1][2].lower().strip()
                if tail_entity not in answer_set:
                    tail_issues.append(
                        f"Cited path {idx} tail '{edges[-1][2]}' not in predicted answers {answers}"
                    )
                    tail_match = False

            # -- Check 2: hallucination detection ----------------------------
            all_entities = get_all_path_entities(paths)
            hallucinated: list[str] = [
                a for a in answers
                if a.strip() and a.lower().strip() not in all_entities
            ]
            hallucination_detected = len(hallucinated) > 0
            if hallucination_detected:
                tail_issues.append(
                    f"Hallucinated entities not found in any path: {hallucinated}"
                )

            # -- Check 3: score confidence -----------------------------------
            log_scores = [p.get("log_score", 0.0) for p in paths]
            top_score = max(log_scores) if log_scores else 0.0
            score_confidence = "high" if top_score >= self._threshold else "low"

            # -- Aggregate ---------------------------------------------------
            consistent = tail_match and not hallucination_detected
            result = {
                "consistent": consistent,
                "issues": tail_issues,
                "details": {
                    "tail_match": tail_match,
                    "hallucination_detected": hallucination_detected,
                    "score_confidence": score_confidence,
                },
            }

            logger.debug(
                "verify_citation: consistent=%s tail_match=%s hallucination=%s confidence=%s",
                consistent, tail_match, hallucination_detected, score_confidence,
            )
            return json.dumps(result, ensure_ascii=False)

        except Exception as exc:
            logger.warning("verify_citation error: %s", exc, exc_info=True)
            error_result = {
                "consistent": False,
                "issues": [f"[Error] verify_citation: {exc}"],
                "details": {
                    "tail_match": False,
                    "hallucination_detected": False,
                    "score_confidence": "unknown",
                },
            }
            return json.dumps(error_result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 4: DecomposeQuestionTool
# ---------------------------------------------------------------------------

class DecomposeQuestionTool(BaseTool):
    """Break a complex question into sub-questions using a single LLM call.

    Uses :data:`~prompts.DECOMPOSE_SYSTEM_PROMPT` as the system prompt and the
    current question from :class:`AgentContext` as the user message.

    Args string
    -----------
    Ignored — the current question is taken from the shared context.
    """

    name = "decompose_question"
    description = (
        "Decompose a complex multi-hop question into simpler sub-questions. "
        "Uses the current question from context. Args: ignored."
    )

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        context: AgentContext,
        device: str = "cuda",
        max_new_tokens: int = 512,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._ctx = context
        self._device = device
        self._max_new_tokens = max_new_tokens

    def call(self, args_string: str) -> str:  # noqa: ARG002
        try:
            from .prompts import DECOMPOSE_SYSTEM_PROMPT

            question = self._ctx.question
            if not question:
                return "[Error] decompose_question: context.question is empty."

            messages = [
                {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]

            enc = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self._device)
            attn_mask = enc.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(self._device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0][input_ids.shape[-1]:]
            raw_output = self._tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            # Count sub-questions in output
            sub_q_matches = re.findall(r"^\s*\d+\.", raw_output, re.MULTILINE)
            count = len(sub_q_matches)

            logger.debug(
                "decompose_question: found %d sub-questions in output", count
            )

            if count > 0:
                return f"Decomposed into {count} sub-questions:\n{raw_output}"
            return f"Decomposed question:\n{raw_output}"

        except Exception as exc:
            logger.warning("decompose_question error: %s", exc, exc_info=True)
            return f"[Error] decompose_question: {exc}"


# ---------------------------------------------------------------------------
# Tool 5: FinishTool
# ---------------------------------------------------------------------------

class FinishTool(BaseTool):
    """Sentinel tool that signals loop termination.

    The :class:`~react_loop.KGReActAgent` loop intercepts ``finish`` calls
    *before* dispatching to the registry, so this tool's :meth:`call` method
    is effectively never invoked during normal operation.  It is registered so
    that :meth:`ToolRegistry.list_tools` accurately reflects all available
    tools.

    Args string
    -----------
    Ignored.
    """

    name = "finish"
    description = (
        "Terminate the agent loop with a final answer. "
        "Args: answer (list), supporting_paths (list), reasoning_trace (str)."
    )

    def call(self, args_string: str) -> str:  # noqa: ARG002
        return "FINISH"


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_default_registry(
    model: Any,
    tokenizer: Any,
    context: AgentContext,
    config: Any | None = None,
    output_format: str = "v2",
) -> ToolRegistry:
    """Create a :class:`ToolRegistry` with all five default tools registered.

    Parameters
    ----------
    model:
        Loaded causal language model.
    tokenizer:
        Corresponding tokenizer.
    context:
        Shared :class:`AgentContext` instance.
    config:
        Optional :class:`~agent_config.AgentConfig`; used for device and
        token-budget defaults.  Falls back to module-level defaults if ``None``.

    Returns
    -------
    ToolRegistry
        Registry with retrieve_paths, reason_and_cite, verify_citation,
        decompose_question, and finish tools registered.
    """
    from .agent_config import AgentConfig

    cfg = config or AgentConfig()

    registry = ToolRegistry()
    registry.register(RetrievePathsTool(context=context))
    registry.register(
        ReasonAndCiteTool(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=cfg.device,
            max_new_tokens=cfg.max_new_tokens_decision,
            output_format=output_format,
        )
    )
    registry.register(
        VerifyCitationTool(
            context=context,
            path_score_threshold=cfg.path_score_threshold,
        )
    )
    registry.register(
        DecomposeQuestionTool(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=cfg.device,
            max_new_tokens=cfg.max_new_tokens_decompose,
        )
    )
    registry.register(FinishTool())
    return registry
