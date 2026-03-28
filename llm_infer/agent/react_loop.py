"""Core ReAct loop for the KG-ReAct Agent.

Classes
-------
AgentResult   -- structured output returned by KGReActAgent.run()
KGReActAgent  -- main agent class that executes the Thought/Action/Observation loop
"""

from __future__ import annotations

import json
import logging
import re
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .agent_config import AgentConfig
from .prompts import DECISION_SYSTEM_PROMPT

if TYPE_CHECKING:
    # Avoid circular imports; ToolRegistry is defined in tools.py (Step 2).
    from .tools import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result contract
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Structured result returned by :meth:`KGReActAgent.run`.

    Attributes
    ----------
    question:
        The original input question.
    answer:
        Final predicted answer entities (list of strings).
    cited_paths:
        1-based indices of paths cited in the final answer.
    reasoning_trace:
        Full step-by-step trace: each entry is a dict with keys
        ``step``, ``thought``, ``action``, ``observation``.
    steps_taken:
        Number of ReAct loop iterations executed.
    terminated_by:
        ``"finish"`` if the agent called finish(), ``"max_steps"`` otherwise.
    final_verification:
        The last verify_citation result dict, or ``None`` if not yet called.
    """

    question: str
    answer: list[str]
    cited_paths: list[int]
    reasoning_trace: list[dict]
    steps_taken: int
    terminated_by: str  # "finish" | "max_steps"
    final_verification: dict | None = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class KGReActAgent:
    """Knowledge Graph ReAct Agent.

    The agent uses a single (model, tokenizer) pair for all LLM calls.
    Behavior is controlled via system prompts:
      - **Decision mode** uses :data:`~prompts.DECISION_SYSTEM_PROMPT` and
        generates Thought/Action text.
      - Tools may internally switch to reasoning mode or decomposition mode
        by passing a different system prompt.

    Model loading and GPU management are the caller's responsibility; this
    class only calls ``tokenizer`` and ``model.generate``.

    Parameters
    ----------
    model:
        A loaded HuggingFace-style causal LM (already on the target device).
    tokenizer:
        Corresponding tokenizer with ``apply_chat_template`` support.
    tool_registry:
        A :class:`~tools.ToolRegistry` instance with all tools registered.
    config:
        :class:`~agent_config.AgentConfig` controlling loop parameters.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        tool_registry: "ToolRegistry",
        config: AgentConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tool_registry = tool_registry
        self.config = config or AgentConfig()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, question: str) -> AgentResult:
        """Execute the ReAct loop for *question* and return an :class:`AgentResult`.

        The loop runs at most ``config.max_steps`` iterations.  It terminates
        early when the agent calls the ``finish`` tool.  If ``max_steps`` is
        reached without a ``finish`` call, the most recent ``reason_and_cite``
        result is used as the fallback answer.

        Parameters
        ----------
        question:
            Natural-language question to answer.

        Returns
        -------
        AgentResult
        """
        logger.info("Starting ReAct loop for question: %s", question)

        steps: list[dict] = []
        final_verification: dict | None = None

        # Seed the conversation history
        history: list[dict] = self._build_history(steps, question)

        for step_idx in range(1, self.config.max_steps + 1):
            logger.debug("Step %d/%d", step_idx, self.config.max_steps)

            # 1. Generate Thought + Action
            raw_thought = self._generate_thought(history)
            logger.debug("Raw model output:\n%s", raw_thought)

            # 2. Parse Action
            try:
                tool_name, args_string = self._parse_action(raw_thought)
            except ValueError as exc:
                logger.warning("Could not parse action at step %d: %s", step_idx, exc)
                observation = f"[ParseError] {exc}. Please output exactly: Thought: ...\nAction: tool_name(args)"
                extracted_thought = self._extract_thought_text(raw_thought)
                if extracted_thought and extracted_thought != raw_thought.strip():
                    thought_for_record = extracted_thought
                else:
                    thought_for_record = f"[PARSE_ERROR] {raw_thought[:200]}"
                step_record = {
                    "step": step_idx,
                    "thought": thought_for_record,
                    "action": None,
                    "observation": observation,
                }
                steps.append(step_record)
                history = self._build_history(steps, question)
                continue

            # Extract thought text (everything before Action:)
            thought_text = self._extract_thought_text(raw_thought)

            # 3. Handle finish tool specially (terminates the loop)
            if tool_name == "finish":
                logger.info("Agent called finish at step %d", step_idx)
                answer, cited = self._parse_finish_args(args_string, steps)
                step_record = {
                    "step": step_idx,
                    "thought": thought_text,
                    "action": f"finish({args_string})",
                    "observation": "[Loop terminated by finish()]",
                }
                steps.append(step_record)
                return AgentResult(
                    question=question,
                    answer=answer,
                    cited_paths=cited,
                    reasoning_trace=steps,
                    steps_taken=step_idx,
                    terminated_by="finish",
                    final_verification=final_verification,
                )

            # 4. Execute the tool
            observation = self._execute_tool(tool_name, args_string)

            # Track verify_citation results for the final_verification field
            if tool_name == "verify_citation":
                try:
                    final_verification = json.loads(observation)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "verify_citation at step %d returned non-JSON observation; "
                        "final_verification was NOT updated. Observation: %s",
                        step_idx,
                        observation[:200],
                    )

            step_record = {
                "step": step_idx,
                "thought": thought_text,
                "action": f"{tool_name}({args_string})",
                "observation": observation,
            }
            steps.append(step_record)

            # Rebuild history with the new observation
            history = self._build_history(steps, question)

        # max_steps reached without finish
        logger.warning(
            "max_steps=%d reached without finish. Extracting best answer from history.",
            self.config.max_steps,
        )
        answer, cited = self._extract_best_answer_from_history(steps)
        return AgentResult(
            question=question,
            answer=answer,
            cited_paths=cited,
            reasoning_trace=steps,
            steps_taken=len(steps),
            terminated_by="max_steps",
            final_verification=final_verification,
        )

    # ------------------------------------------------------------------
    # LLM call helpers
    # ------------------------------------------------------------------

    def _generate_thought(self, history: list[dict]) -> str:
        """Run the LLM in decision mode and return raw generated text.

        Parameters
        ----------
        history:
            Chat messages list (dicts with ``role`` and ``content`` keys)
            already including the system prompt and conversation so far.

        Returns
        -------
        str
            Raw model output (should contain ``Thought: ... Action: ...``).
        """
        input_ids = self.tokenizer.apply_chat_template(
            history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.config.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.config.max_new_tokens_decision,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    # Matches "Action: tool_name(" capturing tool_name and the position of "("
    _ACTION_HEADER_RE = re.compile(r"Action:\s*(\w+)\(", re.MULTILINE)

    def _parse_action(self, text: str) -> tuple[str, str]:
        """Extract ``(tool_name, args_string)`` from model output.

        Expects a line of the form::

            Action: tool_name(arg1, arg2, ...)

        Uses a parenthesis-depth-counting parser so that nested parentheses
        inside the arguments (e.g. ``"What is (something)?"`` or
        ``func(a, g(b))``) are handled correctly.

        Parameters
        ----------
        text:
            Raw model output string.

        Returns
        -------
        tuple[str, str]
            ``(tool_name, args_string)`` where *args_string* is the raw
            content inside the outermost parentheses (not yet parsed into
            Python objects).

        Raises
        ------
        ValueError
            If no parseable ``Action:`` line is found or parentheses are
            unbalanced.
        """
        # Find the last "Action: tool_name(" occurrence so the most recent
        # action wins when the model repeats itself.
        match = None
        for match in self._ACTION_HEADER_RE.finditer(text):
            pass  # keep iterating to land on the last match

        if match is None:
            raise ValueError(
                f"No 'Action: tool_name(...)' found in text: {text[:200]!r}"
            )

        tool_name = match.group(1).strip()
        # Position immediately after the opening "("
        start = match.end()

        depth = 1
        i = start
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1

        if depth != 0:
            raise ValueError(
                f"Unbalanced parentheses in Action args starting at position "
                f"{start}: {text[start:start + 200]!r}"
            )

        # text[start : i-1] is the content between the outermost parentheses
        args_string = text[start : i - 1].strip()
        return tool_name, args_string

    @staticmethod
    def _extract_thought_text(raw: str) -> str:
        """Return the Thought portion of the raw model output (or the full text)."""
        # Take everything up to the first "Action:" line
        lines = raw.splitlines()
        thought_lines: list[str] = []
        for line in lines:
            if re.match(r"^\s*Action:", line, re.IGNORECASE):
                break
            thought_lines.append(line)
        thought = "\n".join(thought_lines).strip()
        # Strip leading "Thought:" prefix if present
        thought = re.sub(r"^Thought:\s*", "", thought, flags=re.IGNORECASE).strip()
        return thought or raw.strip()

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, args_string: str) -> str:
        """Dispatch *tool_name* with *args_string* via the tool registry.

        Parameters
        ----------
        tool_name:
            Registered tool name (e.g. ``"retrieve_paths"``).
        args_string:
            Raw string of arguments as written by the model inside the
            parentheses (e.g. ``'"what is X", K=15'``).

        Returns
        -------
        str
            Observation string to append to the conversation history.
            On tool errors, returns a descriptive error string rather than
            raising, so the agent can attempt recovery.
        """
        logger.debug("Executing tool: %s(%s)", tool_name, args_string[:120])
        try:
            result = self.tool_registry.call(tool_name, args_string)
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False)
            return str(result)
        except Exception as exc:  # ToolNotFoundError, ToolExecutionError, etc.
            error_type = type(exc).__name__
            logger.warning("Tool execution error (%s): %s", error_type, exc)
            return f"[{error_type}] {exc}"

    # ------------------------------------------------------------------
    # History construction
    # ------------------------------------------------------------------

    def _build_history(self, steps: list[dict], question: str) -> list[dict]:
        """Build the chat messages list from the accumulated step history.

        The resulting list follows the standard chat format:

        * ``system`` — :data:`~prompts.DECISION_SYSTEM_PROMPT`
        * ``user``   — ``"Question: {question}"``
        * For each completed step:

          - ``assistant`` — thought text + action line
          - ``user``      — ``"Observation: {observation}"``

        Parameters
        ----------
        steps:
            List of step dicts accumulated so far (may be empty on the first call).
        question:
            The original question string.

        Returns
        -------
        list[dict]
            Chat messages ready for ``tokenizer.apply_chat_template``.
        """
        messages: list[dict] = [
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]
        for step in steps:
            # Reconstruct the assistant's turn
            thought = step.get("thought") or ""
            action = step.get("action") or ""
            if thought and action:
                assistant_content = f"Thought: {thought}\nAction: {action}"
            elif action:
                assistant_content = f"Action: {action}"
            else:
                # ParseError step — include raw output for context
                assistant_content = step.get("thought", "")

            messages.append({"role": "assistant", "content": assistant_content})
            # Observation fed back as the next user turn
            observation = step.get("observation", "")
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return messages

    # ------------------------------------------------------------------
    # Fallback answer extraction
    # ------------------------------------------------------------------

    # Regex patterns for parsing plain-text reason_and_cite observations
    _ANSWER_LINE_RE = re.compile(r"^Answer:\s*(.+)$", re.MULTILINE)
    _PATHS_LINE_RE = re.compile(r"^Supporting Paths:\s*(.+)$", re.MULTILINE)

    def _extract_best_answer_from_history(
        self, steps: list[dict]
    ) -> tuple[list[str], list[int]]:
        """Return the answer and citations from the most recent reason_and_cite step.

        Scans *steps* in reverse order for a step whose action starts with
        ``reason_and_cite``.  The observation of that step is the plain-text
        format produced by :class:`~tools.ReasonAndCiteTool`::

            Answer: entity1 | entity2
            Supporting Paths: 1, 3
            Format OK: True

        If the shared context's ``last_reasoning`` is populated it is used
        directly to avoid re-parsing.

        Parameters
        ----------
        steps:
            Accumulated step records.

        Returns
        -------
        tuple[list[str], list[int]]
            ``(answer_list, cited_indices_list)``.  Both are empty lists if
            no reason_and_cite step is found.
        """
        # Fast path: use shared context if available
        try:
            ctx = self.tool_registry._tools.get("reason_and_cite")
            if ctx is not None:
                reasoning = ctx._ctx.last_reasoning
                if reasoning and (reasoning.get("answers") or reasoning.get("cited_indices")):
                    answers = reasoning.get("answers", [])
                    cited = sorted(int(i) for i in reasoning.get("cited_indices", set()))
                    if answers or cited:
                        return answers, cited
        except Exception:
            pass  # fall through to text-parsing

        for step in reversed(steps):
            action = step.get("action") or ""
            if not action.startswith("reason_and_cite"):
                continue
            obs = step.get("observation", "")

            # Parse plain-text format: "Answer: ...\nSupporting Paths: ...\nFormat OK: ..."
            answer_match = self._ANSWER_LINE_RE.search(obs)
            paths_match = self._PATHS_LINE_RE.search(obs)

            if answer_match:
                raw_answer = answer_match.group(1).strip()
                answers = [a.strip() for a in raw_answer.split("|") if a.strip()]
                # Filter out sentinel "(none)"
                answers = [a for a in answers if a.lower() != "(none)"]
            else:
                answers = []

            if paths_match:
                raw_paths = paths_match.group(1).strip()
                try:
                    cited = sorted(
                        int(t.strip())
                        for t in raw_paths.split(",")
                        if t.strip() and t.strip().lower() != "(none)"
                    )
                except ValueError:
                    cited = []
            else:
                cited = []

            if answers or cited:
                return answers, cited

        logger.warning("No reason_and_cite observation found; returning empty answer.")
        return [], []

    # ------------------------------------------------------------------
    # finish() argument parsing (called internally when agent uses finish)
    # ------------------------------------------------------------------

    def _parse_finish_args(
        self, args_string: str, steps: list[dict]
    ) -> tuple[list[str], list[int]]:
        """Extract answer and supporting_paths from the finish() args string.

        The model may write::

            finish(["entity1", "entity2"], [1, 3], "trace...")

        or just reference them symbolically.  We attempt a best-effort parse;
        on failure, we fall back to :meth:`_extract_best_answer_from_history`.

        Parameters
        ----------
        args_string:
            Raw args content inside the ``finish(...)`` call.
        steps:
            Previous steps, used as fallback.

        Returns
        -------
        tuple[list[str], list[int]]
            ``(answer_list, cited_indices_list)``.
        """
        # Try to parse the first two positional arguments as JSON arrays
        # e.g. ["entity1"], [1, 3], "trace"
        list_pattern = re.compile(r"(\[.*?\])", re.DOTALL)
        matches = list_pattern.findall(args_string)
        if len(matches) >= 2:
            try:
                answer = json.loads(matches[0])
                cited = sorted(int(i) for i in json.loads(matches[1]))
                return answer, cited
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: delegate to shared history-scan helper to avoid duplication
        logger.debug("finish() args not parseable; falling back to history scan.")
        answer, cited = self._extract_best_answer_from_history(steps)
        if not answer:
            logger.warning(
                "finish() fallback also found no answer; returning empty answer and citations."
            )
        return answer, cited
