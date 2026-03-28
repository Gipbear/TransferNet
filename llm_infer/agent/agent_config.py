"""Agent configuration dataclass for KG-ReAct Agent."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Configuration for the KGReActAgent loop and tool defaults.

    Attributes
    ----------
    max_steps:
        Maximum number of ReAct loop iterations before forced termination.
    default_k:
        Default number of paths to retrieve per retrieve_paths call.
    default_lambda:
        Default MMR diversity penalty for the initial retrieval.
    retry_k:
        Number of paths to retrieve on a retry after verification failure.
    retry_lambda:
        MMR lambda used on retry (lower = more diversity).
    path_score_threshold:
        Log-score below which the top-retrieved path is considered low-confidence.
    max_new_tokens_decision:
        Token budget for decision-mode (Thought/Action) generation.
    max_new_tokens_decompose:
        Token budget for decomposition-mode generation.
    device:
        Torch device string for model inference.
    """

    max_steps: int = 8
    default_k: int = 10
    default_lambda: float = 0.5
    retry_k: int = 20
    retry_lambda: float = 0.3
    path_score_threshold: float = -3.0
    max_new_tokens_decision: int = 256
    max_new_tokens_decompose: int = 512
    device: str = "cuda"
