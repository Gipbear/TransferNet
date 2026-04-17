# pathfinder_agent/agent.py
import sys
import os
import logging
from datetime import datetime

# Add parent dir to sys.path to easily import tools if needed globally
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathfinder_agent.tools.query_rewriter import rewrite_question
from pathfinder_agent.tools.dynamic_retriever import retrieve_paths
from pathfinder_agent.tools.llm_reasoner import reason_with_paths
from pathfinder_agent.tools.answer_verifier import verify_answer
from pathfinder_agent.tools.answer_aggregator import aggregate_answers
from pathfinder_agent.tools.question_utils import normalize_question
from pathfinder_agent.config import LOG_DIR, LORA_ADAPTER_PATH

log = logging.getLogger("pathfinder_agent")
if not any(isinstance(handler, logging.NullHandler) for handler in log.handlers):
    log.addHandler(logging.NullHandler())
log.propagate = False
_LOGGING_INITIALIZED = False


def _setup_agent_logging():
    """Configure PathfinderAgent file logging once, without touching the root logger."""
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return log
    if any(getattr(handler, "_pathfinder_agent_file_handler", False) for handler in log.handlers):
        _LOGGING_INITIALIZED = True
        return log

    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = os.path.join(LOG_DIR, f"agent_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    handler = logging.FileHandler(log_filename, encoding="utf-8")
    handler._pathfinder_agent_file_handler = True
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    _LOGGING_INITIALIZED = True
    return log
class PathfinderAgent:
    def __init__(self,
                 model_name: str = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
                 adapter_path: str = None,
                 device: str = "cuda"):
        _setup_agent_logging()
        self.device = device
        log.info("Initializing PathfinderAgent...")

        # Lazy imports to keep module-level imports clean
        from unsloth import FastLanguageModel
        from peft import PeftModel

        log.info(f"Loading base LLM: {model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            local_files_only=True,
        )

        # Load LoRA adapter — prefer caller-supplied path, fall back to config default
        _adapter = adapter_path or LORA_ADAPTER_PATH
        if _adapter and os.path.exists(_adapter):
            log.info(f"Loading LoRA adapter: {_adapter}")
            self.model = PeftModel.from_pretrained(self.model, _adapter)
        else:
            log.warning(f"Adapter path not found: {_adapter}. Running without LoRA adapter.")

        # Switch to fast inference mode (required by unsloth)
        FastLanguageModel.for_inference(self.model)
        self.model.eval()

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # TransferNet wrapper (injected externally via agent.transfernet_wrapper = ...)
        self.transfernet_wrapper = None
        self.last_evidence_paths = []
        self.last_run_metadata = {}
        log.info("PathfinderAgent initialized.")


    def run(self, question, topic_entity):
        if self.transfernet_wrapper is None:
            raise RuntimeError("PathfinderAgent.transfernet_wrapper must be set before run().")

        question = normalize_question(question)
        self.last_evidence_paths = []
        self.last_run_metadata = {
            "agent_mode": "online_pipeline",
            "selected_source": "online_pipeline",
            "fallback_used": False,
            "final_evidence_source": "online_primary",
        }
        log.info(f"\n--- Starting PathfinderAgent for question: {question} ---")

        # 步骤1：问题重写
        queries = rewrite_question(self.model, self.tokenizer, question, topic_entity)
        log.info(f"Rewritten queries: {queries}")

        all_answers = []
        for q in queries:
            q = normalize_question(q)
            # 步骤2: 主路检索
            primary_paths = retrieve_paths(self.transfernet_wrapper, q, topic_entity, fallback=False)
            log.info(f"Primary paths retrieved: {len(primary_paths)}")

            # 步骤3: 推理提取
            cand_ans, indices = reason_with_paths(self.model, self.tokenizer, q, primary_paths)
            log.info(f"Candidate answer proposed: {cand_ans}")

            # 步骤4: 校验拦截
            is_valid, feedback = verify_answer(self.model, self.tokenizer, q, cand_ans, indices, primary_paths)
            accepted_paths = primary_paths

            if not is_valid:
                log.warning(f"Verification failed: {feedback}. Triggering fallback retrieval.")
                # 触发降级检索 (beam=50)
                fallback_paths = retrieve_paths(self.transfernet_wrapper, q, topic_entity, fallback=True)
                cand_ans, indices = reason_with_paths(self.model, self.tokenizer, q, fallback_paths)
                log.info(f"Candidate answer after fallback: {cand_ans}")
                is_valid, feedback = verify_answer(self.model, self.tokenizer, q, cand_ans, indices, fallback_paths)
                if not is_valid:
                    log.warning(f"Fallback verification failed: {feedback}. Dropping candidate answer.")
                    continue
                accepted_paths = fallback_paths
                self.last_run_metadata["final_evidence_source"] = "online_fallback"

            all_answers.append(cand_ans)
            self.last_evidence_paths.extend(accepted_paths)

        # Step 5: Aggregate answers from all rewritten queries
        final_answer = aggregate_answers(self.model, self.tokenizer, all_answers, question=question)
        log.info(f"Final Answer aggregated: {final_answer}")
        return final_answer
