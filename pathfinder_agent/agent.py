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
from pathfinder_agent.config import LOG_DIR, LORA_ADAPTER_PATH

# 初始化日志记录系统
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"agent_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class PathfinderAgent:
    def __init__(self,
                 model_name: str = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
                 adapter_path: str = None,
                 device: str = "cuda"):
        self.device = device
        logging.info("Initializing PathfinderAgent...")

        # Lazy imports to keep module-level imports clean
        from unsloth import FastLanguageModel
        from peft import PeftModel

        logging.info(f"Loading base LLM: {model_name}")
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
            logging.info(f"Loading LoRA adapter: {_adapter}")
            self.model = PeftModel.from_pretrained(self.model, _adapter)
        else:
            logging.warning(f"Adapter path not found: {_adapter}. Running without LoRA adapter.")

        # Switch to fast inference mode (required by unsloth)
        FastLanguageModel.for_inference(self.model)
        self.model.eval()

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # TransferNet wrapper (injected externally via agent.transfernet_wrapper = ...)
        self.transfernet_wrapper = None
        logging.info("PathfinderAgent initialized.")


    def run(self, question, topic_entity):
        logging.info(f"\n--- Starting PathfinderAgent for question: {question} ---")
        
        # 步骤1：问题重写
        queries = rewrite_question(self.model, self.tokenizer, question, topic_entity)
        logging.info(f"Rewritten queries: {queries}")
        
        all_answers = []
        for q in queries:
            # 步骤2: 主路检索
            primary_paths = retrieve_paths(self.transfernet_wrapper, q, topic_entity, fallback=False)
            logging.info(f"Primary paths retrieved: {len(primary_paths)}")
            
            # 步骤3: 推理提取
            cand_ans, indices = reason_with_paths(self.model, self.tokenizer, q, primary_paths)
            logging.info(f"Candidate answer proposed: {cand_ans}")
            
            # 步骤4: 校验拦截
            is_valid, feedback = verify_answer(self.model, self.tokenizer, q, cand_ans, indices, primary_paths)
            
            if not is_valid:
                logging.warning(f"Verification failed: {feedback}. Triggering fallback retrieval.")
                # 触发降级检索 (beam=50)
                fallback_paths = retrieve_paths(self.transfernet_wrapper, q, topic_entity, fallback=True)
                cand_ans, indices = reason_with_paths(self.model, self.tokenizer, q, fallback_paths)
                logging.info(f"Candidate answer after fallback: {cand_ans}")
                
            all_answers.append(cand_ans)
            
        # Step 5: Aggregate answers from all rewritten queries
        final_answer = aggregate_answers(self.model, self.tokenizer, all_answers, question=question)
        logging.info(f"Final Answer aggregated: {final_answer}")
        return final_answer
