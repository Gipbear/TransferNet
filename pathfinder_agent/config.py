# pathfinder_agent/config.py
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "data", "output", "WebQSP", "pathfinder_logs")

# Ensure log dir exists
os.makedirs(LOG_DIR, exist_ok=True)

# Lora Adapter path
LORA_ADAPTER_PATH = os.path.join(BASE_DIR, "models", "webqsp", "ablation", "groupAname_v2")

# Retrieval setting configurations
BEAM_SIZE_PRIMARY = 20
LAMBDA_PRIMARY = 0.2
BEAM_SIZE_FALLBACK = 50
LAMBDA_FALLBACK = 1.0

MAX_PATHS_RETURNED = 30
MAX_DUPLICATE_TAIL_NODES = 2
