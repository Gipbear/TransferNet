#!/usr/bin/env python3
"""
PathfinderAgent 模型服务器 — 加载一次 LLM + TransferNet，持续服务推理请求。

使用方式：
  # 1. 启动服务器（只加载一次模型）
  conda run -n py312_t271_cuda python scripts/model_server.py \\
      --ckpt     data/ckpt/WebQSP/model-29-0.6411.pt \\
      --input_dir data/input/WebQSP \\
      --port 8787

  # 2. 在另一个终端运行评测（跳过模型加载，直接调服务器）
  conda run -n py312_t271_cuda python run_agent_eval.py \\
      --input  data/output/.../eval_run0.jsonl \\
      --output data/output/.../agent_eval_run0.jsonl \\
      --ckpt   data/ckpt/WebQSP/model-29-0.6411.pt \\
      --input_dir data/input/WebQSP \\
      --server-url http://localhost:8787

  # 3. 检查服务器健康状态
  curl http://localhost:8787/health

  # 4. 停止服务器
  curl -X POST http://localhost:8787/shutdown   # 或直接 Ctrl-C

API：
  GET  /health      → {"status": "ok", "n_requests": N}
  POST /run         → body: {"question": str, "topic_entity": str}
                    ← {"pred_answer": [...], "evidence_paths": [...]}
  POST /shutdown    → 优雅关闭服务器
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── 环境配置（必须在 unsloth / transformers 导入前） ────────────────────────
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("UNSLOTH_DISABLE_STATS", "1")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── 日志 ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("model_server")


# ── HTTP 处理器 ───────────────────────────────────────────────────────────────
class InferenceHandler(BaseHTTPRequestHandler):
    """推理请求处理器。当前使用单线程 HTTPServer，串行处理请求。
    类变量（agent / _shutdown_event）在服务器启动前注入，运行时只读，无并发风险。
    """

    # 由 main() 注入
    agent = None
    stats = {"n_requests": 0}
    _lock = threading.Lock()
    _shutdown_event = None

    # ── 路由 ──────────────────────────────────────────────────────────────────
    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "n_requests": self.stats["n_requests"]})
        else:
            self._json(404, {"error": f"unknown path: {self.path}"})

    def do_POST(self):
        if self.path == "/run":
            self._handle_run()
        elif self.path == "/shutdown":
            self._json(200, {"status": "shutting down"})
            if self._shutdown_event:
                self._shutdown_event.set()
        else:
            self._json(404, {"error": f"unknown path: {self.path}"})

    # ── /run 处理 ─────────────────────────────────────────────────────────────
    def _handle_run(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError) as e:
            self._json(400, {"error": f"bad request body: {e}"})
            return

        question = body.get("question", "")
        topic_entity = body.get("topic_entity", "")
        if not question or not topic_entity:
            self._json(400, {"error": "question and topic_entity are required"})
            return

        try:
            with self._lock:
                pred_answer = self.agent.run(question, topic_entity)
                evidence_paths = list(self.agent.last_evidence_paths)
                debug_meta = dict(getattr(self.agent, "last_run_metadata", {}))
                self.stats["n_requests"] += 1
        except Exception as e:
            log.error("Inference error: %s", e, exc_info=True)
            self._json(500, {"error": str(e), "pred_answer": [], "evidence_paths": []})
            return

        self._json(
            200,
            {
                "pred_answer": pred_answer,
                "evidence_paths": evidence_paths,
                "agent_mode": debug_meta.get("agent_mode"),
                "selected_source": debug_meta.get("selected_source"),
                "fallback_used": debug_meta.get("fallback_used"),
                "final_evidence_source": debug_meta.get("final_evidence_source"),
            },
        )

    # ── 辅助 ──────────────────────────────────────────────────────────────────
    def _json(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        log.debug("HTTP %s", fmt % args)


# ── 模型加载 ──────────────────────────────────────────────────────────────────
def load_pipeline(args):
    from pathfinder_agent.agent import PathfinderAgent
    from pathfinder_agent.tools.dynamic_retriever import TransferNetWrapper
    from pathfinder_agent.config import LORA_ADAPTER_PATH

    adapter = args.adapter or LORA_ADAPTER_PATH
    log.info("Loading PathfinderAgent (model=%s, adapter=%s) ...", args.model, adapter)
    agent = PathfinderAgent(
        model_name=args.model,
        adapter_path=adapter,
        device=args.device,
    )
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    logging.getLogger("transformers").setLevel(logging.ERROR)

    log.info("Loading TransferNet (ckpt=%s) ...", args.ckpt)
    transfernet = TransferNetWrapper(data_dir=args.input_dir, ckpt_path=args.ckpt)
    agent.transfernet_wrapper = transfernet

    log.info("Pipeline ready.")
    return agent


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="PathfinderAgent Model Server")
    p.add_argument("--ckpt",       required=True, help="TransferNet checkpoint (.pt)")
    p.add_argument("--input_dir",  required=True, help="TransferNet data dir")
    p.add_argument("--model",      default="unsloth/meta-llama-3.1-8b-instruct-bnb-4bit")
    p.add_argument("--adapter",    default=None,  help="LoRA adapter path (default: config)")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--port",       type=int, default=8787, help="HTTP port (default: 8787)")
    p.add_argument("--host",       default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    agent = load_pipeline(args)

    shutdown_event = threading.Event()

    # 注入到处理器类（类变量，所有实例共享）
    InferenceHandler.agent = agent
    InferenceHandler._shutdown_event = shutdown_event

    server = HTTPServer((args.host, args.port), InferenceHandler)

    def _sighandler(sig, frame):
        log.info("Signal %d received, shutting down...", sig)
        shutdown_event.set()

    signal.signal(signal.SIGINT,  _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    log.info("Model server listening on http://%s:%d", args.host, args.port)
    log.info("Press Ctrl-C or POST /shutdown to stop.")

    # 在后台线程中运行 server.serve_forever()，主线程等待 shutdown_event
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    shutdown_event.wait()
    log.info("Shutting down server...")
    server.shutdown()
    log.info("Bye.")


if __name__ == "__main__":
    main()
