#!/usr/bin/env bash
# 第五章补充实验一键脚本。
#
# 跑完后用 scripts/collect_ch5_results.py 汇总成对照表。
# 每组可用 CH5_GROUPS 环境变量挑选，默认全跑(注意:不能用 GROUPS,它是 bash 内置只读变量)：
#   CH5_GROUPS="canonical check constrained loopback ablation"
#
# 各组说明：
#   canonical   官方完整管线(hybrid + loopback + 守卫 + 全后处理)。也是离线回放源。
#   check       check 策略对照：loose-only / strict-only(hybrid 即 canonical)。
#   constrained 受限解码 off 对照。
#   loopback    同环境 loopback off 对照(重启 path server 设 PATH_DROP_LOOPBACK=0)。
#   ablation    后处理累积消融:base(check 后,无后处理) → +margin → +hop → +expansion。
#               (+守卫 = canonical;首答从任一 run 的 initial_answer.jsonl 离线算)
#
# 用法：
#   bash scripts/run_ch5_experiments.sh
#   CH5_GROUPS="canonical check" LIMIT=50 bash scripts/run_ch5_experiments.sh   # 小样本验证

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# ---- 公共配置 ----
INPUT="${INPUT:-data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/output/WebQSP/checked_batch_agent/ch5_$(date +%Y%m%d_%H%M)}"
LIMIT="${LIMIT:-0}"
BEAM_SIZE="${BEAM_SIZE:-50}"
BATCH_SIZE="${BATCH_SIZE:-20}"
LAMBDA_VAL="${LAMBDA_VAL:-0.2}"
ALPHA_FINAL="${ALPHA_FINAL:-1.0}"
SCORE_MARGIN="${SCORE_MARGIN:-4.0}"
EXPANSION_TOP_GROUPS="${EXPANSION_TOP_GROUPS:-2}"
# 注意:变量名不能用 GROUPS——那是 bash 内置只读数组(当前用户组 GID),会取到 "0"
CH5_GROUPS="${CH5_GROUPS:-canonical check constrained loopback ablation}"

PATH_RETRIEVE_PORT="${PATH_RETRIEVE_PORT:-8789}"
LLM_SERVER_PORT="${LLM_SERVER_PORT:-8788}"
PATH_RETRIEVE_URL="${PATH_RETRIEVE_URL:-http://localhost:${PATH_RETRIEVE_PORT}}"
LLM_SERVER_URL="${LLM_SERVER_URL:-http://localhost:${LLM_SERVER_PORT}}"

mkdir -p "${OUTPUT_ROOT}"
echo "[INFO] 输出根目录: ${OUTPUT_ROOT}"
echo "[INFO] 跑的组: ${CH5_GROUPS}"

# ---- 服务管理 ----
ensure_llm_server() {
  local health_url="${LLM_SERVER_URL%/}/health"
  if curl -sf "${health_url}" >/dev/null 2>&1; then
    echo "[INFO] llm_server ready"
  else
    echo "[INFO] 启动 llm_server"
    PORT="${LLM_SERVER_PORT}" ./scripts/llm_server.sh start
  fi
}

# 轮询 eval 实际访问的地址,直到 cache 真正加载完(restart 自带的 health 是 127.0.0.1,
# 这里再用 localhost + cache_loaded:true 复核,捕获"起来又被 OOM kill"等情况)
wait_path_health() {
  local url="${PATH_RETRIEVE_URL%/}/health"
  local timeout="${1:-120}"
  for _ in $(seq 1 "${timeout}"); do
    if curl -sf "${url}" 2>/dev/null | grep -qE '"cache_loaded":[[:space:]]*true'; then
      return 0
    fi
    sleep 1
  done
  echo "[ERROR] path_retrieve_server 未在 ${timeout}s 内就绪(cache_loaded=true): ${url}" >&2
  echo "[HINT ] 可能多个残留 server 占满内存被 OOM。先清理:pkill -f path_retrieve_server.server" >&2
  exit 1
}

# 用指定的 PATH_DROP_LOOPBACK 值(重)启 path server,并复核就绪
start_path_server() {
  local drop="$1"  # 1=剔除 loopback(默认), 0=保留
  echo "[INFO] (重)启动 path_retrieve_server  PATH_DROP_LOOPBACK=${drop}"
  PATH_DROP_LOOPBACK="${drop}" PORT_BUSY_ACTION=kill PORT="${PATH_RETRIEVE_PORT}" \
    ./scripts/path_retrieve_server.sh restart
  wait_path_health
}

# 复用已就绪的 path server(默认 loopback 开),仅在未就绪时才启动——避免杀掉
# 用户手动起的 server、避免叠加重复进程导致 OOM
ensure_path_server() {
  if curl -sf "${PATH_RETRIEVE_URL%/}/health" 2>/dev/null | grep -qE '"cache_loaded":[[:space:]]*true'; then
    echo "[INFO] path_retrieve_server ready(复用已有;loopback 默认开)"
    return
  fi
  echo "[INFO] path_retrieve_server 未就绪，启动(PATH_DROP_LOOPBACK=1)"
  start_path_server 1
}

# ---- 单次评测 ----
# run_eval <子目录名> <额外 CLI flag...>
run_eval() {
  local name="$1"; shift
  local out="${OUTPUT_ROOT}/${name}"
  if [[ -f "${out}/checked_batch_eval_summary.json" ]]; then
    echo "[SKIP] ${name} 已存在，跳过"
    return
  fi
  echo "============================================================"
  echo "[RUN ] ${name}"
  echo "============================================================"
  local cmd=(
    python -m oh_my_agent.cli.eval_checked_batch_agent
    --input "${INPUT}" --output "${out}"
    --beam_size "${BEAM_SIZE}" --batch_size "${BATCH_SIZE}"
    --lambda_val "${LAMBDA_VAL}" --alpha_final "${ALPHA_FINAL}"
    --expansion_top_groups "${EXPANSION_TOP_GROUPS}"
    --path_retrieve_url "${PATH_RETRIEVE_URL}" --llm_server_url "${LLM_SERVER_URL}"
    "$@"
  )
  [[ "${LIMIT}" != "0" ]] && cmd+=(--limit "${LIMIT}")
  printf ' %q' "${cmd[@]}"; echo
  "${cmd[@]}"
}

# 完整后处理 flag(canonical 用)
FULL_POST=(--check_mode hybrid-reject-list --score_margin "${SCORE_MARGIN}"
           --check_constrained_decoding --hop_filter --large_answer_expansion)

ensure_llm_server
# 复用已就绪的 path server(loopback 默认开);未就绪才启动。不无条件重启,避免
# 杀掉手动起的 server / 叠加重复进程。注:要确保是最新代码,先自行重启一次 path server。
ensure_path_server

for g in ${CH5_GROUPS}; do
  case "${g}" in
    canonical)
      run_eval canonical "${FULL_POST[@]}"
      ;;
    check)
      # loose-only(reject-answer-list)与 strict-only;hybrid 即 canonical
      run_eval check_loose_only \
        --check_mode reject-answer-list --score_margin "${SCORE_MARGIN}" \
        --check_constrained_decoding --hop_filter --large_answer_expansion
      run_eval check_strict_only \
        --check_mode strict-reject-list --score_margin "${SCORE_MARGIN}" \
        --check_constrained_decoding --hop_filter --large_answer_expansion
      ;;
    constrained)
      # 受限解码 off(其余同 canonical)
      run_eval no_constrained \
        --check_mode hybrid-reject-list --score_margin "${SCORE_MARGIN}" \
        --hop_filter --large_answer_expansion
      ;;
    ablation)
      # 后处理累积消融:逐层叠加 margin/hop/expansion,守卫留到最后一档(=canonical)
      # 才加,故前几档统一 --no_topic_guard,保证每档只多一层、归因干净。
      run_eval ablation_base \
        --check_mode hybrid-reject-list --no_topic_guard
      run_eval ablation_margin \
        --check_mode hybrid-reject-list --score_margin "${SCORE_MARGIN}" --no_topic_guard
      run_eval ablation_margin_hop \
        --check_mode hybrid-reject-list --score_margin "${SCORE_MARGIN}" --hop_filter --no_topic_guard
      run_eval ablation_margin_hop_exp \
        --check_mode hybrid-reject-list --score_margin "${SCORE_MARGIN}" \
        --hop_filter --large_answer_expansion --no_topic_guard
      # +守卫 = canonical(全开)
      ;;
    loopback)
      : # 见下方独立处理
      ;;
    *)
      echo "[WARN] 未知组: ${g}（忽略）"
      ;;
  esac
done

# ====== loopback off 对照(单独，需要切换 server）======
if [[ "${CH5_GROUPS}" == *loopback* ]]; then
  echo "[INFO] === loopback off 对照：重启 path server (PATH_DROP_LOOPBACK=0) ==="
  start_path_server 0
  run_eval no_loopback "${FULL_POST[@]}"
  echo "[INFO] === 还原 path server (PATH_DROP_LOOPBACK=1) ==="
  start_path_server 1
fi

echo "============================================================"
echo "[DONE] 全部完成。汇总对照表："
echo "  python scripts/collect_ch5_results.py ${OUTPUT_ROOT}"
echo "============================================================"
