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

# 用指定的 PATH_DROP_LOOPBACK 值(重)启 path server
start_path_server() {
  local drop="$1"  # 1=剔除 loopback(默认), 0=保留
  echo "[INFO] (重)启动 path_retrieve_server  PATH_DROP_LOOPBACK=${drop}"
  PATH_DROP_LOOPBACK="${drop}" PORT_BUSY_ACTION=kill PORT="${PATH_RETRIEVE_PORT}" \
    ./scripts/path_retrieve_server.sh restart
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
# 先以 loopback 开启重启 path server(保证加载最新代码 + PATH_DROP_LOOPBACK=1)
start_path_server 1

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
