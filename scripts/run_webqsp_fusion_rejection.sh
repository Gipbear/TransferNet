#!/usr/bin/env bash
# WebQSP 三强拒答补齐流水线 (Mistral-7B 与 Qwen3.5-4B)
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || exit 1

export NO_PROXY="127.0.0.1,localhost"
LOG="data/output/kgqa/ch4_pfit/webqsp/transfernet_v2/fusion_rejection_queue.log"
mkdir -p "$(dirname "${LOG}")"

echo "=== [$(date '+%F %T')] 启动 WebQSP 三强拒答补齐流水线 ===" | tee -a "${LOG}"
python -m experiments.ch4.run \
  --dataset webqsp \
  --config experiments/configs/ch4/webqsp_rejection_blocks_fusion_v1.json \
  --profile experiments/configs/ch3/webqsp_transfernet_v2.json \
  --phase all >> "${LOG}" 2>&1
RC=$?

echo "=== [$(date '+%F %T')] 流水线结束 (exit code: ${RC}) ===" | tee -a "${LOG}"
exit ${RC}
