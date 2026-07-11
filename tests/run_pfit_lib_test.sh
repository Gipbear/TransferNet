#!/usr/bin/env bash
# run_pfit.sh / run_pfit_lib.sh 的 dry-run 测试:校验命令拼装,不真跑训练。
# 运行: bash tests/run_pfit_lib_test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_PFIT="${PROJECT_DIR}/scripts/run_pfit.sh"
source "${PROJECT_DIR}/scripts/run_pfit_lib.sh"

assert_eq() {
    local actual="$1" expected="$2" label="$3"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "[FAIL] ${label}" >&2
        echo "  expected: ${expected}" >&2
        echo "  actual  : ${actual}" >&2
        exit 1
    fi
}

assert_contains() {
    local haystack="$1" needle="$2" label="$3"
    if [[ "${haystack}" != *"${needle}"* ]]; then
        echo "[FAIL] ${label}" >&2
        echo "  missing : ${needle}" >&2
        echo "  in      : ${haystack}" >&2
        exit 1
    fi
}

assert_not_contains() {
    local haystack="$1" needle="$2" label="$3"
    if [[ "${haystack}" == *"${needle}"* ]]; then
        echo "[FAIL] ${label}" >&2
        echo "  unexpected: ${needle}" >&2
        echo "  in        : ${haystack}" >&2
        exit 1
    fi
}

# ── 库函数:数据集上下文 ──────────────────────────────────────────────────────
if pfit_init_context "/tmp/x" "unknown" 2>/dev/null; then
    echo "[FAIL] unknown dataset should fail" >&2
    exit 1
fi

pfit_init_context "/tmp/x" "webqsp"
assert_eq "${PFIT_RETRIEVE_TRAIN}" "/tmp/x/data/output/kgqa/webqsp/retrieve/train.jsonl" "webqsp retrieve train"
assert_eq "${PFIT_RETRIEVE_TEST}" "/tmp/x/data/output/kgqa/webqsp/retrieve/test.jsonl" "webqsp retrieve test"

pfit_init_context "/tmp/x" "metaqa"
assert_eq "${PFIT_RETRIEVE_TRAIN}" "/tmp/x/data/output/kgqa/metaqa/retrieve/train_20k.jsonl" "metaqa retrieve train_20k"

# ── 库函数:实验目录后缀(FMT / ADAPTER / LIMIT)────────────────────────────
assert_eq "$(pfit_exp_dir "metaqa_main")" \
    "/tmp/x/data/output/kgqa/metaqa/pfit/metaqa_main" "exp dir 无后缀"
assert_eq "$(LIMIT=100 pfit_exp_dir "metaqa_main")" \
    "/tmp/x/data/output/kgqa/metaqa/pfit/metaqa_main_smoke100" "exp dir smoke 后缀"
assert_eq "$(FMT=v1 pfit_exp_dir "metaqa_base_zeroshot")" \
    "/tmp/x/data/output/kgqa/metaqa/pfit/metaqa_base_zeroshot_v1" "exp dir FMT 后缀"
assert_eq "$(ADAPTER=/a/b FMT=v3 LIMIT=50 pfit_exp_dir "metaqa_nopaths")" \
    "/tmp/x/data/output/kgqa/metaqa/pfit/metaqa_nopaths_v3_ft_smoke50" "exp dir 组合后缀"

# ── 入口:参数校验 ───────────────────────────────────────────────────────────
if bash "${RUN_PFIT}" 2>/dev/null; then
    echo "[FAIL] 缺 --exp 应失败" >&2; exit 1
fi
if bash "${RUN_PFIT}" --exp webqsp_main --phase bogus 2>/dev/null; then
    echo "[FAIL] 非法 --phase 应失败" >&2; exit 1
fi
if DRY_RUN=1 bash "${RUN_PFIT}" --exp not_registered 2>/dev/null; then
    echo "[FAIL] 未注册实验应失败" >&2; exit 1
fi

# ── dry-run:训练型实验全流水(webqsp_main, phase all)────────────────────────
ds_root="${PROJECT_DIR}/data/output/kgqa/webqsp"
out="$(DRY_RUN=1 bash "${RUN_PFIT}" --exp webqsp_main --phase all)"
dry_lines="$(grep -c '^DRY:' <<<"${out}")"
assert_eq "${dry_lines}" "3" "webqsp_main all 应有 build/train/eval 三条命令"
assert_contains "${out}" "DRY: python -m kgqa.pfit.build --dataset webqsp --input ${ds_root}/retrieve/train.jsonl --exp_dir ${ds_root}/pfit/webqsp_main --format v2 --path_format chain --entity_repr name" "webqsp_main build 命令"
assert_contains "${out}" "DRY: python -m kgqa.pfit.train --exp_dir ${ds_root}/pfit/webqsp_main --epochs 2" "webqsp_main train 命令"
assert_contains "${out}" "DRY: python -m kgqa.pfit.eval --dataset webqsp --input ${ds_root}/retrieve/test.jsonl --exp_dir ${ds_root}/pfit/webqsp_main --adapter ${ds_root}/pfit/webqsp_main/adapter --format v2 --path_format chain --entity_repr name" "webqsp_main eval 命令(默认 adapter=exp_dir/adapter)"

# ── dry-run:单阶段 phase 只出对应命令 ───────────────────────────────────────
out="$(DRY_RUN=1 bash "${RUN_PFIT}" --exp webqsp_main --phase train)"
assert_eq "$(grep -c '^DRY:' <<<"${out}")" "1" "phase train 仅一条命令"
assert_contains "${out}" "kgqa.pfit.train" "phase train 是训练命令"

# ── dry-run:LIMIT smoke(--sample/--limit 末位覆盖,目录带后缀)────────────
out="$(LIMIT=100 DRY_RUN=1 bash "${RUN_PFIT}" --exp metaqa_main --phase all)"
mq_root="${PROJECT_DIR}/data/output/kgqa/metaqa"
assert_contains "${out}" "--input ${mq_root}/retrieve/train_20k.jsonl" "metaqa build 输入 train_20k"
assert_contains "${out}" "--exp_dir ${mq_root}/pfit/metaqa_main_smoke100" "smoke 目录后缀"
assert_contains "${out}" "--sample 5000 --stratify_by_hop --sample 100" "LIMIT 追加 --sample 覆盖注册表值"
assert_contains "${out}" "--limit 100" "eval 带 --limit"
assert_contains "${out}" "kgqa.pfit.eval --dataset metaqa --input ${mq_root}/retrieve/test.jsonl" "metaqa eval 输入 test"

# ── dry-run:eval_only 实验跳过 build/train,base 零样本无 --adapter ─────────
out="$(DRY_RUN=1 bash "${RUN_PFIT}" --exp webqsp_base_zeroshot --phase all)"
assert_eq "$(grep -c '^DRY:' <<<"${out}")" "1" "eval_only 实验仅 eval 一条命令"
assert_not_contains "${out}" "--adapter" "base_zeroshot 不带 adapter"

# ── dry-run:FMT 变体(末位 --format 覆盖 + 目录后缀)─────────────────────────
out="$(FMT=v1 DRY_RUN=1 bash "${RUN_PFIT}" --exp webqsp_base_zeroshot --phase eval)"
assert_contains "${out}" "--exp_dir ${ds_root}/pfit/webqsp_base_zeroshot_v1" "FMT 目录后缀"
assert_contains "${out}" "--format v2 --path_format chain --entity_repr name --format v1" "FMT 末位覆盖 --format"

# ── dry-run:ADAPTER 覆盖(nopaths 微调变体)─────────────────────────────────
out="$(ADAPTER=/models/some_adapter DRY_RUN=1 bash "${RUN_PFIT}" --exp webqsp_nopaths --phase eval)"
assert_contains "${out}" "--exp_dir ${ds_root}/pfit/webqsp_nopaths_ft" "ADAPTER 目录后缀 _ft"
assert_contains "${out}" "--adapter /models/some_adapter" "ADAPTER 覆盖 adapter 路径"
assert_contains "${out}" "--no_paths" "nopaths 实验带 --no_paths"

# ── dry-run:8 个注册实验全部可解析 ──────────────────────────────────────────
for exp in webqsp_main webqsp_spot_nl webqsp_base_zeroshot webqsp_nopaths \
           metaqa_main metaqa_spot_nl metaqa_base_zeroshot metaqa_nopaths; do
    out="$(DRY_RUN=1 bash "${RUN_PFIT}" --exp "${exp}" --phase all)"
    assert_contains "${out}" "DRY: python -m kgqa.pfit.eval" "${exp} 至少产出 eval 命令"
done

echo "[PASS] run_pfit_lib"
