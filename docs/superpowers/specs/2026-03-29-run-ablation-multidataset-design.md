# Multi-Dataset `run_ablation.sh` Design

## Context

`scripts/run_ablation.sh` currently assumes a single dataset layout rooted at `data/output/WebQSP` and a single baseline adapter rooted at `models/webqsp/webqsp_v2`. That works for the original Chapter 4 WebQSP workflow, but it prevents the same orchestration script from evaluating MetaQA_KB and CWQ retrieval outputs.

The lower-level evaluation stack is already close to dataset-agnostic:

- `llm_infer/eval_faithfulness.py` only requires input JSONL records with `question`, `mmr_reason_paths`, and `golden`
- `MetaQA_KB/predict.py` and `CompWebQ.predict` already emit the same key fields
- `scripts/run_grid.sh` already supports `webqsp`, `metaqa`, and `cwq`

The missing piece is the ablation runner: it needs dataset selection, model-source selection, and predictable fallback behavior when dataset-specific fine-tuned adapters do not exist.

## Goal

Extend `scripts/run_ablation.sh` so one command can run the existing ablation workflow against `webqsp`, `metaqa`, or `cwq` path files while preserving the current WebQSP default behavior.

The script should prefer dataset-matched adapters when available, but automatically fall back to the corresponding WebQSP adapter for the same experiment slot when they are missing.

## Non-Goals

- No changes to `llm_infer/eval_faithfulness.py`
- No changes to `scripts/collect_ablation_results.py` in this task
- No redesign of the Group A / B / C experimental definitions
- No conversion of the bash script into Python

## User-Facing Interface

`scripts/run_ablation.sh` will keep existing defaults and add one new dataset selector plus one model-source selector:

- `--dataset webqsp|metaqa|cwq`
  - Selects which dataset's JSONL inputs and output directories are used
  - Default: `webqsp`
- `--model_dataset webqsp|metaqa|cwq`
  - Selects which dataset's adapter namespace is checked first
  - Default: `webqsp`
- Existing `--group` and `--phase` remain unchanged

Examples:

```bash
# Backward-compatible default
bash scripts/run_ablation.sh

# Evaluate MetaQA_KB paths using MetaQA adapters if present, else WebQSP fallback
bash scripts/run_ablation.sh --dataset metaqa --phase eval

# Evaluate CWQ paths while explicitly preferring WebQSP adapters
bash scripts/run_ablation.sh --dataset cwq --model_dataset webqsp --group C --phase eval
```

## Dataset Mapping

The script will normalize dataset keys to existing directory names:

| CLI value | Output root | Path root | Predict train |
|-----------|-------------|-----------|---------------|
| `webqsp` | `data/output/WebQSP` | `data/output/WebQSP/grid_search/paths` | `data/output/WebQSP/predict_train.jsonl` |
| `metaqa` | `data/output/MetaQA_KB` | `data/output/MetaQA_KB/grid_search/paths` | `data/output/MetaQA_KB/predict_train.jsonl` if present |
| `cwq` | `data/output/CWQ` | `data/output/CWQ/grid_search/paths` | `data/output/CWQ/predict_train.jsonl` if present |

The script should centralize this mapping in one `case` block so the rest of the logic uses dataset-derived variables instead of hard-coded `WebQSP` paths.

## Model Resolution Design

### Target Model Layout

Models are organized under a dataset namespace:

```text
models/
  webqsp/
    webqsp_v2/
    ablation/
      groupA_v1/
      groupA_v3/
      groupA_v4/
      groupA_v5/
      groupB_noshuffle/
      groupB_noscore/
      groupB_dist0.3/
      groupB_dist0.5/
  metaqa/
    metaqa_v2/
    ablation/
      ...
  cwq/
    cwq_v2/
    ablation/
      ...
```

The existing WebQSP baseline adapter should be renamed from `models/webqsp_v2_best` to `models/webqsp/webqsp_v2`.

### Experiment Slots

The adapter lookup must be slot-aware rather than globally pinned to `models/webqsp/webqsp_v2`.

The slots are:

- Baseline slot:
  - Group A `v2`
  - Group C evaluation
- Per-format slot:
  - Group A `v1`
  - Group A `v3`
  - Group A `v4`
  - Group A `v5`
- Per-training-ablation slot:
  - Group B `noshuffle`
  - Group B `noscore`
  - Group B `dist0.3`
  - Group B `dist0.5`

### Resolution Rule

For every slot, resolve the adapter in this order:

1. Dataset-specific adapter for `--model_dataset`
2. Matching WebQSP adapter for the same slot
3. Fail with a clear error if neither exists

Examples:

- If `--dataset metaqa --model_dataset metaqa` and `groupA_v5` is requested:
  - First try MetaQA's `groupA_v5` adapter
  - If absent, fall back to WebQSP's `groupA_v5` adapter
- If `groupC` is requested:
  - First try the model-dataset `v2` adapter
  - If absent, fall back to `models/webqsp/webqsp_v2`

### Path Convention

The script should use dataset-namespaced adapters under `models/<dataset>/...`.

Recommended lookup order:

- Baseline:
  - `models/<model_dataset>/<model_dataset>_v2`
  - `models/webqsp/webqsp_v2`
- Ablation adapters:
  - `models/<model_dataset>/ablation/<config_name>`
  - `models/webqsp/ablation/<config_name>`

This keeps all dataset-specific models under the same top-level namespace and makes fallback behavior explicit.

## Execution Behavior

### WebQSP Default

When no new arguments are provided:

- Dataset remains `webqsp`
- Model dataset remains `webqsp`
- Existing file paths and resulting behavior stay unchanged

This is the backward-compatibility requirement.

### Group A / B

Group A and Group B keep their current semantics:

- `phase=all` or `phase=train` still performs dataset build and training
- `phase=eval` skips build and training
- Evaluation input remains fixed to `beam20_lam0.2.jsonl` for the selected dataset

Training outputs should be written under dataset-aware ablation model/data directories to avoid collisions between WebQSP and non-WebQSP runs.

Recommended output layout:

- Data:
  - `data/output/<DatasetRoot>/ablation/...`
- Models:
  - `models/<dataset_key>/ablation/<config_name>`

### Group C

Group C remains eval-only and uses the selected dataset's `grid_search/paths` directory:

- beam scan: `5, 10, 15, 30`
- baseline: `beam20_lam0.2`
- lambda scan at `beam20`

The adapter for Group C is resolved from the baseline slot, not from any Group A/B training directory.

## Logging and Error Handling

The script should print the resolved runtime context at startup:

- `DATASET`
- `MODEL_DATASET`
- selected roots for train/path/ablation output
- resolved baseline adapter

When resolving an adapter for a slot, log which path was selected and whether it was:

- dataset-specific
- WebQSP fallback

If a dataset lacks a required training input for `phase=all` or `phase=train`, fail with a direct message naming the missing file. `phase=eval` should only require the selected test JSONL plus a resolvable adapter.

If a Group C path file is missing, keep the current behavior of warning and skipping that point.

## Testing Strategy

### Script-Level Checks

- `bash -n scripts/run_ablation.sh`
- Run `--phase eval` on `webqsp`, `metaqa`, and `cwq`
- Verify that startup logs show the correct dataset roots and adapter resolution path

### Functional Checks

1. Backward compatibility:
   - `bash scripts/run_ablation.sh --group C --phase eval`
   - Should behave like the current WebQSP default
2. MetaQA fallback:
   - `bash scripts/run_ablation.sh --dataset metaqa --group C --phase eval`
   - Should use MetaQA path files and fall back to the corresponding WebQSP baseline adapter when no MetaQA adapter exists
3. CWQ fallback:
   - `bash scripts/run_ablation.sh --dataset cwq --group C --phase eval`
   - Should use CWQ path files and apply the same fallback rule

### Safety Checks

- Confirm that non-WebQSP outputs land in dataset-specific directories
- Confirm that default WebQSP execution still works after the model path migration
- Confirm that `v5` resolves to the `groupA_v5` adapter slot rather than the generic `v2` adapter

## Implementation Notes

- Keep the current `run_experiment` / `run_eval_only` structure rather than rewriting the script wholesale
- Add small helper functions for:
  - parsing `--dataset` and `--model_dataset`
  - dataset path initialization
  - slot-based adapter resolution
- Avoid duplicated `case` logic in each group section

## Risks

- The model path migration from `models/webqsp_v2_best` to `models/webqsp/webqsp_v2` must be reflected consistently in the script and any local run instructions
- `predict_train.jsonl` may not exist for every dataset; build/train phases must check this separately from eval-only phases
- If the fallback logic is too coarse, `v5` and other slot-specific experiments could accidentally use the wrong adapter and silently invalidate the comparison

## Open Decisions Already Resolved

- Dataset switching is required: yes
- WebQSP remains the default dataset and model source: yes
- Missing dataset-specific adapters fall back to WebQSP by matching experiment slot: yes
- This task stops at `run_ablation.sh`; result collection stays out of scope: yes
