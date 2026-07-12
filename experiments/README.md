# KGQA 可复现实验入口

本目录是第三、四、五章的现役实验编排入口。历史保留脚本仍在 `scripts/`，仅用于
论文复现核对；新实验不得再向旧目录写入产物。本文档按章节补充运行方式；当前先给出
第三章。

## 术语

- 参数扫描：在固定得分缓存下比较检索参数。
- 已确认检索配置：第三章完成参数扫描后人工确认的配置；第四、五章只能引用此类配置。
- 基准正式评测：完整测试集评测，不等同于冒烟验证。
- 回放消融：读取基准正式评测记录进行确定性后处理，不重复调用语言模型。
- 运行清单：每次运行目录中的 `run_manifest.json`，记录命令、版本、配置和输入来源。

路径和命令行参数使用英文标识，但所有配置说明、注释和结果报告均使用中文。

## 第三章：检索实验

### 实验设计

- 每个数据集先生成 `topk={100,250,500,1000}` 的 train/test score 缓存，用于观察
  top-k 饱和性。默认下游缓存为 `topk=500`，但应根据饱和性结果人工确认是否调整。
- 固定 `threshold=0.01`、终点实体融合权重 `eta=1.0`，完整比较
  `beam_size={20,50,100}` 与 `lambda_val={0.0,0.1,0.2,0.3,0.5}` 的笛卡尔积，共 15 组。
  `lambda_val=0.0` 是无多样性惩罚对照；其余值控制 MMR 的关系集合重叠惩罚。
- `eta` 是论文中的终点实体分数融合权重，替代旧称 `alpha_final`。现役命令和配置使用
  `eta`；旧参数 `--alpha_final` 仍可被读取，仅为兼容历史脚本。
- 不根据测试集指标自动选择配置。完成扫描后，由人工在配置中填写确认理由和
  `selected_candidate`，再发布给第四、五章使用。

### 运行步骤

```bash
# 0. 先核对 WebQSP 配置：在以下文件填写或确认 checkpoint；其余数据集也在 configs/ch3/。
#    如需改默认 top-k 或扫描范围，直接编辑 retrieve 与 parameter_scan 字段。
sed -n '1,160p' experiments/configs/ch3/webqsp_transfernet_v1.json

# 1. 演练：只展示 8 个 score 缓存任务、8 个 top-k 评测任务和 15×2 个参数扫描任务。
python -m experiments.run_ch3 --dataset webqsp --dry_run

# 2. 实际运行：先生成并评测 top-k 饱和性缓存，再运行 beam/λ 完整对比。
python -m experiments.run_ch3 --dataset webqsp --phase all

# 3. 审核每组 train/test 汇总指标与日志（示例为 beam=50、λ=0.2）。
cat data/output/kgqa/ch3_retrieval/webqsp/transfernet/confirmed_profiles/transfernet_v1/\
candidates/beam50_lambda02/test_summary.json

# 4. 人工确认：在 JSON 中填写 confirmation_reason，设 status 为 confirmed，并令
#    selected_candidate 为某个参数组 ID（如 beam50_lambda02）。随后发布正式检索结果。
python -m experiments.run_ch3 --dataset webqsp --phase publish
```

可单独执行 `--phase scores`、`--phase scan` 或 `--phase publish`，便于中断后按阶段恢复。
每个任务目录都有 `run_manifest.json`、`progress.json`、`logs/run.log`、
`logs/events.jsonl` 和 `logs/console.log`。第三章产物如下：

```text
data/output/kgqa/
├── shared/webqsp/backbones/transfernet/scores/
│   └── topk{100,250,500,1000}_{train,test}/
└── ch3_retrieval/webqsp/transfernet/
    ├── topk_saturation/transfernet_v1/
    │   ├── topk{100,250,500,1000}_{train,test}/{score,evaluation}/
    │   └── parameter_scan/<参数组>/<split>/
    └── confirmed_profiles/transfernet_v1/
        ├── candidates/<参数组>/{train,test}.jsonl
        ├── candidates/<参数组>/{train,test}_summary.json
        ├── {train,test}.jsonl                 # 仅人工确认并发布后产生
        └── confirmed_config.json               # 同上
```
