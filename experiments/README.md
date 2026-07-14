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
- 固定 `threshold=0.01`，完整比较 `beam_size`、`lambda_val` 与终点实体融合权重 `eta` 的
  三维笛卡尔积。`lambda_val=0.0` 是无多样性惩罚对照；其余值控制 MMR 的关系集合重叠惩罚。
- `parameter_scan` 填写三个取值列表；编排器自动展开三维网格，并生成稳定候选编号。
  例如 `beam=50，λ=0.2，η=1.0` 的编号为 `beam50_lambda02_eta1`。
- `retrieve.eta` 是 top-k 饱和性评测使用的基准值；参数扫描中每个候选的 `eta` 会覆盖该值。
- `eta` 是论文中的终点实体分数融合权重。`alpha_final` 已废弃，现役命令、配置、接口和
  输出均不接受该字段。
- 不根据测试集指标自动选择配置。完成扫描后，由人工在配置中填写确认理由和
  `selected_candidate`，再发布给第四、五章使用。

### 运行步骤

#### WebQSP 完整复现顺序（本次已执行）

以下顺序以当前 `webqsp_transfernet_v1.json` 为准：`topk=500`、已确认参数组 `beam20_lambda05_eta15`（beam=20，λ=0.5，η=1.5）。耗时来自本机 `py312_t271_cuda` 环境的实际运行记录；checkpoint、GPU、磁盘缓存和数据加载状态不同会使时间波动，建议将其作为容量规划的参考而非严格上限。

| 步骤 | 命令或操作 | 本次实测 / 预计耗时 | 关键产物与完成判据 |
| --- | --- | --- | --- |
| 0. 核对配置 | 阅读配置、确认 checkpoint 与扫描范围 | < 1 分钟 | `experiments/configs/ch3/webqsp_transfernet_v1.json`；首次扫描前应为 `draft`，当前已完成确认的配置为 `confirmed`。 |
| 1. 演练 | `python -m experiments.run_ch3 --dataset webqsp --dry_run` | < 1 分钟 | 仅打印将执行的 score、top-k 评测和参数扫描命令；不加载模型。 |
| 2. score 与 top-k 饱和性 | `python -m experiments.run_ch3 --dataset webqsp --phase scores` | 约 26 分钟 | 生成 train/test 的 8 份 score 缓存，及 8 份 top-k 汇总：`topk_saturation/transfernet_v1/topk{100,250,500,1000}_{train,test}/*_summary.json`。本次 max-topk 前向：train 4 分 40 秒、test 2 分 37 秒；其余小 top-k 由 topk=1000 缓存裁剪。 |
| 3. 检索参数扫描 | `python -m experiments.run_ch3 --dataset webqsp --phase scan` | 约 3 小时 7 分钟 | 当前网格为 6×7×3=126 组；每组输出 1,581 条测试结果与汇总至 `confirmed_profiles/transfernet_v1/candidates/<参数组>/test.{jsonl,summary.json}`。批处理日志在 `topk_saturation/transfernet_v1/parameter_scan/batch/logs/console.log`。 |
| 4. 人工确认 | 比较路径命中、路径 F1、多样性并编辑配置 | 约 5–15 分钟 | 将配置设为 `status=confirmed`，填写 `confirmation_reason`、`selected_candidate` 和对应 `retrieve` 参数。当前已选 `beam20_lambda05_eta15`。 |
| 5. 发布正式上游产物 | `python -m experiments.run_ch3 --dataset webqsp --phase publish` | 首次约 3 分钟；已有候选 train/test 时仅需数秒 | 正式产物为 `confirmed_profiles/transfernet_v1/{train,test}.jsonl` 与 `confirmed_config.json`；发布目录 `publish/progress.json` 应为 `completed`。第四、五章只引用这些正式文件。 |

若希望连续运行第 2、3 步，可使用下列快捷命令；本次总耗时约 3 小时 35 分钟。它**不会**替代第 4 步人工确认，也不会自动执行发布：

```bash
python -m experiments.run_ch3 --dataset webqsp --phase all
```

在进入下一步前，可用以下检查确认上一步完成：

```bash
# 第 2 步：查看 top-k 饱和性汇总
cat data/output/kgqa/ch3_retrieval/webqsp/transfernet/topk_saturation/transfernet_v1/topk500_test/test_summary.json

# 第 3 步：查看已确认候选的扫描汇总（人工确认前仍位于 candidates/）
cat data/output/kgqa/ch3_retrieval/webqsp/transfernet/confirmed_profiles/transfernet_v1/candidates/beam20_lambda05_eta15/test_summary.json

# 第 5 步：确认发布状态和正式测试集
cat data/output/kgqa/ch3_retrieval/webqsp/transfernet/confirmed_profiles/transfernet_v1/publish/progress.json
wc -l data/output/kgqa/ch3_retrieval/webqsp/transfernet/confirmed_profiles/transfernet_v1/test.jsonl
```

#### 通用命令参考

```bash
# 0. 先核对 WebQSP 配置：在以下文件填写或确认 checkpoint；其余数据集也在 configs/ch3/。
#    如需改默认 top-k 或扫描范围，直接编辑 retrieve 与 parameter_scan 字段。
sed -n '1,160p' experiments/configs/ch3/webqsp_transfernet_v1.json

# 1. 演练：只展示 score 缓存、top-k 评测和“参数组数×数据划分数”的参数扫描任务。
python -m experiments.run_ch3 --dataset webqsp --dry_run

# 2. 实际运行：先生成并评测 top-k 饱和性缓存，再运行 beam/λ/eta 完整对比。
python -m experiments.run_ch3 --dataset webqsp --phase all

# 3. 审核每组 train/test 汇总指标与日志（示例为 beam=50、λ=0.2、eta=1.0）。
cat data/output/kgqa/ch3_retrieval/webqsp/transfernet/confirmed_profiles/transfernet_v1/\
candidates/beam50_lambda02_eta1/test_summary.json

# 4. 人工确认：在 JSON 中填写 confirmation_reason，设 status 为 confirmed，并令
#    selected_candidate 为某个参数组 ID（如 beam50_lambda02_eta1）。随后发布正式检索结果。
python -m experiments.run_ch3 --dataset webqsp --phase publish
```

`parameter_scan` 的配置形式如下；新增 beam、λ 或 eta 取值只需在对应列表中添加一个数值。
总参数组数为三个列表长度的乘积：

```json
"parameter_scan": {
  "beam_size": [20, 50, 100],
  "lambda_val": [0.0, 0.1, 0.2, 0.3, 0.5],
  "eta": [0.5, 1.0, 1.5]
}
```

可单独执行 `--phase scores`、`--phase scan` 或 `--phase publish`，便于中断后按阶段恢复。
每个任务目录都有 `run_manifest.json`、`progress.json`、`logs/run.log`、
`logs/events.jsonl` 和 `logs/console.log`。第三章产物如下：

运行时会显示三类进度：第三章任务总数、score 前向样本数和逐题路径检索数；进度文件默认每
50 题原子更新一次。终端不需要进度条时，可传 `--no_progress`；可用
`--progress_interval 100` 调整进度文件更新间隔。

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
