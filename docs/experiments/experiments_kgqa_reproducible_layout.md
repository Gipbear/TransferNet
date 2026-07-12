# KGQA 三章可复现实验与产物约定

## 适用范围

本文定义当前现役的第三章检索、第四章路径监督微调和第五章渐进验证实验。所有新产物
必须写入 `data/output/kgqa/`；`scripts/`、`llm_infer/` 和 `oh_my_agent/` 是历史保留实现，
不改变其旧产物位置。

## 统一目录

```text
data/output/kgqa/
├── shared/<数据集>/backbones/<基础检索模型>/scores/<得分编号>/
├── ch3_retrieval/<数据集>/<基础检索模型>/
│   ├── topk_saturation/<实验编号>/
│   └── confirmed_profiles/<配置编号>/
├── ch4_pfit/<数据集>/<配置编号>/<实验编号>/seed_<随机种子>/
└── ch5_pv_gac/<数据集>/<配置编号>/
    ├── smoke/
    ├── benchmark/
    ├── replay_ablations/
    ├── sensitivity/
    └── reports/
```

每个实际运行目录均写入：

- `run_manifest.json`：运行清单，包含命令、Git 提交、配置及输入来源；
- `progress.json`：可轮询的运行状态；
- `logs/run.log`：人可读日志；
- `logs/events.jsonl`：结构化阶段事件；
- `logs/console.log`：编排脚本所启动子命令的完整输出。

第四章目录中原有的 `manifest.json` 仍仅用于 build/train/eval 的断点续跑。

## 数据集与基础检索模型支持矩阵

| 章节 | WebQSP | MetaQA | CWQ | ReaRev |
|---|---|---|---|---|
| 第三章检索 | 支持 | 支持 | 支持 | 仅 WebQSP 离线得分缓存消费 |
| 第四章路径监督微调 | 支持 | 支持 | 当前不支持 | 当前不支持 |
| 第五章渐进验证 | 支持 | 支持 | 当前不支持 | 当前不支持 |

`topk=500` 是候选默认值，不是固定结论。第三章对 WebQSP、MetaQA、CWQ 均执行
`100/250/500/1000` 饱和性实验，随后由人工确认每个数据集的 top-k 与检索参数。
程序不会根据测试集指标自动选择配置。

## 配置与下游依赖

版本化 JSON 配置位于 `experiments/configs/`。第三章配置初始状态为 `draft`，完成审核后
必须填写中文确认理由并改为 `confirmed`；第四、五章入口会拒绝引用未确认配置。

第四章每个支持的数据集包含主实验、零样本基线、无路径基线、路径格式、输出格式和
训练规模比较。主实验及关键对照使用三个固定随机种子，其他消融使用一个随机种子。

第五章按数据集包含冒烟验证、基准正式评测、回放消融与参数敏感性。回放消融读取
基准正式评测目录，不重新调用语言模型，因此只比较确定性的后处理差异。
