# KGQA 可复现实验入口

本目录是第三、四、五章的现役实验编排入口。历史保留脚本仍在 `scripts/`，仅用于
论文复现核对；新实验不得再向旧目录写入产物。

## 术语

- 参数扫描：在固定得分缓存下比较检索参数。
- 已确认检索配置：第三章完成参数扫描后人工确认的配置；第四、五章只能引用此类配置。
- 基准正式评测：完整测试集评测，不等同于冒烟验证。
- 回放消融：读取基准正式评测记录进行确定性后处理，不重复调用语言模型。
- 运行清单：每次运行目录中的 `run_manifest.json`，记录命令、版本、配置和输入来源。

路径和命令行参数使用英文标识，但所有配置说明、注释和结果报告均使用中文。

## 运行顺序

1. 在 `configs/ch3/` 中填写实际 checkpoint 和 CWQ QA 文件；先执行 top-k 饱和性实验及参数扫描。
2. 审核候选结果，填写 `confirmation_reason`，并将对应第三章配置的 `status` 改为
   `confirmed`，同时填写 `selected_candidate`。这一步是人工确认，脚本不会根据测试集
   指标自动修改这些字段。
3. 发布被确认候选的 train/test JSONL。第四、五章只读取此正式目录。
4. 使用该配置运行第四章或第五章。未确认的配置会被拒绝。

```bash
# 仅展示第三章将运行的命令和输出位置
python -m experiments.run_ch3 --dataset webqsp --dry_run

# 生成 WebQSP 的 top-k 得分缓存和检索参数扫描结果
python -m experiments.run_ch3 --dataset webqsp --phase all

# 人工确认后，将所选候选结果发布为第四、五章的正式上游输入
python -m experiments.run_ch3 --dataset webqsp --phase publish

# 第四章：主实验的三个随机种子
python -m experiments.run_ch4 \
  --dataset webqsp \
  --config experiments/configs/ch4/webqsp_v1.json \
  --profile experiments/configs/ch3/webqsp_transfernet_v1.json \
  --experiment 主实验

# 第五章：基准正式评测；路径服务和语言模型服务需先就绪
python -m experiments.run_ch5 \
  --dataset webqsp \
  --config experiments/configs/ch5/webqsp_v1.json \
  --profile experiments/configs/ch3/webqsp_transfernet_v1.json \
  --phase benchmark
```

所有新产物位于 `data/output/kgqa/`：共享 score 缓存位于 `shared/`，第三章位于
`ch3_retrieval/`，第四章位于 `ch4_pfit/`，第五章位于 `ch5_pv_gac/`。
