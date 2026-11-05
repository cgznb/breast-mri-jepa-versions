# v8 结果复算入口

本目录用于保存论文代码与最终结果数据。复算工具仅读取已复制的评估报告、实验配置、完成凭据和论文聚合表，不需要训练数据、数据 manifest 或模型权重，也不启动训练。

`manuscript/` 保留论文原档，包括全部聚合指标、区间、显著性结果、图表、来源说明及原始复算脚本。原档中的 `regenerate.sh` 包含历史绝对路径及训练数据依赖，本次结果包使用下面的独立入口核验结果。

## 快速核验点估计

在整理目录执行：

```bash
python tools/recompute_metrics.py
```

需要 `numpy` 和 `scikit-learn`，可用 `requirements_analysis.txt` 安装。工具逐项核验：

- 60 个拟合模型与 180 份注册报告；报告、配置和完成凭据按论文原始 provenance 执行 420 项 SHA256 核验。
- 所有 test 报告的患者和标签一致，三个时点均为 175 人、56 阳性。
- 各 fold validation 与 test 患者不重叠；每 seed/模型/时点拼接的五 fold OOF validation 患者唯一，且跨模型和 seed 的患者/标签向量一致。
- 两种阈值政策下的 18 项指标、36 个验证 OOF Youden 阈值、三时点 Macro、四 seed 指标均值和样本 SD；比较论文的 96 行逐 seed 指标及 24 行汇总指标，共 2,664 个数值。

结果写入 `outputs/metric_reproduction/`，只输出聚合 CSV 和 `verification.json`，不导出新的患者级预测。`--output` 必须位于整理目录的 `outputs/` 内，保护归档代码与结果原档。

主结果按“同 seed 五 fold 逐患者概率均值 → 各时点指标 → 该 seed 的 Macro → 四 seed 指标均值及样本 SD”计算。不会跨 seed 平均概率，也不会把 fold、seed 或时点当成独立患者扩充样本量。

## 核验范围

这个入口以独立 sklearn 估计器重算点指标和 seed SD。论文的 bootstrap 区间、99,999 次检验结果、999,999 次 Monte Carlo 敏感性结果及图表均作为冻结最终结果保留，未在本次整理中重新运行。

本工具核验报告层面的 split 标签、患者对齐、OOF 唯一性与 validation/test 排除关系。由于结果包不含训练数据 manifest，不会重新审计原始训练队列成员、临床输入、几何处理或预训练重叠。
