# MIA v8：代码与结果数据整理包

本目录对应 `mia_mean_seed42_44_71_2026_probmean_v8`，总体模型为 **ForecastMean（Medium）**。按作者要求，本包仅整理源码、实验配置和结果数据，不包含 MRI/NPZ、训练输入清单、临床原始表、预训练 codec 或模型权重。

这是独立的文件副本；原论文、实验目录和训练输入均保留。整包位于项目 Git 忽略的 `private_outputs/` 下。逐患者评估报告保留在本地包内，以便从原始预测复算论文结果。

## 目录与使用顺序

| 目录／文件 | 内容 |
|---|---|
| `results/aggregated/` | 论文最终的 29 份 CSV/JSON 结果与统计凭据 |
| `results/figures/` | 8 组论文图的 PDF、PNG 和框架图说明 |
| `results/tables/` | 论文 LaTeX 表格 |
| `experiments/seed*/runs/` | 选定 60 次训练的原始测试／验证报告、诊断、历史和完成凭据 |
| `configs/` | 60 份真实运行配置；`model_index.csv` 对照模型、seed、fold 和报告路径 |
| `code/runtime/src/mri_vla_jepa/` | 实际训练使用的 92 文件冻结 runtime，逐字保留 |
| `code/analysis_original/` | 原论文的指标、统计和绘图源码 |
| `code/original_scheduler_tools/` | Medium976 原调度工具源码快照 |
| `code/environment_reference/` | 当前项目的依赖描述及许可证参考；不把它冒称为历史环境锁文件 |
| `manuscript/` | v8 原稿完整快照，含 PDF、TeX、Overleaf ZIP 和统计分析来源记录 |
| `tools/` | 可独立运行的结果复算及文件校验工具 |
| `provenance/` | 来源映射、哈希、范围说明和整理核验记录 |

先看 `results/aggregated/extended_summary.csv`（完整汇总）、`four_seed_individual_metrics.csv`（逐 seed）、`significance_tests.csv`（统计比较）和 `configs/model_index.csv`（实验索引）。

## 对应的模型与网络

三模型为 NoJEPA、DirectReadout、ForecastMean；训练 seed 为 **42、44、71、2026**，每个 seed 各 5 折，共 **60 个拟合模型**。

主类：`code/runtime/src/mri_vla_jepa/model.py` 的 `RawVLAJEPA`。关键导航：

- `encoder.py`、`vq_encoder.py`：三相共享 3D CNN 连续影像编码，形成 32×768 token。
- `attention.py`：空间、融合和时间因果预测 Transformer。
- `model.py:100`：整体模型；`:237`：ForecastMean 读出；`:443`：自主未来状态 rollout；`:472`：rollout loss。
- `training.py:352`：训练流程；`:748`：评估流程。

空间、融合、预测 Transformer 各 8 层。ForecastMean 拼接历史 pCR query 均值、预测 T3 状态空间均值和“预测 T3 − 当前 EMA 状态”的空间均值，构成 2304 维输入，经 LayerNorm、Linear、sigmoid 输出 pCR 概率。预测对象是未来潜在特征；Mean 指最终空间池化，内部保留 self-attention。

| 模型 | 分类头输入 | 自主 rollout L1 权重 |
|---|---|---:|
| NoJEPA | 历史特征，768 维 | 0 |
| DirectReadout | 历史特征，768 维 | 0.5 |
| ForecastMean | 历史＋预测终末状态＋预测变化，2304 维 | 0.5 |

三模型 teacher-forced 单步 JEPA 权重均为 0。NoJEPA 仍保留共享融合与 transition queries。四个实验 suite 的冻结 runtime 已按相对路径核验为一致，冻结 `model.py` 与 v8 来源审计的 SHA256 相同。

## 独立复算结果

推荐 Python 3.11 或以上。在本目录执行：

```bash
python -m pip install -r requirements_analysis.txt
python tools/recompute_metrics.py
python tools/verify_bundle.py
```

也可使用已有环境：

```bash
/personal_fast_storage/VLA-PCRV1/versions/v1_evolution/.venv/bin/python tools/recompute_metrics.py
```

结果复算只读取包内的评估 JSON 和结果 CSV，不依赖训练数据、模型权重、GPU 或旧绝对路径。它核对报告来源、测试患者标签对齐、验证／测试排除与 OOF 唯一性，复算两种阈值政策、18 项指标、Macro、四 seed 均值及样本 SD。新文件写入 `outputs/metric_reproduction/`；原结果不覆盖。

`tools/verify_bundle.py` 核对 `SHA256SUMS` 中的全部文件。以后新增的 `outputs/` 是派生输出，不纳入原始整理包的文件封印。

原论文分析脚本完整保留，其中部分包含原环境绝对路径，`manuscript/regenerate.sh` 也包含从训练输入读取亚组信息和渲染 MRI 示例的步骤。因此本包提供上面的独立结果复算入口。现成 bootstrap 区间、显著性检验、亚组结果、DCA、诊断和图表全部保留；快速复算工具不重新执行 bootstrap 或亚组分析。

源码和配置可用于检查训练方法；重新训练或执行模型推理还需要本包明确排除的输入数据和权重。原始配置／receipt 内的旧路径作为实验来源记录保留。

## 统计口径和来源边界

每个模型、每个 seed 内，先逐患者平均 5 折概率，再计算 T0/T1/T2 指标；Macro 为三个时点指标的等权均值；最后报告 4 个 seed 指标的均值及样本标准差（ddof=1）。不跨 seed 平均概率，也不先平均五折指标。

所有模型复用同一内部 downstream holdout：175 人、56 阳性；fold、seed、时点不增加独立患者数。模型和 seed 集合为既有结果可见后的回顾性选择。预训练患者重叠、外部独立性等未解决事项沿用原稿说明。

原 `experiments/seed*/matrix.json` 覆盖对应历史 suite 的全部任务；本包实际纳入的 60 个任务以 `configs/model_index.csv` 和 `provenance/selected_runs.json` 为准。训练诊断是固定 training probe 结果，不等同于独立预测验证。

## 本次整理核验

60 个任务、180 份注册报告和 420 项报告来源 SHA256 核验通过。独立复算的 2,664 项指标、阈值和 seed SD 与原稿一致，最大绝对误差为 `2.22e-16`。详情见 `provenance/verification.json`。
