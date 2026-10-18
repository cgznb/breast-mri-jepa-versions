# V1：乳腺 MRI / I-SPY2 VLA-JEPA 原版

本包位于 `versions/v1/`，保留第一版模型、数据管线、训练/评价/预测入口、全部 A0/A1/A2 与 L1–L4 配置及测试。固定 T0 和动态 T0–T3 是本版的两种观察协议，共用代码。原始 MRI 编码器从零训练，没有加载 Qwen3-VL/V-JEPA2 权重。

| 协议 | 可见输入 | pCR 输出 |
| --- | --- | --- |
| 固定 T0 | 仅 T0 三相 MRI + 当时可见临床/Arm | 最终 pCR 概率 |
| 动态 | 截至 T0/T1/T2/T3 的真实观察历史 + 当时可见临床/Arm | 同一最终 pCR 终点的更新概率 |

完整输入为 `[B,4,3,32,128,128]`：4 是纵向阶段，3 是每次检查的 DCE 相位。输入、mask、监督边界、未来特征 shape 与可执行相对路径命令见 **[输入输出说明](docs/INPUT_OUTPUT_ZH.md)**。

## 结果与配置

[全部聚合结果](results_summary/README.md)包含 28 次训练、70 条阶段分类结果和已有世界模型评价。best 按预定开发验证 NLL 选取，不按阶段 AUROC 重选：

| 按两种子平均 NLL 的方案 | seed17 配置 | 平均选模 NLL | 平均 T0 AUROC |
| --- | --- | ---: | ---: |
| 固定 T0：A1 | [A1 T0](configs/registered_roi32_epoch200_a1_t0_seed17.yaml) | 0.5355 | 0.7386 |
| 动态：L1，pCR-only | [L1 dynamic](configs/registered_roi32_epoch200_l1_dynamic_seed17.yaml) | 0.5490 | 0.7406 |

均另有 seed43。动态 L1 未训练世界头；需要世界状态预测应选择有正 JEPA 权重的方案。固定和动态的选模 NLL 定义不同，不能直接用于判断随访收益。[实验框架](docs/EXPERIMENTS_ZH.md)解释损失消融；[架构说明](docs/ARCHITECTURE_ZH.md)解释两条任务路径。

历史成绩来自 764 人训练、102 人开发验证，无独立测试。本次不重新训练或追加患者推理。源代码/配置与聚合结果已整理，正式 checkpoint、真实 manifest、MRI、临床表和患者预测未附带。配置保留历史科学参数和机器路径；按输入输出说明用根目录 `tools/prepare_config.py` 生成本机新配置后运行。

## 安装与入口

各版本使用同名 Python 包 `mri_vla_jepa`，应在各自目录/独立虚拟环境运行，避免互相覆盖。Python ≥3.11：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[raw,test]'
PYTHONPATH=src python train.py --help
PYTHONPATH=src python -m mri_vla_jepa predict-vla-jepa --help
```

正式训练需匹配机器的 PyTorch/CUDA；CLI 的 `evaluate-vla-jepa`/`predict-vla-jepa` 支持 `--device cpu`。只使用虚构数据的 CPU 示例：

```bash
PYTHONPATH=src python train.py smoke-vla-jepa \
  --config configs/raw_vla_jepa_smoke_dynamic.yaml --output runs/my_synthetic_smoke
```

输出目录需是新目录。现有测试为 `PYTHONPATH=src python -m pytest -q`；本次 CPU 验证的环境、后端兼容设置与结果见[发布验证记录](docs/RELEASE_VALIDATION_ZH.md)。真实训练及预测命令见[输入输出说明](docs/INPUT_OUTPUT_ZH.md#本版方案与命令)。

## 阅读与来源

建议按 `contracts.py → data.py → encoder.py → model.py → training.py → cli.py` 阅读。[数据合同](docs/DATA_ZH.md)、[ROI32 适配](docs/REGISTERED_ROI32_ZH.md)和[调试指南](docs/DEBUG_WALKTHROUGH_ZH.md)提供详细上下文。历史说明中的外部运行记录不都随本仓库发布，当前结果导航以 `results_summary/` 为准。

许可见 [LICENSE](LICENSE)；版权声明保持原样。本包没有旧项目的 legacy codec/DiT/MONAI 组件，已删除原 LICENSE 中指向这些未纳入文件的失实例外说明。具体来源与提取范围见 [SOURCES](docs/SOURCES.md)。
