# 乳腺 MRI / I-SPY2：VLA-JEPA 四组版本

整理原始 V1、独立重写的 V2、VQ 前端优化、后续 V1-evolution。**四组包含五个独立项目**：VQ 组同时保留单任务 encoder 和双任务 dualtask，未合并或删减其实现。fixed T0 / dynamic 是各版本的输入模式，不是 V1 / V2 的区别。

本仓库包含模型、训练和推理入口、数据准备与检查脚本、实验队列、配置、测试、许可证及已有聚合报告。原实验目录、患者数据和权重保持原状。发布目录迁移说明见 [来源与验证](docs/RELEASE_ZH.md)。

本机副本名为 **VLA-PCRV1**，位于 `/personal_fast_storage/VLA-PCRV1`，使用第 4 组 `v1_evolution`。
环境激活、迁移数据检查和一条命令运行真实 MRI 推理，见 [本机运行说明](docs/LOCAL_RUN_ZH.md)。
数据处理源码归档和重建输入，见 [数据生成代码](docs/PREPROCESSING_SOURCES_ZH.md)；本地版本管理见 [Git 工作流](docs/GIT_WORKFLOW_ZH.md)。

## 从哪里开始

| 组 | 代码与配置 | 主要区别 | 输入输出 |
| --- | --- | --- | --- |
| 1. 原始 V1 | [versions/v1](versions/v1/) | MRI 编码器；pCR 从观察历史读出，JEPA 可作为辅助任务 | [V1 说明](versions/v1/docs/INPUT_OUTPUT_ZH.md) |
| 2. V2 joint | [versions/v2_joint](versions/v2_joint/) | 多尺度编码器；direct / forecast / gated，未来状态可参与分类 | [V2 说明](versions/v2_joint/docs/INPUT_OUTPUT_ZH.md) |
| 3a. VQ encoder | [versions/vq/encoder](versions/vq/encoder/) | 替换预训练 VQ 前端，冻结 / 微调，pCR 单任务 | [encoder 说明](versions/vq/encoder/docs/INPUT_OUTPUT_ZH.md) |
| 3b. VQ dualtask | [versions/vq/dualtask](versions/vq/dualtask/) | VQ 前端加 pCR + JEPA 双任务；未来预测仍为辅助支路 | [dualtask 说明](versions/vq/dualtask/docs/INPUT_OUTPUT_ZH.md) |
| 4. V1-evolution | [versions/v1_evolution](versions/v1_evolution/) | 未来状态参与分类、历史差分、残差转移、自主多步监督等消融 | [evolution 说明](versions/v1_evolution/docs/INPUT_OUTPUT_ZH.md) |

**想找后期优化版，直接看第 4 组。**该轮最优组为 `vq_forecast_history_residual_free_encoder64`：

- [seed17 完整配置](versions/v1_evolution/configs/evolution_vq_forecast_history_residual_free_encoder64_seed17.yaml)、[seed43 完整配置](versions/v1_evolution/configs/evolution_vq_forecast_history_residual_free_encoder64_seed43.yaml)。
- [模型](versions/v1_evolution/src/mri_vla_jepa/model.py)、[MRI/VQ 编码器](versions/v1_evolution/src/mri_vla_jepa/encoder.py)、[训练与推理](versions/v1_evolution/src/mri_vla_jepa/training.py)。
- [20 次训练的最终汇总](versions/v1_evolution/historical_runs/v1_evolution_20261012/RESULTS_ZH.md)、[跨版本结果与选择依据](docs/EXPERIMENTS_ZH.md)。

它冻结 VQ 编码器，使用压缩前每相 64 通道特征；pCR 头结合观察查询、预测 T3 状态及其相对当前状态的变化。损失为 `BCE + 0.5 × teacher-forcing JEPA + 0.5 × 自主多步 JEPA`。batch=16、lr=1e-4、dropout=0.3、最多 200 epoch、早停 patience=50，种子 17/43。完整参数以链接中的 YAML 为准。

两种子平均开发验证 NLL=**0.545002**，T0 AUROC=**0.731696**。旧动态 V1 L1 对应为 0.549020 / 0.740625；NLL 有小幅降低，T0 AUROC 较低。全部历史实验使用 764 人训练、102 人反复使用的开发验证，**没有独立测试集结果**。这里的“最优”指该轮统一验证准则下的选择，不代表所有指标或外部泛化最好。

## 数据输入与输出

[统一数据说明](docs/DATA_INPUT_OUTPUT_ZH.md)介绍文件格式、张量、可见性和版本差异。核心关系是：

```text
已配准/裁剪/归一化 MRI NPZ + manifest + 临床信息 + 治疗 Arm
    → 当前阶段及之前合法可见的检查
    → 最终 pCR 概率；可选未来 MRI 潜状态（特征，不是图像）
```

一次 MRI 为 `float32 [3,32,128,128]`，三个 DCE 相位；完整 batch 为 `[B,4,3,32,128,128]`，4 表示纵向 T0–T3。临床输入为年龄、HR、HER2、MammaPrint，并有缺失/可见性 mask。正式实验没有生成未来 MRI 像素。推理不需要 pCR 真值或未来真实 MRI。

## 安装与运行

每个项目单独建环境，按对应 README 执行。V1/VQ/evolution 共享 Python 包名 `mri_vla_jepa`，不能把它们同时安装到一个环境。V2 包名为 `response_jepa`。

以最终优化版为例，从仓库根目录执行：

```bash
python3.11 -m venv versions/v1_evolution/.venv
source versions/v1_evolution/.venv/bin/activate
python -m pip install -e './versions/v1_evolution[raw,test]'
python tools/run_cpu_tests.py --project versions/v1_evolution
cd versions/v1_evolution
PYTHONPATH=src python train.py --help
```

真实训练须自行准备符合数据合同的 manifest、ROI32 和兼容外部 codec。历史 YAML 保留原机器路径，使用路径工具创建新的本地配置（下列资源必须实际存在）：

```bash
python ../../tools/prepare_config.py \
  --config configs/evolution_vq_forecast_history_residual_free_encoder64_seed17.yaml \
  --output local_configs/encoder64_seed17.yaml \
  --image-cache /path/to/image_cache \
  --codec /path/to/codec.pt
PYTHONPATH=src python train.py train-vla-jepa \
  --config local_configs/encoder64_seed17.yaml \
  --manifest /path/to/manifest.json \
  --output private_outputs/encoder64_seed17
```

`prepare_config.py` 只替换明确指定的资源路径，保留全部其他 YAML 值，并禁止覆盖已有配置。暖启动方案可用 `--init-checkpoint` 替换原有暖启动路径；不能借此把 fresh-start 变为暖启动。manifest 由训练入口传入。V2 最终实验按更新数运行，使用其文档中的 `scripts/train_overfit.py`。

CPU 工程测试使用合成数据；工具关闭部分 CPU BF16 融合后端以兼容不同处理器，不改变历史 CUDA 训练配置。测试不构成临床复现。正式数据、cache、codec、已训练权重、逐患者预测和原运行日志均未上传。

## 许可证

分别遵循各项目的 LICENSE / NOTICE。V1、V2 的项目代码使用 MIT；VQ 与 evolution 中派生 codec 前端保留 **CC BY-NC 4.0** 及署名，不能将整仓库一概视为 MIT。详见 [总 NOTICE](NOTICE.md)。代码许可证不授予患者数据或外部权重的使用权。
