# VQ encoder：冻结/微调连续前端的 pCR 单任务实验

本包位于 `versions/vq/encoder/`，属于 VQ 优化版本的一组实验。保留完整模型、训练/预测代码、冻结与微调配置、队列工具和已有测试。原 V1 的 MRI 卷积前端被已有 VQ-GAN 的连续 `encoder + pre_quant` 替换；三相共用单通道前端，不运行量化器、码本、解码器或 GAN 判别器。

输入仍是三相纵向 MRI `[B,4,3,32,128,128]`，加临床年龄/HR/HER2/MP、Arm 和可用性 mask。输出是最终 pCR 概率；本实验仅用 BCE 训练，**没有受训练的未来状态预测头**。逐字段/shape、训练与推理边界、冻结→微调命令见 **[输入输出说明](docs/INPUT_OUTPUT_ZH.md)**。

## 结果与配置

四次 dynamic 训练已完成，种子 17/43。下表是 best checkpoint 两种子的平均开发验证 NLL，越低越好：

| 方案 | NLL | seed17 配置 |
| --- | ---: | --- |
| 既有 V1 dynamic L1 | 0.549020 | [V1 L1](../../v1/configs/registered_roi32_epoch200_l1_dynamic_seed17.yaml) |
| 冻结 VQ | 0.583057 | [frozen](configs/vq_frozen_dynamic_seed17.yaml) |
| 冻结 best 后微调 | 0.587042 | [finetune](configs/vq_finetune_dynamic_seed17.yaml) |

VQ 替换未改善本轮主要指标。完整结果见[聚合结果](results_summary/VQ_ENCODER_RESULTS_20261012_ZH.md)。可选 T0 配置也保留，但不是这四次已完成 dynamic 训练。

冻结阶段只固定预训练前端，适配层和后续网络仍训练，学习率 `1e-4`；微调从同种子冻结 best 继承完整模型，前端 `1e-5`、其他参数 `1e-4`，重新建立优化器与训练进度。最多 200 epoch、batch16、patience50，选轮使用患者等权的平均合法前缀 pCR NLL。

历史数据为 764 人训练、102 人开发验证、无独立测试。本包不附正式 codec/checkpoint、临床表、MRI、manifest 或患者预测；机器路径保留在历史配置中，运行时应使用根目录 `tools/prepare_config.py` 生成本机配置。

## 安装与运行

本包与其他版本的导入名均为 `mri_vla_jepa`，建议使用独立环境。Python ≥3.11：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[raw,test]'
PYTHONPATH=src python train.py --help
PYTHONPATH=src python train.py predict-vla-jepa --help
```

[输入输出说明](docs/INPUT_OUTPUT_ZH.md#本版-vq-前端和命令)提供真实参数的训练、两阶段切换和预测命令。正式前端权重须先取得合法的 `symm_world_codec_v2` 安全导出；已完成的 checkpoint 含完整前端，推理无需再次读取 codec。`--forecast-states` 会拒绝本实验 pCR-only 权重。

测试入口为 `PYTHONPATH=src python -m pytest -q`，本次 CPU 验证环境和兼容设置见[发布验证记录](docs/RELEASE_VALIDATION_ZH.md)。原实验的设计与历史工程记录见[实验说明](docs/VQ_ENCODER_EXPERIMENT_20261012_ZH.md)；其中外部 CUDA 报告和运行工件未全部发布。

## 来源与许可

基础新增代码保留 MIT，派生 `vq_encoder.py` 保留 **CC BY-NC 4.0**，整体不是纯 MIT。见 [LICENSE](LICENSE)、[CODEC_NOTICE](licenses/CODEC_NOTICE.md) 和 [来源说明](docs/SOURCES.md)。代码许可不授予临床数据及外部权重的使用权。

批量队列 `scripts/run_vq_encoder_experiment.py` 支持 `--manifest`、`--output`、`--codec` 和 `--image-cache`（`none` 表示直接读取 NPZ）。默认 codec 位于本包 `weights/roi32_codec.pt`，需自行准备；启动时只在新输出的 `input_configs/` 生成本机资源路径副本，原配置和科学参数不改。恢复同一队列时保持相同资源参数，详见输入输出说明。
