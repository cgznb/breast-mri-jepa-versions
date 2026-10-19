# V1-evolution：历史差分、残差转移与自主多步监督

这是乳腺 MRI / I-SPY2 VLA-JEPA 的后续结构优化版本，包含模型、训练、数据准备、实验队列、诊断、配置和原有测试。它逐项比较固定转移槽、冻结 VQ 前端、未来状态参与 pCR、历史差分、残差转移、自主多步监督，以及病灶加权 / 压缩前特征。原始实验与权重没有被此次整理修改。

输入是截至当前阶段真实可用的 MRI、4 项临床变量和治疗 Arm；输出是最终 pCR 概率和可选未来隐状态。正式实验关闭 MRI flow，**最优方案不生成未来 MRI 图像**。完整张量、格式、mask 和命令见 [输入输出说明](docs/INPUT_OUTPUT_ZH.md)。

## 历史最优方案

每次训练按“患者内平均合法观察前缀 NLL，再跨患者平均”选择 best。下表为种子 17 / 43 的均值，不是两模型集成。

| 方案 | 验证 NLL ↓ | T0 AUROC | T1 AUROC | T2 AUROC | T3 AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vq_forecast_history_residual_free_encoder64` | 0.545002 | 0.731696 | 0.730928 | 0.739583 | 0.720443 |

- [seed17 配置](configs/evolution_vq_forecast_history_residual_free_encoder64_seed17.yaml)、[seed43 配置](configs/evolution_vq_forecast_history_residual_free_encoder64_seed43.yaml)
- [20 次训练、40 份世界模型评价的汇总](historical_runs/v1_evolution_20261012/RESULTS_ZH.md)
- [原实验协议与消融定义](docs/V1_EVOLUTION_PROTOCOL_20261012_ZH.md)

数据为 764 人训练、102 人开发验证、0 人独立测试。开发验证集被多轮实验重复使用，结果不能当作独立测试证据或显著性结论。相较旧动态 pCR-only 参考，NLL 从 0.549020 降为 0.545002，T0 AUROC 从 0.740625 降为 0.731696，改善并非覆盖全部指标。不同 checkpoint 的 latent 评价各自在自己的 EMA teacher 坐标中，不能直接跨模型比较 latent L1。

最优方案使用冻结旧 codec 的 `encoder` 输出，**绕过 `pre_quant` 的 8 通道压缩**：每相 64 通道，三相合并 192 通道，池化到 `2×4×4` 后映射为每次检查 `32×128` tokens。融合和状态预测采用 128 维、2 层、4 头 Transformer，dropout 0.3。pCR 头读取 query、预测终末状态和终末相对当前状态变化，共 384 维。

损失为 `BCE + 0.5 × teacher-forcing JEPA L1 + 0.5 × 自主多步 JEPA L1`；重建、方差、flow 权重均为 0。AdamW，batch 16、accumulation 1、学习率 `1e-4`、weight decay 0.01，BF16，最多 200 epoch，早停 patience 50；全部科学参数保留在 YAML。

## 安装与工程检查

四个版本使用同名 Python 包，应分别使用环境，或从各版本目录显式指定 `PYTHONPATH=src`。

```bash
cd versions/v1_evolution
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
PYTHONPATH=src python train.py --help
python ../../tools/run_cpu_tests.py --project . -- -q
PYTHONPATH=src python train.py smoke-vla-jepa --output /tmp/v1-evolution-smoke-new
```

需要 NIfTI / 临床表格读写时安装 `.[raw,test]`。默认 smoke 使用小型合成数据和基础 CNN 配置，仅检查训练、保存、验证与推理链路；它不是 encoder64 的临床复现。`tests/test_evolution_training.py` 另外用合成 codec 检查两种 VQ 特征源、自主联合损失、优化器更新、精确恢复和概率预测。

整理后完整 CPU 测试 **373 项通过**，CLI smoke 与 20 项只读计划通过；环境及边界见 [发布验证记录](docs/RELEASE_VALIDATION_ZH.md)。

## 复现依赖与路径

本机新增 small／medium／large、三相／单相、teacher forcing／hybrid 的48次双种子实验，按患者等权F1选模，并支持双GPU显存与吞吐准入。配置、运行及恢复方法见[容量与相位训练说明](../../docs/CAPACITY_PHASE_TRAINING_ZH.md)。

本仓库不含患者 MRI、临床表格、患者级 manifest、image cache、外部预训练 `codec.pt` 或正式最优训练权重。首次训练需要按原数值契约准备的 ROI32 数据和兼容 codec；已有训练 checkpoint 包含模型权重和训练标准化状态，常规推理无需再次读取 codec 或 image cache。

`configs/` 及打包资源保留原历史路径；运行前**复制配置**，将 `encoder_checkpoint`、`image_cache` 等资源位置改为本机路径，输出放到新的运行目录，不要写入 `historical_runs/`。完整命令见 [输入输出说明](docs/INPUT_OUTPUT_ZH.md)。

历史 `scripts/run_*experiment.py` / `analyze_*experiment.py` 保留多目录队列和回放流程，涉及其他版本与外部工件。新仓库路径适配保留旧 sibling 兼容；仅有源码不足以重新回放患者级历史评价。单配置训练 / 推理入口为 `train.py`。

## 文件与许可证

- [模型与状态读出](src/mri_vla_jepa/model.py)、[MRI / VQ 适配器](src/mri_vla_jepa/encoder.py)
- [输入与监督对象](src/mri_vla_jepa/contracts.py)、[数据加载](src/mri_vla_jepa/data.py)
- [训练 / 选模 / 推理](src/mri_vla_jepa/training.py)、[命令行](src/mri_vla_jepa/cli.py)
- [队列](scripts/run_evolution_experiment.py)、[结果核验](scripts/analyze_evolution_experiment.py)
- [LICENSE](LICENSE)、[VQ 前端来源及 CC BY-NC 4.0 例外](licenses/CODEC_NOTICE.md)

原项目新增代码采用 MIT；派生 VQ 前端保留 CC BY-NC 4.0 及署名，混合包包含非商业组件。代码许可证不授予患者数据或外部预训练权重的权利。
