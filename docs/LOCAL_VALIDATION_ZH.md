# 本机迁移验证记录

验证日期：2026-10-18。目录：`/personal_fast_storage/VLA-PCRV1`。
原处理后的数据已完整迁移，本机 V1-evolution 环境可用；未启动正式训练。

| 项目 | 结果 |
| --- | --- |
| 原患者与划分 | 866 人；训练 764、开发验证 102、测试 0 |
| MRI 文件 | 3039 个 NPZ，17,630,433,243 bytes，缺失和额外文件均为 0 |
| 原服务器逐影像 SHA256 | 3039 / 3039 一致 |
| NPZ 数组头 | 3039 个文件的 image / roi / support 均通过形状、类型及成员大小检查 |
| 原清单、预训练 codec | SHA256 一致 |
| 本机 manifest | 相对原清单只替换影像路径，患者、划分及其他字段一致 |
| seed17 / seed43 本机配置 | 相对原配置只修改 codec 路径和 image_cache，科学参数一致 |
| Python / PyTorch | 3.12.1 / 2.5.1+cu121；独立 `.venv` |
| GPU | NVIDIA A100-SXM4-80GB；CUDA BF16 前后向通过 |
| 现有模型、训练及数据特征测试 | 45 项通过 |
| 真实 MRI 推理 | seed17 原权重、1 位验证患者、T0 输入；pCR 概率与未来潜状态均为有限值 |
| 本次真实数据优化器更新 | 0 |
| 处理源码归档 | 515 个文件 SHA256、391 个 Python 文件语法校验通过 |
| 小型重建输入 | 7423 个文件，134,787,285 bytes，逐文件 SHA256 一致；含原 867 个 T0 预测 mask |

完整影像校验命令：

```bash
versions/v1_evolution/.venv/bin/python tools/check_local_runtime.py --full-image-sha256
```

机器可读结果保存在：

- `data/registered_roi32/migration-status.json`：迁移完成状态。
- `data/registered_roi32/migration-transfer-result.json`：四路 rsync 退出码均为 0。
- `private_outputs/setup_runtime/runtime_check.json`：完整数据、配置、CUDA 汇总检查。
- `private_outputs/setup_runtime/environment_ready.json`：环境、依赖与现有测试结果。
- `private_outputs/reference_inference/20261018T045757.950616Z/summary.json`：单例真实 GPU 推理验证。

SHA256 证明本机影像文件与复制时的原文件一致；本次没有逐个解压检查全部影像数值或独立核验 ZIP CRC。
单例推理和工程测试通过不构成临床性能复现。当前软件版本与历史环境的差异见
[本机环境说明](../versions/v1_evolution/docs/LOCAL_ENVIRONMENT_ZH.md)。
直接运行方法见 [LOCAL_RUN_ZH.md](LOCAL_RUN_ZH.md)，生成数据的代码与重建限制见
[PREPROCESSING_SOURCES_ZH.md](PREPROCESSING_SOURCES_ZH.md)。
