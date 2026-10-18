# V1 发布副本验证（2026-10-18）

本次仅验证整理后的发布副本，不覆盖历史实验，不运行真实患者训练/推理，不重新使用开发验证集选择模型。

## 最终 CPU 测试

在独立进程中使用 Python 3.12.13、PyTorch 2.5.1+cpu 运行全部既有测试，结果 **208 passed（12.09 秒）**。所用解释器已有 pytest、NumPy、PyYAML、scikit-learn、nibabel、pandas 和 openpyxl。示例从仓库根目录运行：

```bash
python tools/run_cpu_tests.py --project versions/v1 -- -q -p no:cacheprovider
```

脚本隐藏 CUDA，设置 CPU 测试环境，关闭 fused MHA 与 oneDNN，保留全部测试。各包单独进程运行，避免同名 `mri_vla_jepa` 包互相覆盖。

初次直接使用 PyTorch 2.5.1 CPU 默认后端时，两个 CPU BF16 测试在 Transformer eval fastpath 中因 BF16/FP32 类型不匹配失败；PyTorch 2.12.1 默认后端则在本机 AVX2 的 oneDNN BF16 backward 中报不支持。关闭上述 CPU 专用后端后，两个版本均通过本包全部 208 项。没有为此修改模型、损失或训练实现，也没有跳过 BF16 测试；这不代表本机默认 CPU BF16 后端可用于正式训练。

## CLI 与发布内容

`train.py --help`、`train.py train-vla-jepa --help`、`evaluate-vla-jepa --help`、`predict-vla-jepa --help`、`scripts/prepare_registered_roi32.py --help` 与 `scripts/evaluate_registered_roi32_world.py --help` 均正常退出（0），没有读取真实数据。

准备脚本只将历史机器默认 inventory/clinical 参数改为必传参数。模型/训练 `src/` 与历史源码一致；新增或调整的内容为发布导航、输入输出文档、可移植资源配置工具说明、历史工件链接及许可证范围说明。V1 LICENSE 删除指向未纳入包内的旧 legacy 组件的例外段，MIT 正文与原版权声明保留；依据是此前已整理发布的同源码 V1 包与实际纳入文件。

本仓库的 CLI 命令需要用户自行准备 manifest、MRI/临床数据及正式权重。测试只说明代码/合同可运行，不能验证历史结果的临床泛化能力。
