# symm-fm · Three-phase Treatment-Response JEPA

这是针对 `cgznb/symm-fm` 三相 MRI 工作流的完整、可独立运行的新实现及非破坏性集成包，不是几段伪代码或仅有模型类的补丁。它包含数据准备、原 VQ 接口、3D masked JEPA、治疗响应与纵向预测、条件生成、临床任务、A–E 训练、恢复、采样、评估和测试。

**范围必须说清楚：** 本包不是原 GitHub 仓库所有历史实验目录的镜像，也没有远程修改你的 GitHub。它提供完整的新三相研究工作流；安装脚本把它装进原仓库并保留旧生成器、旧配置、旧测试和其他工作流。现有 VQ 权重由你提供，不附带真实 MRI、病人标签或经过医学验证的预训练权重。

基准审阅版本：`92265d3b1749ae3b686f2089843c49da129fd4d2`。

## 先读哪份文档

- `docs/MODIFICATION_REPORT_ZH.md`：为什么这样修改，原代码差异，每个模块和梯度路径。
- `docs/RUNBOOK_ZH.md`：真实数据与原仓库接入、训练、采样和恢复命令。
- `docs/COVERAGE_MATRIX.md`：文档建议逐项映射，哪些已实现，哪些需要数据/外部模型，哪些仍是扩展。
- `docs/SOURCES.md`：公开代码来源、具体复用/改写边界、许可证。
- `validation/VALIDATION.md`：实际测试范围及未验证项。

## 快速执行

Python 3.11 或 3.12 适合与原仓库组合。核心只需 PyTorch、NumPy、PyYAML；没有隐式下载。

```bash
cd symm-fm-trjepa
pip install -e '.[test]'
python -m pytest -q
bash scripts/run_smoke.sh /tmp/trjepa-smoke
```

这里生成的是**带显式标记的合成数据**，用于验证代码，不是临床实验。`configs/smoke.yaml` 不能用于论文主实验。

真实模型配置：`configs/three_phase.yaml`。主干是 12 层、384 维 3D ViT + 4 层响应 Transformer + 6 层动态预测器 + 12 层 3D DiT；另有 768 维配置和可选 MONAI U-Net 后端。维度、深度、头数、loss、阶段预算全部可配置。

## 安装到原仓库

```bash
python scripts/install_into_repo.py --repo /absolute/path/to/symm-fm
```

新增 `extensions/trjepa/` 和 `run_trjepa.py`，不覆写原文件。新工作流可通过：

```bash
python /absolute/path/to/symm-fm/run_trjepa.py --help
```

直接在本目录运行 `python -m trjepa.cli ...` 也不依赖原仓库安装。

## 真实数据最重要的限制

原三相是 `pre_aqc0 / first_post_aqc1 / metadata_late`，不是 T0/T1/T2。原缓存是 prequant latent；训练集往往没有可直接用于 JEPA 的三相原始影像。旧 target-centered ROI 不能冒充 source-grid 空间残差数据。

请先按运行手册重建带三相原始影像的 source-grid cache；空间对应未经 QC 确认时，不要把 `spatial_correspondence_verified` 改成 true 来绕过检查。该标记只存在于训练目标/损失，不进入推理条件。

## 研究与许可证边界

采用 VLA-JEPA 的 source-only transition 表示思路，但没有直接搬用 RGB 视频/VLM/机器人权重。I-JEPA/DiT 改写代码有 **CC BY-NC 4.0 非商业限制**；Set Transformer 部分为 MIT；原代码和可选库保留各自许可证。详见 `licenses/`、`docs/SOURCES.md`。

这份实现尚无真实 I-SPY2 精度、临床收益或 CVPR 录用保证。治疗条件生成不是因果最优治疗推荐。
