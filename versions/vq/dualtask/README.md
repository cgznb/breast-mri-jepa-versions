# VQ dualtask：pCR 与未来潜在状态双任务实验

本包位于 `versions/vq/dualtask/`，属于 VQ 优化版本的另一组实验。保留完整源码、raw/冻结 VQ/微调 VQ 六次训练配置、队列/评价/汇总/绘图工具和已有测试。每组种子 17/43，统一训练损失为：

```text
BCE(pCR) + 0.5 × teacher-forcing JEPA latent L1
```

flow、MRI 重构和方差正则关闭。输入是纵向三相 MRI `[B,4,3,32,128,128]`、临床、Arm 和可用性 mask；输出有最终 pCR 概率，以及显式请求时的未来潜在状态 `[B,4,32,128]`。pCR 保持 V1 的观察主干直接分类路径，预测未来状态不回送分类头。字段、mask、监督边界、形状和真实 CLI 命令见 **[输入输出说明](docs/INPUT_OUTPUT_ZH.md)**。

## 结果与配置

六次训练及 best/last 的十二份世界评价已完成。best 按患者等权的平均合法前缀开发验证 NLL 选取。两种子均值：

| 方案 | pCR NLL | T0 AUROC | seed17 配置 |
| --- | ---: | ---: | --- |
| raw CNN | 0.553476 | 0.735714 | [raw](configs/vq_dualtask_raw_dynamic_seed17.yaml) |
| 冻结 VQ A | 0.566432 | 0.715179 | [frozen](configs/vq_dualtask_frozen_dynamic_seed17.yaml) |
| 从双任务 A 微调 VQ B | 0.572514 | 0.717634 | [finetune](configs/vq_dualtask_finetune_dynamic_seed17.yaml) |

本轮 VQ 替换未超过 raw 对照或既有 V1 L1。完整结果见[结果汇总](historical_runs/vq_dualtask_20261012/RESULTS_ZH.md)和[两项任务解释](historical_runs/vq_dualtask_20261012/INTERPRETATION_ZH.md)。latent/copy 比值仅在各自 teacher 坐标内解释，不是共同 MRI 像素质量排名；pCR-best 不表示世界预测最佳。

阶段 A 从外部 codec 加载并冻结 VQ 前端，其余网络从零开始；B 从本实验同种子 A 的完整 best 初始化，前端 `1e-5`、其他参数 `1e-4`，重置优化器/进度。两个阶段都训练 pCR 与 JEPA。历史 764 人训练、102 人开发验证、无独立测试；本包不附真实数据、正式权重或逐患者预测。`historical_runs/` 是只含已发布聚合文档的归档，新的运行产物写入 `runs/`。

## 安装与运行

与其他版本同名导入 `mri_vla_jepa`，建议各自独立环境，Python ≥3.11：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[raw,test,plots]'
PYTHONPATH=src python train.py --help
PYTHONPATH=src python scripts/evaluate_registered_roi32_world.py --help
```

[输入输出说明](docs/INPUT_OUTPUT_ZH.md#本版双任务和命令)给出先生成本机新配置，再进行冻结、微调、pCR/未来状态预测与世界评价的命令。`configs/` 保留历史科学参数/机器路径；根目录 `tools/prepare_config.py` 只替换资源位置。绘图需要 `plots` 可选依赖，读取自己的完整 `analysis.json` 和各 run 的 epoch history；聚合 Markdown 不能替代这些运行工件。

测试入口为 `PYTHONPATH=src python -m pytest -q`，本次 CPU 后端兼容设置及结果见[发布验证记录](docs/RELEASE_VALIDATION_ZH.md)。[历史实验设计](docs/VQ_DUALTASK_EXPERIMENT_20261012_ZH.md)保留旧操作背景，其中部分未随仓库发布的 CUDA 报告/图/分析 JSON 需在原私有运行环境查阅。

## 来源与许可

基础新增代码保留 MIT，派生 VQ 前端保留 **CC BY-NC 4.0**，见 [LICENSE](LICENSE)、[CODEC_NOTICE](licenses/CODEC_NOTICE.md) 和 [来源说明](docs/SOURCES.md)。临床数据和外部预训练权重的权利不由代码许可提供。

批量队列 `scripts/run_vq_dualtask_experiment.py` 支持 `--manifest`、`--output`、`--codec` 和 `--image-cache`（`none` 表示直接读取 NPZ）。默认 codec 位于本包 `weights/roi32_codec.pt`，需自行准备；启动时只在新输出的 `input_configs/` 生成本机资源路径副本，原配置和科学参数不改。恢复同一队列时保持相同资源参数，详见输入输出说明。
