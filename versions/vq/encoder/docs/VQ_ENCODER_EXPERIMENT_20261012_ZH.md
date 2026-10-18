# 2026-10-12：VQ 编码器冻结与微调实验

这次实验回答一个具体问题：在相同第一版 pCR 网络和 dynamic L1 训练设置下，复用已有 VQ-GAN 的连续 MRI 编码器，是否比原有从零训练的前端更好。先测试冻结预训练前端，再测试从冻结阶段最佳权重出发的小学习率微调。结果需要实际训练后比较，目前不能据此认定 pCR 性能提高。

新实验根目录是 `/root/autodl-tmp/line b/breast-mri-vla-jepa-vq-encoder`。这是独立代码包，Python 导入名称仍为 `mri_vla_jepa`；本地运行通过 `PYTHONPATH=src` 使用新目录中的代码。

| 用途 | 本机路径 |
| --- | --- |
| 新代码 | `/root/autodl-tmp/line b/breast-mri-vla-jepa-vq-encoder/src/mri_vla_jepa` |
| 原第一版与已完成 L1 对照 | `/root/autodl-tmp/line b/breast-mri-vla-jepa` |
| 第二版 | `/root/autodl-tmp/line b/breast-mri-response-jepa-v2-joint` |
| 共享数据索引 | `/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32/manifest.json` |
| 共享图像缓存 | `/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32/image_cache_20261004` |
| 旧 VQ-GAN 的安全导出权重 | `/root/autodl-tmp/MAM/symm-fm-world-v2/data/registered_roi32/codec.pt` |
| 新实验输出 | `/root/autodl-tmp/line b/breast-mri-vla-jepa-vq-encoder/runs/vq_encoder_20261012` |

配置里的权重、缓存和阶段 B 初始化路径是本机绝对路径。默认 launcher 直接使用原第一版的 manifest。源代码与数据分开保存，NPZ、缓存及历史对照权重留在原目录；换机器时要准备相同数据，并调整这些路径。改动输出根目录时，阶段 B 配置中的 `initialize_from_checkpoint` 也必须指向对应的新冻结阶段 `best.pt`。

原 VQ-GAN 在单通道 DCE0、`32×128×128`、训练集共同强度尺度上训练；三个 MRI 相位复用同一个编码器。当前 ROI32 图像已经预处理，加载时继续直接读取，不再做每患者 z-score。年龄仍按训练集统计标准化；HR、HER2、MP 保留二值编码，Arm 仍用原类别 embedding。缺失检查仍保留原 T0–T3 槽位、填零并由 mask 屏蔽；dynamic 输入只包含当前阶段及之前合法可见的 MRI。

完整输入形状为 `[B,4,3,32,128,128]`。网络仅编码 mask 标记的真实可见检查。单次检查的前向如下：

```text
三相检查                       [N,3,32,128,128]
  ↓ 各相位分别调用同一个单通道预训练前端
每相 encoder + pre_quant       [N,8,8,32,32]
  ↓ 沿通道拼接
三相连续特征                   [N,24,8,32,32]
  ↓ AdaptiveAvgPool3d(2,4,4)
                               [N,24,2,4,4]
  ↓ 可训练投影 24→128
                               [N,128,2,4,4]
  ↓ 展平空间维 + 位置编码 + 空间 Transformer + LayerNorm
空间 token                     [N,32,128]
  ↓ 原第一版历史/临床/Arm融合与共享 Transformer
4 个 pCR 查询 token            [B,4,128]
  ↓ 查询平均 + LayerNorm + Linear(128,1)
pCR logit                      [B]
```

`N` 表示当前 batch 中需要编码的真实检查数，随合法历史和缺失情况变化。连续特征位于 VQ 量化之前，量化器、码本、解码器和 GAN 判别器均未接入这条路径。当前 MRI token 数和维度保持与第一版一致；改变的是卷积前端和新增的通道适配层。

阅读与打断点可以按以下顺序进行：

1. `data.py` 的 `RawMRIStore.batch`：检查 images、observed mask、clinical 与 Arm。
2. `model.py` 的观察状态编码：查看哪些真实检查被提取并送入 visit encoder。
3. `encoder.py` 的 `VQVisitMRIEncoder.forward`：三个相位分别编码、拼接、池化与投影。
4. `vq_encoder.py` 的 `VQContinuousFrontend.forward`：查看旧卷积编码器和 `pre_quant` 的输出。
5. `model.py` 的 pCR 前向：查看共享 Transformer 的查询特征及最终 logit。
6. `training.py` 的 `_configure`、`_make_optimizer`：确认冻结范围和学习率分组。

默认正式方案采用已完成的 dynamic L1 对照，种子 17/43；四个新 run 为：

| 种子 | 阶段 A | 阶段 B |
| --- | --- | --- |
| 17 | `frozen_dynamic_seed17` | `finetune_dynamic_seed17` |
| 43 | `frozen_dynamic_seed43` | `finetune_dynamic_seed43` |

两阶段都只训练 pCR：`task_weight=1`，JEPA、flow、重构与方差损失权重均为零。辅助 world 模块虽然仍存在于模型定义，正式实验不会训练或评估其预测能力。第一版 pCR 使用共享主干的直接分类头；预测出的未来状态不会再送入这个分类头。

其余对照设置保持一致：patient-uniform 合法阶段采样、batch size 16、accumulation 1、dropout 0.3、AdamW weight decay 0.01、梯度裁剪 1.0、BF16、最多 200 epoch、early stopping patience 50。学习率恒定。每 epoch 对全部合法验证前缀计算 NLL，先在患者内部平均，再对患者平均；据此保存 `best.pt`。

阶段 A 从安全导出的 `codec.pt` 加载 VQ `encoder` 与 `pre_quant`，然后同步完整 EMA teacher 的初始坐标。预训练前端固定为 eval、无梯度，并从优化器中排除；投影、位置编码、空间 Transformer、共享网络和 pCR 头仍以 `1e-4` 训练。日志中的 `pretrained_encoder_frozen=true` 表示这个范围，`encoder_frozen` 则表示包括适配层在内的整个 visit encoder 是否冻结。

阶段 B 从同种子阶段 A 的 `best.pt` 加载完整模型，解冻 VQ 前端。优化器包含 `network=1e-4` 和 `pretrained_encoder=1e-5` 两组。它重新开始计数并建立新的 AdamW、采样器与早停状态；这是从已学到的权重开始的新实验阶段。加载时严格检查模型结构、数据身份、患者划分、归一化、实现源码以及损失/采样设置。正式配置两阶段都启用前端 activation checkpointing；冻结阶段不会为前端建立反向图，微调时通过重新计算减少显存占用。

一个阶段内部使用 `--resume` 时，恢复自己的 `last.pt`、优化器、采样器、随机状态和进度，并严格检查来源一致性。恢复训练和已训练权重推理直接使用 checkpoint 内的模型权重，无需再次读取外部 `codec.pt`，也无需再次读取阶段 A 的初始化文件。`initialization.json` 记录初始化文件路径、大小、修改时间以及阶段来源信息；`history.json`、`epoch_history.json`、`progress.json` 和 `train_report.json` 记录实际学习率组与冻结状态。

先查看状态：

```bash
cd '/root/autodl-tmp/line b/breast-mri-vla-jepa-vq-encoder'
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python \
  scripts/run_vq_encoder_experiment.py --profile dynamic --status
```

真实 GPU 检查已通过。默认启动四个新 run，最多并发两个 worker：

```bash
OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src /root/miniconda3/bin/python \
  scripts/run_vq_encoder_experiment.py --profile dynamic \
  --detach --max-concurrent 2 --gpu-memory-budget-mib 29696
```

若已有队列需要继续运行，在同一命令末尾加 `--resume`。controller 会等待对应冻结 run 完成并验证其 checkpoint，再启动微调 run。`--profile t0` 和 `--profile both` 为另行选择的方案，默认 dynamic 启动不会运行 T0。当前文档中的命令是操作入口；是否已启动、完成或失败请查看实际状态与 `controller.log`，文档不预先记录启动成功。

单 batch CPU 调试可直接加载旧 VQ 权重：

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python \
  scripts/debug_vla_jepa.py \
  --manifest '/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32/manifest.json' \
  --config configs/vq_frozen_dynamic_seed17.yaml --mode loss
```

把 `--mode loss` 改为 `--mode train-step`，可在内存中执行一次更新，检查冻结前端没有梯度而适配层得到梯度；不会保存训练权重。阶段 B 配置依赖阶段 A 权重，调试时通过实际 `--checkpoint` 读取已训练模型：

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python \
  scripts/debug_vla_jepa.py \
  --manifest '/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32/manifest.json' \
  --checkpoint runs/vq_encoder_20261012/finetune_dynamic_seed17/best.pt \
  --split val --mode predict
```

后一个命令需要对应 run 已产生 `best.pt`。最终 CPU 全套为 254 passed，耗时 35.01 秒，包括 230 项核心测试、[19 项 VQ 队列测试](../tests/test_vq_experiment_queue.py)和 [5 项调试脚本测试](../tests/test_debug_vla_jepa.py)。其中 [VQ 编码器测试](../tests/test_vq_encoder_experiment.py)覆盖预训练权重加载、冻结/微调的梯度范围与学习率分组、完整损失反向、断点恢复一致性，以及阶段切换时模型权重继承和优化器重建。

真实 B16 CUDA 检查也已通过：两阶段分别在 T0/T2/T3 执行前向、损失、反向与更新，另检查 T3 的 FP32 验证和两阶段 32 患者/B16 最坏 T3 diagnostics。检查中冻结与微调阶段的 CUDA reserved 峰值分别为 7778 MiB、10422 MiB；队列分别预留 9216 MiB、13312 MiB，总预算 29696 MiB，最多并发两个 worker。详细记录见 vq_encoder_cuda_20261012.json（未随源码发布的历史工件：`../reports/vq_encoder_cuda_20261012.json`）。这些是运行检查与显存调度依据，正式训练结果另行比较。

数据划分仍是训练 764 名、开发验证 102 名、测试 0 名。102 名开发验证患者已用于多次方案选择，新的最佳验证结果只能作为开发阶段比较；判断编码器是否带来稳定提升，至少需要比较同种子冻结、微调和原 dynamic L1 的结果，并明确其未经过独立测试验证。

派生前端的许可和来源见 [CODEC_NOTICE.md](../licenses/CODEC_NOTICE.md)。原有代码保留 MIT；`vq_encoder.py` 保留来自 MeWM 和 `cgznb/symm-fm` 的 CC BY-NC 4.0。
