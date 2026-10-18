# A1、A2 与 ROI32 数据提速

目标目录：`/root/autodl-tmp/line b/breast-mri-vla-jepa`。原四组 A0 已完成，结果保留在 `runs/registered_roi32_epoch200_20261003`；新实验从头训练，输出到独立目录。

## 实验协议

| 实验 | 相对原 A0 的研究变量 | 初始学习率 | Dropout |
|---|---|---|---|
| A1 | 验证 NLL 驱动 ReduceLROnPlateau | 1e-4 | 0.1 |
| A2 | 全局 dropout 增大 | 固定 1e-4 | 0.3 |

A1 的调度器为 `mode=min, factor=0.5, patience=5, min_lr=1e-6, threshold=0, threshold_mode=abs`，每轮验证后调用一次。`patience=5` 是降学习率的等待轮数；训练早停仍为 `patience=50`。固定 T0 使用 T0 NLL；动态使用患者内平均合法阶段 NLL、再对患者平均。

每个实验均为固定 T0、动态 `all_observed` 两版，各 seed17/43：`2 × 2 × 2 = 8` 个 run。最多 200 epoch，物理 batch16，末批保留 12 人，梯度累积 1；每轮 764 名训练患者各一次、48 次更新。CUDA BF16 训练，FP32 验证，验证集仍为 102 名开发患者。

两组都保留原完整损失：`BCE(pCR) + 0.5 × JEPA L1 + 0.1 × 观察影像 MSE 重构 + 0.01 × 原方差正则`，MRI flow 关闭。网络尺寸、数据划分、临床/Arm、影像标准化和世界监督筛选规则沿用 A0。A1、A2 此轮不承担损失消融或方差正则修正。

八份配置为 `configs/registered_roi32_epoch200_{a1,a2}_{t0,dynamic}_seed{17,43}.yaml`，包内资源有相同副本。队列会检查与各自 A0 配置仅有规定研究变量及共同的数据管线设置不同。

## 数据准备

新派生缓存仅保存原 NPZ 的 `image` 数组，为 float32 NPY；3039 次检查全部逐值比对通过，零背景保留，占约 17.81 GiB。原 NPZ、manifest 与临床统计仍作为来源。缓存私有索引记录路径、文件大小和修改时间，以及三相顺序和标准化声明；不记录 checksum 值。

共同的训练设置：

```yaml
image_cache: image_cache_20261004
prefetch_batches: 2
loader_workers: 2
pin_memory: true
non_blocking_transfer: true
```

缓存相对路径从 manifest 所在目录解析。NPY 使用有界的内存映射缓存；影像首次映射检查形状、dtype 和有限值，每次访问核对文件身份。缓存缺检查、声明不符或文件改变时明确拒绝。原路径的 NPY/NIfTI/NPZ 处理仍可单独使用。

后台最多保持两个批次的任务，两个工作线程按计划准备完整批次。大影像直接写入 pinned memory，减少额外复制；主训练线程使用非阻塞 CUDA 传输。患者排列与阶段选择仍由主采样器消费，预取计划使用采样器副本，保存点仅记录已消费位置。中断时舍弃未消费预取，恢复时从保存位置重新规划。

模型公共入口仍校验输入与监督。只去掉同一 `compute_loss` 调用内部重复的全量检查，并合并损失标量的设备回传。传输使用计算流，尚未引入独立 CUDA copy stream。

## 实测与验收

单进程、相同 16 名训练患者、55 次检查、页缓存已热、各 3 次测量，包含张量组装和 CPU 输入校验：

| 路径 | 准备时间中位数 |
|---|---|
| 原 NPZ 串行 | 2.639 秒 |
| NPY 缓存串行 | 0.440 秒 |
| NPY 直接组装 pinned memory | 0.401 秒 |

缓存串行约快 6.0 倍。短程受控计算窗口中的预取等待中位数为 0.041 秒，但仍出现约 0.254 秒的等待；这个受控测量不是正式训练吞吐，不能承诺消除全部 GPU 等待。正式三路吞吐以启动后的实时进度为准。

证据文件：

- `reports/roi32_image_cache_20261004.json`：3039 次检查逐值核对、容量与背景保留。
- `reports/roi32_pipeline_benchmark_20261004.json`：张量逐值一致、准备/传输时间、预取和 BF16 梯度对照。
- `reports/roi32_pipeline_resume_20261004.json`：合成和真实恢复检查。
- `reports/roi32_a1_batch16_20261004.json`、`reports/roi32_a2_batch16_20261004.json`：完整尺寸 T0/T2/T3 的物理 B16 前向、反向、优化器和 EMA 检查。
- `reports/roi32_pipeline_live_20261004.json`：正式三路约 21 秒窗口，GPU 计算利用率均值 53.6%，范围 0%–99%，每个进程约 0.397–0.438 秒/更新；包含验证/保存开销，仅为启动后快照。

完整测试 157 项通过，包括缓存结构/文件变化拒绝、缺失四槽、临床 mask、T0 推理不读取未来或标签、串行/预取的同批次训练及中断恢复、调度器恢复与历史 checkpoint 只读兼容。

CPU 合成训练的模型、优化器、采样器、随机数和历史均可逐值对齐。单批次 BF16 数据路径的损失和梯度在 cuDNN 可复现设置及 math attention 下逐值相同，正式配置保留原内核设置。原默认 CUDA BF16 对相同输入重复运行也有微小梯度差异，且 `adaptive_avg_pool3d_backward_cuda` 不支持 PyTorch 严格确定性；真实 CUDA 恢复检查要求采样、随机数和消费位置完全一致，并用独立连续训练的重复对照量化数值波动，不宣称 GPU 权重逐位恢复一致。

## 启动与查看

在目标目录运行，使用本机 Python 环境：

```bash
env PYTHONPATH=src /root/miniconda3/bin/python scripts/prepare_roi32_image_cache.py

env PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
  /root/miniconda3/bin/python scripts/run_registered_roi32_experiments.py \
  --suite a1_a2 --max-concurrent 3 --detach

env PYTHONPATH=src /root/miniconda3/bin/python \
  scripts/run_registered_roi32_experiments.py --suite a1_a2 --status
```

实时文件为 `runs/registered_roi32_a1_a2_20261004/controller.json` 及各 run 的 `progress.json`、`epoch_history.json`。每轮记录本轮使用的 `lr`、验证后 `next_lr` 和 `lr_reduced`；`best.pt` 按原匹配 NLL 选取，`last.pt` 保存恢复状态。

2026-10-04 北京时间 11:29 已启动三路，首批为 `a1_t0_seed17`、`a2_t0_seed17`、`a1_dynamic_seed17`，其余五组自动接续。启动核验时三组已完成 12/11/10 轮，损失、梯度及权重有限，best/last 的配置、源码、缓存/原始数据身份、标准化和患者分区均匹配。此处进度是核验快照，以实时文件为准。

控制器或训练中断后，用同一启动命令加 `--resume`；它会接管仍存活的匹配子进程，并从各自保存点恢复。已完成 run 必须通过 best/last、配置、数据、源码和选模协议核对才能跳过。不要将 A0 已完成目录当作新源码的续训目录；A0 仍可查看状态与只读评估。

训练期间保持核心源码、新配置、manifest、缓存及原 NPZ 文件身份稳定。新 checkpoint 保存 scheduler、原训练状态、来源文本和数据身份；改变这些内容会拒绝恢复。完整源码改动和验证的聚合记录见 `reports/roi32_a1_a2_launch_20261004.json`。
