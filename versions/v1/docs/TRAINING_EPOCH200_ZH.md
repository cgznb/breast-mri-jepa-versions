# 正式三相 ROI32 训练

用户设置：固定 T0、动态 T0–T3 两版，每版种子 17/43，最多 200 epoch，patience 50，物理 batch size 16。梯度累积为 1，学习率维持 1e-4，使用 CUDA BF16，MRI flow 关闭。

每个 epoch 将 764 名训练患者随机打乱并各使用一次。48 个 batch 中，前 47 个各 16 名，最后一个 12 名；不丢弃患者，也不在 epoch 内重复抽样。动态版每名患者随机选择一个合法 landmark。所有 102 名验证患者在每个 epoch 结束后评估；动态版包含 382 个合法历史前缀。

选模与早停监测 NLL。固定版为 T0 NLL，动态版为每名患者先平均合法阶段、再对患者平均的 NLL。严格改善才保存新的 best 并重置 patience；连续 50 个 epoch 未改善时提前结束，最多 200 epoch。无早停时为 9600 次优化更新。

## 队列与进度

在本项目根目录执行：

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src python scripts/run_registered_roi32_experiments.py --max-concurrent 3 --detach
PYTHONPATH=src python scripts/run_registered_roi32_experiments.py --status
```

输出目录为 `runs/registered_roi32_epoch200_20261003/`。队列顺序为 T0 seed17、dynamic seed17、T0 seed43、dynamic seed43。用户当前选择在同一张 GPU 上最多同时运行三个训练进程，有空位时自动启动后续任务；独占锁拒绝重复启动。启动器默认并发数仍为 2，当前实验使用显式 `--max-concurrent 3`。

`controller.json` 的 `current_jobs` 记录并行任务，各 job 保存 PID、启动时间及四个 run 的状态。各 run 的 `progress.json` 在每个优化步更新，包含当前 epoch、已完成 epoch、batch、损失、最好验证 NLL 和 patience 计数；`epoch_history.json` 保存每个 epoch 的训练与验证汇总。`best.pt` 为最佳 NLL 权重，`last.pt` 为最近恢复点；两个文件均包含优化器、患者排列、阶段采样器、随机状态及早停状态。

控制器发生中断后，可在上述启动命令加 `--resume` 恢复，保留分配器环境变量与 `--max-concurrent 3`。已停止的训练从 `last.pt` 续训；旧记录中仍运行的训练进程会在检查命令参数和进程启动时间后直接接管，保留原 PID 和进度。运行中的控制器会拒绝重复启动。输入、配置和源码的恢复绑定保持严格一致，训练期间不要编辑核心源码或对应配置。

2026-10-03 21:19 Asia/Shanghai 从单路切换为两路；21:52 又提高到三路。原 T0 seed17 与 dynamic seed17 保留 PID 和进度，加入 T0 seed43；dynamic seed43 在出现空位时自动启动。21:54 已验证新增 T0 seed43 完成首轮训练及 102 人验证、保存 best/last 并进入第二轮。实时状态以 `controller.json` 和各 run 的进度为准。

## 显存与验证

启动前物理 B16 实测已通过 T0/T2/T3 的原损失、反向、AdamW 和 EMA 更新：完整输入 `[16,4,3,32,128,128]`，在线编码器单次分别处理 16/48/64 次检查，最高已分配显存约 7.57 GiB。没有使用微批次替代 batch 16。

B16 单步门测最高保留显存约 9.2 GiB；原两路正式运行总占用约 21.9 GiB。提高到三路时，新增训练使用 `PYTORCH_ALLOC_CONF=expandable_segments:True`，减少不同 tensor 形状的缓存碎片；原有两路继续运行。三路首轮训练及验证期间观察到总显存最高 28569 MiB（约 27.9 GiB），当前 RTX 5090 有约 32 GiB 显存。网络、batch16、梯度累积与正式训练配置均保留原设置。

启动前完整 104 项测试通过；新增 6 项并行队列测试通过，覆盖旧 PID 接管、双路填充、进程身份、重复控制器拒绝及失败任务处理。epoch 测试覆盖每轮完整患者覆盖、末批保留、epoch 中间精确续训、patience 50 与旧 checkpoint 兼容。显存报告位于 `reports/registered_roi32_batch16_gate.json`。

这些实验继续使用 764 train / 102 development val，没有独立测试集。四步工程 smoke checkpoint 保存在旧验证目录；正式 run 单独从随机初始化开始。
