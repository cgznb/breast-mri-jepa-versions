# V1 四阶段结构修改与训练协议

## 目标和隔离

本轮在独立包 `/root/autodl-tmp/line b/breast-mri-vla-jepa-v1-evolution` 修改并训练。原 V1、VQ pCR-only、VQ 双任务和第二版保留为只读历史参考。正式输出为 `runs/v1_evolution_20261012/`。

研究问题是：在 V1 其余条件固定时，预测未来状态及其变化是否比同容量当前状态读出更有助于 pCR？其后按顺序检验历史差分、残差转移、自主多步监督，再分别检验病灶区域加权和 VQ 压缩前连续特征。

每组种子 17、43，共 10 组、20 次独立训练。每次重新初始化适配器、融合网络、世界预测器和分类头；VQ 组均重新载入并冻结同一旧 codec 前端。组间依赖仅决定开始顺序，**不加载上一组权重、优化器或采样进度**。

## 实验顺序与单项对照

| 阶段 | 组名（后接 `_seed17` / `_seed43`） | 相对前一对照的变化 | 直接对照 |
| --- | --- | --- | --- |
| 1 | `raw_fixed_direct` | CNN，固定内部未来转移槽位；原 pCR query 读出 | 旧 CNN 双任务；原 L1 是 pCR-only 参考 |
| 1 | `vq_fixed_direct` | 冻结 VQ `pre_quant` 8 通道前端，固定槽位 | 旧冻结 VQ 双任务；同轮 CNN |
| 2 | `vq_copy` | 同容量状态读出，终末状态填当前状态、变化填 0 | `vq_fixed_direct`，用于检测扩充读出容量的影响 |
| 2 | `vq_forecast_detach` | 自主预测到 T3，但 pCR 对预测链停止梯度 | `vq_copy` |
| 2 | `vq_forecast` | pCR 梯度允许经过自主预测链 | `vq_forecast_detach`；同时比较 `vq_copy` |
| 3a | `vq_forecast_history` | 加最近可得历史状态差分及缺失标记 | `vq_forecast` |
| 3b | `vq_forecast_history_residual` | 再加 `下一状态 = 当前状态 + 预测变化` | `vq_forecast_history` |
| 3c | `vq_forecast_history_residual_free` | 再加权重 0.5 的自主多步监督 | `vq_forecast_history_residual` |
| 4 | `vq_forecast_history_residual_free_lesion` | 单独增加 T0 病灶区域的世界损失空间权重 | 阶段 3c；**不改变编码器 pooling** |
| 4 | `vq_forecast_history_residual_free_encoder64` | 单独使用 VQ encoder 压缩前 64 通道连续特征 | 阶段 3c；**不加病灶权重** |

`copy / forecast_detach / forecast` 的分类输入均为 `[Pool(pCR queries), Pool(终末状态), Pool(终末状态−当前状态)]`，使用相同 `LayerNorm(384) → Linear(384,1)`。direct 保留原 `LayerNorm(128) → Linear(128,1)`，因此 direct 与 copy 的比较同时涉及读出容量与状态特征，不能单独归因于未来预测。T3 前缀没有未来转移，终末状态为当前真实 T3。

内部未来转移计算由合法可见前缀固定，`query_mask` 只控制返回哪些预测；所有阶段正式训练仍请求全部未来。分类使用可见前缀开始的自主预测，未来真实 MRI 仅作为辅助损失目标，不进入分类输入。

阶段 2 等阶段 1 的 4 次训练及其 8 份 best/last 世界评价全部通过再开始。阶段 3a 等阶段 2 的 6 次训练和评价全部通过；3b 等 3a，3c 等 3b；阶段 4 两种单项组共同等 3c。任何训练、checkpoint 或世界评价正确性检查失败都会阻止后继步骤。验证 AUROC/NLL 优劣不自动取消后续组。

## 固定数据和优化条件

使用原 registered ROI32 manifest/cache：764 人训练、102 人开发验证、无独立测试。输入三相位 MRI `[3,32,128,128]`，保留真实可见前缀、缺失检查位置、clinical 缺失标记及 Arm 既有可用性规则。旧 cache 路径只读。两种子动态 patient-uniform 前缀采样。

统一 batch 16、accumulation 1、dropout 0.3、AdamW 学习率 `1e-4`、weight decay 0.01、最多 200 **epoch**、耐心 50 epoch。best 由患者等权的前缀开发验证 NLL 选择；不按世界状态损失选择。无 VQ 微调、无 warm start、无 flow/GAN/重建/variance 辅助损失。

基础目标为 `BCE(pCR) + 0.5 × teacher-forcing JEPA`。阶段 3c 及其后两组增加 `0.5 × autonomous rollout JEPA`；真实未来状态目标仍来自本模型自己的 EMA teacher。自主监督从可见当前状态出发，其间使用自己的预测。损失按每患者有效目标平均，再按患者平均。

866 名患者 T0 存档均有合法非空 `roi` 布尔 mask `[1,32,128,128]`（训练 764、验证 102）。其来源是 T0 预测病灶 ROI，位于统一 T0 配准网格；只读取 T0 mask，不使用后续分割。病灶组通过 token 网格上的 mask 占比生成 `1 + 4 × occupancy` 权重并在患者内归一化，保留非病灶区域监督。启用时 store 强制校验 mask，缺失/非法输入直接失败，不制造替代 mask。

`vq_encoder` 每相位读旧冻结 encoder 的 64 通道连续输出，三相位共 192 通道再适配到 128 维/32 tokens；原 `vq_prequant` 为每相位 8 通道/三相位 24 通道。均不运行量化器、codebook、decoder 或 GAN，且从 codec 全新加载前端。现有可靠数据中没有真实检查日期、剂量/周期/治疗进度；本轮不引入伪造的时间间隔或以阶段编号冒充真实天数。

## 运行、状态和恢复

运行前先完成 CPU 回归、真实 CUDA batch-16 训练/诊断和 query 请求不变性检查，再冻结源码及配置。真实训练/FP32 世界与请求不变性检查的最大 reserved 为 CNN 8088 MiB、VQ 各组不超过 7854 MiB、encoder64 8216 MiB；队列预约 CNN/encoder64 各 10240 MiB、其余 VQ 9216 MiB、世界评价 10240 MiB，为 CUDA context/cache 留余量。队列限制最多 3 个训练/评价 worker，29696 MiB 总预算按具体组合允许 2 或 3 个。阶段正确性与资源检查通过后执行：

```bash
cd '/root/autodl-tmp/line b/breast-mri-vla-jepa-v1-evolution'
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python scripts/run_evolution_experiment.py --plan
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python scripts/run_evolution_experiment.py --detach --max-concurrent 3 --gpu-memory-budget-mib 29696
```

查询状态和恢复（`--resume` 会接管仍存活且身份匹配的 worker，并从自己的 last checkpoint 续训；已完成且核验通过的训练不重跑）：

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python scripts/run_evolution_experiment.py --status
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python scripts/run_evolution_experiment.py --detach --resume --max-concurrent 3 --gpu-memory-budget-mib 29696
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python scripts/analyze_evolution_experiment.py --partial
```

`source_freeze.json` 保存源码原文和配置，不生成 checksum。恢复要求源码、配置、数据身份、归一化、人群及 checkpoint 协议一致。正式队列运行后不修改其源文件；后续修订另开输出实验。

## 评价边界

每个 run 完成后自动评价 best/last 两份权重，包含相邻 teacher forcing、自主 1–3 步和各自 copy，共 40 份世界报告。最终聚合必须验证 20 次训练及 40 份世界评价；阶段中间报告明确写 partial，未完成组不排序。汇总保留逐种子配对 NLL/AUROC 变化，同时报告 T0–T3 NLL、AUROC、AUPRC、Brier。

pCR best 的已保存患者预测独立重算并核对 manifest；last 核对 checkpoint 对应 epoch 的保存聚合指标。患者行留在原 run，不进入聚合笔记。历史原 L1 和此前 CNN/冻结 VQ 双任务只读核验，拥有各自源码快照。

世界 copy 比值仅在**该 checkpoint 自己的 EMA teacher 坐标**内有效，不能跨模型或 best/last 将 latent L1/比值排序视为共同坐标上的能力排名。超过 copy 不等于对 pCR 有价值。开发验证集反复使用，两个种子均值也不构成独立测试或统计显著性。阶段 3 的累加结果只支持逐项对照，不能把联合收益全部归因到某一个模块；阶段 4 两组保持独立以避免混淆。
