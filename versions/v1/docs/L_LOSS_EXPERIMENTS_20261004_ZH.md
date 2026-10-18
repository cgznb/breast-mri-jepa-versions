# ROI32 L1-L4 损失消融训练

日期：2026-10-04。工作目录：`/root/autodl-tmp/line b/breast-mri-vla-jepa`。

## 固定的共同策略

按此前方案的两种子平均最佳选模 NLL 固定每个版本的 A*，不按种子分别选超参数：固定 T0 使用 A1（dropout=0.1，Plateau 调度），动态使用 A2（dropout=0.3，恒定学习率 1e-4）。固定 T0 的 A1/A0 最佳 NLL 几乎相同，但 A1 减轻后期恶化；动态 A2 的收益在两个种子间不一致，选择它是遵循预定主指标，不代表稳定改善。依据见 A1/A2 结果（未随源码发布的历史工件：`../reports/A1_A2_RESULTS_20261004_ZH.md`）。

L0 复用 `runs/registered_roi32_a1_a2_20261004/a1_t0_seed{17,43}` 与 `a2_dynamic_seed{17,43}` 的原完整损失结果。L1-L4 分别从相应种子重新初始化，使用新输出；不能用旧权重续训到不同损失配置。

| 实验 | pCR | JEPA | 重构 | 方差 | 定义 |
|---|---:|---:|---:|---:|---|
| L0，已有对照 | 1 | 0.5 | 0.1 | 0.01 | 原定义 |
| L1，pCR-only | 1 | 0 | 0 | 0 | 关闭 |
| L2，去重构 | 1 | 0.5 | 0 | 0.01 | 原定义 |
| L3，去 JEPA | 1 | 0 | 0.1 | 0.01 | 原定义 |
| L4，患者轴方差 | 1 | 0.5 | 0.1 | 0.01 | 同阶段、同 token、同通道，患者轴 FP32 方差 |

L1-L4 各固定 T0/动态两版、seed17/43，总计 **16 次新训练**。均为物理 batch16、梯度累积1、最多200 epoch、patience50、min_delta0、AdamW、clip1.0、CUDA BF16。每轮764训练患者各一次，48批，保留末批12人。所有患者、三相顺序、ROI32、临床处理、已冻结影像标准化和合法阶段采样不变；MRI flow 关闭。

固定版仍按 T0 NLL 选模，动态仍按患者内平均合法前缀 NLL、再对患者平均选模。102人仍是开发验证集，独立测试为空。每个配置报告所有种子，不能按最高 AUROC 重选 epoch。

## 实现边界

- 监督训练允许辅助权重为0；禁用的损失分支不执行。L1/L3 的训练与固定诊断只读取合法已观察 MRI，不读取未来目标影像；pCR 标签仍单独读取。普通 pCR 前向不接收真实未来或目标标签。
- L4 将观察编码放回 `[B,4,N,C]`，按有效患者轴计算 FP32 方差，`unbiased=false`、epsilon=1e-4、标准差目标1；阶段有效患者不足2时跳过，对其余阶段平均。原方差定义继续支持 L0-L3，末批不丢弃。
- 模型结构、维度和粗重构方式保持原样，未同时修改网格、解码器、JEPA 权重幅度或配准质量筛选。L1 是纯监督损失对照，网络仍保留原查询和未激活的辅助模块。
- 配置、数据结构、影像路径/大小/修改时间、训练标准化、患者分区和源码文本继续绑定 checkpoint；源码或配置改变时不能恢复正式 run。历史 A0/A1/A2 只读加载和已有结果保留。新配置和报告不记录 checksum 值。

## 诊断与独立世界评价

所有新实验每5轮使用同一诊断种子20261004固定选择32名训练患者；每名患者固定一个合法阶段，分B16计算。诊断使用 eval 模式、保存并恢复 Python/NumPy/Torch CPU/CUDA RNG及各模块模式，不更新参数、EMA或已有梯度，也不改变下一批采样。

`diagnostics.json` 记录固定训练探针的 FP32 pCR NLL、各加权损失按模块的梯度 L2/与 pCR 的夹角、患者轴方差与有效秩、教师尺度及相对上次诊断的漂移。每轮 `epoch_history.json` 还记录所有批次的裁剪比例；真实 AdamW 参数更新 L2 仅采样诊断轮第一批，并明确标注范围。探针结果不代表全验证集指标，也不用于选模。

世界对照均使用同一 checkpoint 的冻结 EMA 教师坐标：真实连续链上的 teacher forcing、自主1/2/3步预测及对应 copy；同时单列全部合法相邻 teacher-forcing 转移，缺T1患者的真实 T2→T3不会丢弃或拼成T0→T2。按患者先平均再跨患者平均，报告误差/copy比和 `skill=1-ratio`；copy接近0时不计算比值。不同 checkpoint 的 raw latent L1 不属于统一坐标，不据它直接排名。

每组训练结束后，队列独立评估完整102人验证集的 `best.pt` 与 `last.pt`，写 `world_best.json`、`world_last.json`。评估与训练共同计入并发槽位和显存预算；L2/L4 的独立世界评估预留10 GiB。L1/L3 在CPU加载并明确报告世界头未训练、指标不可用，不把随机预测当作有效世界模型。

2026-10-04 用户追加授权增加并行度后，队列采用**最多六路、总任务显存预算29 GiB**的准入规则。当前GPU为32607 MiB的RTX 5090，保留约2.8 GiB余量。单任务预算包括完整阶段训练/诊断峰值及CUDA上下文和分配器余量：L1固定T0为4 GiB，L3固定T0为4.5 GiB，L2/L4固定T0为6.5 GiB；L1/L3动态为7.5 GiB，L2/L4动态为9 GiB。L3固定T0的首次真实并行检查观察到4110 MiB进程显存，因此上调了原先3.75 GiB的估计；GPU进程占用包含CUDA上下文及缓存，不等于PyTorch allocated。动态任务按合法阶段最大峰值预留，不按当前较低的瞬时占用再超额启动。启用预算时先安排预留较小的训练任务，暂时放不下的任务可被较轻任务绕过；只改变队列次序，训练患者排列和阶段采样仍由各训练进程按原配置控制。实际并行数随组合变化，重型动态阶段可回落三路。测量与迁移记录见 GPU并发报告（未随源码发布的历史工件：`../reports/roi32_gpu_concurrency_20261004.json`）。

L0 的四组对照已补齐 best/last 独立世界评价，八份聚合在 reports/l0_world_20261004（未随源码发布的历史工件：`../reports/l0_world_20261004/`）。使用历史权重时公开标明其源码快照与当前损失消融源码不同；评估核对数据、患者分区及训练标准化，编码器、预测器和自主推演结构未改变，原权重未更新。

当前 manifest 未保留逐转移的空间验证字段，因此本轮配准 verified 子组标为不可用；不为此改患者筛选或生成新的质量标签。世界评估失败单独记为 `evaluation_failed`，训练权重保留；恢复队列时仅重试评估，不重新训练已完成组。

## 验证与启动

完整测试和合成/真实门禁结果见 生产设置验证（未随源码发布的历史工件：`../reports/roi32_loss_verification_release_20261004.json`）。门禁包括八个合成配置的随机训练与断点恢复、24个真实完整尺寸 B16 的 T0/T2/T3 损失/梯度/优化器/EMA检查、两条真实 CUDA 恢复路径和八个真实 B16 固定诊断。CPU可逐值恢复；原CUDA BF16内核仍有数值非确定性，恢复检查比较精确采样/RNG/游标及重复连续训练的数值对照，不声称GPU权重逐位相同。

正式配置位于 `configs/registered_roi32_epoch200_l{1,2,3,4}_{t0,dynamic}_seed{17,43}.yaml`，包内资源配置逐字一致。正式输出为 `runs/registered_roi32_l1_l4_20261004`。

在上述工作目录运行：

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
/root/miniconda3/bin/python scripts/run_registered_roi32_experiments.py \
  --suite losses --max-concurrent 6 --gpu-memory-budget-mib 29696 --detach

PYTHONPATH=src /root/miniconda3/bin/python \
  scripts/run_registered_roi32_experiments.py --suite losses --status
```

停止后恢复同一队列：

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
/root/miniconda3/bin/python scripts/run_registered_roi32_experiments.py \
  --suite losses --max-concurrent 6 --gpu-memory-budget-mib 29696 --resume --detach
```

队列会拒绝重复控制器、错误 suite、配置变化和不匹配的 checkpoint。已有控制器运行时不能直接重复上述恢复命令。替换控制器需要先核实PID与启动时间、仅停止控制器，保留并接管训练/评估子进程；权重与进度继续使用原输出目录。训练源码和配置保持冻结。监控以 `controller.json`、各 run 的 `progress.json`/`epoch_history.json` 为准；启动回执和本说明不表示所有实验已经完成。短时GPU利用率或不同任务混合的总更新数不是严格训练加速对照，不据此断言六路恒为最优。

L5、后续权重扫描、MRI-only及临床+Arm输入对照仍按[原计划](OVERFITTING_EXPERIMENTS_20261004_ZH.md)作为后续独立实验，未混入这16次损失消融。
