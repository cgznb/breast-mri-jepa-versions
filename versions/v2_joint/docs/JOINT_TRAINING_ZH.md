# 一阶段联合训练：表征＋世界预测＋pCR

日期：2026-10-09（北京时间）。独立代码目录：`/root/autodl-tmp/line b/breast-mri-response-jepa-v2-joint`。

现在可以从随机初始化开始，在同一次优化器更新中联合训练。无需提前运行A或B，也无需传入初始化checkpoint。原始压缩包、评审副本和原 `breast-mri-vla-jepa` 主线保留。

## 实际联合哪些部分

`train.phase: joint` 使在线MRI编码器、表征辅助头、共享融合主干、世界预测器、直接pCR头、未来pCR头和门控一起优化。教师始终无反向梯度，每次优化器更新后对在线编码器做一次EMA。

同批次的总损失包括原模板的以下系数：

| 损失 | 系数 |
|---|---:|
| 同次masked latent prediction | 1.0 |
| 患者轴 pooled variance | 1.0 |
| pooled covariance | 0.04 |
| 同阶段/token患者轴方差 | 0.1 |
| teacher-forcing latent L1 | 0.25 |
| 自主多步latent L1 | 1.0 |
| 融合pCR BCE | 1.0 |
| 直接pCR BCE | 0.5 |
| 未来状态pCR BCE | 0.25 |

本次没有用真实验证效果调权重；系数仍需实验评估。表征辅助支路只使用每患者当前最新合法已观察MRI，不读取未来检查。训练按实际anchor阶段分批，同批患者不重复，使同阶段不同患者用于方差统计；尾批保留。常规batch16，允许单人尾批合并为17人，但合并也必须满足患者不重复。无法安全合并的单人批次保留，患者轴方差项跳过。

固定T0时，表征辅助输入也仅T0；后续真实MRI进入无梯度的世界损失目标。动态时输入截止当前阶段的历史，辅助支路使用最新合法检查。T3仍训练表征和pCR，世界未来损失为0。

联合模式与分阶段模式的重要差别：教师坐标持续EMA变化。不同epoch的raw JEPA误差不能作为固定坐标误差直接解释；每个评估checkpoint使用该时点相同教师计算预测/copy对照。世界状态仍是latent，不是生成MRI。

## 数据与验证

沿用现有manifest：764训练患者、102验证患者，独立test为空。缺失随访不删除，不重复归一化MRI。临床标准化仅拟合训练患者。

固定T0每轮训练764前缀，验证102个T0前缀。动态原配置每人每轮均衡轮换一个合法前缀（`balanced_cycle`）；新增配置可每轮遍历所有合法前缀（`all_prefixes`，见下文）。两种动态采样均验证所有382个前缀，T0/T1/T2/T3为102/100/94/86。验证和最佳权重选择均按患者等权pCR NLL，先平均患者内合法阶段再平均患者；不按最高AUROC另选epoch。

物理batch16、accumulation1、validation batch4、BF16、activation checkpoint开启，最大60epoch、patience15。优化器沿用模板：分类读出LR1e-4，其余可训练部分3e-5，5epoch warmup后cosine；seed17/43各有固定T0、动态轮换和动态全部前缀配置。

## 动态：每个病人的一种情况作为一个sample

使用 `train.prefix_sampling: all_prefixes`。这里一种情况是截至某个合法阶段已获得的累积观测；完整四次检查的病人每轮贡献4个sample：

| sample | MRI输入历史 | 未来MRI监督目标 | pCR监督 |
|---|---|---|---|
| 病人P、T0 | T0 | T1、T2、T3 | P的最终pCR |
| 病人P、T1 | T0、T1 | T2、T3 | P的最终pCR |
| 病人P、T2 | T0、T1、T2 | T3 | P的最终pCR |
| 病人P、T3 | T0、T1、T2、T3 | 无 | P的最终pCR |

临床和治疗信息也按该sample截止阶段的已知状态遮蔽。未来MRI只用于无梯度教师监督，不能进入输入。缺失检查和延迟可用检查按manifest的 `available_at` 处理，不补造观测；例如没有T1的病人，T2时可见历史可能是T0、T2。每位病人每个合法前缀每轮恰一次，跨轮打乱次序；不枚举任意MRI子集、切片或DCE增强相位组合。

沿用实际数据划分：

| 划分 | 独立病人数 | sample数/轮 | T0 | T1 | T2 | T3 |
|---|---:|---:|---:|---:|---:|---:|
| train | 764 | 2657 | 764 | 736 | 633 | 524 |
| val | 102 | 382 | 102 | 100 | 94 | 86 |

先按病人划分train/val，再在各自划分内枚举前缀；同一病人的所有sample必须留在同一划分。2657是相关观测数，独立训练病人数仍为764。训练沿用每batch的sample平均损失，没有加入每患者前缀数的逆权重；有4个前缀的病人每轮贡献比有2个前缀的病人多一倍。验证继续患者等权，避免检查较全的病人主导最佳模型选择。

物理batch16含不同病人的情况，并按最新实际已观察MRI阶段分组。即使延迟可用使同一病人的两个前缀共用一个anchor，也会分到不同批次。当前真实数据每轮167批，大小9–16、累积1即167次更新；原轮换模式epoch0为49批。每轮sample展示数约3.48倍，更新数约3.41倍，实际耗时需测量。配置仍保留60epoch上限和5epoch warmup，因此总更新预算和EMA更新次数增加；比较两种采样的效果应对齐更新数或数据展示预算，不能直接把相同epoch当作相同训练预算。

从本目录启动全部前缀模式：

```bash
cd '/root/autodl-tmp/line b/breast-mri-response-jepa-v2-joint'
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
python train_response.py train \
  --config configs/joint_dynamic_all_prefixes_batch16_seed17.yaml \
  --manifest '/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32/manifest.json' \
  --output runs/joint_dynamic_all_prefixes_seed17
```

第二种子用 `configs/joint_dynamic_all_prefixes_batch16_seed43.yaml` 和新的seed43输出目录。也可在原动态命令增加 `--set train.prefix_sampling=all_prefixes`；改变采样方式需新建run，不能作为旧采样checkpoint的原样resume。此开关也支持动态B/C，B仍剔除没有未来MRI监督的前缀；A保持原单次MRI采样。

## 固定T0启动命令

从本目录运行：

```bash
cd '/root/autodl-tmp/line b/breast-mri-response-jepa-v2-joint'
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
python train_response.py train \
  --config configs/joint_t0_batch16_seed17.yaml \
  --manifest '/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32/manifest.json' \
  --output runs/joint_t0_seed17
```

无需 `--init`。动态版使用 `configs/joint_dynamic_batch16_seed17.yaml`，输出目录换为 `runs/joint_dynamic_seed17`。第二种子使用对应seed43配置和新的seed43输出目录。不要让两个进程同时写同一输出目录。

恢复同一训练使用相同命令附加 `--resume`，从 `last.pt` 的完整epoch边界恢复。源码、配置、数据和运行环境不同会拒绝恢复。本地checkpoint schema独立，不兼容交付包旧格式。

## 验证结果与范围

完整CPU测试：**74 passed in 14.18s**，包括原分阶段路径、联合各模块梯度、未来信息隔离、教师无梯度及EMA一次、全部前缀覆盖与延迟可用性、同批患者不重复、精确CPU断点恢复、联合权重推理、旧归一化元数据兼容与语义变化拒绝。全部前缀的连续2epoch与1epoch后恢复，模型、优化器、随机状态、训练损失和验证结果逐值一致；CPU合成测试不代表真实MRI效果。

实际764/102数据采样检查：固定T0 epoch0为48批（12–16人）；动态轮换epoch0为49批（6–16人），每患者恰一次；动态全部前缀每轮167批（9–16人），每合法患者前缀恰一次。联合训练均同批actual anchor stage一致、患者不重复。没有导出患者记录。全部前缀报告见 `reports/joint_all_prefixes_sampling_20261009.json`，包含epoch0/1/59计划检查。

RTX5090真实训练缓存四个阶段各完成一次可丢弃权重的完整尺寸BF16前向、反向、AdamW、EMA：

| 阶段 | batch | peak allocated MiB | peak reserved MiB |
|---|---:|---:|---:|
| T0 | 16 | 6556.4 | 6716.0 |
| T1 | 16 | 7368.8 | 7516.0 |
| T2 | 16 | 7923.6 | 8018.0 |
| T3 | 16 | 8262.6 | 8384.0 |

全部损失、梯度和更新后参数有限；教师无梯度、EMA逐值符合预期。T3世界参数无梯度是没有未来监督的预期行为，表征和pCR正常反传。这里是四个真实训练批次的门禁，未保存模型权重、未开始完整正式训练，没有真实性能提升结论。CUDA注意力和3D池化仍有非确定性，不保证GPU逐位恢复。

报告：`reports/joint_cuda_batch16_20261009.json`、`reports/joint_sampling_20261009.json`、`reports/joint_validation_20261009.json`记录最初联合实现检查；`reports/joint_all_prefixes_sampling_20261009.json`记录新增采样及当前74项CPU回归检查。采样变化没有改动模型/损失张量路径，实际新增配置最大batch仍为16，本次没有重复GPU门禁或启动正式训练。重现GPU检查：`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/check_joint_cuda.py`。

## 新增源码位置

- `config.py`：joint显式合同与联合配置检查。
- `model.py`：联合trainable策略，教师保留no-grad。
- `objectives.py`：joint_loss，同批表征＋世界＋pCR。
- `engine.py`：同anchor阶段组批、从零初始化、统一更新/EMA、验证和断点恢复。
- `data.py`：实际已观察阶段查询；新增动态全部合法前缀采样。
- `cli.py`、`diagnostics.py`：联合checkpoint预测、评估和诊断。
- `provenance.py`：源码文本/文件元数据身份绑定，符合本工作区不保存校验值规则。

102患者仍是反复使用的开发验证集。后续应比较联合/分阶段和原L1/L0；独立泛化结论需要独立患者测试或严格折内重新训练。

## 已启动的三个种子正式实验

2026-10-09用户授权正式训练。固定T0与动态全部合法前缀均使用seed17/43/71，共6个run；动态每轮2657sample，固定T0每轮764sample。配置的batch16、累积1、BF16、学习率、损失系数、60epoch上限及15epoch早停均沿用上述设置。

`scripts/run_joint_suite.py`创建独立输出、配置快照和日志，在RTX5090上按显存预算最多3路并发，并自动填补结束后的空位。控制器和训练进程独立于终端存活；重复启动时有文件锁，重启能识别并接管本队列仍在运行的训练。每3秒记录整卡GPU利用率、功耗、显存和训练进程显存。默认预留固定T0 8192MiB、动态10240MiB，初始整卡预算29696MiB；正式运行实测动态各约9158MiB、固定T0约7410MiB后，控制器预算提高至31232MiB，以支持三个动态任务同时运行，原训练进程完整保留。训练配置不随并发变化。

训练循环新增及时释放已完成微批的GPU输入/输出/损失引用，避免下批传输及验证期间额外占用；没有改动目标、梯度、优化器或EMA时机。多步真实训练门禁：固定T0进程NVIDIA采样峰值7406MiB，动态9156MiB；随后FP32验证未抬高reserved峰值。见 `reports/joint_sequence_cuda_20261009.json`。

运行与状态：

```bash
cd '/root/autodl-tmp/line b/breast-mri-response-jepa-v2-joint'
python scripts/run_joint_suite.py \
  --output runs/joint_three_seed_20261009 \
  --manifest '/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32/manifest.json' \
  --seeds 17,43,71 --max-concurrent 3 \
  --gpu-memory-budget-mib 31232 --detach
```

`runs/joint_three_seed_20261009/status.json`是队列实时状态；各run的 `history.json`在每轮完成后更新。退出0且真实达到60epoch或早停、最佳/最终权重和指标齐全才算完成，`metrics_best.json`存在不能单独证明完整训练完成。

结果汇总命令（训练未全部完成时生成明确标注的partial报告）：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/analyze_joint_suite.py \
  --suite-root runs/joint_three_seed_20261009 \
  --output reports/joint_three_seed_20261009 \
  --baseline-csv '/root/autodl-tmp/line b/breast-mri-vla-jepa/reports/l_loss_analysis_20261004/pcr_per_seed.csv'
```

汇总独立回放最佳权重预测的三读头分阶段AUC/AP/NLL/Brier、患者等权NLL、各profile三个种子的均值与样本标准差，以及best到last的变化。世界预测在同一checkpoint教师下比较free/copy；不同epoch教师坐标仍不等同。旧主线参照只匹配seed17/43：固定T0的locked A1/L0、动态L1，seed71没有旧对照；训练预算也不相同，不把三种子与旧两种子均值当成严格匹配实验。正式启动核查见 `reports/joint_three_seed_launch_20261009.json`。

已后台启动 `scripts/finalize_joint_suite.py`，每30秒仅读队列；六任务进入终态后自动执行上述分析并调用 `scripts/plot_joint_suite.py`生成PNG/PDF。该进程使用CPU、屏蔽CUDA、不修改队列或停止训练；等待状态在 `runs/joint_three_seed_20261009/finalize_status.json`。全部六个完成且核对通过才标记completed；失败或仅部分结果会明确标记failed/partial。

自动汇总还会执行 `scripts/compare_joint_old_predictions.py`，直接核对新旧manifest、标准化和完整验证病人/标签；只比较已完成且存在旧对照的seed17/43。在阶段内按病人配对bootstrap5000次，并对动态整体NLL/Brier先平均病人内前缀、再平均病人。结果见 `reports/joint_three_seed_20261009/paired_baseline/aggregate.json` 和 `paired_metrics.csv`。bootstrap条件于已经选定的best模型，不包含种子、划分、选模或多重比较的不确定性；不把它当作独立泛化证明。

2026-10-09最后一次完整CPU回归为 **112 passed in 15.47s**，包括原模型训练、全部前缀、持久队列、独立结果回放、历史参照和自动汇总逻辑。图形脚本已对实际部分结果生成并检查六面板曲线，累计更新数横轴避免把不同epoch预算混为一谈。当前正式训练尚未全部结束，结果报告不是最终效果结论。
