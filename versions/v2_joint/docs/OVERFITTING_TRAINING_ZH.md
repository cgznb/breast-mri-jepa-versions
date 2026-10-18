# 过拟合改进训练已启动

启动时间：2026-10-09 19:14:13（北京时间）。代码目录：`/root/autodl-tmp/line b/breast-mri-response-jepa-v2-joint`。

实时状态以 `runs/overfit_joint_20261009/research_status.json` 和 `screening/status.json` 为准；本文记录启动协议，不是最终效果报告。

## 当前筛选

动态模型 seed17，ABC 联合、实际 batch16、累积1、BF16，从相同初始化和相同数据比较五组：

| 组 | 改动 |
|---|---|
| B0 | 原算法，在新的共同更新预算下训练 |
| B1 | 患者等权逐样本损失 |
| B2 | forecast_detach，截断分类到终末预测特征的直接梯度 |
| B3 | 同一前缀两次 dropout 分类的一致性约束 |
| B4 | 三项组合 |

训练患者内部固定611人训练、153人验证，动态前缀数2138/519。原102人验证集完全排除于筛选。影像与临床归一化仅由611名训练患者拟合，三相位2657组缓存全部通过像素检查。

学习率沿用 head1e-4 / backbone3e-5；共同 warmup200 updates、最多3000 updates、每100 updates验证、连续500 updates未改善早停。按患者等权 fused NLL 选 best.pt。一致性系数从0增加到0.1，用500 updates预热。所有组共用参考 batch 分母16，批统计正则独立计算。

初始启动 B0/B1/B2 三路并行，B3/B4自动排队。实测真实 B16 进程峰值：B0约9180MiB，B3约12956MiB，B4约13018MiB。队列最多三路、整卡预算31232MiB，普通动态任务预留10240MiB，一致性动态任务预留14336MiB。每3秒记录GPU状态并自动填补空位。

## 自动复验

五组全部完成后，CPU分析器核对配置、源码、数据、归一化、环境、更新游标、选模和保存预测；仅完整核对通过的结果进入选择。候选在B1/B2/B3/B4中按内部验证患者等权NLL最小选择，字母编号打破相等值。候选未达到NLL降低0.01的预设筛选目标时，也会明确记录并作探索性复验，不声称已有改进。

锁定候选后，在原764训练／102开发验证划分上自动训练：

- 选中方案：固定T0、动态各seed17/43/71。
- 相同预算的B0对照：固定T0、动态各seed17/43/71。

通常为12次复验。若B1胜出，其固定T0权重全部为1，数值算法与B0相同，复用B0的三个T0 run，因此实际9次复验。旧六次实验保持历史参照，不代替新的预算匹配对照。

本轮先做单个严格内部划分筛选，再作三种子开发复验。没有自动开展额外三个内部数据划分或嵌套交叉验证；独立泛化验证仍是后续研究工作。原102人已反复使用，不能称为独立测试集。

## 路径与后台进程

- 总研究状态：`runs/overfit_joint_20261009/research_status.json`
- 固定协议：`runs/overfit_joint_20261009/protocol.json`
- 筛选计划：`runs/overfit_joint_20261009/screening/experiment_plan.json`
- 筛选队列／日志：`screening/status.json`、`screening/logs/`
- 内部数据凭证：`data/overfit_development_20261009/COMPLETE.json`
- 完成后分析：`reports/overfit_joint_20261009/{screening,replication}/`
- 启动核对：`reports/overfit_joint_launch_20261009.json`

后台研究控制器、显存队列和训练进程独立于对话终端运行。重复启动命令会检查现有研究；队列有锁和精确PID身份检查，成功及失败任务均不擅自重训，中断的本次checkpoint可按协议恢复。

```bash
cd '/root/autodl-tmp/line b/breast-mri-response-jepa-v2-joint'
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python scripts/run_overfit_research.py --detach
```

## 验证与训练期间约束

启动前完整CPU回归198项通过（23.61秒），真实MRI B16门禁B0/B3/B4全部通过：多次前向、反向、AdamW、教师EMA和FP32验证均有限，教师无反向梯度。更新级CPU中断恢复逐值通过。结果分析器与后台研究恢复逻辑另有针对性测试通过。

训练期间包源码和正式配置冻结；报告、外部分析脚本和项目记忆可更新，不能混入新的训练实现。初期各组验证值仅代表进度，最终结论以核对后的完整三种子结果为准。

## 初筛收尾故障与恢复

五组均完成600次更新，在100次更新达到最佳，随后连续500次未改善而正常早停。权重、训练记录与预测结果均已保存；旧包CLI在最后打印结果时读取不存在的`best_epoch`，导致五组退出码均为1，研究队列于2026-10-09 19:46:36（北京时间）停止，尚未进入复验。

严格CPU回放五组全部通过，内部验证最佳患者等权NLL：B0=0.632282、B1=0.632169、B2=0.632341、B3=0.632293、B4=0.632301。B1仅降低0.000113，未达到预设0.01目标；继续复验属于探索性验证，尚无有效改善结论。

恢复工具`scripts/recover_overfit_finalization.py`只接受这一明确的终末打印错误，并再次核对冻结源码和完整训练产物；持有研究、队列及worker锁后记录恢复凭证，再修正队列完成状态。原始退出码1、日志、历史状态保留，已完成训练不重跑。凭证为`screening/post_training_recovery.json`。

后续队列使用外部入口`scripts/train_overfit.py`直接调用同一冻结训练函数，正确打印`best_update`。训练包、网络、损失、配置与数据不变；训练异常仍直接报错。原包`response_jepa.cli train`仍保留旧接口，更新制实验请使用上述队列或外部入口，以保持既有checkpoint的源码严格可回放。

2026-10-09 20:04（北京时间）已恢复研究控制器，锁定B1进入9次复验。20:05:45核对3路运行、6路排队；三路进程身份、配置、包源码、数据、归一化、环境及best/last checkpoint身份全部匹配。显存25803MiB、瞬时GPU利用率92%。证据：`reports/overfit_joint_recovery_20261009.json`。新增入口/新旧队列51项、分析/研究23项、恢复16项测试全部通过。
