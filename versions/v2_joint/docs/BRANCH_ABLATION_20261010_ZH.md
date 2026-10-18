# 直接分支、未来分支和融合读出的消融

2026-10-10 08:42:19（北京时间）启动。目标：`/root/autodl-tmp/line b/breast-mri-response-jepa-v2-joint`。

## 已有消融和已有单分支预测

此前B0–B4比较患者等权、forecast梯度截断、同前缀一致性及组合，属于训练方法消融；没有分别从头重训单一预测读出。

当前两条pCR预测路径是：直接分支读取合法影像历史的融合特征；未来分支读取世界模型自由递推至T3的状态及条件；门控按概率混合两者。原B0动态模型的三个种子，均按融合NLL选每个种子的最佳checkpoint。其已有各head的结果如下，**不是单分支重新训练结果**。

| 输出 | 整体患者等权NLL（均值 ± 标准差） | T0 AUC均值 | T1 AUC均值 | T2 AUC均值 | T3 AUC均值 |
|---|---:|---:|---:|---:|---:|
| direct | 0.626541 ± 0.009262 | 0.523214 | 0.576593 | 0.592187 | 0.595649 |
| forecast | 0.615466 ± 0.049002 | 0.635565 | 0.672181 | 0.651736 | 0.648810 |
| fused | 0.607351 ± 0.017117 | 0.592708 | 0.636029 | 0.629340 | 0.621100 |

未来head在各阶段平均AUC更高，融合输出整体NLL更低；不能由此断言单独训练未来分支更好。NLL与AUC衡量不同性质，且现有checkpoint由融合NLL选出。三个种子全部指标与标准差见报告CSV。

## 新增结构消融

| 方案 | 最终分类输出 | 分类监督 |
|---|---|---|
| D_direct | 直接head | 1 × BCE(direct) |
| F_forecast | 未来head | 1 × BCE(forecast) |
| G_gated | 两head的门控概率融合 | 1 × BCE(fused) |
| O_original（复用） | 原B0门控融合 | 原有融合1 + 直接0.5 + 未来0.25 |

三种新方案每种均训练固定T0和动态、seed17/43/71，共18次。O_original复用上一轮六个B0参考，无需重跑。

D/F/G均关闭额外direct/forecast BCE，统一最终输出BCE系数1，使三组之间的分类监督强度一致。G也需要新训练，不能直接用带额外分支监督的旧B0替代。G与O_original可另外观察移除辅助分类监督的影响；D/F与O差异不能全部归因于结构。

三组保留同样A表征学习与B世界预测损失，继续ABC联合训练。D只切换pCR输出路径，世界模型仍接受B的辅助监督；本轮不等同于移除全部世界模型或只做pCR监督。F中直接head没有分类梯度，备用诊断输出不参加主比较。测试实际验证各head和门控的梯度路径。

完整网络按相同种子初始化，不使用最佳权重热启动。原764训练/102开发验证划分、影像和临床归一化、缓存、all_prefixes、batch16/累积1、BF16、LR1e-4/backbone3e-5、EMA.996及A/B系数均沿用B0。

最多3000 updates、warmup200、每100验证、patience500、min_delta1e-5。所有组按患者等权最终输出NLL选模。单读出模型结果文件中的fused字段是最终所选输出：D时等于direct，F时等于forecast，并不表示仍使用混合。

## 运行与核验

- 冻结协议与18份配置：`runs/branch_ablation_20261010/experiment_plan.json`、`input_configs/`。
- 总控制器：`runs/branch_ablation_20261010/research_status.json`。
- 显存队列、日志、监测：同目录`status.json`、`logs/`、`gpu_monitor.jsonl`。
- 报告：`reports/branch_ablation_20261010/README_ZH.md`、`aggregate.json`、`primary_per_seed.csv`、`primary_stage_aggregate.csv`。
- 原head完整统计：`original_branch_nll.csv`、`original_branch_stage_aggregate.csv`。
- 真实CUDA检查：`reports/branch_cuda_gate_20261010.json`。

后台最多三路并行，整卡预留31232MiB；动态每路10240MiB，固定T0每路8192MiB。真实batch16三组都验证T0/T3前反向、AdamW、教师无梯度、EMA更新、FP32验证及梯度路径。采样进程峰值D/F/G分别9420/9334/9334MiB。单独采样峰值不是精确运行上限，队列仍保留余量。

相关72项CPU测试全部通过（18.43秒），包括三种实际小规模更新制训练完整落盘与严格回放、损失强度及梯度隔离、计划不可覆盖、恢复控制器复用活队列、错误不能宣称完成、已有队列/入口/分析回归。

训练包源码保持不变；只新增外部编排、分析、检查与测试。完成后严格CPU核对配置、冻结源码、数据、归一化、环境、游标、选模和保存预测。部分报告会明确完成数量；全部18次核验通过才将研究标记完成。

```bash
cd '/root/autodl-tmp/line b/breast-mri-response-jepa-v2-joint'
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=src \
/root/miniconda3/bin/python scripts/run_branch_ablation.py --detach
```

重复启动会保留现有控制器；队列不会自动重训完成或失败任务。训练结束自动写最终报告，无需手动重训或GPU推理。

## 解释范围

T0/T1/T2/T3的验证人数为102/100/94/86，不能把不同阶段总体均值视为同一患者纵向增益。T3没有更晚影像，forecast读出真实T3教师特征及条件，不代表预测未来。原102人已经反复用于开发验证，所有结果仍不构成独立测试结论。
