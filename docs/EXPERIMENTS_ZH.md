# 历史版本、选择结果和最终优化版

“四个版本”按代码演进分为 V1、V2、VQ 优化、V1-evolution，VQ 优化保留两个完整独立项目。下面都是已有开发验证结果；整理发布期间没有训练真实模型或重新评估患者数据。

## 已有结果

共同历史数据为 764 人训练、102 人开发验证，无独立测试。动态 NLL 定义为先在每位患者内平均合法前缀 NLL，再跨患者平均；固定 T0 仅评价 T0，不能将两种 NLL 直接解释成动态方法的收益。

| 版本 / 方案 | 种子 | 开发验证 NLL ↓ | T0 AUROC | 原始聚合证据 |
| --- | --- | ---: | ---: | --- |
| V1 动态 L1，pCR-only | 17/43 | 0.549020 | 0.740625 | [V1 汇总](../versions/v1/results_summary/README.md) |
| V1 固定 T0 A1 | 17/43 | 0.535521 | 0.738616 | [V1 汇总](../versions/v1/results_summary/README.md) |
| V2 动态 F_forecast | 17/43/71 | 0.571275 | 0.684821 | [V2 最终消融](../versions/v2_joint/results_summary/branch_ablation_20261010/README_ZH.md) |
| VQ encoder frozen，pCR-only | 17/43 | 0.583057 | 0.705580 | [encoder 结果](../versions/vq/encoder/results_summary/VQ_ENCODER_RESULTS_20261012_ZH.md) |
| VQ encoder finetune，pCR-only | 17/43 | 0.587042 | 0.697545 | [encoder 结果](../versions/vq/encoder/results_summary/VQ_ENCODER_RESULTS_20261012_ZH.md) |
| VQ dualtask raw 参考 | 17/43 | 0.553476 | 0.735714 | [dualtask 结果](../versions/vq/dualtask/historical_runs/vq_dualtask_20261012/RESULTS_ZH.md) |
| VQ dualtask frozen | 17/43 | 0.566432 | 0.715179 | [dualtask 结果](../versions/vq/dualtask/historical_runs/vq_dualtask_20261012/RESULTS_ZH.md) |
| VQ dualtask finetune | 17/43 | 0.572514 | 0.717634 | [dualtask 结果](../versions/vq/dualtask/historical_runs/vq_dualtask_20261012/RESULTS_ZH.md) |
| evolution encoder64 | 17/43 | **0.545002** | **0.731696** | [evolution 最终结果](../versions/v1_evolution/historical_runs/v1_evolution_20261012/RESULTS_ZH.md) |

V2 三种子 NLL 标准差为 0.008247；同 17/43 的 V2 均值 NLL=0.567000、T0 AUROC=0.693304，未超过原动态 L1。表中 V2 的种子数与其他行不同，且各模型结构、目标和历史开发过程不同，不把本表当作独立测试排名。各 landmark 的患者集合也可不同。

V1 固定 T0 的单个最低 NLL 是 L3 seed17=0.530131（T0 AUROC=0.748661），但按两种子平均比较，固定方案最好是 A1。单个 best seed、最优组和后续结构版本是不同概念。

## 后期优化版的完整定位

最终 evolution 汇总记录完成 20/20 次训练和 40/40 份世界模型评价，报告时间 2026-10-13 01:48:15+08。最优组为 `vq_forecast_history_residual_free_encoder64`，两次独立运行不构成已评估集成：

| seed | 验证 NLL | 配置 |
| --- | ---: | --- |
| 17 | 0.5398512091 | [完整 YAML](../versions/v1_evolution/configs/evolution_vq_forecast_history_residual_free_encoder64_seed17.yaml) |
| 43 | 0.5501530880 | [完整 YAML](../versions/v1_evolution/configs/evolution_vq_forecast_history_residual_free_encoder64_seed43.yaml) |

两种子平均 T0/T1/T2/T3 AUROC 为 0.731696 / 0.730928 / 0.739583 / 0.720443。对旧动态 L1，NLL 0.549020→0.545002，T0 AUROC 0.740625→0.731696，不能宣称全面改善或统计显著。

主要代码在 [model.py](../versions/v1_evolution/src/mri_vla_jepa/model.py)、[encoder.py](../versions/v1_evolution/src/mri_vla_jepa/encoder.py)、[training.py](../versions/v1_evolution/src/mri_vla_jepa/training.py)。损失实现是模型的 `compute_loss` 方法。结构与配置摘要：

| 项目 | 最优组设置 |
| --- | --- |
| VQ 前端 | 冻结兼容 codec，读取 encoder 的压缩前特征，每相64通道；三相共192通道 |
| token / Transformer | 2×4×4=32 tokens，dim128，4 heads；encoder/fusion/predictor depth2；dropout0.3 |
| 转移 | 固定转移槽、历史差分/缺失信息、残差转移、自主 rollout |
| pCR 输入 | query 池化、预测 T3 池化、T3−当前状态池化，共384维 |
| 损失 | BCE + 0.5×teacher-forcing JEPA L1 + 0.5×free-rollout JEPA L1 |
| 未启用项 | flow、重建、方差、病灶加权 / 病灶池化 |
| 优化器 | AdamW，lr1e-4，weight_decay0.01，grad_clip1，batch16，accumulation1 |
| 训练 / 选择 | BF16，最多200epoch，patience50，每epoch验证；患者等权前缀 pCR NLL 选best |

## V2 最终实验入口

V2 是独立重写，不是 evolution 的另一个名称。最终 branch ablation 完成 D/F/G × fixed/dynamic × 3种子共18次训练，动态 F_forecast 在该轮内部最好。

正式冻结配置在 [branch_ablation_20261010/input_configs](../versions/v2_joint/historical_runs/branch_ablation_20261010/input_configs/)，例如 [F_forecast_dynamic_seed17.yaml](../versions/v2_joint/historical_runs/branch_ablation_20261010/input_configs/F_forecast_dynamic_seed17.yaml)。使用 [scripts/train_overfit.py](../versions/v2_joint/scripts/train_overfit.py)，实际上限3000次优化器更新、warmup200、每100次验证、patience500；旧 epoch 字段不控制这轮实验。架构、联合损失和全部命令见 [V2 输入输出](../versions/v2_joint/docs/INPUT_OUTPUT_ZH.md)。

## 权重位置与使用范围

最优训练权重保留在原研究服务器 `seetacloud-bjb2`，未上传 GitHub：

```text
# evolution，seed分别为17、43
/root/autodl-tmp/line b/breast-mri-vla-jepa-v1-evolution/runs/v1_evolution_20261012/vq_forecast_history_residual_free_encoder64_seed17/best.pt
/root/autodl-tmp/line b/breast-mri-vla-jepa-v1-evolution/runs/v1_evolution_20261012/vq_forecast_history_residual_free_encoder64_seed43/best.pt

# V2，seed分别为17、43、71
/root/autodl-tmp/line b/breast-mri-response-jepa-v2-joint/runs/branch_ablation_20261010/F_forecast_dynamic_seed17/best.pt
```

相同架构不同 seed 使用各自完整 checkpoint，加载其配置、标准化和 EMA 等状态。没有新增外部测试、阈值优化或跨 seed 集成评估。新实验应重新明确数据划分、候选、损失及统一验证选择规则；独立测试只在选择冻结后使用。
