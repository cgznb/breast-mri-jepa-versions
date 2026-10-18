# V1：原始 MRI VLA-JEPA 的输入输出

## 文件与可见性

实际实现以 [`contracts.py`](../src/mri_vla_jepa/contracts.py)、[`RawMRIStore.batch`](../src/mri_vla_jepa/data.py) 和 [`predict`](../src/mri_vla_jepa/training.py) 为准。

输入 JSON manifest 使用 `schema=responsewm_raw_mri_v1`。每位患者有独立 `split`、四槽 `visits`、临床值及各自 `known_at`、Arm 及可用阶段；训练/评价另有最终二分类 `target.pcr`。T0 必须存在，缺失随访用 `null`，不能把缺失 T1 后的 T2 移到 T1 槽。所有检查使用 T0 定义的 RAS 网格及 ROI；本包不含从 DICOM 发现序列的完整流程。

**三相和纵向是两个不同维度**：一次检查的三个通道按 `pre_aqc0`（增强前）、`first_post_aqc1`（首个增强后）、`metadata_late`（元数据确定的晚期相）排列；四个纵向槽为 T0/T1/T2/T3，不是四个 DCE 相位。

| 输入字段 | 批量形状/类型 | 含义 |
| --- | --- | --- |
| `images` | float `[B,4,3,D,H,W]` | 仅截至 landmark 合法可见的 MRI，隐藏槽为全零 |
| `observed_mask` | bool `[B,4]` | 每个纵向槽是否真实可见 |
| `landmark` | int64 `[B]` | 当前阶段 0/1/2/3 |
| `clinical` | float `[B,4]` | 年龄、HR、HER2、MammaPrint（MP）；不可见项为零 |
| `clinical_mask` | bool `[B,4]` | 临床字段此刻是否已知；区分缺失与真实数值 0 |
| `arm_id` | int64 `[B]` | 0=未知；1–13 为代码内固定顺序的官方治疗 Arm |
| `arm_mask` / `arm_known_at` | bool / int64 `[B]` | 是否可见及可用阶段；不可见时 ID=0、known_at=-1 |
| `query_mask` | bool `[B,4]` | 所求未来阶段，即阶段大于 landmark；不透露未来 MRI 有无 |

正式 ROI32 输入为 **`[B,4,3,32,128,128]`**：每次检查从 NPZ 的 `image` 数组读取，必须为 float32 `[3,32,128,128]`。保持原已声明的训练强度尺度与零背景，不再次 resize 或逐患者 z-score。NPZ 可能另有准备阶段使用的 support/ROI 信息，但 `RawMRIInput` 没有分割 mask 通道；上表中的 mask 均是可用性 mask。

原始 MRI 模式还支持单个堆叠 NPY 或三个 3D NPY/NIfTI：按照 T0 的每相均值和标准差归一化，并变换到配置的 `image_shape`。这是另一种预处理合同，不可与已经归一化的 ROI32 缓存混用。年龄只按训练集中 T0 已知值的统计量标准化；HR/HER2/MP 保留 0/1。推理从 checkpoint 恢复归一化统计。

Arm 是治疗分配类别，不含剂量、疗程和真实阶段间治疗变化。历史实验使用 `assigned_arm_scenario`、从 T0 给定 Arm；这需要明确作为实验假设。各字段的可用阶段受 manifest 约束。

## 训练监督与推理边界

训练在输入 `inp` 之外构造 `RawMRISupervision`：`future` 为 `[B,4,3,D,H,W]`，`future_mask` 为 bool `[B,4]`；`label` 为最终 pCR 的 `[B]` 0/1，`label_mask` 为 bool `[B]`。世界损失只监督真实存在的未来检查，EMA teacher 的目标状态为 `[B,4,N,d]`。没有未来转移的 T3 仍可训练 pCR。

固定 T0 使用 `[I0,0,0,0]`；动态版按患者采样一个合法阶段，只输入截至该阶段的实际观察历史。推理执行 `batch(..., supervised=False)`，不读取 pCR 真值或未来 MRI 内容。评价计算分类指标需要 pCR 真值；世界模型评价还需要未来真值，它们只用于监督/评分。

## 模型输出与保存文件

默认 `token_grid=(2,4,4)`，每次检查有 `N=32` 个空间 token，维度 `d=128`。直接前向 `model(inp)` 返回：

| 输出字段 | 默认形状 | 解释 |
| --- | --- | --- |
| `pcr_logit` | `[B]` | 最终 pCR 终点的 logit；sigmoid 后为概率 |
| `pcr_features` | `[B,4,128]` | 四个 pCR 查询特征；这里的 4 是查询数 |
| `dynamics_features` | `[B,4,8,128]` | 每阶段八个状态转移查询特征 |

`forecast_states`/`forecast` 才执行自主多步预测，未来状态为 **`[B,4,32,128]`**；未查询槽为零。它是 encoder/EMA teacher 坐标中的连续特征，不是 MRI 图像、分割或离散码本 ID。V1 及本 VQ 系列的 pCR 头直接读取观察主干，未来预测状态不回送 pCR 头。

CLI `predict-vla-jepa` 保存 JSON，主要字段为 `pcr_probability`、`landmark`、`observed_stages`、`requested_stages` 和 Arm 可见性。加 `--forecast-states` 保存同名 `.states.npz`，包含 `state_prediction`、`query_mask`。单患者 CLI 的 `B=1`，因此状态数组为 `[1,4,32,128]`，JSON 中 pCR 概率为标量。正式配置均关闭 flow，不能用这些权重生成 MRI；可选 `--generate` 需要另行训练开启 flow 且有正 flow 损失的模型。

训练目录含 `best.pt`、`last.pt`、`config.json`、`normalization.json`、`progress.json`、`history.json`、`epoch_history.json`、`best_validation.json`、`train_report.json`，按配置还会有 diagnostics。评价 JSON 包含聚合 AUROC/AUPRC/NLL/Brier 及患者预测 `rows`；新生成文件应保留在个人数据环境中。本仓库发布代码、配置和已有聚合结果，不附临床数据、正式权重或患者预测行。

固定 T0 按 T0 验证 NLL 选轮；动态版先在每位患者内平均合法前缀 NLL，再跨患者平均。不同定义的选模 NLL 不能解释随访收益；pCR 选出的 best 也不是世界预测最优权重。历史实验 764 人训练、102 人开发验证、无独立测试；新实验的数据与评价方案需另行确定。

## 本版方案与命令

V1 原始编码器从三相 MRI 学习空间 token。A0/A1/A2 和 L1–L4 都在本包；损失由配置决定。按两种子平均预定验证 NLL，固定 T0 最好为 A1，动态最好为 L1。**L1 的 JEPA 权重为零，不能导出受训练的未来状态**；需要状态预测时应使用有正 JEPA 权重的方案，例如动态 L2。

以下命令从仓库根目录进入本包；`data/registered_roi32/manifest.json`、可选缓存和权重须由使用者准备。先把历史配置的资源路径生成到新文件，科学参数保持原值：

```bash
cd versions/v1
python ../../tools/prepare_config.py \
  --config configs/registered_roi32_epoch200_l1_dynamic_seed17.yaml \
  --output local_configs/l1_dynamic_seed17.yaml \
  --image-cache data/registered_roi32/image_cache_20261004
PYTHONPATH=src python train.py train-vla-jepa \
  --config local_configs/l1_dynamic_seed17.yaml \
  --manifest data/registered_roi32/manifest.json \
  --output runs/my_l1_dynamic_seed17
PYTHONPATH=src python train.py evaluate-vla-jepa \
  --checkpoint runs/my_l1_dynamic_seed17/best.pt \
  --manifest data/registered_roi32/manifest.json \
  --split val --device cpu --output results/my_l1_validation.json
PYTHONPATH=src python train.py predict-vla-jepa \
  --checkpoint runs/my_l1_dynamic_seed17/best.pt \
  --manifest data/registered_roi32/manifest.json \
  --patient-key PATIENT_DEMO --landmark 1 --device cpu \
  --output results/patient_demo_T1.json
```

把 `PATIENT_DEMO` 替换为自己的 manifest 中的患者键；T1 示例假定该患者的 T1 MRI 当时实际可见。固定 T0 配置将首条命令的输入换成 `registered_roi32_epoch200_a1_t0_seed17.yaml`，输出另起新目录；固定 T0 权重只允许 `--landmark 0`。动态 L2 等训练过 JEPA 的 checkpoint 可在最后一条命令加 `--forecast-states`。要复核帮助而不读取数据，运行 `PYTHONPATH=src python train.py predict-vla-jepa --help`。

仅使用虚构数据的 CPU 工程示例为 `PYTHONPATH=src python train.py smoke-vla-jepa --config configs/raw_vla_jepa_smoke_dynamic.yaml --output runs/my_synthetic_smoke`，需新输出目录。它不能复现临床成绩。
