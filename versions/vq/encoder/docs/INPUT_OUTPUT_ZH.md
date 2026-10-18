# VQ encoder：pCR 单任务的输入输出

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

## 本版 VQ 前端和命令

单次可见检查 `[O,3,32,128,128]` 的三个相位分别调用同一个单通道 VQ 连续前端。每相 `encoder + pre_quant` 输出 `[O,8,8,32,32]`，拼接为 `[O,24,8,32,32]`，池化到 `[O,24,2,4,4]`，投影 24→128，再经过空间 Transformer，得到 `[O,32,128]`。这里 `O` 是整个 batch 中可见检查总数。没有离散量化、码本查找、解码器或 GAN 判别器。

正式四次 dynamic 实验均为 **pCR-only BCE**：VQ 冻结阶段训练适配层/共享网络/分类头；微调从同种子冻结 best 初始化完整模型，VQ 前端用 `1e-5`，其他部分 `1e-4`，重置优化器与进度。JEPA/flow/重构/方差权重均为 0；因此 `predict-vla-jepa --forecast-states` 会被拒绝，不能将存在于定义中的 world 模块当成受训练预测器。

从仓库根目录进入本包。先在私有数据目录准备 manifest、缓存以及合法取得的 `symm_world_codec_v2` 安全导出权重，路径示例如下：

```bash
cd versions/vq/encoder
python ../../../tools/prepare_config.py \
  --config configs/vq_frozen_dynamic_seed17.yaml \
  --output local_configs/frozen_dynamic_seed17.yaml \
  --image-cache ../../v1/data/registered_roi32/image_cache_20261004 \
  --codec weights/roi32_codec.pt
PYTHONPATH=src python train.py train-vla-jepa \
  --config local_configs/frozen_dynamic_seed17.yaml \
  --manifest ../../v1/data/registered_roi32/manifest.json \
  --output runs/my_frozen_dynamic_seed17
python ../../../tools/prepare_config.py \
  --config configs/vq_finetune_dynamic_seed17.yaml \
  --output local_configs/finetune_dynamic_seed17.yaml \
  --image-cache ../../v1/data/registered_roi32/image_cache_20261004 \
  --init-checkpoint runs/my_frozen_dynamic_seed17/best.pt
PYTHONPATH=src python train.py train-vla-jepa \
  --config local_configs/finetune_dynamic_seed17.yaml \
  --manifest ../../v1/data/registered_roi32/manifest.json \
  --output runs/my_finetune_dynamic_seed17
PYTHONPATH=src python train.py predict-vla-jepa \
  --checkpoint runs/my_finetune_dynamic_seed17/best.pt \
  --manifest ../../v1/data/registered_roi32/manifest.json \
  --patient-key PATIENT_DEMO --landmark 1 --device cpu \
  --output results/patient_demo_T1.json
```

`PATIENT_DEMO` 要换成自己的患者键；上述 T1 示例假定该患者的 T1 MRI 当时实际可见。评价命令为 `PYTHONPATH=src python train.py evaluate-vla-jepa --checkpoint runs/my_finetune_dynamic_seed17/best.pt --manifest ../../v1/data/registered_roi32/manifest.json --split val --device cpu --output results/validation.json`。仅查看帮助可运行 `PYTHONPATH=src python train.py --help`，不会读取数据。

已完成 checkpoint 含完整前端权重，普通推理无需再次读取外部 codec。阶段 B 初始化和严格恢复会校验来源/配置，移动代码后的旧 checkpoint 续训不能视为自动兼容。可选 T0 配置在 `configs/vq_*_t0_*.yaml`，不属于已完成四次 dynamic 实验。

### 批量队列的资源路径

`run_vq_encoder_experiment.py` 接受 `--manifest`、`--output`、`--codec` 和 `--image-cache`；后者可设为 `none`，直接读原 NPZ。默认 codec 为当前包 `weights/roi32_codec.pt`。启动时在新输出目录的 `input_configs/` 生成仅替换资源位置的运行配置，原始 YAML、损失、模型和训练参数保持原值；自定义输出会使阶段 B 指向同一新队列的阶段 A best。`--resume` 必须保持相同资源参数，资源变化会被拒绝。

例如在本包目录查看队列状态（只读）：

```bash
PYTHONPATH=src python scripts/run_vq_encoder_experiment.py --profile dynamic \
  --manifest ../../v1/data/registered_roi32/manifest.json \
  --output runs/my_vq_queue --codec weights/roi32_codec.pt \
  --image-cache none --status
```

实际启动使用相同参数去掉 `--status`，可按机器设置 `--max-concurrent` 与 `--gpu-memory-budget-mib`。正式资源与 CUDA 环境准备后才运行训练；本次发布检查仅执行帮助及合成 CPU 测试。
