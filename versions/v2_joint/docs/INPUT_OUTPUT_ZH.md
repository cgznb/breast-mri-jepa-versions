# V2 的数据输入、监督及输出

按本包实际 `contracts.py`、`data.py`、`model.py`、`objectives.py`、`engine.py`、`update_engine.py` 和 CLI 核对。下列维度以历史正式 `F_forecast_dynamic` 配置为例；合成测试使用更小体积和模型。

## 输入文件与预处理边界

入口读取 JSON manifest，schema 为 `responsewm_raw_mri_v1`，时间轴为阶段索引 T0/T1/T2/T3（0/1/2/3），不是连续日历时间。每名患者固定保留 4 个 visit 槽位，缺失随访为 `null`，基线 T0 必须存在。每次实际检查读取一个 NPZ，数组键为 `image`，类型为有限值 `float32`，形状为 **`[3,32,128,128]`**。

三个通道是同次 DCE-MRI 的三个相位：

| 通道 | phase_order | 含义 |
|---|---|---|
| 0 | `pre_aqc0` | 对比前相位 |
| 1 | `first_post_aqc1` | 首个增强相位 |
| 2 | `metadata_late` | 元数据指定的晚期增强相位 |

本包读取上游已配准的 ROI32，**不直接读取 DICOM/NIfTI**。真实影像模式必须为 `preprocessed_roi32`，声明 T0 决定 ROI/参考网格、RAS、source_grid；检查需共享患者的 T0 网格。影像已按训练集拟合的规则归一化，背景零保持为零；manifest 要记录 mean/std、fit_split=`train`、scope、normalization_policy、already_normalized=true、background_policy=`preserve_zero`。本包不会再次 z-score、调整真实 ROI 尺寸或依未来检查重新裁剪。可选 image cache 是上游 NPZ 对应的 float32 NPY 缓存，须通过元数据及文件身份校验。

manifest 每名患者还包含 `patient_key`、`split`（train/val/test）、geometry、visits、clinical、arm 和可缺失的 target.pcr。visit 记录 stage、available_at、grid_id、image；`available_at` 控制检查何时实际可用。四项临床信息依次为 `age_at_screening`、`hr_positive`、`her2_positive`、`mammaprint_binary`，各有 known_at；未知值和时间用 null。年龄仅用训练集 T0 已知的年龄计算 mean/std，三个二元变量保持 0/1。推理使用 checkpoint 内保存的临床尺度，影像尺度也必须与训练一致。

Arm 是 13 个官方治疗组类别及一个未知类别，不是药物剂量序列。manifest 需声明 `label`、`known_at_stage` 和 `semantics`（`prospective_verified` 或 `assigned_arm_scenario`）。只有当前阶段已知的类别才进入输入；类别及顺序见 `data.py:ARMS`。指定治疗组下的条件预测不能解释为因果治疗效果。

## 模型接收的张量

`B` 为 batch size；`Inputs` 不包含未来监督或 pCR 标签。

| 字段 | 形状/类型 | 约定 |
|---|---|---|
| `images` | `[B,4,3,32,128,128]`，浮点数 | 四阶段 MRI；不可观察槽位必须全零 |
| `observed` | `[B,4]`，bool | 截止 landmark 合法且已到达的 MRI |
| `landmark` | `[B]`，int64 | 当前预测阶段 0–3 |
| `clinical` | `[B,4]`，浮点数 | 归一化年龄、HR、HER2、MammaPrint；不可用值为零 |
| `clinical_mask` | `[B,4]`，bool | 各项临床变量是否已知 |
| `arm_id` | `[B]`，int64 | 0 未知，1–13 已知治疗组 |
| `arm_known_at` | `[B]`，int64 | 已知阶段；未知为 -1 |
| `query_mask` | `[B,4]`，bool | 只选择返回哪些未来潜状态，不改变分类或训练损失 |

这些 mask 表示阶段/字段可用性和输出选择，**不是肿瘤分割标签**。三个 DCE 相位始终共同输入。模型从 observed 中最新合法的真实 MRI 初始化自由递推，缺失中间检查时仍可预测后续潜状态。

固定 T0 配置仅使用 T0；动态配置读取当前 landmark 截止的全部合法历史，例如 T2 时可读取已到达的 T0/T1/T2。MRI、clinical 和 Arm 均按可用时间遮蔽。动态训练 `all_prefixes` 把每个合法前缀作为一个样本，同批按最新实际观察阶段分组并避免同患者重复。历史 F_forecast 使用 `sample_weighting: sample`、固定参考 batch=16，训练是前缀样本等权；验证仍先在每名患者内部平均各前缀 NLL，再跨患者平均。

## 训练监督与最优配置

独立的 `Targets` 提供 `future=[B,4,3,32,128,128]`、`future_mask=[B,4]`、二元最终 pCR `label=[B]` 和 `label_mask=[B]`。未来真实 MRI 只用于损失：EMA 教师提取目标 latent，以及 teacher-forcing 支路。未来监督不进入最终 pCR 分类前向输入；推理通过 `future=False, labels=False` 采样，无需读取未来影像内容或 pCR 标签。

联合训练包括最新合法真实检查的遮蔽表征学习、未来潜状态预测、最终 pCR 分类。最优动态 `F_forecast` 的实际总目标是：

```text
1.0 × 同次检查 masked latent L1
+ 1.0 × pooled variance + 0.04 × covariance + 0.1 × token variance
+ 0.25 × teacher-forcing future latent L1
+ 1.0 × free-rollout future latent L1
+ 1.0 × 最终 forecast pCR BCE
```

`repr_pcr/direct/forecast/align/consistency` 额外损失系数均为零。配置 `loss.fused=1` 表示最终所选输出 BCE：F 模式等于 forecast BCE；CSV 中 `fused` 也表示最终输出，不一定存在概率融合。所有教师参数无反向梯度，每次 AdamW 更新后 EMA=.996。模型为 128 维、4 头、256 个空间 token、多尺度 3D 编码器、2 层空间/融合 Transformer、4 层世界预测器；dropout=.1。

冻结配置见 [`historical_runs/branch_ablation_20261010/input_configs`](../historical_runs/branch_ablation_20261010/input_configs/)。batch=16、累积=1；分类头学习率 1e-4、其余可训练主干 3e-5，weight decay=.01、BF16、梯度裁剪=1。**实际按优化器更新数运行**：max_updates=3000、warmup_updates=200、eval_every_updates=100、patience_updates=500、min_delta=1e-5。`epochs: 60`、warmup_epochs 和 epoch patience 不控制该更新制实验。按开发验证患者等权最终 pCR NLL 选 checkpoint；跨候选统一此定义，不比较组成不同的训练总损失。

该消融内部动态 F_forecast 的三种子（17/43/71）均值 NLL 为 0.5713±0.0082，见[最终报告](../results_summary/branch_ablation_20261010/README_ZH.md)。102 名患者反复参与开发验证，没有独立测试结论；不同 landmark 的患者集合也不同。

## 推理输出

单个患者 `predict` 写 JSON：

| 字段 | 含义 |
|---|---|
| `pcr_probability` | 最终所选读出的 pCR 概率，0–1 |
| `direct_probability` | 直接 head 的 sigmoid 输出 |
| `forecast_probability` | 未来状态 head 的 sigmoid 输出；direct-only 时为 null |
| `landmark`、`readout_mode` | 当前阶段及 direct/forecast/gated 等模式 |
| `future_ground_truth_read` | predict 固定为 false |
| `target_coordinate_origin` | 教师坐标的运行/阶段信息 |

`direct` 从合法历史的融合特征读出；`forecast` 从自由递推至 T3 的状态及条件读出；`gated` 使用 `p_final=(1−g)×p_direct+g×p_forecast`。**F_forecast 的最终输出等于 forecast，备用 direct head 未受分类监督，不能将其数值作为另一个已验证分类器。** 当前已观察 T3 时，forecast 读取真实 T3 的教师编码状态，无未来一步可预测。

可选 `--states-out` 写 NPZ，包含 `states` 与 `query_mask`。正式模型 `states` 的 shape 为 **`[1,4,256,128]`**，一般 batch 前向为 `[B,4,256,128]`；256=`4×8×8`，128 为 latent 维度。四阶段槽位保留，未请求槽位全零，只能请求 landmark 之后的阶段。`query_mask` 改变返回范围，不改变分类、内部递推或监督。模型没有解码器，潜状态不能直接当作 `[3,32,128,128]` MRI 图像，没有像素重建/生成接口。

训练目录通常输出 `best.pt`、`last.pt`、`config.json`、`provenance.json`、`data_audit.json`、`history.json`、`progress.json`、`completion.json`、`metrics_best.json`；启用保存预测还有 `predictions_val.csv` 与 `world_val.csv`。前者含各前缀 direct/forecast/fused 概率，后者含潜空间 free/copy L1。checkpoint 保存配置、权重、临床尺度、源/数据/环境身份、EMA 坐标与恢复游标。逐患者 CSV、清单、拟合患者键及权重属于本地研究产物，不在发布包内。

## 安装与运行

在本版本目录单独创建环境，避免其它版本同名 Python 包覆盖：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
PYTHONPATH=src python -m response_jepa.cli --help
PYTHONPATH=src python scripts/train_overfit.py --help
```

依赖：Python≥3.11、torch≥2.5,<3、numpy≥1.24,<3、PyYAML≥6,<7、scikit-learn≥1.3,<2；测试另需 pytest≥8,<10。GPU 正式训练使用适配机器的 CUDA/PyTorch 环境。原许可证与版权见 [LICENSE](../LICENSE)。

只使用合成数据的 CPU 自检：

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m response_jepa.cli smoke --output /tmp/response-jepa-v2-smoke
PYTHONPATH=src python -m response_jepa.cli predict \
  --checkpoint /tmp/response-jepa-v2-smoke/C/best.pt \
  --manifest /tmp/response-jepa-v2-smoke/synthetic_data/manifest.json \
  --allow-synthetic --patient-key SYNTHETIC-0000 --landmark 0 --request 1,2,3 \
  --output /tmp/response-jepa-v2-smoke/prediction.json \
  --states-out /tmp/response-jepa-v2-smoke/states.npz
```

`smoke` 是已有 A→B→C 合成集成测试；联合训练和更新制路径由现有 pytest 覆盖。合成潜状态维度小于正式模型，成绩没有临床含义。上面的输出目录必须为新目录；已存在时换路径。

真实训练前，使用总仓库 `tools/prepare_config.py` 将历史配置生成独立本地配置，仅替换机器路径。下面将 image cache 指向自己的有效缓存；历史原配置不覆盖，路径替换不改变模型、损失或更新协议。manifest 仍由训练命令传入。

```bash
mkdir -p private_outputs
python ../../tools/prepare_config.py \
  --config historical_runs/branch_ablation_20261010/input_configs/F_forecast_dynamic_seed17.yaml \
  --output private_outputs/F_forecast_dynamic_seed17.yaml \
  --image-cache /path/to/image_cache
PYTHONPATH=src python -m response_jepa.cli preflight \
  --manifest /path/to/manifest.json --output private_outputs/preflight.json
PYTHONPATH=src python scripts/train_overfit.py \
  --config private_outputs/F_forecast_dynamic_seed17.yaml \
  --manifest /path/to/manifest.json --output private_outputs/F_forecast_dynamic_seed17
```

将示例占位路径换成自己的文件；其它种子使用各自冻结配置。更新制训练用现成 `scripts/train_overfit.py`，因为原通用 CLI 的训练结束打印假定存在 `best_epoch`，而更新制返回 `best_update`。训练/恢复逻辑未修改；继续同次运行时追加 `--resume`，要求配置/数据/源码/环境身份一致。

对自己的 joint checkpoint 推理（manifest 中须有该患者；无需标签和未来监督）：

```bash
PYTHONPATH=src python -m response_jepa.cli predict \
  --checkpoint private_outputs/F_forecast_dynamic_seed17/best.pt \
  --manifest /path/to/inference_manifest.json --patient-key EXAMPLE-001 \
  --landmark 1 --request 2,3 --device cpu \
  --output private_outputs/prediction.json --states-out private_outputs/states.npz
```

只加载自己可信的 checkpoint（含优化器/RNG 的原格式使用 pickle）。外部推理可用不同患者集合，但必须沿用训练影像归一化声明及 checkpoint 的临床归一化；固定 T0 checkpoint 不支持 landmark>0。独立测试集仅在候选、权重与选择结果冻结后用于最终评估。
