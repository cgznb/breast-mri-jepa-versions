# V1-evolution 的数据输入与输出

本说明依据本目录的 `contracts.py`、`data.py`、`encoder.py`、`model.py`、`training.py` 和 `cli.py` 核对。最优配置是 `evolution_vq_forecast_history_residual_free_encoder64_seed17.yaml` / `seed43.yaml`。命令从 `versions/v1_evolution/` 执行。

## 任务与信息边界

在 T0、T1、T2 或 T3，利用截至当前真实可用的 MRI / 临床信息 / Arm，预测疗程结束时的二分类 pCR 概率。新增真实随访后使用同一套权重更新预测。阶段仅是离散 `0..3`；数据没有可靠真实日期、剂量或周期。

```text
合法观察前缀 MRI + clinical/masks + Arm/known_at
   ├─ student 编码 + 融合 → pCR query、三个固定转移槽
   └─ EMA teacher 编码已观察 MRI → 当前 / 过去状态
             ↓ 历史差分 + 残差转移，递推至 T3
       预测 T3 latent 与相对当前的变化
             ↓ 与 pCR query 拼接
           pCR logit → sigmoid 概率

训练时独立的未来真实 MRI → EMA teacher 目标
             ↓ 只用于 TF JEPA / 自主多步 JEPA 损失
```

## 磁盘格式与预处理

入口为 `responsewm_raw_mri_v1` JSON manifest，声明 `canonical_stages: [0,1,2,3]`、`time_basis: stage_index` 和下列固定相位 / 临床顺序。MRI 路径可相对 manifest 目录，也可为绝对路径。

| 内容 | 格式 / 语义 |
| --- | --- |
| 一次正式 MRI | `.npz` 的 `image` 成员，必须是 `float32 [3,32,128,128]`，轴为相位、D、H、W |
| 三相顺序 | `pre_aqc0`、`first_post_aqc1`、`metadata_late`；第三相按原元数据选择，不能一概称为固定 DCE2 |
| 预处理 | `image_preprocessing: {mode: preprocessed_roi32}`；manifest 声明原训练集的 `image_normalization`；加载保持已标准化数值及零背景，不再 z-score / resize |
| 几何 | T0 source-only 固定 ROI / RAS / source grid；各期共享 T0 网格，不利用未来病灶确定裁剪 |
| visits | 长度为 4，缺失检查为 `null`；每次含 `stage`、`available_at`、`grid_id`、`image` 路径；T0 必须存在且阶段 0 可用 |
| clinical | `values` / `known_at` 均长 4，顺序为年龄、HR、HER2、MammaPrint；缺失值及其 known_at 为 `null` |
| Arm | 官方 13 类之一，显式 `known_at`，声明 `prospective_verified` 或 `assigned_arm_scenario`；分配 Arm 情景不等于完整实际给药史 |
| target | 训练时 `pcr: 0/1` 或缺失，并声明 `label_source`；推理不需要读取标签 |
| split | 患者级 `train` / `val` / `test`；患者与 MRI 工件不能跨 split 重复 |

临床字段精确名称为 `age_at_screening`、`hr_positive`、`her2_positive`、`mammaprint_binary`。年龄按训练集 T0 可用值标准化；HR / HER2 / MP 保持 0/1。推理从 checkpoint 恢复训练标准化参数。

最优 encoder64 的 `lesion_world_weighting=false`、`lesion_pooling=false`，不加载病灶 mask。病灶消融可读取 **T0** `.npz` 的 `roi: bool [1,32,128,128]`，这是原预测病灶区域，不是人工 core/ring 标注。`support` 表示采集覆盖，不能当作原强度标准化前景或直接作为损失 mask。

ROI32 标准化声明字段为 `mean`、`std`、`fit_split: train`、`scope`、`background_policy: preserve_zero`、`already_normalized: true`、`normalization_policy`；必须使用实际预处理声明，不能用示意统计量替代。代码另支持 raw 模式 `.npy` / NIfTI，按患者 T0 每相均值 / 标准差处理并插值到配置大小；**正式 VQ 实验要求原 `preprocessed_roi32` 数值契约，不能用 raw 模式代替。**

## 张量与 mask

设 `B` 为 batch，正式 `D,H,W = 32,128,128`。

| `RawMRIInput` 字段 | 形状 / 类型 | 要求 |
| --- | --- | --- |
| `images` | `[B,4,3,D,H,W]` 浮点 | 只填真实可见检查，隐藏 / 未来槽全零 |
| `observed_mask` | `[B,4]` bool | 真实存在且当前可用；至少一个为真；不可包含当前之后阶段 |
| `landmark` | `[B]` int64 | 当前决策阶段 `0..3` |
| `clinical` | `[B,4]` 浮点 | 按训练规则处理；不可用值为零 |
| `clinical_mask` | `[B,4]` bool | 变量已知且 `known_at <= landmark` |
| `arm_id` | `[B]` int64 | `0` 未知，`1..13` 对应官方枚举 |
| `arm_mask` | `[B]` bool | 当前 Arm 是否可用 |
| `arm_known_at` | `[B]` int64 | 可用时为 `0..landmark`；不可用时为 `-1` 且 ID 为 0 |
| `query_mask` | `[B,4]` bool | 请求导出的未来阶段，仅 `stage > landmark`；不表示未来真实检查存在 |
| `lesion_weights` | 可选 `[B,1,D,H,W]` | 仅特定病灶消融使用 T0 ROI；最优 encoder64 关闭 |

`RawMRISupervision` 独立保存 `future: [B,4,3,D,H,W]`、`future_mask: [B,4]`、`label: [B]`、`label_mask: [B]`。`future_mask` 只标记真实存在的未来检查且必须属于请求阶段；缺失标签 / 未来检查不参与对应损失。`store.batch(..., supervised=False)` 不读取未来 MRI 或标签。

例如 T1 决策，T0 / T1 均可用：`observed_mask=[1,1,0,0]`，`query_mask=[0,0,1,1]`；若实际仅有 T3 随访，`future_mask=[0,0,0,1]`。T2 缺失不阻止递推 T2 latent 后预测 T3，但不会构造真实 T2 MRI，也不对缺失 T2 施加目标损失。

## encoder64 与世界状态如何参与 pCR

三相共享单通道预训练前端。`encoder_type=vq_encoder` 使用压缩前特征：每相 `[64,8,32,32]`，三相拼为 `[192,8,32,32]`，池化至 `[192,2,4,4]`，投影和空间 Transformer 后得到每检查 `[32,128]` tokens。`vq_prequant` 对照每相 8 通道、三相合计 24；两者均使用连续特征，不用 codebook / quantizer / decoder。

`fixed_transition_queries=true` 在每个前缀始终建立 T0→T1、T1→T2、T2→T3 三个内部槽，每槽 8 个 128 维 query。内部槽与用户输出请求分离，因此改变 `query_mask` 不改变 pCR。融合后的 `dynamics_features` 是 `[B,4,8,128]`，T0 占位为零；`pcr_features` 是 `[B,4,128]`，其中第二轴是 4 个 pCR query，**不是四个时间点**。

自主预测从前缀**最新真实可用**的 EMA teacher 状态开始，保留历史，由自己的预测递推到 T3。`history_delta=true` 使用源状态与最近可用前驱的差分及有效位；缺少前驱则差分为零。`world_residual=true` 预测 `S_next=S_current+ΔS`，残差末层零初始化，初始模型等同 copy。转移 Transformer 在阶段块间保持因果注意力。

`pcr_readout=forecast` 拼接 `mean(pCR query)`、`mean(S_T3)`、`mean(S_T3−S_current)` 得到 `[B,384]`，再经 `LayerNorm(384) → Linear(384,1)`。分类梯度可沿自主预测反传到世界模型，真实未来不进入这条路径。已真实观察 T3 时无需转移，终末状态取当前真实状态，变化项为零。

TF JEPA 用相邻、目标可用的 teacher 状态做 L1；自主多步 JEPA 比较合法前缀自由推进所得各可用未来状态与 teacher 目标。两者先在患者内平均，再在有监督患者间平均。最优方案为 `BCE + 0.5 × TF-L1 + 0.5 × free-L1`。用于选轮的验证指标始终是患者等权的前缀 pCR NLL，不用多项训练总损失选轮。

## 输出

| 入口 | 输出 |
| --- | --- |
| `model(inp)` | `pcr_logit: [B]`、query 特征；forecast readout 还含自主状态 / 384 维分类特征；概率为 `sigmoid(pcr_logit)` |
| `model.forecast(inp)` | 分类结果加 `state_prediction: [B,4,32,128]`、`query_mask: [B,4]`；未请求阶段为零 |
| `predict-vla-jepa` | JSON 含 `pcr_probability`、当前 / 已观察 / 请求阶段及 Arm 可见性；可选 `.states.npz` 含 `state_prediction`、`query_mask` |
| `train-vla-jepa` | 新目录内 `best.pt`、`last.pt`、训练 / 配置 / 归一化等记录；checkpoint 含 student、teacher、优化器、恢复状态 |
| `evaluate-vla-jepa` | 分类指标 JSON；世界状态 / copy 对照由 `scripts/evaluate_registered_roi32_world.py` 单独评价 |

latent 没有 MRI 灰度单位，也不是分割图。正式 `enable_flow=false`、`flow_weight=0`，`--generate` 会报错；保留的可选 flow 源码不表示最优模型已训练 MRI 生成功能。

## 命令示例

CLI 和合成链路不需要患者数据或真实 codec：

```bash
PYTHONPATH=src python train.py --help
PYTHONPATH=src python train.py predict-vla-jepa --help
PYTHONPATH=src python train.py smoke-vla-jepa --output /tmp/v1-evolution-smoke-new
PYTHONPATH=src python -m pytest -q tests/test_evolution_model.py tests/test_evolution_data_features.py tests/test_evolution_training.py
```

默认 smoke 使用基础 CNN 小配置；evolution 测试另建随机合成 codec，验证 encoder64 和自主训练路径。它们仅验证工程行为。

正式训练先复制历史配置，替换本机资源位置。下例需指向实际授权数据 / 权重；`image_cache=None` 表示直接读取原 `.npz`。

```bash
python - <<'PY'
from pathlib import Path
import yaml
source = Path('configs/evolution_vq_forecast_history_residual_free_encoder64_seed17.yaml')
cfg = yaml.safe_load(source.read_text())
cfg['encoder_checkpoint'] = '/absolute/path/to/codec.pt'
cfg['image_cache'] = None
Path('config.encoder64.local.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
PYTHONPATH=src python train.py train-vla-jepa \
  --manifest /absolute/path/to/manifest.json \
  --config config.encoder64.local.yaml \
  --output /absolute/path/to/new-run-encoder64-seed17
```

如使用缓存，先运行下面命令，再将运行副本配置的 `image_cache` 指向新缓存。缓存仅精确导出原图以加速读取，不做新归一化。

```bash
PYTHONPATH=src python scripts/prepare_roi32_image_cache.py \
  --manifest /absolute/path/to/manifest.json \
  --output /absolute/path/to/new-image-cache \
  --report /absolute/path/to/new-image-cache-report.json
```

已训练 checkpoint 的单例预测，可切换 `--landmark 0/1/2/3`：

```bash
PYTHONPATH=src python train.py predict-vla-jepa \
  --checkpoint /absolute/path/to/new-run-encoder64-seed17/best.pt \
  --manifest /absolute/path/to/inference-manifest.json \
  --patient-key example_subject --landmark 0 \
  --forecast-states --device cpu \
  --output /absolute/path/to/prediction.json
```

生成 `prediction.json` 与 `prediction.states.npz`。推理 manifest 可不含训练患者，标准化从 checkpoint 恢复；`patient-key` 须匹配用户自备 manifest。不要将患者级输出提交到代码仓库。

开发验证显式指定 `val`，因为通用 CLI 默认 split 是 `test`：

```bash
PYTHONPATH=src python train.py evaluate-vla-jepa \
  --checkpoint /absolute/path/to/new-run-encoder64-seed17/best.pt \
  --manifest /absolute/path/to/manifest.json \
  --split val --landmark all_observed --bootstrap 0 --device cpu \
  --output /absolute/path/to/development-evaluation.json
```

完整 20 次消融的队列也已适配新仓库目录。`--plan` 仅显示协议，不启动训练或创建输出目录；实际运行可显式提供 manifest、codec 与缓存位置：

```bash
PYTHONPATH=src python scripts/run_evolution_experiment.py \
  --plan --output /absolute/path/to/new-evolution-suite
PYTHONPATH=src python scripts/run_evolution_experiment.py \
  --manifest /absolute/path/to/manifest.json \
  --codec /absolute/path/to/codec.pt --image-cache none \
  --output /absolute/path/to/new-evolution-suite \
  --detach --max-concurrent 3 --gpu-memory-budget-mib 29696
```

队列仅在新输出的 `input_configs/` 生成资源路径副本，保持原配置和科学参数。默认运行 codec 路径是本包 `weights/roi32_codec.pt`；外部权重不随仓库提供。`--image-cache none` 直接读取原 NPZ，或传已准备缓存目录。恢复时需保持相同资源参数；已有副本与新请求不一致时拒绝继续，以防恢复到另一资源坐标。队列的阶段正确性屏障、源码绑定和 checkpoint 验证保持原有规则。

本次整理没有运行真实训练或患者评价。新实验需另定数据集、划分、候选和验证规则；有独立测试集时，方案和选模结果冻结后才作最终评价。
