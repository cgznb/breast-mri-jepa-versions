# 第一版 VLA-JEPA：数据、网络与单批逐行调试

本说明核对于 2026-10-12。代码目录：

```text
/root/autodl-tmp/line b/breast-mri-vla-jepa
```

它是第一版独立代码包，当前目录没有 Git 仓库。第二版位于旁边的 `breast-mri-response-jepa-v2-joint`。第一版使用直接 pCR 分类头，世界预测是辅助任务；第二版才加入未来状态分类头与门控融合。第一版采用自行实现的 3D MRI encoder 和 Transformer，没有加载原 VLA-JEPA 的 Qwen3-VL/V-JEPA2 预训练权重。

下面先读第一版原始完整损失 A0 配置。A1/A2、L1–L4 也是第一版的实验配置；尤其 L1 关闭所有辅助损失，用它调试时不会进入 JEPA 的计算分支。

## 1. 先跑一个 batch

本机已有 `/root/miniconda3/bin/python`，核实为 Python 3.12.3、PyTorch 2.12.1+cu130，CUDA 可用，debugpy 已安装。现有 manifest 和权重可以直接使用。

```bash
cd '/root/autodl-tmp/line b/breast-mri-vla-jepa'
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python scripts/debug_vla_jepa.py --mode loss
```

这条命令使用真实训练数据、物理 batch=1、T0、CPU/FP32、原完整损失配置，打印关键模块输入输出的 shape。默认模型随机初始化，因此输出仅用于理解计算。它不保存 run、预测文件或 checkpoint。

调试脚本调用现有 `RawMRIStore.batch` 和 `RawVLAJEPA.compute_loss`，没有另写一套数据或模型实现。为便于同步逐行跟踪，它同步组装一批数据；不会执行正式训练的患者 shuffle、后台预取、整个验证集、scheduler 或 checkpoint 保存。

### VS Code

直接用 VS Code 打开上述**代码目录**。新增的 [.vscode/launch.json](../.vscode/launch.json) 提供三项：

1. `V1: T0 数据、前向与完整损失 (CPU)`：第一次阅读选它。
2. `V1: 动态 T2 单步反向、AdamW、EMA (CUDA)`：完整走一次参数更新，更新只留在此进程内。
3. `V1: 已训练 T0 best.pt 的 pCR 推理 (CPU)`：用历史 A0 的训练权重看预测路径。

在下面列出的代码行左侧打断点，按 F5；F10 执行当前行，F11 进入调用，Shift+F11 返回，F5 继续至下一断点。配置中的 `${workspaceFolder}` 指本代码包根目录；如果当前打开的是父工作区，请打开本代码目录后再使用这些配置。

### 终端 pdb

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python scripts/debug_vla_jepa.py --mode loss --break
```

它停在组装数据之前，可以依次设置：

```text
b mri_vla_jepa/data.py:343
b mri_vla_jepa/encoder.py:72
b mri_vla_jepa/model.py:124
b mri_vla_jepa/model.py:131
b mri_vla_jepa/model.py:139
b mri_vla_jepa/model.py:144
b mri_vla_jepa/model.py:181
b mri_vla_jepa/model.py:291
c
```

`n` 执行当前行，`s` 进入调用，`r` 返回，`c` 继续，`p expression` 查看变量，`q` 退出。上面 pdb 的文件名按 `PYTHONPATH=src` 搜索，所以省略 `src/`；也可写完整绝对路径。断点停在该行**执行前**；要看该行赋值后的变量，先执行 `n`。建议查看 shape 和 mask，而不是直接打印整个 MRI。

## 2. 当前数据是什么

- 数据索引：data/registered_roi32/manifest.json（未随源码发布的历史工件：`../data/registered_roi32/manifest.json`）。包含每名患者的 split、四检查槽、临床字段、Arm、最终 pCR 和数据来源。
- 真实影像资产在 `/root/autodl-tmp/MAM/symm-fm-trjepa-v2/data/registered_roi32/views`，manifest 引用其 NPZ。
- 866 名患者：train=764、val=102、test=0；共 3039 次 MRI。102 人是开发验证集。
- 按 T0/T1/T2/T3 计，实际检查数是 866/836/727/610。缺失的检查保留槽位，不把 T2 挪成 T1。
- 每次 MRI 的 `image` 为 float32 `[3,32,128,128]`，三通道依次是 `DCE0`、`first_post`、`late_post`。三相是同一次检查内的对比剂相位；T0–T3 是不同治疗阶段。
- 这些是已经配准、按 T0 定义 ROI 的体素 MRI。训练读取 NPZ 的 `image`，没有读取 VQ latent，也没有把肿瘤 mask 当成额外输入通道。
- 当前 manifest 声明 `preprocessed_roi32`。运行时直接保持缓存的强度刻度与零背景，不再对每名患者做 MRI z-score，也不再 resize。上游归一化声明和准备检查见 [REGISTERED_ROI32_ZH.md](REGISTERED_ROI32_ZH.md)。本调试不重做上游配准、分割、裁剪或强度统计。

上游数组生成代码可继续往前追：`MAM/MeWM-ISPY2/mewm_ispy2/registered_three_phase_data.py` 选择 `(phase0, phase1, metadata_late_index)`，late 相位由元数据指定。它把 MRI 配准/重采样到 T0 crop，MRI 用线性插值、coverage 用最近邻。`registered_roi32_data.py:141–156` 根据 T0 预测肿瘤 mask 的最大 26 连通分量的物理 bbox 中心确定固定 ROI；配置 spacing 为 Z/Y/X=`2.0/0.7032/0.7032 mm`。三个 phase 共享由训练集唯一 precontrast crop 的非零 foreground 聚合得到的 frozen mean/std，在有效 foreground 上标准化，其他位置保持零。

`MAM/symmv4/symm-fm-trjepa/scripts/prepare_registered_roi32.py:50–62` 把三相堆叠并保存 NPZ。当前代码包的 `scripts/prepare_registered_roi32.py` 则审计已有 NPZ、合并官方临床表、建立 manifest，没有重新处理体素。以上是核对的上游实现与当前数据声明；本次没有重新计算全数据的强度统计。

不要把另一条通用原始 NPY/NIfTI 路径和当前 NPZ 路径混淆：通用路径会使用该患者 T0 的均值/标准差并插值；当前 `preprocessed_roi32` 分支会直接返回现有数组。

## 3. 数据逐行阅读

先打开 [data.py](../src/mri_vla_jepa/data.py)、[contracts.py](../src/mri_vla_jepa/contracts.py)。

| 位置 | 执行什么 | 建议查看 |
|---|---|---|
| `data.py:51` | 建立 RawMRIStore，读 manifest、验证合同、建立 split，准备临床标准化 | `store.image_shape`、`store.by_split` 的长度 |
| `data.py:275` | `_read_scan` 读取一次检查 | `paths`，在本地查看具体数据位置 |
| `data.py:278` | 当前 NPZ 分支，可选择逐值一致 NPY cache | `self.image_cache` |
| `data.py:285` | 取出 NPZ 的 `image` | `array.shape`、`array.dtype` |
| `data.py:329` | `_resize_normalized`：当前模式直接返回 tensor | `self.preprocessed_roi32` |
| `data.py:335` | `batch(tasks)`，tasks 是 `(patient_index, landmark)` 列表 | `tasks`、`b` |
| `data.py:343` | 分配全零 `[B,4,3,32,128,128]` | `images.shape` |
| `data.py:355` | 读取此人的 T0，用作源几何参考 | `baseline.shape` |
| `data.py:362` | 只填入不晚于 landmark 且当时已可用的检查 | `stage`、`landmark`、`observed[row]` |
| `data.py:367` | 请求全部未来阶段，不参考未来检查是否真实存在 | `queries[row]` |
| `data.py:368` | 按可用阶段取临床字段，准备可用性 mask | `mask`、`values.shape` |
| `data.py:369` | 按训练集坐标标准化临床值，缺失值填零 | `standardized.shape`、`cmask[row]` |
| `data.py:372` | 取得当前可见的 Arm id/mask/known_at | `available`、`when` |
| `data.py:374` | 构造并验证 `RawMRIInput` | `inp` 的各字段 |
| `data.py:379` | 普通推理在这里返回，完全不读取未来监督或 pCR 标签 | `supervised` |
| `data.py:381` | 训练才建立独立未来监督 tensor | `future.shape`、`fmask` |
| `data.py:387` | 若 JEPA/flow 需要监督，读取 landmark 之后真实存在的 MRI | `future_supervision` |
| `data.py:392` | 取最终二分类 pCR，放进 supervision | `labels.shape`、`lmask` |
| `data.py:400` | 返回 `(inp, sup)` | `inp.observed_mask`、`sup.future_mask` |

四个临床字段按 `age_at_screening, HR, HER2, MP` 排列。Age 使用训练患者 T0 可见值拟合的均值/标准差；三个二值字段保持 0/1。所有字段都有独立可用性 mask，缺失或尚不可用的值为零。

Arm 是官方 13 类治疗分配的类别，unknown 为 0，网络使用 embedding。当前数据采用 `assigned_arm_scenario`，把分配 Arm 作为 T0 已给定的治疗场景；它没有阶段间剂量、周期或治疗修改轨迹，也没有从表中验证真实分配时间。

完整四次检查的示意：

```text
landmark=T0: inp.images=[I0, 0,  0,  0 ], observed=[1,0,0,0], query=[0,1,1,1]
landmark=T2: inp.images=[I0,I1, I2,  0 ], observed=[1,1,1,0], query=[0,0,0,1]
landmark=T3: inp.images=[I0,I1, I2, I3 ], observed=[1,1,1,1], query=[0,0,0,0]
```

真实缺失会额外令 `observed_mask`/`future_mask` 为 false。`query_mask` 表示“请求预测哪些未来阶段”，`future_mask` 表示“哪些未来检查有真值可算损失”，二者不能互换。例如请求 T1/T2/T3，但 T1 缺失时，future mask 可以是 `[0,0,1,1]`。

`RawMRIInput` 的检查要求未来 MRI 槽实际全零。最终 pCR 和未来真实 MRI 只在 `RawMRISupervision` 中，不是 pCR 输入。

## 4. MRI 怎么变成 token

打开 [encoder.py](../src/mri_vla_jepa/encoder.py) 的 `VisitMRIEncoder.forward`，断在第 72 行。令 `O = observed_mask.sum()`，即这批实际已观察的检查总数。

```text
输入                  [O,  3,32,128,128]
stem.0 Conv3d s=2     [O, 64,16, 64, 64]
stem.1/2 GN + GELU    同上
stem.3 Conv3d s=2     [O,128, 8, 32, 32]
stem.4/5 GN + GELU    同上
stem.6 网格平均池化    [O,128, 2,  4,  4]
flatten(2)            [O,128,32]
transpose(1,2)        [O, 32,128]
加空间位置 embedding  [O, 32,128]
2 层空间 Transformer  [O, 32,128]
LayerNorm             [O, 32,128]
```

32 个 token 对应池化后的 `2×4×4` 空间网格，每个 token 是 128 维。这里是两次 3D 卷积后池化，不是直接把原始体积切成 ViT patch。编码器每次只处理单次 MRI 的空间 token，不在这里混合多次检查。

在 [model.py](../src/mri_vla_jepa/model.py) 第 117 行实际传入 `inp.images[inp.observed_mask]`。因此缺失/未来的全零槽没有送进 encoder；它们之后用 padding mask 屏蔽。

## 5. 共享主干和 pCR 逐行阅读

配置：[registered_roi32_epoch200_t0_seed17.yaml](../configs/registered_roi32_epoch200_t0_seed17.yaml)。`dim=128`、`state_queries=8`、`pcr_queries=4`、`fusion_depth=2`。

| model.py 行 | 代码含义 | 张量变化 |
|---|---|---|
| 117 | 编码真正已观察的 MRI | `[O,3,32,128,128] → [O,32,128]` |
| 118–119 | 把 observed token 放回患者×四阶段的槽 | `[B,4,32,128]` |
| 120 | 加对应 T0/T1/T2/T3 的 stage embedding | 同上 |
| 121 | clinical 值和可用性 mask 拼接，线性投影为一个 token | `[B,8] → [B,128]` |
| 122 | Arm 类别 embedding 加可用阶段 embedding | `[B,128]` |
| 124 | 128 个 MRI token + 1 clinical + 1 Arm | `context[B,130,128]` |
| 125 | 给缺失/未观察 MRI token 设置 padding | `[B,130]` |
| 128–129 | 每阶段 8 个转移查询，加 stage embedding | `dynamic[B,4,8,128]` |
| 130 | 4 个 pCR 查询，加当前 landmark embedding | `pcr[B,4,128]` |
| 131 | context + 32 dynamics queries + 4 pCR queries | `packed[B,166,128]` |
| 132 | 屏蔽过去/无效转移 query，保留 pCR queries | `padding[B,166]` |
| 137 | 构造整体上三角因果 mask | `causal[166,166]` |
| 138 | context 内部允许双向注意力 | 合法历史检查之间可互看 |
| 139 | 2 层 fusion Transformer，再 LayerNorm | `hidden[B,166,128]` |
| 140–142 | 提取转移特征，屏蔽不请求的阶段 | `dynamics_features[B,4,8,128]` |
| 143 | 提取最后 4 个 pCR tokens | `pcr_features[B,4,128]` |
| 144 | token 平均、LayerNorm、Linear、squeeze | `[B,128] → [B,1] → pcr_logit[B]` |

`sigmoid(pcr_logit)` 才是最终 pCR 概率。T0/T1/T2/T3 预测的是同一个手术后的最终 pCR 终点，变化的是当前可见的历史。

两个概念要分别看：

- **主干 dynamics query 特征**：由合法历史、clinical、Arm 产生。它们排列在 pCR queries 前面，所以 pCR queries 可以注意到它们。
- **world predictor 的未来状态预测**：由后面的辅助分支产生，不送回第一版 pCR head。

普通 `model(inp)` 调用 `forward → backbone → _backbone`，在直接 pCR 输出结束；不会执行 EMA teacher、世界预测器、重构或 flow。

## 6. 训练里的 JEPA、损失和 EMA

从 `model.py:246` 的 `compute_loss(inp, sup, ...)` 开始。第 260 行首先完成上述共享主干。

1. 第 264–266 行：用直接头的 logit 和 `sup.label` 计算 BCE，只用有效标签。
2. 第 267–269 行：若 JEPA 权重大于零，进入 `_teacher_forcing_loss`。
3. 第 156–164 行：冻结的 EMA teacher 分别编码已观察 MRI 与真实未来 MRI，构造 `[B,4,32,128]` teacher states。这里 `no_grad`，teacher 不接收梯度。
4. 第 183–184 行：只监督从当前 landmark 往后的真实相邻检查对，源和目标必须存在。缺 T1 不会伪造 T0→T2。
5. 第 185 行：world predictor 输入真实前序 teacher state `[B,3,32,128]` 与转移特征 `[B,3,8,128]`。内部为 root + 三块 `(8 action + 32 state)`，共 `[B,121,128]`；按阶段因果注意力，输出 `[B,3,32,128]` 下一状态预测。
6. 第 186–189 行：与下一检查 teacher state 算 latent L1；先平均每个患者自己的有效相邻对，再平均有有效对的患者。
7. 第 271–275 行：online observed tokens 变回 `[O,128,2,4,4]`，1×1×1 卷积到 3 通道，插值回完整影像大小，算 observed MRI 重构 MSE。
8. 第 276–277、300–303 行：完整基线 `legacy` 方差正则把“检查×空间 token”混在一起统计每个 feature 的标准差，罚 `relu(1-std)`。这不能解释成严格按患者轴防塌缩；L4 才换成患者轴的 FP32 定义。
9. 第 291 行：合并加权损失。完整基线是：

```text
loss = 1.0 × BCE(pCR)
     + 0.5 × JEPA latent L1
     + 0.1 × observed MRI reconstruction MSE
     + 0.01 × variance penalty
```

T3 没有未来 query，所以 JEPA 没有有效 pair，但完整损失配置仍可计算 pCR、观察重构和方差正则。ROI32 正式配置 `enable_flow=false`、`flow_weight=0`，所以项目里有 flow.py，但这条实验路径不使用它。

训练的 JEPA 是 **teacher forcing**：用真实前序 state 预测下一 state。`forecast_states` 的自主推演才会把自己的预测反复回喂，普通 pCR 推理不会运行它。

正式更新位置在 [training.py](../src/mri_vla_jepa/training.py)：

```text
354       取出 inp/sup
355–356   搬到设备
357–361   CUDA BF16 autocast 内算完整损失
364       backward（按 accumulation 缩放）
369       梯度裁剪，阈值默认 1
378       AdamW.step
383       更新 EMA teacher
```

`model.py:322` 的 teacher 更新为 `teacher = .995 × teacher_old + .005 × online_new`。teacher 始终 eval、冻结；只跟随 online encoder，不更新 clinical/fusion/pCR head 的 teacher 副本，因为这些没有 teacher 副本。

## 7. 动态阶段、反向和训练权重命令

真实动态 T2 单步训练计算，B1/CUDA BF16：

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python scripts/debug_vla_jepa.py \
  --config configs/registered_roi32_epoch200_dynamic_seed17.yaml \
  --landmark 2 --device cuda --precision bf16 --mode train-step
```

`train-step` 完成一次 backward、clip、AdamW、EMA，只修改进程内模型。正式实验 B16；B1 是便于看变量的调试批次，不用于复现正式指标。它使用全新 AdamW，不恢复历史 optimizer、RNG、scheduler 或训练进度；这也适用于加载 checkpoint 后演示一次更新。损失/更新调试仅支持当前 one-stage 配置，单步更新要求 accumulation=1。

改 `--landmark 3` 可观察零未来 query 与零 JEPA 有效 pair；改 `--mode predict` 可观察 teacher/world/reconstruction 断点不触发。调试器会从指定 split 中选择在该 landmark 真实有检查可用的人，`--patient-offset` 是这个筛选后列表里的偏移。

查看已经训练好的 A0 T0 权重：

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python scripts/debug_vla_jepa.py \
  --checkpoint runs/registered_roi32_epoch200_20261003/t0_seed17/best.pt \
  --split val --mode predict
```

checkpoint 模式自动使用其保存的模型配置与临床标准化坐标。`--config` 与 `--checkpoint` 二选一。在 `probability` 赋值后的行打断点可查看预测值；默认终端只输出 shape，不列患者身份、临床值或标签。

动态 A0 权重位于 `runs/registered_roi32_epoch200_20261003/dynamic_seed17/best.pt`。固定 T0 配置/权重只接受 T0；看 T2/T3 要换 dynamic 配置或权重。

L1 dynamic 的候选权重在 `runs/registered_roi32_l1_l4_20261004/l1_dynamic_seed17/best.pt`；它关闭全部辅助损失，依然保留第一版共享主干和直接 pCR 头。原始 A0 的 best.pt 用于本指南演示完整结构，不能称为所有消融中最佳权重。

## 8. 正式训练入口怎么连接

```text
train.py
→ cli.py:69–72 (load_config → RawMRIStore → train)
→ training.py:188 (BatchLoader)
→ training.py:196 (_train)
→ training.py:335–353 (患者/阶段采样)
→ training.py:354 (loader.fetch)
→ data_pipeline.py:154–159 或后台预取
→ data.py:335 (RawMRIStore.batch)
→ model.py:246 (compute_loss)
→ backward → AdamW → EMA → 验证/保存
```

原 A0 epoch 配置默认同步读取 NPZ，`image_cache=None/prefetch_batches=0`。后续 A1/A2/L 配置可打开逐值一致 NPY cache、有界预取与 pinned transfer；它们调用同一个数据合同，影像内容和可见性规则保持一致。

epoch 模式每轮打乱 764 名训练患者，每人取一次。固定 T0 全部取 landmark=0；dynamic 对该患者真实合法阶段均匀随机取一个，不是在训练每轮穷举所有前缀。B16 时 48 批/epoch，最后一批 12 人。动态验证才穷举合法前缀，选模 NLL 先按患者内部平均，再按患者平均。

若需要从正式入口启动一个**新的** A0 训练：

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python train.py train-vla-jepa \
  --config configs/registered_roi32_epoch200_t0_seed17.yaml \
  --manifest data/registered_roi32/manifest.json \
  --output runs/your_new_t0_run
```

它会正式训练最多 200 epoch、patience 50，并写 checkpoint。dynamic 换成 `registered_roi32_epoch200_dynamic_seed17.yaml`。本指南没有启动这类长训练。

`--stop-after 1` 表示在一次优化更新后暂停，但仍会执行保存与原训练入口的验证工作，不能把它理解成轻量的“只看一批前向”。逐行理解优先用上面的独立单批调试入口。

## 9. 本次调试入口核验

真实 B1 的 CPU T0 完整损失、CUDA/BF16 动态 T2/T3 的 backward/AdamW/EMA、历史 best.pt 的 CPU pCR 推理、L1 的 CPU pCR-only 损失均运行通过。T3 未来数量和 JEPA pair 数均为零，L1 未触发 teacher/world/reconstruction；实际输出的编码器与 fusion shape 和上文一致。脚本语法、VS Code JSON、文档链接及终端 pdb 断点已检查。这是调试工具的工程核验，没有重算正式性能指标。
