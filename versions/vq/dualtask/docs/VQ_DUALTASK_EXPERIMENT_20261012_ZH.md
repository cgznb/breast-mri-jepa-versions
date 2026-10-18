# 2026-10-12：VQ 编码器下的 pCR + JEPA 双任务实验

这次实验同时观察两个任务：最终 pCR 是否预测得更好，以及未来 MRI 的潜在状态是否比简单 copy 更容易预测。原始编码器、冻结 VQ 前端、微调 VQ 前端都使用同一双任务损失和相同数据划分。这样可以在同种子下比较编码器变化，并把 pCR 与世界预测的结果分别报告。

目标目录已核实为 `/root/autodl-tmp/line b/breast-mri-vla-jepa-vq-dualtask`。它是独立代码包，导入名称仍为 `mri_vla_jepa`。本目录运行命令使用 `PYTHONPATH=src`；数据继续借用原 V1 的文件，不在新目录复制患者影像。

| 用途 | 本机路径 |
| --- | --- |
| 当前双任务代码 | `/root/autodl-tmp/line b/breast-mri-vla-jepa-vq-dualtask/src/mri_vla_jepa` |
| 原 V1 与共享数据 | `/root/autodl-tmp/line b/breast-mri-vla-jepa` |
| 已建立的 pCR-only VQ 实验 | `/root/autodl-tmp/line b/breast-mri-vla-jepa-vq-encoder` |
| 数据索引 | `/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32/manifest.json` |
| 图像缓存 | `/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32/image_cache_20261004` |
| 旧 VQ 前端权重 | `/root/autodl-tmp/MAM/symm-fm-world-v2/data/registered_roi32/codec.pt` |
| 当前输出 | `/root/autodl-tmp/line b/breast-mri-vla-jepa-vq-dualtask/runs/vq_dualtask_20261012` |

配置中的缓存、权重与阶段 B 初始化路径都是本机绝对路径。迁移机器需要准备相同数据并更新路径；修改输出根目录时，B 的 `initialize_from_checkpoint` 也要指向对应 A 的 `best.pt`。

输入数据处理继续采用原 V1：图像已经配准、裁剪为 `32×128×128` 并按训练集共同强度尺度预处理，加载时直接读取数组，不重新做每患者图像归一化。年龄按训练集统计标准化，HR/HER2/MP 保留二值编码；Arm 使用原类别 embedding。dynamic 输入允许 T0 到当前阶段的真实可见历史，缺失阶段保留原槽位、填零并用 mask 屏蔽。最终 pCR 标签与未来真实影像保存在监督对象中。

完整输入为 `[B,4,3,32,128,128]`，其中 4 是治疗阶段，3 是一次检查的三个 MRI 相位。VQ 方案只编码被 observed mask 标记的真实检查；单次检查的路径为：

```text
一次三相 MRI              [N,3,32,128,128]
  ↓ 各相位共用单通道旧 VQ encoder + pre_quant
每相连续特征              [N,8,8,32,32]
  ↓ 通道拼接
                          [N,24,8,32,32]
  ↓ 池化到 2×4×4，投影 24→128，展平空间维
                          [N,32,128]
  ↓ 位置编码 + 空间 Transformer + LayerNorm
合法历史空间 token        [N,32,128]
```

`N` 是 batch 中真实可见检查的数量。这里取得量化前连续特征，不运行 VQ 量化器、码本、解码器或 GAN 判别器。raw 对照则保留原 V1 从零训练的 3D 卷积前端，同样向后续网络提供 32 个、每个 128 维的空间 token。

两条任务路径从共享融合网络分开：

```text
合法 MRI 历史 + 临床 + Arm + 查询 token
                    ↓ 共享 fusion Transformer
         ┌──────────┴────────────┐
         ↓                       ↓
4 个 pCR 查询特征            状态转移查询特征
平均 + LayerNorm + Linear   world predictor + 真实前一期 teacher 状态
         ↓                       ↓
pCR logit                   下一期 32×128 latent 状态
         ↓                       ↓
BCE                         与 EMA teacher 下一期状态计算 L1
```

pCR 查询 token 和状态转移查询 token 是可学习的查询，不能直接等同于四次 MRI。共享主干中的查询彼此按原有 attention 规则交互，两种损失会更新共同参数。world predictor 接收状态转移特征；pCR 头仍使用共享主干的直接分类特征，未来 latent 预测不会回送到 pCR 头。这保留了第一版的任务耦合方式。

JEPA 在训练时使用 teacher forcing：EMA teacher 无梯度地编码真实 MRI 状态，world predictor 根据真实前一期 latent 与状态转移特征预测下一期 latent。合法相邻 pair 要求真实来源、真实目标和请求 mask 同时成立；缺失阶段不会虚构成相邻转移。每名患者先平均自己的有效 pair loss，再在有有效 pair 的患者间平均。

训练总损失固定为：

```text
loss = 1.0 × BCEWithLogits(pCR)
     + 0.5 × mean |predicted_state - EMA_teacher_target|
```

`flow_weight=0`、`reconstruction_weight=0`、`variance_weight=0`、`variance_definition=off`、`enable_flow=false`。因此预测任务是 latent state JEPA，当前实验不训练体素生成、MRI 重构或方差正则。两个任务在 raw、A、B 中都同时启用；A/B 仅描述 VQ 前端是否冻结，`schedule=one_stage`。T3 没有未来目标，或某个 batch 没有合法未来 pair 时，JEPA 分支的有效 loss 为零，pCR 仍正常训练。

六次训练的来源关系为：

| 种子 | raw 双任务对照 | 冻结双任务 A | 微调双任务 B |
| --- | --- | --- | --- |
| 17 | `raw_dynamic_seed17` | `frozen_dynamic_seed17` | `finetune_dynamic_seed17` |
| 43 | `raw_dynamic_seed43` | `frozen_dynamic_seed43` | `finetune_dynamic_seed43` |

配置名为 `vq_dualtask_{raw,frozen,finetune}_dynamic_seed{17,43}.yaml`。raw 对照从零建立原始编码器、共享网络和两个头；A 只从旧 `codec.pt` 加载 VQ 前端，其余部分随机初始化，并同步完整 EMA teacher。A 没有从之前的 pCR-only VQ run 继承权重。

A 中旧 VQ `encoder + pre_quant` 冻结为 eval、无梯度、排除在优化器之外，适配层与后续网络以 `1e-4` 训练。B 从同种子 A 的双任务 `best.pt` 加载完整模型，包括适配层、fusion、pCR 头、world predictor 和 EMA teacher，再解冻 VQ 前端。B 新建 AdamW、采样器、训练计数与早停状态，学习率组为 `network=1e-4`、`pretrained_encoder=1e-5`。B 初始化会严格检查结构、数据、患者划分、归一化、源码以及损失/采样设置。

全部 run 采用 dynamic 的 patient-uniform 合法阶段采样、batch size 16、accumulation 1、dropout 0.3、恒定学习率、AdamW weight decay 0.01、梯度裁剪 1.0、BF16、最多 200 epoch、early stopping patience 50。EMA teacher 始终无梯度；即使 VQ 前端冻结，适配层与空间 Transformer 的 teacher 坐标仍会随其学生权重更新。微调前端启用 activation checkpointing，以重新计算减少显存占用。

保存 `best.pt` 时，只使用 pCR 的开发验证选择指标：对每名患者的全部合法时间前缀计算 NLL，患者内部先平均，再对患者平均。world loss 没有参与 checkpoint 选择。因此报告中的 best 只能写为“pCR NLL 选出的 checkpoint”；latent prediction 的最佳模型需要单独的选模方案，本实验没有采用该方案。

训练完成后，每个 run 的 `best.pt` 与 `last.pt` 都会进行完整开发验证的独立 FP32 world 评价，结果为 `world_best.json`、`world_last.json`，日志为 `world_evaluation.log`。六个 run 共十二份 world 报告。队列把世界评价与训练分别记录；对应 A 的训练与 checkpoint 验证完成后，B 可进入队列，world 评价待运行或失败不会触发重新训练 A。训练和评价按显存预算共享 worker 名额，报告尚未全部完成时，应继续检查评价状态。

评价实现是 `scripts/evaluate_registered_roi32_world.py` 和 `diagnostics.py` 的 `evaluate_world`。它读取完整验证 split，dynamic 使用所有合法 prefix，而非训练 diagnostics 的小探针。预测、真实目标和 copy 基线都使用同一 checkpoint 的 EMA teacher 坐标，包含：

| 项目 | 模型如何输入 | copy 对照 |
| --- | --- | --- |
| 相邻 teacher forcing | 每次使用真实前一期状态 | 复制真实前一期状态 |
| 连续链 teacher forcing | 在相同 1–3 步完整真实链上逐步比较 | 每步复制真实前一期状态 |
| 自主 1–3 步预测 | 从起点真实 latent 出发，后续只反馈自己预测的 latent | 始终保持起点真实 latent |

连续链要求起点、目标和所有中间真实检查均存在；缺失检查的链被排除。全部相邻 TF 单独汇总，即使早期有缺失，后面存在的真实相邻 pair 仍可评价。各 horizon 分别统计患者数与 prefix-target 数，同一患者多个 prefix 不是多个独立患者。

每项误差先在患者内平均，再在患者间平均；`ratio=预测 L1 / copy L1`，`skill=1-ratio`，ratio 小于 1 才表示在该评价中优于 copy。若 copy 的汇总 L1 不超过 `1e-8`，ratio 和 skill 为 null，报告同时记录近零 copy 患者数。TF 与自主预测的 copy 定义不同，不能直接比较各自 ratio 来量化累计误差；应在同一完整链上同时查看模型的原始 TF 与自主 L1。

结果解释有四个限制：

1. 每个 checkpoint 的 EMA teacher 都有自己的 latent 坐标。跨 raw/VQ、A/B 或 best/last 的原始 latent L1 不能直接作为共同尺度排名；copy ratio 也只反映各自坐标内的相对预测能力。
2. 方差正则在本方案关闭。latent loss 变小或 ratio 改善，需要结合 teacher 方差、effective rank、漂移和 copy 分母检查，不能单独证明表示具有更多临床信息。
3. B 继承 A 并增加一段训练预算。B 对 A/raw 的差异包含额外训练时长与初始化影响，不能全部归因于小学习率解冻。
4. 数据划分是训练 764 名、开发验证 102 名、独立测试 0 名；开发验证已多次用于方案选择，两个种子也不足以代替独立测试。manifest 没有逐转移配准 QC，world 报告不能提供该 QC 子组的结论。

pCR 结果分别列出同种子 raw、A、B 的选择 NLL、各阶段 AUC/AP/Brier，以及 best 到 last 的变化。world 结果另列同种子、checkpoint 和 horizon 的 TF、自主误差、copy、ratio 与覆盖人数。改善一个任务不自动说明另一个任务改善，也不能由 pCR 的 best 推断未来状态预测达到最佳。

**本实验已完成。** 最终 analysis.json（未随源码发布的历史工件：`../runs/vq_dualtask_20261012/analysis.json`） 于 **2026-10-12 16:05:43（北京时间）**记录 `status=completed`、训练 6/6、已验证 world 报告 12/12；结果见 [RESULTS_ZH.md](../historical_runs/vq_dualtask_20261012/RESULTS_ZH.md) 与 [解释报告](../historical_runs/vq_dualtask_20261012/INTERPRETATION_ZH.md)。两种子 pCR-best 的 NLL / T0 AUC 均值为 raw **0.553476 / 0.735714**、冻结 A **0.566432 / 0.715179**、微调 B **0.572514 / 0.717634**。冻结双任务的 NLL 优于旧冻结 VQ 单任务，但尚未超过原 L1 或新 CNN 双任务，微调未进一步改善主指标 NLL。仍是 102 名开发验证患者上的结果，不能视为独立测试表现。

控制器在六次训练和十二份 world 评价全部通过后，自动检查来源并汇总，生成输出根目录的 `analysis.json` 与 `RESULTS_ZH.md`。分析失败单独写入 `analysis.json`，不会将已经完成的训练变成需要重训的 run。判断整套实验完成时，同时检查各 job 的训练/world 状态和分析文件状态；仅分析失败时，优先直接重跑分析脚本，避免通过整个队列 resume 再评价全部 world checkpoint。

独立分析入口为 [analyze_vq_dualtask_experiment.py](../scripts/analyze_vq_dualtask_experiment.py)，默认读取 `runs/vq_dualtask_20261012/` 并要求结果全部完成：

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python \
  scripts/analyze_vq_dualtask_experiment.py
```

训练期间需要阶段性汇总时，在命令末尾增加 `--partial`。输出仍为该实验根目录的 `analysis.json` 与 `RESULTS_ZH.md`，需按其中的完成状态解释。

分析脚本的 15 项测试已通过，最终严格汇总已完成并验证全部六次训练和十二份 world 报告。训练期间的 partial 汇总已由最终 completed 汇总替换。

可独立重新生成结果图：

```bash
/root/miniconda3/bin/python scripts/plot_vq_dualtask_results.py
```

绘图脚本只读取完成的 `analysis.json` 与六份 `epoch_history.json`；若分析为 partial，或 epoch、最佳 NLL、world 身份验证与 horizon 覆盖不一致，则拒绝生成正式图。输出目录为 `runs/vq_dualtask_20261012/figures/`，每张图保存为 300 dpi PNG 和 PDF。

正式图已生成并通过视觉核查：六组训练曲线（未随源码发布的历史工件：`../runs/vq_dualtask_20261012/figures/pcr_learning_curves.png`）；自主预测相对自身 copy（未随源码发布的历史工件：`../runs/vq_dualtask_20261012/figures/autonomous_copy_ratios.png`）。对应 PDF 位于同目录。

`pcr_learning_curves` 展示六次训练的未加权 pCR BCE 与患者平均前缀验证 NLL，标出 pCR 最佳 epoch，raw、A、B 的 epoch 分别从 1 起。`autonomous_copy_ratios` 展示每个方案、种子与 best/last 的自主 1–3 步预测相对于其自身 copy 的比值，水平基线为 1；空 cohort 或 copy 近零导致的 null 比值明确标注。不同 teacher 下的原始 L1 不被合并到图中。绘图已通过临时合成汇总检查：四份 PNG/PDF 有效，11 项异常输入被拒绝，null 比值正常显示；正式实验目录未生成测试图。

查看已完成队列：

```bash
cd '/root/autodl-tmp/line b/breast-mri-vla-jepa-vq-dualtask'
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python \
  scripts/run_vq_dualtask_experiment.py --profile dynamic --status
```

以下保留本实验原始启动命令作为复现记录。当前已完成，请使用上面的 `--status` 查看，勿在同一输出目录重复启动：

```bash
OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src /root/miniconda3/bin/python \
  scripts/run_vq_dualtask_experiment.py --profile dynamic \
  --detach --max-concurrent 3 --gpu-memory-budget-mib 29696
```

中断队列可在同一命令末尾加 `--resume` 恢复；当前队列已完成，无需 resume。默认输出为 `runs/vq_dualtask_20261012/`，这个 launcher 只接受 dynamic profile。状态以 `controller.json`、`controller.log`、各 run 的 `progress.json`、world 文件和最终 `analysis.json` 为准。

队列于 **2026-10-12 14:20:59（北京时间）正式 detached 启动**，初始 controller PID 为 49690；六个 run、并发上限 3 与总预算 29696 MiB 已记录在 launch.json（未随源码发布的历史工件：`../runs/vq_dualtask_20261012/launch.json`）。全部训练、完整 world 评价与严格分析现已完成。

真实 CUDA 验证已完成：raw、冻结与微调各自进行 B16 的 T0/T2/T3 前向、损失、反向和更新，共九项训练检查；另做各方案 T0/T3 diagnostics，共六项检查，以及 FP32 teacher forcing、1–3 步自主 latent 预测和 copy 对照检查。均通过，记录见 vq_dualtask_cuda_20261012.json（未随源码发布的历史工件：`../reports/vq_dualtask_cuda_20261012.json`）。

检查的 CUDA reserved 峰值为 raw 8088 MiB、冻结 7904 MiB、微调 10568 MiB。队列分别预留 10240、9216、13312 MiB，world evaluator 预留 10240 MiB；总预算 29696 MiB，最多三个训练或评价 worker。只有预留总和允许时才并发三个，微调通常最多两个。CPU 完整套件 291 项通过，自动分析 hook 接入后另有 67 项相关检查通过。这些运行检查不代表两个任务的性能已经提高。

单 batch CPU 调试需要显式借用原 V1 manifest：

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src /root/miniconda3/bin/python \
  scripts/debug_vla_jepa.py \
  --manifest '/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32/manifest.json' \
  --config configs/vq_dualtask_frozen_dynamic_seed17.yaml --mode loss
```

这会显示 pCR BCE 与 JEPA 分量。使用 `--landmark 0`、`1` 或 `2` 可检查不同当前阶段，并通过 `--patient-offset` 选择有真实相邻随访的样本；仅调试 T3 时 JEPA 为零是预期行为。把 `--mode loss` 改为 `--mode train-step`，可检查冻结前端不更新、适配层及有监督信号的两个分支得到梯度；这只在内存中执行一步，不保存训练权重。读代码可按 `data.py → encoder.py/vq_encoder.py → model._backbone → model._teacher_forcing_loss → training._configure/_make_optimizer` 的顺序进行。

原有代码保留 MIT；派生的 `vq_encoder.py` 保留 MeWM 与 `cgznb/symm-fm` 的 CC BY-NC 4.0，见 [CODEC_NOTICE.md](../licenses/CODEC_NOTICE.md)。当前文档不更改许可范围。
