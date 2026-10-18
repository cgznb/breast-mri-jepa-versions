# Response-JEPA v2：训练、评价与消融完整手册

版本：0.2.0。参照上游提交：`[commit reference omitted]`。

## 1. 这个包如何与现有项目衔接

本包是独立、完整的 v2 源代码，不依赖上游 Python 包。解压到新目录，保留原仓库、原配置和原实验结果。它复用上游 **数据合同**，不是复用上游 checkpoint。原来的 `best.pt`、`last.pt` 和 YAML 不能直接交给 v2；v2 会明确拒绝，而不是忽略不匹配参数。

你需要已有的已配准 ROI32 manifest 和其引用的 NPZ，以及原临床字段/Arm/pCR。数据准备和配准仍由原数据管线负责。这里没有 DICOM 整理、NIfTI 配准或未来病灶分割功能。上游原始实验的精确对照，请继续用原仓库运行；本包的 `legacy_direct` 只是旧编码器拓扑控制，训练/融合/采样并非整个历史流程。

研究主问题是：稳定状态空间中的治疗条件自主推演，是否优于 copy，并为早期 pCR 判断提供稳定增益。运行代码不是已证明这个主张。

## 2. 安装与最小检查

建议使用独立虚拟环境或已有干净的 PyTorch CUDA 环境。PyTorch 的 CUDA wheel 应匹配你本地硬件与驱动，不要盲目安装某个固定 CUDA 版本。

```bash
cd breast_mri_response_jepa_v2
python -m venv .venv
source .venv/bin/activate
# 此处先在自己的环境安装合适的 PyTorch。
python -m pip install -e '.[test]'
python -m pytest -q
python train_response.py --help
python train_response.py smoke --output runs/smoke_cpu_01
```

`smoke` 使用虚构的 16/8/4 名 train/val/test 病例，各阶段仅 2 epoch、小网络和 `[8,16,16]` 图像。它检查 A→B→C 初始化、优化、评价和权重保存，不用于医学效果比较。测试输出目录要新建，不要覆盖前一次。

运行时只支持 CPU 与单进程 CUDA；没有 DDP、FSDP 或 `torchrun`。不要用 `torchrun` 启动本入口，否则可能多进程写同一结果。多卡可用于彼此独立的实验，各自设置 `CUDA_VISIBLE_DEVICES` 和不同输出目录。

## 3. 数据预检查：先做，再训练

```bash
export MANIFEST=/absolute/path/to/registered_roi32/manifest.json
# 有缓存时额外定义；没有缓存就省略所有 image-cache 参数。
export CACHE=/absolute/path/to/image_cache_20261004

python train_response.py preflight --manifest "$MANIFEST" \
  --verify-pixels --output reports/preflight.json

# 使用现成缓存的版本：
python train_response.py preflight --manifest "$MANIFEST" \
  --image-cache "$CACHE" --verify-pixels --output reports/preflight_cached.json
```

正式图像要求：每次检查读取 NPZ 中的 `image`，dtype 必须 float32，shape 必须 `[3,32,128,128]`。三个通道依次为增强前、首次增强后、元数据指定晚期相位；T0/T1/T2/T3 是纵向检查，不是这三个相位。

本包不重新标准化已处理的 MRI，不缩放，不重新裁剪。临床年龄仅从 train 且 T0 已知的数据拟合标准化；HR/HER2/MP 保留二元值。未知临床值和未知 Arm 使用明确的 mask，不能把缺失当成阴性。

每名患者所有检查必须处于同一个 split。缺失 T1 时保持 `visits[1]=null`，不要把 T2 前移。`available_at` 必须反映实际信息可见阶段。ROI 和参考网格应由 T0 定义；JSON 声明只能检查合同，不能证明上游实际未使用未来病灶，也不能证明解剖配准质量。

预检查会统计患者、检查、标签缺失及路径；`--verify-pixels` 会实际检查形状、dtype、有限值。现成缓存还检查 source/cached 文件路径、大小、mtime 与元数据是否一致，拒绝过期缓存。数据指纹包含 manifest 与资产身份，但不是每个 MRI 像素的完整密码学哈希。

数据可以有缺失 pCR：A 不依赖 pCR 优化；B 不用 pCR；C 的分类项屏蔽缺失标签，仍可按配置参与世界监督。分类评价只纳入有标签患者。训练世界模型不要求人人四期齐全，但必须有可监督的未来目标。

**新的交叉验证划分不能仅改 split。**原 ROI32 已用原训练集强度尺度处理；若新外层测试患者曾参与该尺度拟合，需要在上游按新训练折重做或重新应用合法的强度标准化。每一折的 A、B、C 都只能使用该折训练患者的影像进行参数学习。验证影像可以用于预定选模，测试影像不能用于 SSL 预训练或选模。

## 4. 三个阶段究竟训练什么

### A：同次 MRI 表征学习，不做未来转移

每名训练患者每轮选一次实际存在的检查，跨轮轮转阶段；同阶段病例放在一起计算物理患者轴统计。将约 35% 的网格对应体素区域在 **进入 CNN 前**置零，三个 DCE 相位同步遮挡。在线编码器学习恢复同次完整检查的 EMA 特征。

损失为：

`masked_L1 + 1.0*pooled_variance + 0.04*pooled_covariance + 0.1*same_stage_token_patient_variance`。

没有重建整幅 MRI 像素，没有 T0→T1 的监督，也不默认加入 pCR。`loss.repr_pcr=0` 时不计算该分类损失；改成非零就必须在实验记录中称为标签辅助预训练，而不是无标签 SSL。

EMA 系数默认 0.996，每次优化器更新后只在 A 更新。方差/协方差统计显式 FP32。梯度累积不等于把多次物理 batch 合并做协方差：batch4×accum4 的优化器有效 batch 约16，但表征统计仍基于物理 batch4。不要用物理 batch1 企图靠累积解决方差统计。

A 的最低验证目标不自动代表医学表示最好。必须做冻结表征探针、MRI-only 对照，防止学到不具备下游价值的坐标。

### B：固定目标坐标，训练世界模型

将 A/best.pt 的 EMA encoder 同时装入在线和目标编码器，二者冻结。共享融合模块和世界预测器学习治疗条件变化。默认 `0.25*teacher_forcing_L1 + 1.0*free_rollout_L1`，不训练 pCR。

残差预测器最后的增量投影初始化为零，因此初始输出是 exact copy。free rollout 用自己的前一步预测继续推进，不中途读取真实未来。T1 缺失但 T2 存在时，能通过预测 T1 再用真实 T2 监督终点；teacher forcing 没有该真实相邻对，就不虚构监督。

B 全部变体都按验证集 **free-rollout L1** 选模，包括 TF-only 变体，保证选模准则一致。固定坐标下 `ratio=预测L1/copyL1`，小于1才优于 copy；分母极小会输出 null，不造一个很大的或很小的数。

### C：分类与预测状态联合训练

默认冻结 MRI encoder 和目标 encoder，微调融合/世界预测器以及直接、未来与门控读出。

`L = 1.0*BCE_fused + 0.5*BCE_direct + 0.25*BCE_forecast + 0.25*L_TF + 1.0*L_rollout`。

`p_fused=(1-g)*p_direct+g*p_forecast`；是概率混合，不是把两个 logit 简单求平均。门控初始偏向直接分支，`sigmoid(-2)≈0.119`；这个 gate 不是经过校准的不确定性估计器。

未来分类分支读取自主预测的 S3，不读取 teacher-forced 预测。T3 输入时，S3 已被观察，因此这时的所谓 future 分支实际上是末期状态读出；不能把其表现当作提前预测成功的证据。

直接分类消融只计算一次 BCE，不重复计算同一个 direct/fused 损失。T3 的世界项为零；损失先在患者内部平均有效目标，再对物理 batch 全部患者平均，无有效目标者贡献零。这与上游“只平均有未来监督患者”的缩放口径不同，需在方法中写清楚。

## 5. 推荐先跑固定 T0 主任务

这样最容易回答“不获得新的 MRI，能否借助预测治疗反应改进最终 pCR”。A 阶段可用训练患者的各期 MRI 学习表征，B/C 的合法输入始终仅为 T0 和 T0 可见临床/Arm；未来影像只作为训练目标。

### 5.1 A 表征训练

```bash
python train_response.py train \
  --config configs/A_representation.yaml \
  --manifest "$MANIFEST" \
  --output runs_v2/A_shared/seed17 \
  --set train.seed=17
```

有缓存时上面加 `--set "train.image_cache=$CACHE"`；B、C 也加相同配置。GPU 不能用 BF16 时加 `--set train.precision=fp32`。不要把真实 ROI32 的 `image_shape` 改小来省显存，这会违反合同。

### 5.2 A 完成后先看冻结探针

```bash
python train_response.py linear-probe \
  --checkpoint runs_v2/A_shared/seed17/best.pt \
  --manifest "$MANIFEST" --stage 0 --device cuda \
  --output reports/probe_T0_seed17.json

python train_response.py linear-probe \
  --checkpoint runs_v2/A_shared/seed17/best.pt \
  --manifest "$MANIFEST" --stage 3 --device cuda \
  --output reports/probe_T3_seed17.json

python train_response.py tabular --manifest "$MANIFEST" \
  --output reports/clinical_logistic.json
python train_response.py tabular --manifest "$MANIFEST" --include-arm \
  --output reports/clinical_arm_logistic.json
```

这些命令默认只用 train 拟合、val 选超参数和评价，不自动查看 test。只有已锁定最终方案时，才显式加 `--evaluate-test`。

线性探针仅使用冻结教师的空间平均 MRI 特征，不加临床或 Arm；它是可解释的诊断，不是全部 MRI 判别能力的上界。T3 探针必须有真实 T3，不能放入 T0 方法排名。

### 5.3 B 固定坐标世界训练

```bash
python train_response.py train \
  --config configs/t0/B_default.yaml \
  --manifest "$MANIFEST" \
  --init runs_v2/A_shared/seed17/best.pt \
  --output runs_v2/t0/B_default/seed17 \
  --set train.seed=17

python train_response.py evaluate \
  --checkpoint runs_v2/t0/B_default/seed17/best.pt \
  --manifest "$MANIFEST" --split val --device cuda \
  --save-predictions --output reports/B_t0_seed17_val.json
```

重点查看 JSON：`world.all.ratio`、`world.horizons.1/2/3.ratio`、`world.complete_chain`、`world.teacher_forcing`、`target_coordinate_hash`。B 没有分类指标；它的随机任务头不应被报道为 pCR 模型。

不同条件应复用同种子的 A，使目标坐标相同。不同 A/不同种子的 latent L1 不是同一测量坐标，不要简单按误差大小排名。

若 B 仍不能超过 copy，先检查表征、输入配准和阶段可用性，再调世界模型。可以继续 C 做失败分析，但不能预先把它写成有效未来预测。队列提供 `--require-positive-world-skill` 自动拦截 ratio>=1 或不可定义的 B；这只是工程检查，不是统计显著性证明。

### 5.4 C 分类与预测状态读出

```bash
python train_response.py train \
  --config configs/t0/C_full.yaml \
  --manifest "$MANIFEST" \
  --init runs_v2/t0/B_default/seed17/best.pt \
  --output runs_v2/t0/full/seed17 \
  --set train.seed=17

python train_response.py evaluate \
  --checkpoint runs_v2/t0/full/seed17/best.pt \
  --manifest "$MANIFEST" --split val --device cuda \
  --save-predictions --output reports/full_t0_seed17_val.json
```

查看 direct、forecast、fused 三条分支，以及 C 的世界指标是否仍优于 copy。C 按患者等权 pCR NLL 选模，可能牺牲 B 的世界能力；如果发生，必须报告并调整/消融世界约束，而不是只展示 B 的最好 world 与 C 的最好 pCR 拼成一个不存在的模型。

## 6. 动态任务怎么训练

动态任务是在真实 MRI 到来时，用最新合法前缀重新预测同一最终 pCR。它不要求使用动态模型的人一定获得未来检查：只给 T0 也能运行；要展示 T1/T2/T3 更新的收益，才输入当时实际已有的 MRI。

同种子和同训练折可复用 A。重新训练动态的 B 和 C：

```bash
python train_response.py train --config configs/dynamic/B_default.yaml \
  --manifest "$MANIFEST" --init runs_v2/A_shared/seed17/best.pt \
  --output runs_v2/dynamic/B_default/seed17

python train_response.py train --config configs/dynamic/C_full.yaml \
  --manifest "$MANIFEST" --init runs_v2/dynamic/B_default/seed17/best.pt \
  --output runs_v2/dynamic/full/seed17
```

不要把只训练固定 T0 的 checkpoint 在 T1/T3 上当作动态模型测试。代码会拒绝不匹配的单患者请求。

默认 dynamic 采用 balanced_cycle：每患者每轮一次，跨轮轮转全部合法 landmark，起点由患者ID确定，种子控制训练顺序打乱。它减少随机抽样导致的阶段暴露不均衡，但不是“每轮把每患者全部前缀当独立人训练”。验证时读取全部合法前缀，先患者内平均 NLL，再患者间平均。仍应报告各 stage，不能只报告一个混合 AUROC。

固定 T0 和动态模型的任务、训练暴露和选模目标不同；比较真实随访收益优先用同一个动态 checkpoint、同一组患者配对，而非简单比较两个表格列。

## 7. 全部训练参数及硬件调整

| 参数 | A | B | C full |
|---|---:|---:|---:|
| 最大 epoch | 80 | 80 | 60 |
| patience / min_delta | 15 / 1e-5 | 15 / 1e-5 | 15 / 1e-5 |
| 物理 batch / 累积 | 4 / 4 | 4 / 4 | 4 / 4 |
| 学习率 | 1e-4 | 1e-4 | 分类相关头 1e-4；共享主干/世界 3e-5 |
| warmup / scheduler | 5 epoch / cosine | 同左 | 同左 |
| 最低 LR 比例 | 0.05 | 0.05 | 0.05 |
| weight decay / clip | 0.01 / 1.0 | 同左 | 同左 |
| MRI encoder | 可训练 | 冻结 | 默认冻结 |
| 目标 encoder | EMA 0.996 | 固定 | 固定 |
| dropout | 0.1 | 0.1 | 0.1 |
| 默认训练精度 | CUDA BF16 | CUDA BF16 | CUDA BF16 |
| 评价精度 | FP32 | FP32 | FP32 |

这是可用起始方案，不是已调优的最佳配置。参数分组对 bias、归一化一维参数和位置/查询不施加普通权重衰减。固定 RNG 的 CPU epoch 恢复已测试；CUDA 上可能仍有不可确定算子，不能承诺所有硬件逐位一致。

遇到显存不足：先降低物理 batch 到2并把 accumulation改到8，eval_batch_size到1或2；表征预训练物理batch不能1。A 同阶段尾部 singleton 会并入上一批，所以物理批量可能比设定大1，提前留余量。A 与可训练多尺度 C 默认启用 activation checkpoint；`encoder_chunk` 控制一次编码的实际检查数。改变这些设置要在新目录重跑对照组，不能中途改配置 resume。

```bash
python train_response.py train --config configs/A_representation.yaml \
  --manifest "$MANIFEST" --output runs_lowmem/A_seed17 \
  --set train.batch_size=2 --set train.accumulation=8 \
  --set train.eval_batch_size=2 --set model.encoder_chunk=2
```

不要默认 batch4 一定能放进某张显卡；交付时没有 GPU 显存实测。梯度累积保留优化器批量近似，但不同物理批量会改变 A 方差统计，因此属于需要完整记录的协议变化，不可称完全等价。

## 8. 每个训练目录会输出什么

`config.json` 是实际生效配置；`data_audit.json` 是数据合同/覆盖统计；`provenance.json` 记录来源、运行环境、参数数、数据指纹和 warm-start 父 checkpoint；`history.json` 记录每轮损失、验证与优化器更新；`best.pt` 用于报告预定选模模型；`last.pt` 用于恢复；`metrics_best.json` 是 best 的验证结果。

默认不在每次训练自动写患者级预测，避免误公开；用 `evaluate --save-predictions` 导出。`--set train.save_predictions=true` 可在训练结束导出私有的 `predictions_val.csv/world_val.csv`。这些 CSV 不应直接放到公共 GitHub。

## 9. 断点恢复

```bash
python train_response.py train --config configs/t0/C_full.yaml \
  --manifest "$MANIFEST" --output runs_v2/t0/full/seed17 --resume
```

恢复时不要再传 `--init`。第一次训练用了哪些 `--set`，恢复时也必须一致，例如缓存、seed、batch。代码要求源文件哈希、配置、数据身份和 Python/PyTorch/NumPy 版本一致。checkpoint 在完整 epoch 结束时保存；进程中途失败只能回到上个完成 epoch，不承诺逐 minibatch 恢复。

修改源代码、模型结构、最大 epoch、数据划分或标准化后，应启动新实验。`--stop-after-epoch` 只用于测试中断，不改变整个预定 LR 日程。只加载可信的 v2 checkpoint，PyTorch 完整优化器/RNG 恢复使用可执行的 Python 序列化格式。

## 10. 如何进行消融：18 个条件分别回答什么

| ID | 初始化 | 主要设置 | 对比目的 |
|---|---|---|---|
| clinical | 从零 | 无 MRI、无 Arm 的直接网络 | 临床信息本身的能力 |
| clinical_arm | 从零 | 无 MRI，有 Arm | Arm 的预测贡献；不是因果效应 |
| mri_direct | 从零 | 仅 MRI，新编码器、直接分类 | MRI 能否独立提供信息 |
| legacy_direct | 从零 | 两卷积旧拓扑、2×4×4 网格 | 历史编码瓶颈近似控制，非完整历史复现 |
| legacy_large_grid | 从零 | 同旧拓扑、4×8×8 网格 | 与上一项比网格分辨率 |
| multiscale_direct | 从零 | 新多尺度、直接分类、encoder可训练 | 与上一项比编码前端 |
| pretrained_direct_ft | A | 直接分类，encoder微调 | 与 multiscale_direct 比预训练 |
| pretrained_direct | A | 直接分类，encoder冻结 | 与上一项比冻结策略 |
| tf_aux | A→B_tf_only | direct；TF，无自由推演损失 | teacher-forcing 辅助基线 |
| rollout_aux | A→B_default | direct；TF＋free rollout | 与 tf_aux 比自主训练 |
| full | A→B_default | gate/direct/forecast；完整 C 损失 | 主方法 |
| copy_readout | 同 full 的 A/B | 同样训练世界模型，但分类末态改读 anchor copy | 预测变化的价值，而非多加一个头 |
| stopgrad_forecast | 同 full 的 A/B | 只截断 pCR→预测状态梯度 | 结局监督是否塑造了推演 |
| no_residual | A→B_absolute | 下一状态直接预测，去 residual | copy 初始化/残差结构的作用 |
| no_tf | A→B_free_only | 只有 free rollout，无 TF | TF 是否稳定优化 |
| no_arm | A→B_no_arm | B/C 移除 Arm | 治疗类别是否被使用 |
| linear_world | A→B_linear | token-wise 线性残差世界头 | 复杂转移器是否必要 |
| full_encoder_ft | 同 full 的 A/B | C 微调在线encoder＋固定教师对齐1.0 | 稳定坐标与适应下游的权衡 |

`clinical`/`clinical_arm` 使用与主方法一致的任务框架；另提供更简单且值得首先报告的 `tabular` 逻辑回归，不要拿过大的纯临床 Transformer 作为唯一临床基线。

**copy_readout 不是把世界模型替换为 copy。**它与 full 使用同一个 B 初始化、同样世界损失和分类分支容量，只把未来分类头看到的 S3 换为当前 anchor 状态；世界预测仍学习并正常评价。因此它能比“完整模型 vs 少一个头”更直接检验预测变化的读出价值。训练以后因为 task 梯度不同，两个模型的世界参数当然可能分化。

`rollout_aux` vs `full` 同时增加预测读出及其分类监督，是一个组件包的效果，不是严格等参数单因素；应结合 `copy_readout` 和 `stopgrad_forecast` 才能解释。`no_residual` 同时取消零增量的 copy 起点，属于合理结构消融，不应声称仅改变一个数学符号。

`pretrained_direct_ft` vs `multiscale_direct` 具有匹配的 C 微调策略，但 A 引入额外训练计算；若论文要声称预训练效率，还需要报告总更新量/计算，并增加计算预算匹配的直接训练对照。当前模板没有宣称所有条件总训练 FLOPs 相同。

### 建议的执行顺序

首先跑 tabular、mri_direct、legacy_direct、legacy_large_grid、multiscale_direct，判断 MRI 和编码前端是否真的起作用。然后固定 A，跑 pretrained_direct、tf_aux、rollout_aux、full、copy_readout、stopgrad_forecast。只有主链条出现有重复性的正向结果，再扩展 no_residual、no_tf、no_arm、linear_world、full_encoder_ft。

初筛可用17、43两个种子；确认后增加71，再进行预定的更多种子/划分。不要只发表最好的 seed。种子差异不等于患者抽样或跨医院泛化不确定性，最终仍需要适当的独立验证。

## 11. 自动生成与运行消融队列

```bash
# 仅生成明确的实验依赖、配置与命令。
python scripts/run_ablation_suite.py \
  --manifest "$MANIFEST" --workdir runs_v2 --profile t0 \
  --seeds 17,43 \
  --experiments pretrained_direct,tf_aux,rollout_aux,full,copy_readout,stopgrad_forecast

# 实际同步执行，并恢复/复用已经合法完成的公共阶段。
python scripts/run_ablation_suite.py \
  --manifest "$MANIFEST" --workdir runs_v2 --profile t0 \
  --seeds 17,43 \
  --experiments pretrained_direct,tf_aux,rollout_aux,full,copy_readout,stopgrad_forecast \
  --execute --resume
```

每个 seed 是1个 A、2个 B（TF-only/default）、6个 C，共9次；两个种子计划18次，不是每个 C 重练一遍相同 A/B。`--profile dynamic` 生成动态 B/C。同一个 `workdir` 的 A_shared 可在两种 profile 间共享；完成过 A 时应使用 `--resume`。

加 `--image-cache "$CACHE"` 使用原缓存；`--require-positive-world-skill` 会在进入相关 C 之前检查 B 的验证 ratio。该开关会中止依赖无效 B 的队列，不应据此偷偷剔除不好的种子；报告所有失败和终止。

队列默认只打印计划，只有 `--execute` 才训练。它不自动运行 test。可以用 `--experiments all` 生成全部18条件，但不建议第一轮盲跑。科学变量修改应编辑独立 YAML；队列 `--set` 只允许硬件/批量等列明的覆盖，避免“某个消融偷偷换损失”。

```bash
python scripts/summarize_runs.py --workdir runs_v2 \
  --output reports/all_seed_stage_branch_metrics.csv
```

汇总保存每个 seed、每个 stage 和每条分支，不自动挑最佳 seed、不伪造显著性。最终报告均值/离散度时要保持同样的 seed 和患者集合。

## 12. 共同患者配对比较

先为两个预定 checkpoint 导出 val 或最终 test 的逐患者预测。文件名由 `--output foo.json` 自动派生为 `foo_predictions.csv`。

```bash
python train_response.py evaluate \
  --checkpoint runs_v2/t0/pretrained_direct/seed17/best.pt \
  --manifest "$MANIFEST" --split val --device cuda --save-predictions \
  --output reports/direct_t0_seed17_val.json

python train_response.py paired \
  --a reports/direct_t0_seed17_val_predictions.csv \
  --b reports/full_t0_seed17_val_predictions.csv \
  --stage-a 0 --stage-b 0 --repetitions 2000 \
  --output reports/paired_full_minus_direct_T0_seed17.json
```

输出差值是 **B−A**：AUROC/AUPRC 为正更好；NLL/Brier 为负更好。bootstrap 以共同患者为单位，不把一名患者的多个访视当多个独立人。不满足两类标签的重采样会跳过无法定义的 AUROC/AP，并返回有效次数。

对动态模型，比较同一个 checkpoint 的 T0 与 T3：

```bash
python train_response.py evaluate \
  --checkpoint runs_v2/dynamic/full/seed17/best.pt \
  --manifest "$MANIFEST" --split val --device cuda --save-predictions \
  --output reports/full_dynamic_seed17_val.json

python train_response.py paired \
  --a reports/full_dynamic_seed17_val_predictions.csv \
  --b reports/full_dynamic_seed17_val_predictions.csv \
  --stage-a 0 --stage-b 3 --repetitions 2000 \
  --output reports/paired_dynamic_T3_minus_T0_seed17.json
```

这只使用两阶段均有合法记录的共同患者，因此回答的是该共同子集的随访信息增益，不自动代表所有患者；还应报告缺失随访的人数和特征。bootstrap 区间仅反映给定已训练模型的患者采样变异，不包含重新训练/重新选模的不确定性。多个种子的预测不能直接追加成重复患者记录再 bootstrap；应逐种子分析，或预先定义每患者集成方式后再做患者级分析。

旧仓库只发布的聚合 CSV 不足以做配对。若你本地保留包含患者级 `patient_key/landmark/pcr_probability/pcr` 的旧评价 JSON，可运行：

```bash
python scripts/convert_upstream_predictions.py \
  --input /path/to/upstream_evaluation.json --output reports/legacy_predictions.csv
```

转换器拒绝不存在或含多个歧义预测列表的数据。即便可以配对，历史优化器、抽样和选模策略不同也要明确披露。

## 13. 单患者推理：只有 T0 也可以

```bash
python train_response.py predict \
  --checkpoint runs_v2/t0/full/seed17/best.pt \
  --manifest "$MANIFEST" --patient-key PATIENT_ID \
  --landmark 0 --request 1,2,3 --device cuda \
  --states-out private_outputs/patient_states.npz \
  --output private_outputs/patient_prediction.json
```

预测只读取合法输入，不读真实未来图像、不读 pCR 标签；已测试未来图像文件不存在仍可进行合法早期推理。`--request` 仅决定返回哪些 latent，不改变当前 pCR。只请求 T3 时内部仍会按模型需要经过T1/T2，不会把T3伪装成一步。

NPZ 的 `states` shape为 `[B,4,256,128]`（默认）；未请求槽为零，`query_mask` 描述哪些输出有效。它没有图像 decoder 和物理 affine，不能当做生成 MRI。直接分类且未训练世界头的 checkpoint 请求 latent 会被拒绝。

动态模型在 T1 得到真实新 MRI 后，可指定 landmark1 重新推理；这不是强制纠正网络，也不是运行时更新参数。固定 T0 路线完全不需要后续观测。

Arm是当时已知/明确给定的分配场景类别，不包含真正核验过的逐周期治疗动作。替换Arm后得到的差异只是条件模型输出，不是已识别的个体因果治疗效应。

## 14. 梯度与表示诊断

```bash
python train_response.py diagnose \
  --checkpoint runs_v2/t0/full/seed17/best.pt \
  --manifest "$MANIFEST" --device cuda --patients 32 \
  --output reports/diagnostics_full_seed17.json
```

诊断只取训练患者，不更新参数。输出分类/世界目标在共享 fusion 的加权梯度范数与夹角，以及 anchor 表征的患者方差和有效秩。负夹角不自动证明任务冲突导致泛化失败；它需要结合消融、重复实验和训练曲线解释。动态探针中 anchor 阶段可能不同，不能把混合阶段的方差当成严格同阶段坍塌指标。

## 15. 最终测试与论文结果组织

开发期间只用 train/val。原102人验证集已经支持过多轮方法选择，不应重新命名成盲测测试集。真正独立的 test 准备好、预处理合法、方法和阈值/分支选取锁定后，才运行：

```bash
python train_response.py evaluate \
  --checkpoint runs_v2/t0/full/seed17/best.pt \
  --manifest /absolute/path/to/final_manifest.json \
  --split test --device cuda --save-predictions \
  --output final_results/full_t0_seed17_test.json
```

外部推理允许不同患者，但必须使用训练时保存的临床尺度与影像标准化约定。模型会拒绝不一致的影像标准化声明。不能在外部测试患者上拟合新尺度后又称完全一致输入。

主表应报告固定早期任务的 pCR AUROC/AP/NLL/Brier 与共同患者差值；世界表报告固定坐标、各 horizon、copy ratio 和有效患者/目标数；动态更新单独成表；真实T3参考、临床基线、分支读出分别标明信息时间点。不得把不同checkpoint的最好 pCR 与最好世界指标拼成同一模型，也不要把多个seed的患者记录当独立样本。

这一版不提供真实医学提升结果，不提供未来 MRI 生成结果，也不承诺投稿成功。是否继续扩展，应由“表征可读 → 自主预测优于copy → 预测变化相较copy_readout有增益 → 独立数据泛化”这条链条决定。

## 16. 常见问题

**为什么不直接训练 C？** direct-only 从零基线可以；世界版本需要稳定状态和已训练B，代码故意阻止冻结随机编码器以及没有世界监督的假预测读出。

**A 用了未来 MRI 会不会泄漏？** 训练患者的后续检查用作预训练/监督不等于测试输入泄漏；任何验证/测试患者的后续检查都不能参与参数学习。A不意味着部署时需要那些检查。

**B 的 loss 下降但 ratio还大于1？** 说明绝对误差下降仍未超过不变基线，或学习状态缺少可预测的反应信息。不能把loss下降当世界模型成功。

**C 的 forecast 很差但 fused 还行？** 可能gate主要依赖direct。必须看full-v-copy_readout和分支结果，不能仅凭fused正常就认定预测状态有效。

**为什么没有分割/FTV损失？** 当前数据合同没有经过核验的病灶目标。本包不凭空生成病灶标签，也不把latent拆成两段就声称解剖/反应解耦。

**能不能随便加大 encoder、dim、grid？** 可以形成新研究配置，但会改变A/B权重形状和统计意义，必须重训，不能旧checkpoint续训。首先完成这里的可归因对照。
