# VLA-JEPA / Cancer JEPA 与当前 pCR 双任务结构复核

日期：2026-10-12。范围：第一版新双任务包 `breast-mri-vla-jepa-vq-dualtask`，兼顾已完成的 V1/V2 对照。本文是代码与论文复核及研究建议，不是新实验结果；没有修改模型、配置、数据或权重，也没有启动新训练。

## 1. 两个链接对应两篇不同论文

| 来源 | 实际内容 | 与当前任务的关系 |
| --- | --- | --- |
| [ginwind/VLA-JEPA](https://github.com/ginwind/VLA-JEPA) / [2602.10098](https://arxiv.org/abs/2602.10098) | VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model，作者公开机器人实现，基于 starVLA | 未来表征监督动态 token，并通过 token 顺序辅助动作学习 |
| [2610.09194](https://arxiv.org/abs/2610.09194) | Patient, Place, Prior (P³): What Counts as Personalization in Medical World Models?，提出 Cancer JEPA | 乳腺纵向 latent 预测、病灶定位和个体化预测审计；没有 pCR 实验 |

已阅读用户精确论文全文，以及仓库对应原论文的方法和实际 forward/loss/inference。引用缓存位于 `/root/autodl-tmp/.codex/references/vla_jepa_review_20261012/`；论文逐节阅读记录见其中 `paper/review.md`。

## 2. 原仓库与咱们已经一致的核心连接

上游 `starVLA/model/framework/VLA_JEPA.py` 的 179–180 行分离动态 token 与动作读出 token；204–215 行用固定 V-JEPA2 教师状态训练因果 world predictor，损失为 latent L1；241 行直接把动作读出 token 送入 Flow Matching 动作头。机器人批次 world 权重为 0.1，人类视频批次单独训练 world。实际 `predict_action`（277–305 行）不运行 world predictor，也不把预测出的未来 latent 输入动作头。

上游 [§3.3 / Eq.6](https://arxiv.org/html/2602.10098v2#S3.SS3) 将动作读出 token 放在动态 token 后面，利用因果注意力传递约束。咱们已经实现相同方向：

- [model.py](../src/mri_vla_jepa/model.py) 150–152 行建立上三角因果遮罩，仅放开观测 context 内部的双向注意力。
- dynamics query 能读合法观测 context，pCR query 能读 context 和 dynamics；dynamics 不能反向读 pCR。
- 157 行的四个 pCR query 求均值，再经 `LayerNorm(128) → Linear(128,1)` 输出 logit。
- 190–202 行的未来预测只提供辅助 L1；真正生成的未来 latent 没有回送分类。

因此，“加入 dynamics → pCR 有向注意力”已经存在；“加入预测状态 → pCR”则是比原 VLA-JEPA 更强的任务连接，需要单独证明价值。

## 3. 完成结果给出的约束

下表是种子 17/43 的 pCR-best 开发验证均值，非集成；完整证据见 [最终解释](../historical_runs/vq_dualtask_20261012/INTERPRETATION_ZH.md)。

| 方案 | 患者等权 NLL ↓ | T0 AUROC ↑ |
| --- | ---: | ---: |
| 原 L1：CNN pCR-only | 0.549020 | 0.740625 |
| 新 CNN 双任务 | 0.553476 | 0.735714 |
| 旧冻结 VQ pCR-only | 0.583057 | 0.705580 |
| 新冻结 VQ 双任务 | 0.566432 | 0.715179 |
| 新微调 VQ 双任务 | 0.572514 | 0.717634 |

冻结 VQ 加 JEPA 比此前冻结 VQ 单任务的 NLL 两种子都改善，但没有超过原 CNN L1。微调 B 相比自己的 A-best，NLL 两种子都变差；各阶段 AUROC 有混合变化。六组训练后期分类明显过拟合，所以没有证据支持先扩大分类头、换巨大 VLM 或扩大整体 Transformer。

冻结/微调 VQ 在各自 teacher 坐标中，自主各步均优于 copy；这不等于状态预测已提高 pCR，也不允许跨模型或 best/last 比较原始 latent L1。当前目标是各 checkpoint 自身的 EMA encoder，冻结 VQ 前端仍会训练适配层与空间 Transformer。

全部来自反复使用的 102 人开发验证、两个种子；test 为 0。

## 4. 建议先做的结构清理：输出请求不应改变分类

当前 `query_mask` 同时表示“请求返回哪些未来状态”和“fusion 中哪些 dynamics query 有效”（model.py 145 行）。标准训练每次请求所有未来阶段，因此 T0/T1/T2/T3 的有效动态 query 数为 24/16/8/0。

CPU 合成输入检查使用当前 `frozen_dynamic_seed17/best.pt`，seed20261012，eval/no-grad，输入 `[1,4,3,32,128,128]`，只有同一个 T0 MRI 可见，临床、Arm、landmark 固定：

| 仅改变输出请求 | pCR 概率 |
| --- | ---: |
| 请求 T1/T2/T3 | 0.3933896 |
| 仅请求 T3 | 0.4471138 |
| 不请求状态 | 0.4850651 |

三个请求都通过输入 contract。前两者自主 S3 预测逐值相等，但分类不同：rollout 会补全内部查询，分类 backbone 不补全。该检查不涉及真实患者或临床指标；标准训练恒定请求全部未来，所以不能据此解释当前 AUROC。

建议在新实验里固定三个内部 transition query 组，输出 `query_mask` 只裁剪返回状态；MRI/临床/治疗可见性继续由输入 availability 决定。不能只修改旧 checkpoint 的 mask 后把分数视作公平新结果。V2 已有这种接口解耦，可借鉴实现。

## 5. 最值得验证的任务连接：自主未来变化 → pCR

保留当前直接分类作对照，在新实验中加入小型预测读出：

```text
合法已观察 MRI + 临床/Arm → encoder/fusion → dynamics queries
                                    └────→ direct pCR（现有）
真实当前状态 + dynamics → 自主 rollout → 预测 T3 状态/变化 → 小型 pCR 读出（候选）
```

候选特征可为 `pool(S_hat_T3)` 与 `pool(S_hat_T3 − S_current)`，再经小型投影/LayerNorm/Linear。训练与推理都从合法可见前缀自主生成，分类绝不能读真实未来 MRI、未来 teacher target 或 teacher-forced 未来前驱。T3 没有预测链，需明确只用当前状态，并单独报告该阶段。

只比较“加头前后”不能隔离未来演化的作用。建议在同一 V1 backbone、相同容量/预算下逐个比较：直接读出；当前状态复制到终点的读出；自主预测但 detach 的读出；自主预测且允许 BCE 反传的读出。各组统一可见输入、条件 token、teacher 起点、pool/head 容量、辅助监督与训练预算；forecast 与 copy 的同容量比较用于判断未来演化增益，原 direct 用于实际分类参考。先回答预测特征是否有用，再回答分类梯度是否改善预测链，最后再考虑 gate。

V2 已有残差 world、自主多步 loss、forecast readout 和 gate，不能称为尚未实现的全新思路。其 dynamic F（真实 forecast 分支）三种子 NLL 0.571275，优于 V2 D 0.616710 / G 0.602856；但未超过 V1 原 L1 0.549020，整套协议不同，也未隔离状态表示、读出容量和未来演化贡献。参见 `/root/autodl-tmp/line b/breast-mri-response-jepa-v2-joint/reports/branch_ablation_20261010/INTERPRETATION_ZH.md`。

## 6. Cancer JEPA 更值得迁移的是变化与空间信息

[论文 §4](https://arxiv.org/html/2610.09194v1#S4) 实际使用冻结 3DINO 目标（1024×7×7×7）、latest latent、此前的 latent 差分、真实时间间隔、T0 病灶 core/ring；用训练集拟合的 shrinkage/RRR 做 baseline，再预测 core 门控残差。它在 latest 与差分的同一位置遮挡，只监督 masked∩core，推理按相同遮挡分布平均。不是咱们当前全 token、无空间权重的 L1。

适合后续分开测试的改动：

1. 显式历史变化 `ΔS = S_latest − S_previous` 加上缺失标记；T0 没有历史，差分为零并保留有效性 mask。真实 `Δt` 和逐步治疗进度仅在有可靠、预测时可得元数据时加入，不能把离散 T0–T3 当实际天数。
2. 残差转移 `S_hat_next = S_current + R`，残差输出层零初始化，再单独增加自主 horizon1–3 的训练监督。目前仅 TF L1，与部署自主多步条件有差异。但在同一目标空间，`||(S_current + R) − S_next||₁ = ||R − (S_next − S_current)||₁`；收益若有，来自结构/初始化/条件或监督方式，而非改写公式本身。改变 stop-gradient 路径也可能改变优化梯度，需要保持对照一致。
3. 用预测时可得的 T0 病灶 occupancy 作 soft attention/区域加权，并单独检验空间 token 数。当前输入 contract 不含病灶 mask；T0 ROI 不等于模型显式使用 core/ring。不能使用未来 mask，也不宜直接硬裁掉所有病灶外围变化。
4. 单独对比 VQ `pre_quant8` 与其之前的 `encoder64` 连续特征。现在三相位每相位8通道，拼24通道后平均池化到32 token；扩大到128维不能恢复已经丢失的细节。保留较密 state 网格、再池化到32个 fusion token 是候选，尚未验证。保持 CNN 对照。
5. 使用经过训练且有意义的固定目标 encoder，或预先定义共同训练/评价的目标空间，便于解释 world 误差。不能直接把旧模型在自身 EMA 坐标中的预测向量与另一 encoder 的目标向量计算 L1；需共同目标空间训练，或预先规定并验证坐标映射。EMA 是合法 JEPA 设计，并非 bug；直接冻结随机适配层不能变成语义教师。固定 VQ 重构特征也不等同论文 3DINO。

避免一轮同时替换编码器、RRR、病灶 gate、遮挡 loss 和分类读出。优先在保留现有 CNN/VQ 对照的情况下回答一个明确问题。

## 7. 预测能力还需要超过哪些基线

论文 Table1 primary core 对 RRR 的增益区间跨零；Table2 自己历史优于替换历史、自己病灶支持优于替换支持，但相对 population-average neural correction 的区间仍跨零。论文没有证明 pCR 增益。

当前优于 copy 也不足以说明学到患者特异病程。冻结已有模型后可加入 train-only 均值/低秩预测基线，以及固定目标、时间/Arm/临床条件下的 donor-history 控制。针对模型哪个分支替换历史需明确，不能把整体替换审计冒充原论文只替换 neural correction 的协议。Prior 应在匹配条件下先平均训练患者的预测，再计算误差；不能用平均输入代替。Place 审计仅在有可独立替换的合法空间支持时适用。

现有32患者固定训练 probe 的 weighted JEPA/BCE 共享 encoder+fusion 梯度范数比，中位数约 0.9%–4.4%。它提示辅助信号量级偏小，但不足以证明加权必有效、发生梯度冲突或表征塌缩；不宜据此直接加复杂梯度优化器。

建议的下一步研究问题是：**在现有小型 V1 上，实际自主预测的变化，是否比同容量的当前状态读出更有助于 pCR？** 先清理内部 query/输出请求耦合，再做该受控读出消融；时间、病灶、目标表征和 free rollout loss 后续逐项加入。
