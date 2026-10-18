# 四组代码的数据输入与输出

这里描述已完成的乳腺 MRI / I-SPY2 实验，而非为新实验预设数据方案。所有维度按历史正式 ROI32 配置；合成 smoke 使用更小尺寸。

## 文件入口

主入口是 JSON manifest（`schema: responsewm_raw_mri_v1`）及其指向的 MRI 文件。每位患者有四个固定纵向槽 T0/T1/T2/T3、split、临床字段、治疗 Arm 及各信息的可用阶段；缺失随访保留 `null`，不移动后续检查的槽位。训练/评价还可有最终 `target.pcr`。T0 必须存在。

正式一次 MRI 从 NPZ 中读取 `image` 数组，必须为有限值 `float32 [3,32,128,128]`：

| 维度 / 通道 | 含义 |
| --- | --- |
| 3 个通道 | 同次 DCE-MRI 的 `pre_aqc0`、`first_post_aqc1`、`metadata_late` |
| 32×128×128 | 深度×高度×宽度的已预处理 ROI |
| 4 个纵向槽 | T0、T1、T2、T3 四次随访；与三个 DCE 相位是不同维度 |

晚期相由元数据指定，不能一律当作 DCE2。各次检查使用 T0 决定的参考网格与 ROI，已经完成配准、裁剪和按训练集规则归一化，零背景保持为零。manifest 需声明几何、归一化与相位顺序。可选 image cache 是经元数据检查的 float32 NPY 加速副本，不是另一个独立数据集。

仓库保留各版本已有的数据准备/检查脚本，但正式训练入口不直接读取原始 DICOM。本仓库也不包含上游外部项目的全部 DICOM 发现、配准和 codec 训练流程。V1 系列支持的另一种 raw NPY/NIfTI 合同见各包说明，不能与已归一化 ROI32 的处理规则混用。

## 模型实际输入

| 信息 | batch 形状 | 可见性规则 |
| --- | --- | --- |
| MRI | `[B,4,3,32,128,128]` | 截止当前 landmark 已到达的真实检查；不可见槽全零 |
| MRI 观察 mask | `[B,4]` bool | V1 系列叫 `observed_mask`；V2 叫 `observed` |
| 当前 landmark | `[B]` 整数 | 0/1/2/3，表示当前观察阶段 |
| 临床值 | `[B,4]` | 年龄、HR、HER2、MammaPrint；不可见值为零 |
| 临床 mask | `[B,4]` bool | 区分未知值和真实数值零；按 known_at 遮蔽 |
| 治疗 Arm | `[B]` 整数 | 0 未知；1–13 为固定顺序的官方分组，另有可用阶段 |
| 未来输出请求 `query_mask` | `[B,4]` bool | 请求哪些未来阶段，不表示未来检查是否实际存在 |

年龄标准化只拟合训练集；二元临床变量保留 0/1。推理恢复 checkpoint 的训练尺度。治疗 Arm 是类别，不能解释为剂量序列或治疗因果效应；历史 `assigned_arm_scenario` 表示从 T0 给定已分配 Arm 的情景假设。当前未知的信息必须遮蔽。

这些 mask 是可用性或输出选择，不是肿瘤分割图。最优 evolution encoder64 不使用病灶加权；仓库中 lesion 消融不代表该最优组的输入必需项。

固定 T0 只输入 `[I0,0,0,0]`。动态模式在当前时间只输入已观察历史：例如 T1 且 T0/T1 都已到达时，输入 `[I0,I1,0,0]`、观察 mask `[1,1,0,0]`，可以请求 T2/T3 的状态。T2 检查缺失时不应将 T3 移入 T2 槽。

## 训练监督与推理

监督对象与前向输入分开：最终 pCR 为 `[B]` 二元标签及 label mask；真实未来 MRI 为 `[B,4,3,32,128,128]` 和 `[B,4]` future mask。未来影像由 EMA teacher 提取目标特征，按各版本的正权重损失监督世界模型。V2 / evolution 的 teacher-forcing 训练支路可以使用真实中间状态；最终分类预测使用合法历史和自主 rollout。

推理不需要患者最终 pCR 标签，也不需要尚未发生的未来影像内容。所有阶段输出预测的是**同一个最终 pCR 终点**，不是四个不同的阶段标签。评估分类成绩需要 pCR 真值；评估未来状态误差另需未来影像真值，两者仅用于评分。

## 版本输出对照

| 项目 | 最终分类输出 | 可选未来状态 | 与分类的关系 |
| --- | --- | --- | --- |
| V1 | `pcr_probability` | `[B,4,32,128]` | pCR 从观察历史读出；world 为辅助支路。动态 L1 没有训练 world head |
| VQ encoder | `pcr_probability` | 正式单任务权重不提供有效预测；底层容器同 V1 | JEPA 权重为零，world head 没有训练 |
| VQ dualtask | `pcr_probability` | `[B,4,32,128]` | 正 JEPA 训练未来支路，仍不回送 pCR 分类头 |
| V2 joint | 最终/direct/forecast 概率及 readout mode | `[B,4,256,128]` | forecast 状态进入分类；gated 可混合 direct 和 forecast；最优 F 使用 forecast |
| V1-evolution 最优组 | `pcr_probability` | `[B,4,32,128]` | 自主 T3 状态及其相对当前状态的变化进入分类 |

最后的 128 是特征维度，32 或 256 是空间 token 数。状态的四槽布局保留 T0–T3，未请求槽置零。模型内部所需 rollout 与对外请求范围应区分：特别是 evolution 使用稳定的三个转移槽，V2 也不会因减少返回阶段而跳过分类所需 rollout。

这些是连续 latent，**不是 MRI 像素、分割或 VQ 离散码本 ID**。旧 VQ 使用每相 8 通道的连续 pre-quant 特征；最优 evolution encoder64 使用压缩前每相 64 通道特征，两者最终都适配为 32×128 tokens。正式实验关闭未来 MRI flow 生成。V1 部分方案保留已观察 MRI 的重建辅助损失，这不提供未来 MRI 生成能力。

CLI 推理保存概率和阶段信息的 JSON；V1 系列 `--forecast-states` 另存 `.states.npz`（`state_prediction`、`query_mask`），V2 `--states-out` 另存 NPZ（`states`、`query_mask`）。字段及完整命令以各版本输入输出文档为准。概率不自动等于临床决策；阈值应按具体新任务预先确定。

## 训练产物与仓库范围

训练产生 `best.pt` / `last.pt`、配置副本、标准化统计、训练/验证汇总、恢复状态，某些版本还可保存逐患者预测。best 按该实验的固定验证准则选择；动态通常先患者内平均合法前缀 NLL，再跨患者平均。

GitHub 只发布代码、历史配置和聚合报告；不发布患者 MRI、临床表、患者 manifest、cache、codec、正式权重、逐患者预测或原始训练日志。历史输入路径和原运行目录见 [来源说明](RELEASE_ZH.md)。仅凭仓库可进行合成工程检查，复现临床结果还需原数据和外部工件。

各项目更细的字段、命令、损失与输出文件见 [V1](../versions/v1/docs/INPUT_OUTPUT_ZH.md)、[V2](../versions/v2_joint/docs/INPUT_OUTPUT_ZH.md)、[VQ encoder](../versions/vq/encoder/docs/INPUT_OUTPUT_ZH.md)、[VQ dualtask](../versions/vq/dualtask/docs/INPUT_OUTPUT_ZH.md)、[evolution](../versions/v1_evolution/docs/INPUT_OUTPUT_ZH.md)。
