# pCR 外部对比复现建议与统一协议

> **主协议已更新为同一患者划分、各方法保留原论文网络输入。** 当前以 [NATIVE_INPUT_PROTOCOL_ZH.md](NATIVE_INPUT_PROTOCOL_ZH.md) 和 `native_input_policy.json` 为准。下文涉及统一三相ROI、额外临床输入或改写prefix的旧建议仅保留为历史/补充控制。更新前全文在 `history_before_native_input_policy/`。

核验日期：2026-11-02（北京时间）。本轮完成论文、代码、数据接口与实验设计核验；正式模型训练尚未开始。下列“首批”表示优先开发和验证的目标，不表示代码已适配或模型已训练。所有外部论文的结果都需在本队列重新获得，不从原文复制进主结果表。

## 1. 参考论文能复现到什么程度

参考文件：`../2610.25333v1.pdf`，题目为 *CoRe-WM: Conditioned Response-State World Modeling for Longitudinal Treatment-Response Prediction in Breast DCE-MRI*。arXiv v1 发布于2026-10-18。当前官方页面/API未核实期刊录用或期刊DOI，不能因PDF有IEEE TMI页眉就称为已发表TMI论文。

PDF第6页及Table I的比较包含：BreastDCEDL ViT、ViT两阶段临床/几何融合、匹配3D ResNet、PrefixGRU、Temporal representation learning、Jing/Wang two-stage dual-task、BSTNet；作者另提供CoRe-WM。3D ResNet和PrefixGRU的结构可直接重实现，其他方法需要按照原文和现有代码进行适配。

**可以复现多数方法的机制并在我们的队列上重新比较，但目前不能承诺精确复现CoRe-WM原表数值。** PDF第9页仍写代码将在发表时公开；官方arXiv页面和针对性GitHub检索尚未定位到完整官方训练仓库。原808人名单、原折划分、确切训练seed和部分预处理/几何细节也未完全公开。未找到代码不等于证明代码不存在。

它与我们的数据/协议不同：

| 项目 | CoRe-WM参考论文 | 当前MIA稿件 |
|---|---|---|
| 内部人群 | 808位I-SPY2 | 976位，801开发+175固定内部留出 |
| 影像输入 | 8通道，32×96×96，含mask及增强派生图 | 3相，32×128×128，pre/firstpost/metadata-late |
| 显式几何 | 每访视9维病灶描述和响应分支 | 当前分类器没有这些显式动态几何特征 |
| 学习方式 | outcome-free预训练，冻结后balanced logistic读出 | 端到端pCR BCE+自主未来潜在状态辅助监督 |
| 内部主表 | 互不重叠测试折的指标均值±SD | 同175人的五折概率先平均，再算指标 |
| 外部测试 | 独立I-SPY1，156人/41阳性 | 当前175人为内部holdout；不是外部验证 |

原文Table I的0.7620不能与我们概率集成的0.7507直接相减。参考论文外部Table II也不是所有时点最佳：T0–T1的dual-task AUROC为0.7635，CoRe-WM为0.7321。应学习其比较结构，保留所有不利结果。

## 2. 借鉴对比逻辑，并补强关键对照

按“要回答的问题”选择模型：

| 科学问题 | 推荐比较 | 说明 |
|---|---|---|
| 临床/治疗条件本身能预测多少 | ClinicalOnly Logistic；可加ClinicalOnly XGBoost | 同4临床变量；分别报告含/不含assigned arm的场景 |
| 影像是否带来临床以外信息 | 3D ResNet+相同临床输入 | 另保留image-only控制，区分额外信息与网络收益 |
| 直接使用已观察纵向变化是否足够 | 同3D backbone的PrefixGRU/TemporalTransformer+相同临床 | 与未来状态模型同T0/T0–T1/T0–T2可见范围 |
| 近期纵向pCR方法能达到什么水平 | BSTNet；TemporalRL | 原文机制、输入及监督变化必须逐项说明 |
| 未来状态监督及其读出是否有增益 | 已有NoJEPA/DirectReadout/ForecastAttention | 这些是内部消融，不充当外部SOTA方法 |
| 影像合成是否有额外价值 | 后续Seq2Seq-pCR适配 | 跨序列合成与跨访视预测需分别描述 |

参考论文Table III中，加入patient profile使AUROC从0.5816变为0.7135，之后加几何为0.7380。这说明clinical-only及clinical+image强对照尤其必要，不能只与缺少临床信息的裸ResNet/GRU比较。若后续加入clinical+geometry对照，必须先获得相应访视的合法病灶分割，不能把当前重复使用的T0 ROI误作随访病灶。

## 3. 近期文章筛选结果

### 首批纵向目标：BSTNet，2026，npj Precision Oncology

- 论文：*Longitudinal MRI-based deep learning model for predicting pathological complete response in breast cancer: a multicenter, retrospective cohort study*。
- DOI：https://doi.org/10.1038/s41698-025-01256-2 。DOI含2025，正式文章年份为2026。
- 官方代码：https://github.com/XmySz/BreastPCRClassification_Github ，固定commit `723a09ef114240b2f78638b893d89dd2227b6b4c`。
- 价值：ResNet+注意力+LSTM、时序排列预训练，能检验“学习时序”与“显式预测未来状态”的差别；也是参考论文已有对手。
- 真实状态：有训练和fine-tuning脚本，但必须做方法指导下的适配。当前代码硬编码4访视；注意力接收二维[B,F]，会将不同患者当序列；训练脚本每epoch评test并保存best-test权重，val/test还`drop_last=True`。应修正患者/时间维、合法prefix/missing mask、val-only选择和完整测试分母，并记录与论文/源码的差异。
- 结果命名：`BSTNet-adapted`，不能声称未经改变的原论文数值复现。合成张量的注意力语义检查见 `bst_attention_semantics_check.json`。

### 首批纵向目标：Temporal Representation Learning / ART，2025

- 论文：*Temporal Representation Learning of Phenotype Trajectories for pCR Prediction in Breast Cancer*，https://arxiv.org/abs/2509.14872 。当前按预印本标注。
- 代码：https://github.com/cirmuw/temporal-representation-learning ，固定commit `07ffd8b652105291d9bc49e877f9b0ebff501930`。
- 价值：重建+时间连续性+responder alignment，直接针对I-SPY2纵向pCR，并与世界模型机制不同。
- 真实状态：有模型、预训练引擎和ART loss；未核实完整的下游pCR线性评估脚本，需要补齐。ART使用训练pCR标签，不可称完全outcome-free学习。
- 输入：原文四访视PEearly/PElate/SER的2D MIP。当前NPZ保存raw裁剪强度，具有计算派生增强图的基础，但须核对原文相位时刻、比值公式、分母/clip和MIP规则；若使用现有三相MIP，应明确为输入适配。
- 评估：在training患者上学习表示；验证/测试预测只使用当时可见prefix。原4访视读出不可直接用于T0–T2早期预测。

### 第二批高相关目标：STI，2025，EClinicalMedicine

- 论文：*Development and validation of an MRI spatiotemporal interaction model for early noninvasive prediction of neoadjuvant chemotherapy response in breast cancer: a multicentre study*。
- DOI：https://doi.org/10.1016/j.eclinm.2025.103298 。
- 代码：https://github.com/JBnoJB/STIModel ，固定commit `7d24ba6f7635089ed9533f0f1eb4bab918d43343`。
- 价值：治疗前/早期治疗双分支及时间交互，任务高度相关，应争取加入。
- 阻碍：三类habitat mask、相应预训练权重/recipe和实际elapsed time仍需准备。官方main先过采样后分折，有重复患者跨折风险；device条件永远落到CPU；分割标签需避免强度预处理破坏离散类别。先修复这些问题再正式比较。
- 不应把三个habitat全部换成同一个T0 mask后还称原版STI复现。

### 已发表但适配量更大的备选

- **Attention-based multimodal prediction，2025，Journal of Translational Medicine**：https://doi.org/10.1186/s12967-025-06617-w ；代码 https://github.com/TaishiNishizawa/bc_pcr_prediction 。有模型/训练脚本，但原输入含治疗前和治疗后（我们的T0/T3）全乳MRI。改作T0–T2预测、ROI与临床集合均是适配；不能直接拿原结果作早期baseline。
- **MRP，2024，Nature Communications**：https://doi.org/10.1038/s41467-024-53450-8 ；代码 https://github.com/yawwG/MRP 。完整方法还用mammography；我们只能优先考虑MRI分支。源码有两个时点都取date2等问题，需修复并检查临床字段的可见时点。
- **Seq2Seq，2024，Medical Image Analysis**：https://doi.org/10.1016/j.media.2023.103044 ；代码 https://github.com/fiy2W/mri_seq2seq 。原论文代码在 `publications/src/seq2seq`，当前主仓库的新nnseq2seq不能冒充原版。已核实的原目录偏BraTS合成/差异图，缺乳腺pCR训练入口与划分；适合作为合成机制补充，需要自己补齐pCR流程。

### 近期但暂不作为立即复现主力

- **GNN-pCR，2026**：https://arxiv.org/abs/2607.04912 ，https://github.com/hannesk95/GNN_pCR_ISPY2 。当前仓库主要README/图，仍承诺发表后发布完整代码。可阅读并独立重实现，不能称已具官方训练代码。
- **LMF**：https://github.com/martin-bro/LMF 。README自称MICCAI2025，但当前仅README/LICENSE等占位文件；本轮未独立核实完整出版信息，不能作为成熟开源实现推荐。
- **MESN，2025，Breast Cancer Research**：https://doi.org/10.1186/s13058-025-02009-6 。需要DWI/ADC等当前DCE张量没有的模态，未核实完整专用训练仓库，不列首批。
- **CoRe-WM**：机制很近，值得比较，但原完整代码、动态几何和原始split尚缺，先列论文方法重实现目标。

## 4. 首批执行顺序与验收

1. **建立共同评估接口和强控制**：ClinicalOnly Logistic、ResNet3D+clinical、PrefixGRU+clinical。核验五折训练/验证互斥、同175患者、完整分母、合法prefix、batch内其他患者不影响预测。临床标化/缺失处理仅拟合train。
2. **BSTNet适配**：先保证4时点时序预训练与患者独立性，再增加缺失mask和T0–T2前缀分类；记录修正，不保留best-test选择机制。
3. **TemporalRL适配**：对齐raw派生图/MIP定义、训练label使用和下游linear readout；严格隔离test，再验证合法prefix。
4. **STI第二批**：在habitat/时间和预训练来源可复现后开发。若暂时缺失，应如实报告未完成，而不是产生简化后冒名的STI结果。
5. 固定每种方法的recipe、超参数搜索范围和checkpoint规则后进行fold1兼容性/资源验证，再扩展五折；验证集选择及调整不能看test改善。最终指标按既有统一协议重算。原作者recipe与匹配recipe的差异记入每个方法的reproduction card。

新方法统一使用 **seed44、既有五折、801开发/175内部留出、同一56阳性、五折概率先平均、T0/T1/T2指标及Macro**。阈值主结果仍为0.5，其他阈值仅验证敏感性；现有14指标、患者置信区间与配对统计复用同一实现。新增比较后必须重新明确多重比较的范围，不能沿用只含旧3模型的九项校正当作全新baseline表的校正。

MRI-only方法应保留原版信息范围，并另给加入相同临床变量的匹配控制；额外临床/几何、监督标签、预训练来源和计算预算均在主表说明。使用病灶真值或更大FOV的原版行，与统一3相ROI适配行分开描述。若做外部I-SPY1，需单独冻结独立患者与预处理协议；不能把I-SPY2/ACRIN命名空间当两个独立外部中心。

## 5. 数据接入的重要细节

- 当前976数据的NPZ存raw，不是已经z-score的旧ROI32。必须使用冻结Medium976配套loader或等价的前景内、fold-train统计一次归一化，避免透传或重复归一化。
- NPZ里的每个访视`roi`都来自同一个T0预测mask，不是随访肿瘤分割，不能用于声称真实动态体积、9D随访geometry或habitat。
- 当前只读已核验NPZ不涉及重做几何。若后续为8通道/新mask/全乳输入重建数据，则要新建版本，并按项目AGENTS要求保持metadata、mask header、变换与原裁剪几何配套；不能直接混用新版public JSON与旧mask header。
- 当前划分和统计身份见 `frozen_data_protocol.json`，只保存聚合计数及manifest哈希，没有导出患者名单。
- 所有模型代码先完成最小前向/梯度、患者独立性、未来遮罩、源数据与split核验，再作资源测量和正式训练。代码获取、适配、验证、开训、完成训练和正式评估是不同状态，不能混称“已复现”。

## 6. 证据位置

- `reference_paper/AUDIT_ZH.md`：参考论文逐页审查与比较逻辑。
- `reference_paper/comparison_inventory.json`：原文每种baseline及可复现边界。
- `recent_literature/candidate_methods.json`：近期候选的文章、代码、commit和缺口。
- `recent_literature/`：论文全文XML、仓库README及关键源码快照/获取凭据。
- `local_readiness/`：本地数据与训练代码接入核验、BSTNet/Seq2Seq源码证据。
- `frozen_data_protocol.json`：本轮核对后的五折计数、归一口径和manifest SHA256。

本文件是研究与实施建议，不是已取得的baseline性能报告。当前稿件保留原有结果，待外部方法完成公平重训和正式评估后再增加结果表。
