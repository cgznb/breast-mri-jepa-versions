# 近期乳腺MRI pCR预测对比方法：论文与公开代码审查

> 用户已明确主复现采用同一患者划分和各原论文输入；当前执行规则见 `../NATIVE_INPUT_PROTOCOL_ZH.md`。以下来源核验仍有效，统一三相ROI或改写原生访视组合的适配建议只作为补充候选。

检索截止用户日期：2026-11-02。该报告只核验论文、官网代码链接和源文件，没有运行外部代码、启动训练或改写MRI。仓库链接存在不代表训练实现已发布。

## 建议优先级

首批建议采用 **BSTNet（2026，依据论文修正简化代码并标为paper-guided reimplementation）+ TemporalRL/ART（2025，明确输入/下游读出适配）**。并用 ClinicalOnly、同临床输入的ResNet/GRU作稳健地基。若ART读出/2D MIP适配需要较久，可用Nishizawa2025注意力多模态分类器作为有完整网络与训练循环的监督备选，但必须移除未来T3输入并标注适配。

STI2025应保留为有说服力的第二批纵向对照：其habitat和contrastive预训练并非仓库一键可得，且官方脚本有split前过采样等问题。Seq2Seq2024适合作为合成表示补充，不属于同义的未来时点world model。GNN2026和LMF当前仅占位仓库，不能当已可运行训练方法。

## 候选汇总

| 方法 | 年份/发表 | DOI或论文 | 公开代码/版本 | 训练覆盖与结论 |
|---|---|---|---|---|
| BSTNet | 2026 / npj Precision Oncology | https://doi.org/10.1038/s41698-025-01256-2 | [XmySz/BreastPCRClassification_Github](https://github.com/XmySz/BreastPCRClassification_Github) `723a09ef1142` | Official simplified release contains ResNet18+MHSA+LSTM, train/finetune and temporal permutation data scripts. Training protocol and tensor axes require repairs; not exact original training reproduction. |
| TemporalRL_ART | 2025 / arXiv:2509.14872 | https://arxiv.org/abs/2509.14872 | [cirmuw/temporal-representation-learning](https://github.com/cirmuw/temporal-representation-learning) `07ffd8b65210` | Models, ART/TESSL losses, data loaders and pretraining training/validation engines present; no downstream pCR linear-evaluation script found in exact archive. |
| Attention_Multimodal | 2025 / Journal of Translational Medicine | https://doi.org/10.1186/s12967-025-06617-w | [TaishiNishizawa/bc_pcr_prediction](https://github.com/TaishiNishizawa/bc_pcr_prediction) `712e068955b9` | Four source scripts include model.py,data_utils.py,train_test.py,main.py; main placeholders for JSON paths, split indices and checkpoint; import/environment adaptation required. |
| STI | 2025 / EClinicalMedicine | https://doi.org/10.1016/j.eclinm.2025.103298 | [JBnoJB/STIModel](https://github.com/JBnoJB/STIModel) `7d24ba6f7635` | Main, fivefold trainer, 3D ResNet, Siamese/temporal/habitat architecture and dataset present. No habitat segmentation generation or contrastive pretrain recipe/weights found. |
| MRP_iMRrhpc | 2024 / Nature Communications | https://doi.org/10.1038/s41467-024-53450-8 | [yawwG/MRP](https://github.com/yawwG/MRP) `ac79736164ca` | MRI+MG model, data loaders, run.py and Lightning training/test routines present. Official environment Python3.7/PyTorch1.8.1/Lightning1.5.9; private clinical schema and checkpoint paths. |
| Seq2Seq | 2024 / Medical Image Analysis | https://doi.org/10.1016/j.media.2023.103044 | [fiy2W/mri_seq2seq](https://github.com/fiy2W/mri_seq2seq) `0414c0766678` | Large active official repo; exact2024 publication source is under publications/src/seq2seq; pCR downstream completeness audited separately by code_inventory. |
| GNN_pCR | 2026 / arXiv:2607.04912 | https://arxiv.org/abs/2607.04912 | [hannesk95/GNN_pCR_ISPY2](https://github.com/hannesk95/GNN_pCR_ISPY2) `4acf6f3208b5` | Current root has README and figures only; README says full source code will be released upon publication. |
| MESN | 2025 / Breast Cancer Research | https://doi.org/10.1186/s13058-025-02009-6 | 未核实完整官方实现 | Article links generic self-attention-cv library only; no full model training repository verified. |
| LMF | 2025 / MICCAI2025 claimed by README; publication metadata not independently confirmed | 未独立核实出版页 | [martin-bro/LMF](https://github.com/martin-bro/LMF) `0fab332210fb` | Archive/root only.gitignore/LICENSE/README; timeline promises full code by end2025 but not present. |

## 关键适配与代码问题

### BSTNet

**原始输入：** Peak DCE phase, ROI64x128x128, T0-T3 for temporal permutation pretraining; ResNet18+MHSA+LSTM; downstream clinical integration optional.

**适配：** Same-split observed-prefix variant; no test future MRI; reproduce temporal-order stage using development patients only.

**风险/限制：** Official README explicitly calls release simplified. code_inventory audit found best_test checkpoint selection and test evaluation each epoch, validation/test drop_last=True, random file splits, hardcoded four visits; 2D MHSA input mixes patient axis, finetune channel mismatch, paper256d/8heads versus code512d/4heads. Reimplement paper method with correct temporal attention, patient-disjoint manifests and validation-only selection; do not reproduce leaked protocol.

**许可：** README claims MIT, but no LICENSE file in root snapshot; do not automatically redistribute large code excerpts in public package.

### TemporalRL_ART

**原始输入：** 585 ISPY2, four visits, early PE/late PE/SER 2D MIP; reconstruction+temporal continuity+responder alignment; frozen representation linear readout.

**适配：** Supplement linear readout and evaluate legal prefixes. Existing NPZs store raw cropped pre/firstpost/metadata_late intensities (normalization occurs in loader), providing a basis for PE/SER derivation, but original phase timing, denominator, clipping, units and MIP generation must be audited. Start with explicitly labeled matched three-phase-MIP input adaptation.

**风险/限制：** Alignment uses training pCR labels; do not label it outcome-free SSL. README December2025 acknowledges sensitivity to loss weights and recommends tuning on new tasks.

**许可：** MIT file present.

### Attention_Multimodal

**原始输入：** Whole-breast firstpost MRI80x256x256 at pre-NAC and post-NAC (our T0/T3), clinical age/race/HR/HER2/menopausal.

**适配：** Restrict to observed T0/T1/T2 histories, match available covariates and specify ROI-vs-whole-breast adaptation; original T3 model is not early-prediction baseline.

**风险/限制：** Original I-SPY2 660, external I-SPY1 114. No treatment arms. Private data paths alone are routine adapter work, not absence of architecture.

**许可：** No LICENSE file in complete archive.

### STI

**原始输入：** Peak-enhancement DCE at baseline and early treatment; three-class tumor habitat masks; elapsed time; Siamese+temporal attention.

**适配：** Supply valid habitat masks/elapsed time; restrict T0-T1 or legal T0-T2 pair; preserve three habitats or label spatial-prior simplification; repair split-before-oversampling.

**风险/限制：** Source device condition always selects CPU. Main oversamples before k-fold risking same-patient leakage. Masks pass through image preprocessing and must preserve integer labels. Do not run official pipeline blindly.

**许可：** README claims MIT but full archive has no LICENSE file.

### MRP_iMRrhpc

**原始输入：** MRI branch iMRrhpc: longitudinal subtraction MRI+stage embeddings+rhpc clinical factors; full MRP additionally mammography, unavailable in current task.

**适配：** Use MRI branch only, align observed prefixes and available pre-outcome covariates; cannot claim reproducing full mammography+MRI fusion MRP.

**风险/限制：** Current dataset x_idx1 and x_idx2 both use date2; INB_COMPETITION_TASKS undefined; cfg.data.frac/transform mismatches. Fix/audit before training. Do not automatically use post-operative pathology columns as predictors.

**许可：** MIT file present.

### Seq2Seq

**原始输入：** Multi-sequence3D/4D MRI cross-sequence synthesis and differentiation map representation; breast pCR downstream.

**适配：** Reuse exact publication method, common development/holdout and legal longitudinal prefixes; distinguish cross-sequence synthesis from future-visit prediction.

**风险/限制：** Recent nnseq2seq main is a newer method; cannot call current main exact2024 reproduction.

**许可：** MIT licence file present.

### GNN_pCR

**原始输入：** Longitudinal graph representation of real multi-timepoint MRI embeddings; not generative.

**适配：** Paper-based independent reimplementation possible; cannot claim official training-code reproduction now.

**风险/限制：** Repository existence is not release of training implementation.

**许可：** No LICENSE file visible.

### MESN

**原始输入：** Pre/early-treatment T1/DWI/ADC tumor/peritumor features plus clinic.

**适配：** Requires additional DWI/ADC feature inputs absent from current DCE-only tensor; DCE-only variant is a substantial adaptation.

**风险/限制：** Do not mistake generic attention library for official MESN training code.

**许可：** Not applicable; no verified official repo.

### LMF

**原始输入：** README claims longitudinal MRI/clinical ISPY2.

**适配：** No code to port currently; would require independent paper verification and reimplementation.

**风险/限制：** Do not rank as mature open-source method merely from repository description.

**许可：** MIT file present.

## 补充的生成/两阶段工作

- **Jing & Wang，2025，arXiv:2502.00051**：双阶段双任务早期pCR方法，原论文页已实时访问；摘要页没有官方GitHub链接，本轮未核实完整训练代码。不能替换成“现成开源”首批方法；可按文章独立重实现，但应另报reimplemented。https://arxiv.org/abs/2502.00051
- **LISPCN，BMC Cancer 2026**：Joint synthesis of longitudinal MRI and enhanced prediction of neoadjuvant chemotherapy response in breast cancer: a multicohort study，DOI10.1186/s12885-026-15889-4。原文XML已保存PMC13134087.xml；本轮全文未出现GitHub/源码发布段落。与未来影像合成+pCR非常接近，工程优先级受没有完整官方代码影响。https://doi.org/10.1186/s12885-026-15889-4

## 公平比较和报告边界

所有对照应保持咱们固定开发/留出患者、种子44五折与逐患者概率集成口径，允许的输入时点为T0、T0+T1、T0+T1+T2。训练患者可用未来状态作预训练/辅助目标，测试患者禁止未来MRI/未来mask输入。原文含T3的分类器必须重新设计合法prefix，不能复制其已发表AUC到咱们结果表。

当前数据磁盘NPZ存储原始裁剪强度，归一化在loader中完成。raw pre/firstpost/metadata_late为PE/SER计算提供基础，但并不自动等于原文PE_early/PE_late/SER；需逐项审查相位时刻、分母阈值、clip、百分比单位、MIP方向和固定FOV。对比先以matched-input adapted版为主，若保留original recipe则单列更忠实复现。

TemporalRL的alignment显式使用训练pCR标签，不能把它描述为完全无结局标签SSL。STI的habitat不能用随意阈值三个区域替代后仍声称full STI。Gao完整MRP还使用mammography；只复现MRI分支应命名iMRrhpc/adapted。

这些是适配和可复现性判断，没有产生任何新的模型性能证据；不能保证咱们模型优于它们。论文声称临床效益或高AUC不代表在咱们固定holdout可取得同样结果。

## 证据文件

- `candidate_methods.json`：结构化候选、DOI、commit、输入/训练覆盖与适配。
- `fulltext_code_availability.json` 和五篇 `PMC*.xml`：论文全文代码发布说明。
- 各`owner__repo/fetch_receipt.json`：README和GitHub root树实抓、当前commit。
- 已读取实现的四个repo另保留 `source.zip`、`archive_receipt.json` 与代码文本快照，所有来源都是公开仓库，未执行。
- `arxiv_code_links.json`：原文arXiv页面及源码链接核查。

GitHub REST API在本轮返回速率限制403，随后改用官方GitHub HTML、raw内容与codeload按不可变commit抓取；未把API受限误报为“仓库不存在”。

## BSTNet补充审查（code_inventory交叉核验）

Official README explicitly calls release simplified. code_inventory audit found best_test checkpoint selection and test evaluation each epoch, validation/test drop_last=True, random file splits, hardcoded four visits; 2D MHSA input mixes patient axis, finetune channel mismatch, paper256d/8heads versus code512d/4heads. Reimplement paper method with correct temporal attention, patient-disjoint manifests and validation-only selection; do not reproduce leaked protocol.

因此第一批选择BSTNet是因为任务与机制相关、可从论文定义独立实现，不能称为官方源码一键忠实复现。README声明当前是simplified release。
