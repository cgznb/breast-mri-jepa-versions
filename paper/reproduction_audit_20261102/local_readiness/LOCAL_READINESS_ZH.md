# 本地复现准备审查：2026-11-02

> 当前主复现规则已改为同一患者划分、各原论文网络输入，见 `../NATIVE_INPUT_PROTOCOL_ZH.md`。本文为之前的本地接入与源码审查记录，统一ROI或改写prefix不再是主方案。

当前条件足以构建统一输入的 ClinicalOnly、3D ResNet、PrefixGRU 和 PrefixTransformer；可以对 BSTNet 等公开方法作受控适配。不能把当前 ROI32 直接视为 CoRe-WM 原文输入，也不能把已迁移的 Pillar 依赖文件当作完整训练仓库。

本次仅检查文件、源码、manifest 和实际 CPU 数据读取，未修改 MRI、模型、实验输出或稿件，未启动训练、下载权重。

## 冻结队列与实际接口

- 数据根：`/personal_fast_storage/VLA-PCRV1/data/registered_roi32_medium976_v1`。
- 固定五折 manifest：`cv_manifests/fold1/manifest.json` 至 `fold5/manifest.json`。
- 五折均为同一976人、3641次真实访视。训练/验证/测试人数分别为折1的640/161/175，折2–5的641/160/175。每折测试均为相同175人、56阳性；五折并非五个独立测试集。
- 生产实验配套运行时：`/projects/xts0030/public/cxx/planned_tasks/medium976_t0_t2_no_tf_cv5_multiseed_v1/runtime/src`。
- 类：`mri_vla_jepa.data.RawMRIStore`；`store.by_split` 提供患者索引；`store.allowed_landmarks(index)` 只返回真实可用访视。
- `store.batch([(patient_index, landmark)], supervised=True, future_supervision=False)` 返回输入和监督。输入包含 `images [B,4,3,32,128,128]`、`observed_mask [B,4]`、`landmark [B]`、`clinical [B,4]`、`clinical_mask`、`arm_id`、`arm_mask` 等；监督中 `label`/`label_mask` 承载 pCR。分类基线应设置 `future_supervision=False`，减少误读未来图像的机会。
- 真实 CPU 读取核对了同一测试患者在T0/T1/T2的三个合法输入；观察mask分别为1000、1100、1110，T3始终不作输入，全部数值有限。未导出病例身份或概率。
- NPZ数组：`image float32[3,32,128,128]`；`foreground/support bool[3,32,128,128]`；`roi bool[1,32,128,128]`；`crop_affine_ras float64[4,4]`；`phase_indices int16[3]`。
- 三期顺序为pre-contrast、first-post、metadata-late。磁盘是原始强度，只有配套loader在foreground内做每折训练集拟合的同一标量 `(raw-mean)/std`；背景保持零。旧 `versions/v1_evolution/src/mri_vla_jepa/data.py` 不支持 `preprocessed_roi32_raw`，不可错用或重复归一化。
- 临床字段为age、HR、HER2、MammaPrint；assigned arm另作编码。以训练患者拟合临床尺度；未来的实际剂量、周期变化和治疗调整并不存在于该输入合同。

完整只含聚合统计的凭据：`local_data_readiness.json`。

## 第一批可执行的模型与匹配原则

1. **ClinicalOnly logistic/MLP**：age/HR/HER2/MammaPrint，加相同 assigned-arm 编码；用于判断生物学/治疗先验贡献。图像不输入，T0–T2预测应保持一致，可在图表展示为参考而不制造三个独立模型结果。
2. **同临床变量的3D ResNet**：共享当前三期ROI32，T0为单时点基线；如果扩展到T1/T2，应明确“latest visit”或“history pooling”，不能把只读T2称为纵向序列。
3. **同编码器+临床的PrefixGRU**：逐时点编码再GRU聚合，T0/T0–T1/T0–T2。支持不完整历史和真实missing mask，不读取T3。
4. **同编码器+临床的PrefixTransformer**：序列位置编码与key-padding/causal输入约束相匹配；同样支持三个前缀。只有上述基础模型中的机制差异足够可控，不能把“更多临床变量”误归因为时序模型增益。
5. **BSTNet 2026 adapted**：保留3D encoder、时序attention、LSTM、时序排序预训练思想，按下述源码审查修正明显问题，并在最终论文准确标注适配。

各模型固定seed44、相同五折开发划分与175人留出集；五折概率逐患者平均后再计算14指标，Macro等权平均三个时点指标。模型选择和超参均只能使用训练/验证患者。为减少输入差异，主比较应给所有方法同一临床变量；论文原生image-only模型另列，或增加清楚命名的clinical late-fusion变体。

现有训练是每epoch每位患者随机选一个合法T0/T1/T2前缀；验证使用患者等权前缀指标选checkpoint。新共享前缀模型若沿用它，需要确认每位患者曝光和输入可用性一致。按时点独立训练的原生模型则应标明独立模型与额外训练预算。不能把训练次数相同等同于参数量或预训练资源相同。

## BSTNet：有训练代码，但不宜原样运行

官方仓库：<https://github.com/XmySz/BreastPCRClassification_Github>，HEAD `723a09ef114240b2f78638b893d89dd2227b6b4c`。

实核存在 `DeepLearning/Experiment/models.py`、`datasets.py`、`train.py`、`train_with_fine_tuning.py`，以及 `DeepLearning/Preprocess/生成微调数据集映射表.py`、`构建微调数据集.py`。后两者生成正序/置换序列标签，因此并非只有网络定义或推断演示。README同时明确当前是simplified version，official version尚待发布。

需要明确修正/记录的部分：

- `models.py` 在每访视池化后把二维 `[B,F]` 直接传给 `MultiheadAttention(batch_first=True)`。PyTorch把二维输入视为无batch序列，实际会沿患者维度进行注意力，而不是沿时点。若按论文时序attention意图适配，应先组合为 `[B,T,F]`，再沿T计算并处理缺失mask。`bst_attention_shape_check.json` 以独立小型MHA演示了只改变其他行也会改变首行输出；没有训练完整模型。
- forward硬编码 `x.chunk(4,dim=1)`，缺少合法前缀/缺失序列处理；必须防止T3输入泄漏。
- 原生正式数据构建给 `[4,H,W,D]`（四访视单通道），而finetune脚本构造 `ResNet18Encoder(in_channels=4)`，按forward拆成四段后每段实际单通道，配置不一致；需修正成与实际输入合同一致的通道数。
- README描述256维、8-head，代码默认512维、4-head。应明确采用论文描述还是公开实现，不可称其完全相同。
- `train.py` 每epoch评外部test，且保存best-test-AUC和best-test-accuracy checkpoint；应删除基于test的checkpoint选择，将最终test评估放在训练完成后。
- validation/test DataLoader设置 `drop_last=True`，会丢患者；新benchmark必须为False并核验175人完整覆盖。
- `datasets.py` 在文件列表上自行进行80/20随机划分，不使用当前患者manifest；时序置换造数据后再按样本随机分割也可能使同一患者置换样本跨训练/验证。应先按患者划分，再在该fold训练患者内造自监督任务。
- 原生归一化为per-volume zscore再minmax，与当前fold-train尺度不同；同输入主表应使用统一loader。若另报告native preprocessing，必须独立命名并解释。
- README声称MIT并指向LICENSE文件，但根目录未观察到该许可正文。本次只保存小文件源码审查凭据，不把尚未核验完整许可正文的源码整包打进公开投稿包。

以上是当前可见源码与本次目标协议的差异，不等同于对论文所用内部训练代码或原结果作判断。最终名称建议 `BSTNet (adapted to matched prefixes)`，同时提供原生网络/预训练思想及字段差异表。

源码节选和URL/SHA256均在本目录，统一索引为 `network_receipts.json`。

## Seq2Seq：原版合成实现可用，pCR端到端脚本未见

官方仓库 <https://github.com/fiy2W/mri_seq2seq>，HEAD `0414c0766678719b0f5c50bfb1ad67e71b218770`，对应 MIA2024 DOI `10.1016/j.media.2023.103044`。

论文原版目录是 `publications/src/seq2seq/`；根目录 `nnseq2seq/` 是更新的方法实现，不宜替代原版而仍称2024原文复现。已核验原版README、BraTS 2D dataloader/train、PSNR/SSIM/LPIPS测试、序列贡献和imaging-differentiation map生成脚本。公开配置/权重指向BraTS。

在该目录与完整当前文件树中，未找到breast/pCR分类训练入口、对应pCR划分和pCR分类权重；README明确自定义数据需适配loader和train文件。因此能复现它的序列生成/差异表征机制，再独立适配pCR读出，不能称已有一条命令就完整复现其乳腺pCR表格。它是跨MR序列合成，不是沿NAC访视预测未来患者状态；适合作为次级生成表征基线，优先级低于直接纵向pCR工作。

## 其他代码核验

- **Pillar-0+TDN 2026**：官方 <https://github.com/omarftt/longitudinal_temporal_pillar>，HEAD `89e9279d474ddb650be0e76c3973ea4b8c2d4657`。官方完整树有 `src/tdn.py`、`scripts/run_experiments.py`、ViT2D与Clinical baseline。当前本地 `preprocessing/upstream/longitudinal_temporal_pillar` 仅10个预处理依赖文件；SOURCE_MANIFEST的旧copy pin为 `da390dc5e9e19beb0c6546cb5e5b57242578a98c`，不能把这份局部拷贝当可运行完整训练包。Pillar权重gated，官方输入还有17维临床/治疗与真实elapsed days，当前合同没有days字段。冻结foundation encoder与预训练患者来源需另审；可作第二批。
- **BreastDCEDL ViT**：<https://github.com/naomifridman/BreastDCEDL>，HEAD `961c458a393b3173ce923e7bc78409790153a15e`。存在pCR预测notebook与checkpoint加载演示；所检查的 `BrestDCEDL_vit_predict.ipynb` 主要载入ViT-MAE及pCR权重，未见完整训练loop。不能把已训练权重直接评当前175当公平独立泛化，训练队列可能有I-SPY2重叠；应重新训练或明确审计预训练来源。
- **GNN-pCR 2026**：<https://github.com/hannesk95/GNN_pCR_ISPY2>。当前官方README明确“Full source code will be released upon publiction”，仅数据下载接口可见；不能列为已具可训练官方代码的第一批。
- 参考CoRe-WM PDF正文中未观察到公开代码链接；不能仅因其对照方法声称“released architectures”而推定其所有复现脚本可用。

## CoRe-WM严格复现仍缺的合同

参考论文使用808位完整四访视I-SPY2、八通道32×96×96、真实访视ROI-derived九维几何、同折约525/121/162训练/验证/测试；另156位I-SPY1作跨队列。当前976为不完整历史可训练队列，三通道32×128×128，固定175人内部holdout。两者统计口径和病例集合均不同，原表AUROC不能与咱们0.7507直接比较。

当前每时点NPZ的 `roi` 都是同一T0预测mask，不能拿它计算真实随访肿瘤变化、动态FTV或九维q，否则模型只会看到不变的基线几何。当前冻结manifest也没有对应q/FTV字段；是否能从其他配套历史mask/原始资源恢复，需要单独几何与标签来源审核。原始来源可能存在，并不等于本报告已证实全项目完全没有这些资源。

若开展八通道/随访mask/跨队列原文复现，必须先检查与旧MAMA-MIA mask及变换配套的版本；不能把新public JSON与旧mask header直接组合。现阶段统一ROI32的适配复现无需改动这些上游文件。

## 范围结论

第一批建议先准备ClinicalOnly、相同clinical/arm的ResNet与GRU，再加入经明确适配的BSTNet及TemporalRL。所有新实验都须重新训练并回到相同175患者、同一seed44五折概率集成评价，不能用对方论文的现成AUC填入主表。没有官方完整训练脚本、checkpoint provenance或必要输入的工作放在后续队列，并准确标为重实现或适配。
