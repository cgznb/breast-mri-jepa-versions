# MIA 英文初稿 v5：seed44，五折分别计算指标

本版沿用从Elsevier官网下载并保留来源凭据的`elsarticle`官方源包，使用`5p,twocolumn,authoryear`双栏排版。主文件为`main.tex`，PDF为`main.pdf`，目录为`mia_seed44_foldmean_v5`，Overleaf源码包为`mia_seed44_foldmean_overleaf.zip`。页数和校验值以当前PDF与`notes/build_verification.json`为准。此前版本保留。

英文标题为 **Forecast-aware latent modeling for longitudinal prediction of pathological complete response from breast MRI**。本版全文、图表和显著性检查均按下述口径重写。

## 唯一评价口径

主比较为NoJEPA、DirectReadout、ForecastAttention；ForecastMean为补充池化对照。所有模型仅使用 **seed44**，每个模型有五个fold：

`每个fold独立对175位患者计算T0/T1/T2指标 → 在同fold内等权平均三个时点指标得到Macro → 等权平均五fold指标`。

主表报告 **五折指标均值 ± 样本标准差（ddof=1）**。不平均任何模型的预测概率，不把五个fold组成概率集成，也不引入其他训练种子。主三模型共15个拟合模型，补充ForecastMean共5个，总计20个模型。

所有模型、fold和timepoint重复使用同一内部留出集：**175人，56人pCR阳性**，主阈值 **0.5**。五个fold的训练患者重叠，测试患者完全相同；不能当作五个独立临床队列。fold SD描述五个拟合模型/分区的差异，不是训练种子标准差、患者标准误或95%患者区间。Macro的SD从每fold自己的Macro值计算，不能用各时点SD的均值替代。

## 当前结果

| 模型 | Macro AUROC | Macro AP | Macro F1 |
|---|---:|---:|---:|
| NoJEPA | 0.7219 ± 0.0113 | 0.5792 ± 0.0202 | 0.5437 ± 0.0383 |
| DirectReadout | 0.7303 ± 0.0124 | 0.5888 ± 0.0214 | 0.5539 ± 0.0233 |
| ForecastAttention | 0.7365 ± 0.0159 | 0.5990 ± 0.0151 | 0.5469 ± 0.0334 |
| ForecastMean（补充） | 0.7348 ± 0.0092 | 0.5845 ± 0.0145 | 0.5458 ± 0.0225 |

ForecastAttention平均Macro AUROC和AP最高，DirectReadout平均Macro F1最高。Attention的T0平均AUROC为0.7294，略低于Direct的0.7304及Mean的0.7317，不能称所有时点都最好。第五折Attention的Macro AUROC为0.7170，低于NoJEPA的0.7347和Direct的0.7477；这类相反结果保留在逐fold表中。

## 患者区间与显著性

2,000次有放回患者bootstrap计算95%百分位区间，每次所有模型、fold与timepoint共享同一患者抽样。每次先分别重算各fold指标，再平均fold指标。模型固定，不重抽fold、不重新训练；区间条件于当前五个拟合模型，不包含seed44选择、汇总口径或重新训练的不确定性。

全部14项指标均采用 **99,999次零中心配对患者bootstrap近似检验**，随机数种子20261103。双侧原始p为 `(1 + 满足 abs(bootstrap差值−观察差值)≥abs(观察差值) 的次数)/100000`，包含并列。加一约定避免Monte Carlo零值，不使近似检验成为有限样本精确检验。AP、F1、ECE等非光滑指标尤其需谨慎。没有做fold t检验，不将重复预测当作增加的独立患者。

主要比较组为三个主模型两两比较的Macro AUROC/AP/F1，共9项；次组81项包括逐时点主要指标27项、其他Macro指标33项、Attention与Mean池化比较12项、三主模型T2–T0变化9项。组内分别Holm校正，另提供全部90项共同Holm校正。分组校正不控制两组合并后的整体错误率。

表格报告差值、未作多重校正的95%区间、原始p、组内Holm p及显著标记：`*`为校正p<0.05，`**`为<0.01，`***`为<0.001，`ns`为≥0.05。患者百分位区间并非这些近似检验的严格反演，不能由是否跨零反推p。显著性结果见`data/significance_tests.csv`，方法及输入凭据见`data/significance_provenance.json`。

本版显著性结果：

- 主要9项Macro AUROC/AP/F1比较全部未经校正即未达到0.05，组内及全部90项Holm校正也均为`ns`。Attention对NoJEPA的Macro AUROC差为 **+0.0146**，患者95%区间 **[−0.0073, 0.0376]**，原始p=**0.19246**，主要组Holm p=**1.0000**。Attention对Direct的AUROC差+0.0062，原始p=0.54536，校正p同样为1。
- 次要81项中只有三项达到组内及全部90项校正标准：Attention对NoJEPA的Macro敏感性差 **+0.0940**，原始p=**0.00038**，次组Holm p=**0.03002**，全部90项p=**0.03344**；Attention对NoJEPA的特异性差 **−0.1109**，对Direct的特异性差 **−0.0969**，这两项原始p均为 **0.00001**，次组Holm p均为 **0.00081**，全部90项p均为 **0.00090**。敏感性获益应与明显的特异性代价一起报告。
- 两项特异性原始p是99,999次抽样中零次超出后的加一Monte Carlo分辨率下限，不是p=0，也不代表真p恰好等于0.00001。表格显示为`<0.0001`。
- Attention的T2−T0 AUROC变化+0.0127，未经多重校正区间[0.0026, 0.0235]，原始p=0.01892，但次组和全部90项Holm p均为1，不能称校正后显著的纵向提升。

本次seed44、指标汇总与统计政策仍是结果可见后的选择，不能称为预注册或独立验证性证据。未显著也不等于模型等效或没有效应。亚组、阈值敏感性、DCA与训练probe不新增显著性检验。

## 阈值敏感性与扩展指标

本版计算AUROC、AP、F1、accuracy、sensitivity、specificity、PPV、NPV、balanced accuracy、MCC、Youden J、Brier、NLL、十等宽分箱ECE，另保留TP/FP/FN/TN。平均混淆计数是五fold整数计数的均值，不是一个合并分类器的人数。

替代阈值政策为 **`fold_validation_youden`**：每个模型、fold和landmark分别使用本fold验证预测选Youden阈值，再应用于同一个fold模型的测试预测。并列时取最高阈值，不拼接其他fold验证概率，不使用测试标签选阈值。验证患者还参与了最佳checkpoint选择，所以这只是阈值敏感性分析，不是独立或nested校准。患者bootstrap保持所选阈值固定，不包含选阈值的波动。

八组图使用统一模型颜色、出版尺寸字体，提供矢量PDF与600dpi PNG：

1. `framework`：匿名开发病例MRI，以及连续编码、融合、自主预测、EMA目标和消融设计；评估按fold分别计算。
2. `discrimination`：每fold独立计算ROC/PR，再在共同网格上平均曲线。逐fold直接计算的平均AUROC/AP不必等于显示曲线面积。
3. `paired_comparisons`：五fold均值指标的配对差值及共享患者区间。
4. `calibration_utility`：每fold校准组摘要和DCA分别计算后平均，附treat-all/none参考。
5. `subgroup_operating`：亚组折均值判别差异及敏感性/特异性。
6. `forecast_mechanism`：seed44归档固定训练probe的copy-normalized skill与梯度路径。
7. `landmark_performance`：六项指标的fold均值；描述性fold SD与条件患者CI分别展示。
8. `outcome_score_changes`：先按fold计算每位患者T2−T0概率变化及其直方图，再平均直方图；重复fold预测不会把175人扩大为875人。

临床亚组保留分母，小样本或结局不足时抑制AUROC/AP；来源命名空间同属一次临床发布，不是外部队列。四个HR/HER2组的Attention对NoJEPA平均Macro AUROC差依次为+0.0324、−0.0415、+0.0082、+0.0217，对应未经多重校正的区间全部跨零；其中HR+/HER2+点估计转负，也完整保留。没有亚组交互检验。机制诊断在每fold snapshot自己的EMA坐标内计算，再作描述性折均值；没有潜在状态集成，不是最佳pCR checkpoint的独立留出forecast评价。NoJEPA无world diagnostic时记为不可用，不填零。epoch20三个horizon的Direct平均skill为0.0258/0.0500/0.1935，Attention为−0.0368/−0.0154/0.0242；三步目标人数仅5–9人，不能据此把判别差异归因于未来状态预测更准确。

## 编译与本地重建

在本目录运行：

```bash
bash build.sh
```

需要常见TeX Live组件，包括pdfLaTeX、BibTeX、lmodern、graphicx、booktabs和microtype。官方`elsarticle.cls`与`elsarticle-harv.bst`已包含。将ZIP上传Overleaf，主文件设为`main.tex`，编译器设为pdfLaTeX即可；预生成图表不需要Python或本地数据。

完整本地重建：

```bash
cd /personal_fast_storage/VLA-PCRV1/manuscripts/mia_seed44_foldmean_v5
bash regenerate.sh
```

生成脚本使用NumPy和Matplotlib，读取冻结评估报告和已验证manifest；不启动训练、不加载checkpoint、不改写MRI。聚合源表为`data/extended_summary.csv`，固定0.5主表为`data/seed44_foldmean_summary.csv`，逐fold结果为`data/seed44_individual_fold_metrics.csv`。来源凭据见`data/extended_provenance.json`及`data/subgroup_provenance.json`。

从聚合CSV无法独立重建患者bootstrap或亚组，这些步骤需要本地患者级评估报告。源码包不发布患者ID、逐患者预测、原MRI/NPZ或checkpoint；方法图仅包含匿名开发病例的渲染切片。

## 待补与研究边界

175人是内部downstream holdout，其中29位新增患者曾属于旧开发集。当前pCR模型按新fold重新训练，但codec与定位分割预训练是否包含这些患者尚未完整审计。

NoJEPA保留公共fusion和transition queries，只关闭未来辅助损失。三主模型teacher-forced单步JEPA权重均为0，实际未来监督为自主rollout L1。Attention相对Direct同时改变预测特征、池化和分类头宽度；Mean是更直接的池化对照。训练probe不能替代最佳checkpoint的独立未来状态预测验证。

T0 MRI名义上早于随机分配，使用最终assigned arm属于回顾性条件场景，不能据此主张分配前部署或个体因果疗效预测。MIA专属Guide此前返回HTTP403，官方模板可用，但匿名、字数和投稿文件细则仍需作者最终核对。

作者、伦理、pCR病理定义、预训练队列、实际扫描参数和公开发布承诺等未知事项保留红色`TO COMPLETE`，详见`notes/TODO_ZH.md`。本稿尚未提交。
