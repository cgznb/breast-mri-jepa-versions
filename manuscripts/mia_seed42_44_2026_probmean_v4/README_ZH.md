# MIA 英文初稿 v4：seed42、44、2026

本版沿用从 Elsevier 官网下载、保留来源凭据的 `elsarticle` 官方源包，使用 `5p,twocolumn,authoryear` 双栏排版。主文件为 `main.tex`，PDF 为 `main.pdf`，目录为 `mia_seed42_44_2026_probmean_v4`，Overleaf 源码包为 `mia_seed42_44_2026_probmean_overleaf.zip`。页数和校验值以当前 PDF 与 `notes/build_verification.json` 为准。此前版本保留，不覆盖。

英文标题为 **Forecast-aware latent modeling for longitudinal prediction of pathological complete response from breast MRI**。已重写摘要、实验与统计方法、结果、讨论、结论和补充说明；所有结果依照下面的三种子口径重算，不能与旧版单seed44数字混用。

## 唯一汇总口径

主比较为 NoJEPA、DirectReadout、ForecastAttention；ForecastMean 为补充池化对照。所有模型均使用种子 **42、44、2026**，每个种子五折：

`同一种子内五折概率逐患者平均 → 在175位患者上计算各时点指标 → T0/T1/T2指标等权平均得到该种子Macro → 三种子指标等权平均`。

主表报告 **三种子指标均值 ± 样本标准差（ddof=1）**。这里的标准差只描述三次已选训练实现的变化，不是患者标准误，也不是患者95%置信区间。没有先平均五折指标，没有跨种子平均预测概率，没有将十五个模型组成一个新的概率集成。主三模型共45个拟合模型，补充ForecastMean共15个，总计60个种子×折模型。

所有模型、种子、时点重复使用同一内部留出集：**175人，56人pCR阳性**。fold、seed、timepoint均不会增加独立患者数。主阈值为 **0.5**。

| 模型 | Macro AUROC | Macro AP | Macro F1 |
|---|---:|---:|---:|
| NoJEPA | 0.7243 ± 0.0040 | 0.5947 ± 0.0027 | 0.5425 ± 0.0132 |
| DirectReadout | 0.7353 ± 0.0107 | 0.6022 ± 0.0052 | 0.5471 ± 0.0206 |
| ForecastAttention | 0.7454 ± 0.0047 | 0.6169 ± 0.0102 | 0.5556 ± 0.0143 |
| ForecastMean（补充） | 0.7454 ± 0.0043 | 0.6132 ± 0.0048 | 0.5723 ± 0.0048 |

ForecastAttention在主三模型中平均AUROC、AP、Macro F1最高；与ForecastMean的平均Macro AUROC精确相等，ForecastMean的平均Macro F1更高。不能称Attention在全部模型或全部指标全面最好。seed2026中DirectReadout的Macro AUROC低于NoJEPA，seed42中Attention略低于DirectReadout；这些相反结果在单种子补充表保留。

## 患者区间与显著性

2,000次有放回患者bootstrap计算95%百分位区间，每次所有模型、三个种子与三个时点共享同一患者抽样。重采样内先分别计算各seed的指标，再对seed指标平均。这里不重抽种子、不重新训练，区间条件于当前固定的三个seed模型；不能用它量化完整训练算法或事后选种子的波动。

全部14项指标均使用 **99,999次零中心配对患者bootstrap近似检验**，随机数种子20261103。每次同样先按seed计算指标、再平均，双侧原始p为 `(1 + 满足 abs(bootstrap差值−观察差值)≥abs(观察差值) 的次数)/100000`，包含相等情形。加一用于避免Monte Carlo零值，不使近似检验成为有限样本精确检验。AP、F1、ECE等非光滑指标尤其需谨慎。没有做五折或三种子的t检验，也没有将三次训练当作三个独立临床队列。

主要比较组为三个主模型两两比较的Macro AUROC/AP/F1，共9项；次组81项包括逐时点主要指标27项、其他Macro指标33项、Attention与Mean池化比较12项、三主模型T2–T0变化9项。组内分别Holm校正，另导出全部90项共同Holm校正作为敏感性检查。分组校正不控制两组合并后的整体错误率。

表格同时提供差值、未经多重校正的95%区间、原始p、组内Holm p及显著标记：`*`为校正p<0.05，`**`为<0.01，`***`为<0.001，`ns`为≥0.05。区间和检验并非严格反演，不能由是否跨零反推p。

本版的重要结果与边界：

- Attention相对NoJEPA的Macro AUROC差 **+0.0211**，95%患者区间 **[0.0090, 0.0345]**，原始p=**0.00224**，主要9项Holm p=**0.02016**。它在主组校正下达到标准，但全部90项校正p=**0.19488**，未达到标准。
- Attention相对DirectReadout的Macro特异性差 **−0.0411**，区间 **[−0.0649, −0.0202]**，原始p=**0.00048**，次组81项Holm p=**0.03888**，全部90项Holm p=**0.04320**。不利的操作点差异应与有利的判别结果同时报告。
- 主组只有上述AUROC比较达到组内校正标准，次组只有上述特异性比较达到组内标准。未显著不等于模型等效或没有效应。

本次种子组合、聚合与统计政策仍是结果可见后的事后选择，不能称为预注册或独立验证性证据。完整结果见 `data/significance_tests.csv`，方法与输入凭据见 `data/significance_provenance.json`。正文主比较在 `tables/bootstrap.tex`；其他统计表分别为逐时点、扩展Macro、池化和纵向比较。OOF阈值、亚组、DCA和训练probe不新增显著性检验。

对接近0.05的特异性比较另做一次固定999,999次重采样的Monte Carlo稳定性检查，独立随机数种子20261104。原始p为0.000434；只替换该项后，次组Holm p=0.035154、全部90项Holm p=0.03906，结论不变。主表保留统一99,999次结果，额外检查记录在 `data/monte_carlo_stability.json`，可运行 `scripts/check_monte_carlo_stability.py` 复算。

## 扩展分析与图表

本版计算AUROC、AP、F1、accuracy、sensitivity、specificity、PPV、NPV、balanced accuracy、MCC、Youden J、Brier、NLL、十等宽分箱ECE，另保留TP/FP/FN/TN。平均混淆计数是多个seed整数计数的算术均值，不是一个合并分类器的人数。

替代阈值 `oof_validation_youden` 在每个seed内拼接五折互不重叠验证预测，以每个开发患者仅有的一次验证折预测选择Youden阈值，然后应用于该seed的测试五模型概率均值。验证患者还参与了最佳epoch选择；OOF为单模型分数，测试为五模型均值，分布可能不同。因此它是阈值敏感性分析，不是独立、嵌套或与集成匹配的校准。

八组图使用统一模型颜色、白底、面板标号、出版尺寸字体，提供矢量PDF与600dpi PNG：

1. `framework`：匿名开发病例三期×三时点MRI，以及连续编码、融合、自主预测、EMA目标和消融设计。
2. `discrimination`：先按seed计算再平均的ROC/PR曲线及患者重采样区间；显示曲线面积不必等于逐seed直接计算的平均AUROC/AP。
3. `paired_comparisons`：三seed均值指标的配对差值及共享患者区间。
4. `calibration_utility`：各seed校准组摘要和DCA分别计算后平均，附treat-all/none参考。
5. `subgroup_operating`：亚组平均判别差异及敏感性/特异性。
6. `forecast_mechanism`：归档固定训练probe的copy-normalized skill与梯度路径。
7. `landmark_performance`：六项指标的三seed均值随时点变化。
8. `outcome_score_changes`：各seed先计算患者T2−T0概率变化分布，再平均分布，不能把525条重复预测当作525位患者。

临床亚组保留分母，小样本或单类结局不足时抑制AUROC/AP；来源命名空间同属一次临床发布，不是外部队列。HR+/HER2−的Attention对NoJEPA平均Macro AUROC差为+0.0333，未经多重校正区间[0.0072, 0.0653]，其他三个HR/HER2组区间跨零；没有交互检验，不能声称确定的亚型获益。该组Attention敏感性仍仅0.1414。机制诊断在各自snapshot的EMA坐标内计算、每seed先平均五折再汇总三个seed；不集成潜在状态，不是最佳pCR检查点的独立留出forecast评价。NoJEPA没有对应world diagnostic时记为不可用，不填零。epoch20三个horizon的平均copy-normalized skill为Direct 0.0068/0.0551/0.1724，Attention −0.0576/−0.0584/0.0089；三步目标人数只有5–9人，不能据此将pCR判别提高归因于未来状态预测更准确。

## 编译与 Overleaf

在本目录运行：

```bash
bash build.sh
```

需要常见TeX Live组件，包括pdfLaTeX、BibTeX、lmodern、graphicx、booktabs和microtype。官方`elsarticle.cls`与`elsarticle-harv.bst`已包含。将ZIP上传Overleaf，主文件设为`main.tex`，编译器设为pdfLaTeX即可；预生成图表不需要Python或本地数据。

完整本地重建：

```bash
cd /personal_fast_storage/VLA-PCRV1/manuscripts/mia_seed42_44_2026_probmean_v4
bash regenerate.sh
```

重建顺序为分析、显著性、亚组、表格、完整指标区间、方法图、扩展图、机制图，最后编译。生成脚本使用NumPy和Matplotlib，读取冻结评估报告和已验证manifest；不训练、不加载检查点、不改写MRI。可选独立核验使用已有训练环境中的scikit-learn，不是论文生成依赖。

聚合源表为 `data/extended_summary.csv`，固定0.5主表导出为 `data/three_seed_ensemble_summary.csv`；逐seed指标为 `data/three_seed_individual_metrics.csv`。结果与报告身份见 `data/extended_provenance.json` 和 `data/subgroup_provenance.json`。源包保留可复算脚本，但从聚合CSV无法独立重建患者bootstrap或临床亚组；这些步骤需要本地患者级评估报告。源包不发布患者ID、逐患者预测、原MRI/NPZ或检查点；方法图仅包含匿名开发病例的渲染切片。

## 待补与研究边界

175人是内部downstream holdout，其中29位新增患者曾属于旧开发集。当前pCR模型按新fold重新训练，但codec与定位分割预训练是否包含这些患者尚未完整审计。

NoJEPA保留公共fusion和transition queries，只关闭未来辅助损失。三主模型teacher-forced单步JEPA权重均为0，实际未来监督为自主rollout L1。Attention相对Direct同时改变预测特征、池化和分类头宽度；Mean是更直接的池化对照。训练probe不能替代最佳检查点的独立未来状态预测验证。

T0 MRI名义上早于随机分配；使用最终assigned arm属于回顾性条件场景，不能据此主张分配前部署或个体因果疗效预测。MIA专属Guide此前返回HTTP403，官方模板可用，但匿名、字数和投稿文件细则仍需作者最终核对。

作者、伦理、pCR病理定义、预训练队列、实际扫描参数和公开发布承诺等未知事项保留红色 `TO COMPLETE`，详见 `notes/TODO_ZH.md`。本稿尚未提交。
