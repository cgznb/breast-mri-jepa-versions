# MIA 英文初稿 v7：seed44、2026、3407

本版从v6排除seed42，仅使用seed44、2026、3407，并重写全文、图表、显著性检查和分析；保留所有历史版本。排除是在holdout结果可见后的回顾性报告政策变更，不能称预注册排除。采用已从Elsevier官网下载并保存来源凭据的`elsarticle`官方源包，使用`5p,twocolumn,authoryear`双栏排版。主文件为`main.tex`，PDF为`main.pdf`，Overleaf源码包为`mia_seed44_2026_3407_probmean_overleaf.zip`。页数、独立编译与校验值见`notes/build_verification.json`。

英文标题为 **Forecast-aware latent modeling for longitudinal prediction of pathological complete response from breast MRI**。

## 唯一汇总口径

主比较为NoJEPA、DirectReadout、ForecastAttention；ForecastMean为补充池化对照。所有模型均使用 **seed44、2026、3407**，每个seed五折：

`同一seed内五折概率逐患者平均 → 在175人上计算各时点指标 → 等权平均T0/T1/T2指标得到该seed的Macro → 三seed指标等权平均`。

主表报告 **三seed指标均值 ± 样本标准差（ddof=1）**。Macro的SD从三个seed各自的Macro值计算，不能平均三个时点的SD。seed SD只描述这三次已选训练实现的变化，不是患者标准误或患者95%区间。不先平均五折指标，不跨seed平均预测概率，也不把15个模型合成一个概率集成。主三模型共45个拟合模型，ForecastMean共15个，总计60个seed×fold模型，核验60份完成凭据及180份登记报告。

所有模型、seed与时点重复使用同一内部留出集：**175人，56人pCR阳性**，主阈值 **0.5**；重复预测不增加独立患者数。队列划分的固定seed2026与训练seed是不同概念，三seed训练共用同一个划分。

| 模型 | Macro AUROC | Macro AP | Macro F1 |
|---|---:|---:|---:|
| NoJEPA | 0.7293 ± 0.0050 | 0.5914 ± 0.0056 | 0.5546 ± 0.0123 |
| DirectReadout | 0.7339 ± 0.0094 | 0.6046 ± 0.0028 | 0.5533 ± 0.0231 |
| ForecastAttention | 0.7458 ± 0.0042 | 0.6188 ± 0.0097 | 0.5663 ± 0.0056 |
| ForecastMean（补充） | 0.7446 ± 0.0057 | 0.6096 ± 0.0038 | 0.5667 ± 0.0077 |

ForecastAttention在四模型中平均Macro AUROC/AP最高，在主三模型中平均Macro F1最高。ForecastMean的平均Macro F1仅高0.00032；不能将该差值写成明显优势，也不能称两者等效。Attention对Mean的Macro AUROC差+0.0013，未经多重校正患者区间[−0.0091, 0.0126]；AP及F1的区间也跨零，未建立确定的注意力优势。

逐seed相反结果仍保留：seed2026的Direct Macro AUROC低于NoJEPA（0.7230对0.7251），seed44的Attention F1低于Direct（0.5604对0.5678）。seed3407的NoJEPA/Direct/Attention/Mean Macro AUROC分别为0.7348/0.7386/0.7432/0.7381，但Direct的F1略低于NoJEPA（0.5654对0.5661）。三seed整体Direct的Macro F1也比NoJEPA低0.00136，不能把辅助监督写成三项指标均改善。逐seed表拆为direct和forecast两张补充表，各24行。

## 患者区间与显著性

2,000次有放回患者bootstrap计算未经多重校正的95%百分位区间。每次所有模型、seed与时点共享同一患者抽样；先分别重算各seed指标，再平均seed指标。模型固定，不重抽seed、不重新训练。区间条件于当前三个seed模型，不包括训练算法、seed选择、汇总方式、模型选择或阈值选择的不确定性。

全部14项指标均采用 **99,999次零中心配对患者bootstrap近似检验**，随机数种子20261103。原始双侧p为`(1 + 满足 abs(bootstrap差值−观察差值)≥abs(观察差值) 的次数)/100000`，包含并列；加一避免Monte Carlo零值，但不使该近似检验成为有限样本精确检验。AP、F1和分箱ECE等非光滑指标尤其需谨慎。不做seed或fold t检验，不把三次训练当作三个独立临床队列。

主要比较组为主三模型两两比较的Macro AUROC/AP/F1，共9项；次组81项包括逐时点主要指标27项、其他Macro指标33项、Attention对Mean的池化比较12项、主三模型T2−T0变化9项。分别作组内Holm校正，另报告全部90项共同Holm校正。分组校正不控制两组合并后的整体错误率。

表格提供差值、未经多重校正的95%区间、原始p、Holm p和标记：`*`为校正p<0.05，`**`为<0.01，`***`为<0.001，`ns`为≥0.05。区间与近似检验不是严格反演，不能仅由区间是否跨零推断p。

- Attention对NoJEPA的Macro AUROC差 **+0.0166**，患者区间 **[0.0035, 0.0304]**，原始p=**0.02107**，主要9项Holm p=**0.18963（ns）**，全部90项Holm p=1.0000。**主要9项检验全部经校正后未达到0.05**，不能沿用v6“AUROC组内显著”的结论。
- 同一对比的Macro AP差+0.0275，区间[0.0044, 0.0528]，原始p=0.02811，主组Holm p=0.22488，全部90项p=1.0000；F1也未达到校正标准。未校正区间不跨零不能代替多重校正后的判断。
- Attention对Direct的Macro特异性差 **−0.0430**，区间 **[−0.0667, −0.0224]**，原始p=**0.00032**，次81项Holm p=**0.02592（*）**，全部90项Holm p=**0.02880（*）**。它是次组及全部90项中唯一显著的比较，方向为不利的特异性下降。对应敏感性差+0.0556，但次组及全部90项Holm p均为1.0000。

沿用既有的Attention对Direct Macro特异性对照，另作999,999次独立Monte Carlo稳定性敏感检查，RNG为20261104。新原始p=0.000345，模拟标准误约0.0000186；只替换该项时次组及全部90项Holm p为0.027945/0.031050，结论保持。主表仍统一采用99,999次结果，不能用追加结果覆盖主表标记。完整凭据为`data/monte_carlo_stability.json`。

种子集合、聚合与统计政策均为结果可见后的事后选择；显著性校正不能将其变成预注册或独立验证性证据。未显著也不等于模型等效。完整检验及方法凭据见`data/significance_tests.csv`与`data/significance_provenance.json`；亚组、阈值、DCA和训练probe不新增显著性检验。

## 扩展分析与图表

本版计算AUROC、AP、F1、accuracy、sensitivity、specificity、PPV、NPV、balanced accuracy、MCC、Youden J、Brier、NLL及十等宽分箱ECE，另保留TP/FP/FN/TN。平均混淆计数来自三seed各自的整数计数，不是一个合并分类器的人数。

Attention的Macro敏感性/特异性为0.6925/0.6452，Direct为0.6369/0.6881。全部主模型Macro accuracy低于全阴性参考0.68。Attention的NLL/Brier/ECE为0.5959/0.2057/0.1579，三项点估计均差于两个direct对照及补充Mean（0.5882/0.2027/0.1379）。排序获益不能等同于概率质量或操作点全面改善。

`oof_validation_youden`在每seed内拼接五折互不重叠的验证预测，每个开发患者只提供其验证fold模型的一次预测；按landmark最大化Youden指标，并列取最高阈值，再应用于该seed测试五模型概率均值。验证患者同时参与最佳epoch选择，且OOF单模型分数与测试五模型均值的分布不同，因此只是阈值敏感性分析，不是独立或嵌套校准，不用测试标签选阈值。Macro F1固定阈值→替代阈值分别为NoJEPA 0.5546→0.5574、Direct 0.5533→0.5485、Attention 0.5663→0.5653、Mean 0.5667→0.5599。

Attention的T2−T0 AUROC差+0.0022，区间[−0.0037, 0.0089]；AP差−0.0028、F1差仅+0.00003，其区间也均跨零。不能把后续访视写成确定的性能提升。DCA使用未校准概率与假设性正向操作；模型排序随阈值改变，不验证具体临床治疗决策。

八组图使用统一颜色、出版尺寸字体，提供矢量PDF和600dpi PNG：`framework`、`discrimination`、`paired_comparisons`、`calibration_utility`、`subgroup_operating`、`forecast_mechanism`、`landmark_performance`、`outcome_score_changes`。曲线、校准组、DCA及概率变化分布都先分别按seed计算再汇总；不跨seed平均概率。landmark图分别展示描述性seed SD和条件患者区间。

四个HR/HER2组的Attention对NoJEPA Macro AUROC差分别为+0.0138、+0.0034、+0.0567、+0.0296，**未经多重校正区间全部跨零**。HR+/HER2−组Attention敏感性仍仅0.1818。没有亚型交互检验；来源命名空间同属一次临床发布，不是外部验证。临床亚组保留分母，样本或结局不足时抑制AUROC/AP。留出集11例crop部分截断者仅1个阳性，不能据此评价可靠的亚组判别能力。

机制诊断在每个snapshot自己的EMA坐标内计算，先平均每seed内五fold再汇总三seed；不集成潜在状态，不是最佳pCR检查点的独立留出forecast验证。epoch20三个horizon的copy-normalized skill为Direct 0.0081/0.0454/0.1312，Attention −0.0462/−0.0474/0.0107；三步目标人数只有5–9人。该诊断不支持将pCR排序差异归因于未来状态预测更准确。缺失的NoJEPA world diagnostic不填零。

## 编译和本地重建

在本目录运行：

```bash
bash build.sh
```

将ZIP上传Overleaf，主文件设为`main.tex`，编译器设为pdfLaTeX；预生成图表不需要Python或本地患者数据。官方`elsarticle.cls`和`elsarticle-harv.bst`已包含。

完整本地重建：

```bash
cd /personal_fast_storage/VLA-PCRV1/manuscripts/mia_seed44_2026_3407_probmean_v7
bash regenerate.sh
```

脚本读取冻结评估报告和核验过的manifest，不启动训练、不加载checkpoint、不改写MRI。聚合源表为`data/extended_summary.csv`，固定0.5主表为`data/three_seed_ensemble_summary.csv`，逐seed指标为`data/three_seed_individual_metrics.csv`，逐seed DCA为`data/three_seed_dca.csv`。来源与一致性见`data/extended_provenance.json`、`data/subgroup_provenance.json`及`notes/source_audit.json`。

源码包保留可复算脚本，完整数值重算需要本地患者级评估报告和manifest；仅凭聚合CSV不能重建患者bootstrap或亚组。包内不发布患者ID、逐患者预测、原MRI/NPZ或checkpoint；方法图只包含匿名开发病例的渲染切片。

## 待补与研究边界

175人是内部downstream holdout，其中29位新增患者曾属于旧开发集。当前pCR模型按新fold重新训练，但codec与定位分割模型预训练是否包含这些患者尚未完整审计。

NoJEPA保留公共fusion及transition queries，只关闭未来辅助损失。主模型teacher-forced单步JEPA权重均为0，实际未来监督为自主rollout L1。Attention对Direct同时改变预测特征、池化与分类头宽度；Mean是更直接的池化对照。训练probe不能替代独立未来状态验证。

T0 MRI名义上早于随机分配，使用最终assigned arm属于回顾性条件场景，不能据此主张分配前部署或个体因果疗效预测。MIA专属Guide此前返回HTTP403；官方模板可用，字数、匿名及投稿文件细则仍需作者核对。作者、伦理、pCR病理定义、预训练队列、实际扫描参数与公开发布承诺等未知事项保留红色`TO COMPLETE`，详见`notes/TODO_ZH.md`。本稿尚未提交。
