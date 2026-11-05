# MIA 英文初稿 v8：ForecastMean，seed42、44、71、2026

本版使用ForecastMean作为总体模型，与NoJEPA、DirectReadout共同构成三个主模型。每个模型均使用seed42、44、71、2026，所有数字、图表和显著性按新口径重算。总体模型及seed集合是在既有holdout结果可见后的回顾性变更；不能称为预注册分析。历史版本保留。

沿用已从Elsevier官网下载并保留凭据的`elsarticle`官方源包，采用`5p,twocolumn,authoryear`双栏排版。主文件为`main.tex`，PDF为`main.pdf`，Overleaf包为`mia_mean_seed42_44_71_2026_probmean_overleaf.zip`。最终页数、独立编译及校验值见`notes/build_verification.json`。

## 模型与唯一汇总口径

ForecastMean对32个变换后的空间token取算术均值，分别汇总预测T3终末状态及其相对当前EMA状态的变化，再与当前融合特征拼接。三个向量各768维，分类头输入为2304维；direct分类头为768维。Mean指分类readout的空间汇总方式，公共空间、融合及预测Transformer仍保留self-attention。ForecastMean对Direct的比较同时改变预测特征输入与分类头宽度，不能归因于单一机制。

三个主模型共 **3模型×4seed×5fold=60个拟合模型**，核验60份完成凭据及180份时点报告；没有额外补充模型。全部模型、seed与时点重复使用同一内部留出集：**175人，56人pCR阳性**，主阈值 **0.5**。

`每seed内逐患者平均五fold概率 → 分别计算T0/T1/T2指标 → 等权平均三个时点指标得到该seed的Macro → 等权平均四seed指标`。

主表为 **四seed指标均值 ± 样本标准差（ddof=1）**。Macro SD从四个seed各自的Macro值计算，不能平均时点SD。这里的SD描述所选训练实现的变化，不是患者标准误或95%区间。不先平均五fold指标，不跨seed平均概率，不把20个模型合成一个新的概率集成。fold、seed和时点不会增加独立患者数。固定队列划分seed2026与训练seed是不同概念，四seed共用相同划分。

| 模型 | Macro AUROC | Macro AP | Macro F1 |
|---|---:|---:|---:|
| NoJEPA | 0.7258 ± 0.0044 | 0.5905 ± 0.0086 | 0.5382 ± 0.0138 |
| DirectReadout | 0.7354 ± 0.0087 | 0.5972 ± 0.0107 | 0.5446 ± 0.0176 |
| ForecastMean | 0.7483 ± 0.0067 | 0.6197 ± 0.0137 | 0.5760 ± 0.0083 |

ForecastMean在三个主模型中平均AUROC、AP、F1最高，但不利seed结果仍保留：seed42的Mean Macro AUROC低于Direct（0.7407对0.7428），seed2026的Direct低于NoJEPA（0.7230对0.7251）。seed71的Mean Macro AUROC/AP/F1为0.7569/0.6394/0.5869，Direct为0.7358/0.5825/0.5369。补充逐seed表分为direct两模型32行和Mean16行；不只展示有利训练实现。

## 患者区间与显著性

2,000次共享患者bootstrap计算未经多重校正的95%百分位区间。每次同一175人有放回抽样共享于所有模型、seed与时点，先分别重算各seed指标，再平均seed指标。固定已训练模型，不重新训练或重抽seed；区间不包括总体模型、seed、checkpoint、阈值及汇总政策选择的不确定性。

全部14项指标均采用 **99,999次零中心配对患者bootstrap近似检验**，RNG为20261103。双侧原始p为`(1 + 满足 abs(bootstrap差值−观察差值)≥abs(观察差值) 的次数)/100000`，包含并列。加一避免Monte Carlo零值，但不使近似bootstrap成为有限样本精确检验。AP、F1与分箱ECE等非光滑指标尤其需谨慎。不做fold或seed t检验，不把重复预测当独立患者。

主要组9项为三个模型两两比较的Macro AUROC/AP/F1；次组69项包括逐时点主要指标27项、其他Macro指标33项和T2−T0变化9项，共 **78项**。分别作Holm9/Holm69校正，另提供全部78项共同Holm校正。分组校正不控制两组合并后的整体错误率。

表格给出差值、未经多重校正的95%区间、原始p、Holm p及标记：`*`为校正p<0.05，`**`为<0.01，`***`为<0.001，`ns`为≥0.05。百分位区间与近似检验不是严格反演，不能只由是否跨零判断校正p。

- Mean对NoJEPA的Macro AUROC差 **+0.0225**，患者区间 **[0.0106, 0.0347]**，原始p=**0.00050**，Holm9 p=**0.00450**（`**`），Holm78 p=**0.03850**（`*`）。这是主9项中唯一达到组内标准的比较。
- 同一比较的Macro AP差+0.0292，区间[0.0078, 0.0508]，原始p=0.01020，Holm9 p=0.08160；F1差+0.0377，区间[0.0034, 0.0782]，原始p=0.04513，Holm9 p=0.22565。两者都未达到校正标准。
- Mean对Direct的Macro AUROC/AP/F1组内校正p为0.13699/0.13699/0.22565，均为ns。不能把所有主要指标写成显著改善。
- 次69项中仅Mean对NoJEPA的T1和T2 AUROC达到组内标准。T1差+0.0243，原始p=0.00036，Holm69/78 p=0.02484/0.02808；T2差+0.0224，原始p=0.00058，Holm69/78 p=0.03944/0.04408。全部78项中共三项达到标准，均为该模型对比的AUROC；没有操作点或概率质量指标达到次组校正标准。

针对接近0.05且受模拟误差影响的T2 AUROC比较，另作999,999次独立患者bootstrap，RNG为20261104。新原始p=0.000722，Monte Carlo标准误约0.0000269；只替换该项并固定其余77项后，Holm69 p=0.049096，Holm78 p=0.054872。**T2的联合校正结论从主表0.04408跨过0.05，不能称稳健显著**。主表保留统一99,999次结果及标记，附注完整报告敏感性，不替换成追加值。Macro与T1在该敏感性计算中保持原值，其模拟未被重跑。凭据为`data/monte_carlo_stability.json`。

显著性不能消除回顾性总体模型与seed选择偏倚；未显著也不等于模型等效。完整结果和方法凭据见`data/significance_tests.csv`及`data/significance_provenance.json`。亚组、阈值、DCA和训练probe不新增显著性检验。

## 扩展指标、图与分析

报告AUROC、非插值AP、F1、accuracy、sensitivity、specificity、PPV、NPV、balanced accuracy、MCC、Youden J、Brier、NLL及十等宽分箱ECE，另保留TP/FP/FN/TN。平均混淆计数来自四seed各自的整数计数，不代表单个合并分类器的人数。

Mean的Macro敏感性/特异性为0.6786/0.6814，Direct为0.5967/0.7213；获益与代价并存。Mean对Direct敏感性的原始/次组p为0.00127/0.08509，特异性为0.00191/0.12415，均未通过次组校正。Direct与Mean Macro accuracy为0.6814/0.6805，仅略高于全阴性参考0.68，NoJEPA为0.6748。

Mean的NLL/Brier/ECE为0.5829/0.2002/0.1405，三项点估计均差于NoJEPA（0.5812/0.1996/0.1157）及Direct（0.5771/0.1979/0.1207）。较高AUROC不保证更好的概率质量。示例阈值0.30/0.50/0.70的DCA中Direct平均净获益均高于Mean；分析使用未校准概率与假设性操作，不验证具体临床治疗决策。

`oof_validation_youden`按每seed每landmark拼接互不重叠的验证OOF预测，每个开发患者只贡献其验证fold模型的一次预测；最大化Youden指标，并列取最高阈值，再应用于该seed测试五模型概率均值。共36个阈值。验证患者也参与最佳epoch选择，且OOF单模型与测试五模型均值分布可能不同，因此只是阈值敏感性，不是独立或嵌套校准。Macro F1固定阈值→替代阈值为NoJEPA 0.5382→0.5418、Direct 0.5446→0.5420、Mean 0.5760→0.5743。

Mean的T2−T0 AUROC差+0.0002，区间[−0.0054, 0.0061]；AP差+0.0062、F1差−0.0020的区间也均跨零。AUROC/F1在T1达到均值峰值，不能写成增加访视必然改善性能。

八组图为`framework`、`discrimination`、`paired_comparisons`、`calibration_utility`、`subgroup_operating`、`forecast_mechanism`、`landmark_performance`、`outcome_score_changes`，均提供矢量PDF与600dpi PNG。曲线、校准组、DCA和概率变化分布先按seed分别计算再汇总；不跨seed平均概率。landmark图分别展示描述性seed SD与条件患者区间。

四个HR/HER2组的Mean对NoJEPA Macro AUROC差分别为+0.0385、+0.0143、+0.0272、+0.0380；第一和第四组未经多重校正区间不跨零，另两组跨零。没有亚型交互检验，不能称确认的亚型获益。HR+/HER2−组Mean敏感性仍仅0.1515。来源命名空间不是外部验证；小样本或结局不足时抑制AUROC/AP。留出集11例crop部分截断者仅1个阳性，不能支持可靠的亚组判别结论。

机制诊断在各snapshot自己的EMA坐标内计算，先平均每seed内五fold，再汇总四seed；不集成潜在状态，不是最佳pCR检查点的独立留出forecast验证。epoch20三个horizon的copy-normalized skill为Direct 0.0129/0.0621/0.1635，Mean −0.0469/−0.0479/0.0283，三步目标只有5–9人。Mean的pCR→predictor梯度均值为0.9867，Direct为0；这支持预期梯度路由，不能将pCR排序差归因于更准确的未来预测。NoJEPA缺失world diagnostic不填零。

## 编译与本地重建

在本目录运行：

```bash
bash build.sh
```

将ZIP上传Overleaf，主文件设为`main.tex`，编译器设为pdfLaTeX。预生成图表不需要Python或本地患者数据；官方`elsarticle.cls`和`elsarticle-harv.bst`已包含。

完整本地重建：

```bash
cd /personal_fast_storage/VLA-PCRV1/manuscripts/mia_mean_seed42_44_71_2026_probmean_v8
bash regenerate.sh
```

脚本读取冻结评估报告和已核验manifest，不启动训练、不加载checkpoint、不改写MRI。聚合表为`data/extended_summary.csv`，主表导出为`data/four_seed_ensemble_summary.csv`，逐seed指标为`data/four_seed_individual_metrics.csv`，逐seed DCA为`data/four_seed_dca.csv`。来源见`data/extended_provenance.json`、`data/subgroup_provenance.json`及`notes/source_audit.json`。

源码包只包含聚合CSV、匿名开发病例渲染图和复算脚本；不发布患者ID、逐患者概率、MRI/NPZ原档或checkpoint。重算患者bootstrap和亚组需要本地患者级评估报告和manifest，不能只从聚合CSV重建。

## 待补与研究边界

175人是内部downstream holdout，其中29位新增患者曾属于旧开发集。当前pCR模型按新fold重新训练，但codec与定位分割预训练的患者重叠尚未完整审计。NoJEPA保留公共fusion及transition queries，仅关闭未来辅助损失。三模型teacher-forced单步JEPA权重均为0，实际未来监督为自主rollout L1。

T0 MRI名义上早于随机分配，最终assigned arm是回顾性条件场景，不能据此主张分配前部署或个体因果疗效预测。官方模板可用，但MIA专属Guide此前返回HTTP403，字数、匿名及上传细则仍待作者核对。作者、伦理、pCR病理定义、预训练队列、实际扫描参数及公开发布承诺等未知事项保留红色`TO COMPLETE`，详见`notes/TODO_ZH.md`。本稿尚未提交。
