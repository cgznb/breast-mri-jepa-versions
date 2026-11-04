# MIA 英文初稿 v3：seed44 五折概率集成

本版采用从 Elsevier 官网下载并保留来源凭据的 `elsarticle` 官方源包，使用 `5p,twocolumn,authoryear` 双栏排版。主文件为 `main.tex`，PDF 为 `main.pdf`，目录为 `mia_seed44_probmean_v3`，Overleaf 源码包为 `mia_seed44_probmean_overleaf.zip`。最终页数和文件校验值以 PDF 及 `notes/build_verification.json` 为准。

英文标题为 **Forecast-aware latent modeling for longitudinal prediction of pathological complete response from breast MRI**。初稿包含摘要、引言、相关工作、方法与公式、队列和预处理、实验设计、扩展结果、讨论、局限、结论、声明占位、补充分析、Highlights 和已核实的参考文献。

## 本版唯一结果口径

主比较为 NoJEPA、DirectReadout、ForecastAttention；ForecastMean 是补充池化对照。所有模型均只使用 **seed44、五折模型**。对每位患者、每个时点，先将五个折模型的 pCR 预测概率作算术平均，再用该概率计算指标：

`五折概率逐患者平均 → 在175位患者上计算指标 → T0/T1/T2指标等权平均得到Macro`。

本版没有平均五折的指标，也没有混入 seed21。原 `mia_seed44_v1`（seed44 五折指标平均）及 `mia_seed21_44_v2`（折指标再作两种子平均）均保留，三版不能混用。主三模型对应15个拟合模型，补充 ForecastMean 对应5个，共20个种子×折模型。每一时点均覆盖同一批内部留出患者：175人，其中56人 pCR 阳性；五折或多个时点不会增加独立患者数。

| 模型 | Macro AUROC | Macro AP | Macro F1 |
|---|---:|---:|---:|
| NoJEPA | 0.7279 | 0.5968 | 0.5416 |
| DirectReadout | 0.7399 | 0.6075 | 0.5678 |
| ForecastAttention | 0.7507 | 0.6283 | 0.5604 |
| ForecastMean（补充） | 0.7465 | 0.6139 | 0.5742 |

ForecastAttention 在主三模型中 Macro AUROC 和 AP 最高；固定0.5阈值下，DirectReadout 的 F1 高于 ForecastAttention。补充模型 ForecastMean 的 F1 更高。不能将判别排序、阈值指标和概率质量合并成一个“全面最佳”的结论。

本版比较14项指标：AUROC、AP、F1、accuracy、sensitivity、specificity、PPV、NPV、balanced accuracy、MCC、Youden J、Brier、NLL、十等宽分箱 ECE。另导出 TP/FP/FN/TN 计数。阈值指标默认使用0.5，完整14指标表同时提供患者 bootstrap 区间。

替代阈值 `oof_validation_youden` 先拼接五折互不重叠的验证集 OOF 预测，每位开发患者只使用其验证折模型的预测；每个模型、时点用这些预测选择 Youden 阈值，再应用于测试集五模型概率集成。阈值从未用测试标签选择。但验证 OOF 是单模型分数，测试是五模型均值，分布可能不同；因此它是阈值敏感性分析，不是与集成匹配的验证或校准。

## 八组图与扩展分析

图采用统一模型配色、面板编号、白底和按出版尺寸设置的字体，提供矢量 PDF 与600 dpi PNG。当前正文和补充材料共使用八组图：

1. `framework`：匿名开发病例的三期×三时点 MRI 示例、连续编码、融合、自主 rollout、EMA目标和消融设计。
2. `discrimination`：概率集成的 T0/T1/T2 ROC与PR曲线及共享患者重采样区间。
3. `paired_comparisons`：读出/辅助监督对照的逐时点和 Macro 配对差值。
4. `calibration_utility`：集成概率校准图与假设性决策曲线，附 treat-all/none 参考。
5. `subgroup_operating`：HR/HER2、年龄组的探索性差异及敏感性/特异性。
6. `forecast_mechanism`：固定训练 probe 的 copy-normalized forecast skill 和梯度路径。
7. `landmark_performance`：六项互补指标随时点的变化。
8. `outcome_score_changes`：按真实结局分层的 T0 至 T2 集成预测概率变化分布。

临床亚组、影像来源命名空间和预测定位截断敏感性保留分母；小样本或结局不足时抑制不可靠的 AUROC/AP。区间未校正多重比较，不能证明亚组间交互。机制图来自归档的固定32人训练 probe，以 seed44 五折 snapshot 的诊断值作描述性均值；不是潜在状态集成，也不是最佳 pCR 检查点或独立留出患者上的未来预测验证。长 horizon 的连续目标可用人数仅5–9人。NoJEPA 无对应训练 diagnostic 时记为不可用，不填零。

2,000次患者 bootstrap 在模型与时点间共享抽样，直接对固定概率集成重新计算指标。区间条件于现有拟合模型，不包含重新训练、事后选种子/模型或重新估计验证阈值的不确定性。配对差值区间与各模型边际区间分别报告。决策曲线采用未重新校准的概率与通用风险阈值，属于假设性效用分析。

## 配对显著性检查

主要检验针对“给定固定拟合模型及评估口径后，所报告总体指标差为零”的零假设，采用双侧检验。AUROC使用配对DeLong渐近正态检验；Macro AUROC先将三个时点的阳性患者placement向量、阴性患者placement向量分别在患者内平均，再估计模型配对差值的方差，保留时点与模型间关联。各时点模型比较及同模型T2–T0比较均保留患者配对，不把五折模型或重复时点视为独立样本。

其余13项指标使用以零假设为中心的配对患者bootstrap近似检验，重采样99,999次，随机数种子20261103；每次有放回抽取175位患者，所有模型和时点共用同一患者抽样，重新计算指标及差值。所有阈值指标固定使用0.5，Macro仍为三个时点指标的等权均值。原始 `p=(1+满足abs(bootstrap差值−观察差值)≥abs(观察差值)的次数)/(99999+1)`，包含相等情形。加一是避免Monte Carlo结果为零的约定，不使bootstrap检验成为有限样本精确检验；AP、F1、ECE等非光滑指标尤其存在有限样本近似限制。没有重新训练模型。

主组为三个主模型两两比较的Macro AUROC/AP/F1，共9项；次组共81项，包含三个主模型各时点AUROC/AP/F1的27项比较、其余11个Macro指标的33项比较、ForecastAttention与ForecastMean在三个时点和Macro的AUROC/AP/F1共12项比较，以及三个主模型各自T2–T0的AUROC/AP/F1共9项比较。分别在主组和次组内使用Holm多重比较校正；CSV另提供全部90项一起Holm校正的敏感性分析。分组校正不保证两组合并后的整体错误率控制。

表格报告原始p值、组内Holm校正p值及标记：`*`表示校正p<0.05、`**`表示<0.01、`***`表示<0.001，`ns`表示校正p≥0.05。配对差值仍使用原有2,000次重采样、未经多重比较校正的95%患者bootstrap百分位区间。它们不是DeLong或中心化bootstrap检验的严格反演；区间是否跨零与校正后的显著性可以不同，不从区间反推p值。

完整聚合结果见 `data/significance_tests.csv`，方法与输入凭据见 `data/significance_provenance.json`。主组p值加入 `tables/bootstrap.tex`；逐时点、其余Macro指标、池化对照及纵向变化分别见 `tables/significance_landmarks.tex`、`tables/significance_extended.tex`、`tables/significance_pooling.tex` 和 `tables/significance_longitudinal.tex`。OOF阈值、临床亚组、决策曲线与训练probe不新增显著性检验。

最初的完整概率轨迹交换检验保留在 `data/significance_score_swap_sensitivity.csv`，对应脚本为 `scripts/significance_score_swap.py`。它在患者内交换两模型完整T0–T2概率轨迹，或同模型T0/T2概率，检验比指标相等更强的“给定结局后分数可交换”零假设，采用99,999次随机置换及加一p值约定；它不用于主要表格的显著性判决。看到初始轨迹交换结果后，主要方法修订为针对指标差的配对DeLong与中心化bootstrap检验，原检查完整保留以供审查。

本次检验仍然是探索性分析。seed44、集成方式、比较组及主要检验方法的选择均发生在性能结果已可见之后；Holm校正不能消除这种事后选择偏倚，也不能用单种子患者检验证明训练稳定性。

当前结果：主组9项和次组81项均未达到各自Holm校正后的0.05标准；将90项一起校正也无显著结果。ForecastAttention相对NoJEPA的Macro AUROC差为+0.0228，配对DeLong原始p=0.0462，主组校正p=0.4161。不能将该差异写成经多重比较校正的显著提升。统计细节见 `notes/significance_checks_ZH.md`，独立数值验证见 `notes/significance_verification.json`。

## 编译与 Overleaf

从本目录执行：

```bash
bash build.sh
```

需要常见 TeX Live 组件，如 `pdflatex`、`bibtex`、`lmodern`、`graphicx`、`booktabs` 和 `microtype`。根目录 `elsarticle.cls`、`elsarticle-harv.bst` 来自保留的官方源包。将源码 ZIP 上传至 Overleaf，主文档设为 `main.tex`，编译器选 pdfLaTeX。图表均已生成，Overleaf 不需要 Python 或本地评估目录。

## 重算分析与制图

完整重建需要 NumPy 和 Matplotlib，依赖见 `requirements_analysis.txt`：

```bash
cd /personal_fast_storage/VLA-PCRV1/manuscripts/mia_seed44_probmean_v3
bash regenerate.sh
```

依赖顺序为：

```bash
python scripts/extended_metrics.py
python scripts/significance_analysis.py
python scripts/significance_score_swap.py
python scripts/subgroup_analysis.py
python scripts/build_results.py
python scripts/build_metric_intervals.py
python scripts/draw_framework.py
python scripts/plot_expanded.py
python scripts/plot_mechanism.py
bash build.sh
```

脚本读取冻结评估报告和 manifest，不启动训练、不加载模型检查点、不改写 MRI。系统 Python 具有 NumPy 和 Matplotlib；已有训练环境 `/personal_fast_storage/VLA-PCRV1/versions/v1_evolution/.venv/bin/python` 有 NumPy 和 scikit-learn，但没有 Matplotlib。scikit-learn 只用于可选的独立数值交叉核验，不是上述生成脚本的依赖。

外部接收者可用已生成的图、表和聚合 CSV 编译论文。患者 bootstrap、临床亚组及 MRI 示例选择需要原本地患者级评估记录与 manifest，不能从聚合 CSV 完整重算。源码包不含患者ID名单、患者级预测、原始 MRI/NPZ 或检查点。方法图包含一个匿名开发病例的渲染切片，原始图像数据不随包发布。当前结果和报告身份见 `data/extended_provenance.json`、`data/subgroup_provenance.json`，主表导出为 `data/seed44_ensemble_summary.csv`。

## 报告边界与待补事项

seed44 是其他种子与结果已可见后的回顾性选择，不能称为预注册选种子或全部种子稳定性证据。175人是内部 downstream holdout；其中29位新增患者曾属于旧开发集。当前 pCR 模型按新 fold 重训，但 codec 和定位分割预训练与留出患者的重叠尚未完整审计。

NoJEPA 保留公共 fusion 与 transition queries，只关闭未来辅助监督。三主模型的单步 JEPA 损失权重均为0，实际未来监督为自主 rollout L1。ForecastAttention 相对 DirectReadout 同时改变预测特征读出和分类头宽度；ForecastMean 是更接近单独池化比较的补充对照。训练 probe 的全部 horizon 不支持“完整模型未来状态预测均更准确”的主张。

TCIA 名义协议的 T0 MRI 早于随机分配。T0 使用最终 assigned arm 属于回顾性条件场景，不能据此主张分配前可前瞻部署。MIA 专属 Guide 当前返回 HTTP403；官方模板与出版社通用政策已归档，但匿名方式、字数和提交文件细则仍待核对。

作者、伦理、pCR 病理定义、预训练队列、实际扫描参数与发布承诺等未知事项保留红色 `TO COMPLETE`，详见 `notes/TODO_ZH.md`。本稿尚未提交至期刊或其他外部系统。
