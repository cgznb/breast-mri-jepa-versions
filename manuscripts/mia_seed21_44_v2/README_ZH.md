# MIA 英文初稿 v2：Medium976，seed21 与 seed44

本版使用 Elsevier 官网实际下载的 `elsarticle` 官方源包，采用 `5p,twocolumn,authoryear` 双栏排版。主文件为 `main.tex`，PDF 为 `main.pdf`。最终目录名为 `mia_seed21_44_v2`；此前仅报告 seed44 的 `mia_seed44_v1` 保留，不能将两版结果的汇总范围混用。最终页数与编译记录以 PDF 和 `notes/build_verification.json` 为准。

英文标题为 **Forecast-aware latent modeling for longitudinal prediction of pathological complete response from breast MRI**。本目录包含摘要、引言、相关工作、方法与公式、队列和预处理、实验设计、扩展结果、讨论、局限、结论、声明占位、补充分析、Highlights 和核实过的参考文献。

## 模型、患者和结果口径

主比较为 NoJEPA、DirectReadout、ForecastAttention；ForecastMean 为补充池化对照。四组均使用 **seed21 和 seed44，各五折**。主结果先在每折独立计算指标，再取同一种子的五折均值，最后对两个种子等权平均；Macro 对 T0/T1/T2 再等权平均。没有平均预测概率后重新计算指标。折、种子、同一患者的多个时点也不作为独立患者样本。

所有时点均使用同一批内部留出患者：175 人，56 人 pCR 阳性。40 个模型×种子×折评估回执及其120份注册报告身份已核验。正文模型结果不混入其他种子、旧866人队列、VQ预测头实验或未运行的外部算法分数。

| 模型 | Macro AUROC | Macro AP | Macro F1 |
|---|---:|---:|---:|
| NoJEPA | 0.7157 | 0.5773 | 0.5333 |
| DirectReadout | 0.7277 | 0.5904 | 0.5425 |
| ForecastAttention | 0.7360 | 0.5969 | 0.5490 |

本版比较14项指标：AUROC、AP、F1、accuracy、sensitivity、specificity、PPV、NPV、balanced accuracy、MCC、Youden J、Brier、NLL、十等宽分箱ECE。另外导出TP/FP/FN/TN四项计数；它们不额外计作四种指标。阈值指标默认0.5，并提供**每折、每时点仅用验证集选择的Youden阈值**对照，未在测试集寻找最优阈值。

新的两种子汇总中，ForecastAttention 的主要三项宏均值最高，但敏感性提高伴随特异性和PPV下降；DirectReadout的balanced accuracy、MCC和概率质量仍有优势。不能沿用v1单seed44中的F1排序。部分AUROC配对区间不跨0，AP/F1及部分模型间区间仍跨0；不能把v1“主要区间均跨0”的描述直接复制到本版，也不能把探索性条件区间解释为确认性疗效证据。

## 八组图及增加的分析

图均采用统一模型配色、清晰面板编号、白底和按出版尺寸设置的字体；提供矢量PDF及600 dpi PNG。当前正文和补充材料共使用八组图：

1. `framework`：真实匿名开发病例的三期×三时点MRI示例、连续编码、融合、自主rollout、EMA目标和消融设计。
2. `discrimination`：T0/T1/T2 ROC与PR曲线，带共享患者重采样区间。
3. `paired_comparisons`：不同读出/辅助监督对照的逐时点与Macro配对差值。
4. `calibration_utility`：校准图及假设性决策曲线，附treat-all/none参考。
5. `subgroup_operating`：HR/HER2、年龄组的探索性差异和敏感性/特异性。
6. `forecast_mechanism`：现有固定训练probe的copy-normalized forecast skill和梯度路径。
7. `landmark_performance`：六项互补指标随时点的变化。
8. `outcome_score_changes`：按真实结局分层的T0到T2预测概率变化分布。

临床亚组、影像来源命名空间和预测定位截断敏感性均保留分母；小样本或结局不足时抑制不可靠的AUROC/AP。亚组区间未作多重性校正，不证明亚组间交互。机制图来自已归档的固定32人**训练probe**，不是holdout或最佳pCR检查点的未来预测验证；NoJEPA缺少对应diagnostic时记为不可用，不填零。使用同一snapshot内的copy归一化可减少EMA坐标影响，仍不能证明各模型潜在空间具备相同的生物学意义。

2,000次患者bootstrap在模型、折、两个种子及多个时点间共享患者抽样，每次保持原来的汇总顺序。区间条件于现有训练模型；不包含重新训练、事后选种子、选择模型或重新估计验证阈值的不确定性。决策曲线使用未重校准的概率和通用风险阈值，仅属假设性效用分析。

## 编译与Overleaf

从本目录执行：

```bash
bash build.sh
```

需要常见TeX Live组件：`pdflatex`、`bibtex`、`lmodern`、`graphicx`、`booktabs`、`microtype`等。根目录的`elsarticle.cls`和`elsarticle-harv.bst`来自保留的官方源包。将提供的Overleaf源码ZIP上传后，把主文档设为`main.tex`，编译器选pdfLaTeX。图、表已生成，不必在Overleaf运行Python或连接本地评估目录。

## 重算分析与制图

运行分析脚本需要NumPy和Matplotlib；`requirements_analysis.txt`记录依赖。scikit-learn仅用于已经实施的独立数值交叉核验，不是生成脚本的运行依赖。当前系统`python`具有NumPy和Matplotlib。已有训练环境`/personal_fast_storage/VLA-PCRV1/versions/v1_evolution/.venv/bin/python`具有NumPy和scikit-learn，但没有Matplotlib，不能直接用它运行制图步骤。

完整重建入口为：

```bash
cd /personal_fast_storage/VLA-PCRV1/manuscripts/mia_seed21_44_v2
bash regenerate.sh
```

手工依赖顺序如下；前两步读取当前冻结评估记录与manifest，后续步骤依赖它们生成的聚合CSV：

```bash
python scripts/extended_metrics.py
python scripts/subgroup_analysis.py
python scripts/build_results.py
python scripts/build_metric_intervals.py
python scripts/draw_framework.py
python scripts/plot_expanded.py
python scripts/plot_mechanism.py
bash build.sh
```

如仅重算统计，可使用包含scikit-learn的已有环境执行前两步和表格生成步骤；图仍使用具有Matplotlib的解释器：

```bash
MIA_STAT_PY=/personal_fast_storage/VLA-PCRV1/versions/v1_evolution/.venv/bin/python
"$MIA_STAT_PY" scripts/extended_metrics.py
"$MIA_STAT_PY" scripts/subgroup_analysis.py
"$MIA_STAT_PY" -c 'import sklearn; print(sklearn.__version__)'
```

最后一条仅核对可选数值验证库，不替代完整结果验证。脚本直接读取已有报告，不启动训练、不加载模型检查点、不改写影像或运行输出。

外部接收者能够用已生成的图、表和聚合CSV编译论文，但不能只靠聚合文件重做患者bootstrap、临床亚组或MRI示例选择；这些步骤需要原始本地患者级评估记录和manifest。源码包不包含原始MRI/NPZ、患者级预测、检查点或患者ID名单。方法图包含一个匿名开发病例的渲染切片，原始图像数据不随包发布。

## 报告边界与待补信息

seed21和44是多个种子结果已可见后选择的两个种子，属于事后选择，不能写成预注册或全种子稳定性证据。当前175人是内部downstream holdout；其中新增患者里29人曾属于旧开发集。当前pCR模型按新fold重新训练，但codec/定位分割预训练与留出患者的重叠尚未完整审计。

NoJEPA保留公共fusion和transition queries，只关闭未来辅助监督；三组的单步JEPA损失权重均为0，实际未来监督为自主rollout L1。ForecastAttention与DirectReadout同时改变预测特征读出和分类头宽度；ForecastMean补充对照更接近单独的池化比较。训练probe中，完整模型较好的pCR排序并不对应全部horizon上更高的copy-normalized预测skill，负向证据已保留。

TCIA源协议的T0 MRI早于随机分配。T0使用最终assigned arm属于回顾性条件场景，不能声称分配前就可前瞻部署。MIA专属Guide当前返回HTTP403；官方模板和出版社通用政策已获取，但匿名方式、字数和提交文件细则仍待最终核对。

作者、伦理、pCR病理定义、预训练队列、实际扫描参数及发布承诺等未知事项以红色`TO COMPLETE`保留，详见`notes/TODO_ZH.md`。没有向期刊或其他外部系统提交本稿。
