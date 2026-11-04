# MIA 英文初稿：Medium976 / seed44

主文件为 `main.tex`，已编译的审阅文件为 `main.pdf`。使用从 Elsevier 官网当前链接实际下载的 `elsarticle` 官方源包，采用 `preprint,12pt,authoryear` 审阅格式。下载 URL、原包 SHA256、许可证和版本来源保存在 `notes/template_and_journal.md`、`sources/official/template_download_receipt.json` 和 `vendor/`。这不是自制的 MIA 仿版模板。

## 本版内容与模型范围

英文标题：**Forecast-aware latent modeling for longitudinal prediction of pathological complete response from breast MRI**。

已写摘要、引言、相关工作、方法与公式、队列和预处理、实验方案、结果、讨论、局限、结论、声明占位、补充分析，以及 Highlights。包含方法示意图、纵向指标图、患者 bootstrap 图、队列特征表、主结果表、校准表、池化补充表与配对区间表。`references.bib` 收录 19 条通过官网或文献元数据服务实时核实的文献；并非每条备选文献都已经在正文引用。

正文主比较为 NoJEPA、DirectReadout、ForecastAttention，全部使用 **seed44 × 五折**。ForecastMean 同样只使用 seed44，作为补充池化对照。未混入其他种子、旧866人队列、VQ生成头实验或外部算法的未运行分数。

主指标口径：同一175人/56阳性的内部留出集，每折分别计算指标 → 五折均值 → T0/T1/T2等权 Macro。没有将五折概率平均后计算指标。主要结果为：

| 模型 | Macro AUROC | Macro AP | Macro F1 |
|---|---:|---:|---:|
| NoJEPA | 0.7219 | 0.5792 | 0.5437 |
| DirectReadout | 0.7303 | 0.5888 | 0.5539 |
| ForecastAttention | 0.7365 | 0.5990 | 0.5469 |

当前20个seed44评估回执及其60份报告身份已核对，60组折×时点的主指标从原始预测重算并与报告一致。新增2,000次患者配对bootstrap：同一患者抽样在模型、折、时点之间共享；每次仍按原口径汇总，未把折视为独立患者。主要差值的95%区间均跨0，因此稿件没有统计显著性或确定性胜出的表述。bootstrap实现另与scikit-learn在包含并列分数的加权重采样例子上核对，最大误差约2.22e-16。

## 编译和修改

在本目录运行：

```bash
bash build.sh
```

环境需要常见 TeX Live 组件，包括 `pdflatex`、`bibtex`、`lmodern`、`tikz` 和 `microtype`。本工作区已安装并成功编译。`elsarticle.cls` 与 `elsarticle-harv.bst` 已包含在根目录，均来自下载的官方源包。Overleaf 上传 `mia_seed44_overleaf.zip` 后，将主文档设为 `main.tex`，编译器设为 pdfLaTeX。生成的图表已包含在压缩包，无需在 Overleaf 运行 Python 或访问本地数据目录。

分章节修改 `sections/*.tex`；结果表由 `scripts/build_results.py` 从受保护的已有评估输出生成。需要重新生成结果时，本地运行：

```bash
python scripts/build_results.py
bash build.sh
```

Python生成脚本依赖 NumPy、Matplotlib，且需要访问原始评估路径；外部接收者可以直接使用已生成的聚合CSV和图表编译论文。原始患者行、MRI、检查点及患者ID名单没有打入论文包。

## 事实边界与待补信息

- seed44是在多个种子结果已经可见之后为本稿选定的。正文已明确为事后选择的探索性分析，没有写成预注册或盲选。所有训练结果仍仅按用户要求使用44。
- NoJEPA保留共同fusion和transition queries，只关闭未来辅助损失；主三组的单步JEPA损失均为0，实际未来监督是自主rollout L1。
- ForecastAttention与DirectReadout同时改变预测特征读出和分类头宽度，不能把差值全部归因于注意力。ForecastMean补充对照更接近单独的池化比较。
- 完整模型的校准较差，DirectReadout平均F1更高，正文保留这些负结果。
- 175人属于内部downstream holdout。新增留出患者中29人曾属于旧开发集；当前模型按新划分重新训练，但codec/分割预训练与留出患者的重叠尚未完全审计。
- TCIA协议的T0 MRI位于随机分配前。当前T0输入eventual assigned-arm属于回顾性条件场景，不能当作治疗分配前即可部署的前瞻性预测。
- MIA专属Guide在当前环境返回403；已成功下载出版社官方模板，但摘要字数、匿名类型及上传文件细则仍需最后核对。未把未读取的指南当成已验证要求。

红色 `TO COMPLETE` 仅用于未知事项，详见 `notes/TODO_ZH.md`。作者信息、伦理/豁免号、精确pCR病理定义、预训练队列、实际扫描参数和数据/代码公开承诺均未编造。此文件夹是一份可继续修改的初稿，尚未向任何期刊或外部系统提交。
