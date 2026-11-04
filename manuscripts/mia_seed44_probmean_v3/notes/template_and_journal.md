# v3 官方模板、投稿要求与文献核验

官网和文献元数据核查日期：2026-11-02。当前扩展v3只使用seed44；逐患者先平均五折模型概率，再计算指标，Macro等权平均T0/T1/T2指标。采用官方elsarticle的5p双栏布局。v1的单seed44折指标均值稿和v2的seed21/44折指标汇总稿均保留。本文件记录模板来源、格式选择和可核验的政策范围，不代表已经完成投稿要求确认；最终页数以编译PDF为准。

## 已从出版社官网实际下载的模板

- 官方入口：https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions
- 此页面上实际的 **Download the journal article template package** 链接：
  https://assets.ctfassets.net/o78em1y1w4i4/4MpsJHO0MOJ2xZuwGTAbOZ/7bc64af36477c5d6cfce335a1f872363/elsarticle.zip
- 下载成功（HTTP 200），原始压缩包：`vendor/elsarticle-official.zip`，1,430,199 bytes。
- SHA256：`0b093093e84db49f99bcc9a7c3f69ed1fb61b0147c6d296427aff7963e7f50f6`。
- 完整下载凭据：`sources/official/template_download_receipt.json`。
- 官方原包完整保留并解压至 `vendor/elsarticle/elsarticle/`。它包含 `elsarticle.dtx`、`elsarticle.ins`、Harvard/数字参考文献样式、示例 `.tex`、文档和manifest。原包未预生成`.cls`；通过官方`elsarticle.ins`的docstrip步骤生成根目录使用的类文件。
- 当前v3使用`\documentclass[5p,twocolumn,authoryear]{elsarticle}`，采用双栏全文与跨栏图表，并使用Harvard作者年份参考文献样式。v1曾使用`preprint,12pt,authoryear`单栏审阅格式。这些均来自出版社通用类的合法选项，不能据此称作已经核对过全部MIA专属格式要求。
- 新图按182 mm宽度设置字体和布局，保存矢量PDF及600 dpi PNG；模型配色统一。图的具体风格是本稿的出版呈现选择，不是MIA官网强制风格规定。
- Elsevier 页面也给出单栏和双栏 CAS 模板；本稿没有混用 CAS 类文件。
- 官方版权和 LPPL 授权已保留于原源文件；另保存 `vendor/LPPL-1.3c.txt`，来源为 LaTeX Project 的官方许可文本。

## MIA 专属要求的可核验边界

已尝试下列官方入口，但当前环境均返回 HTTP 403，保存的是失败响应，不能把它当作已读取的投稿指南：

1. https://www.sciencedirect.com/journal/medical-image-analysis/publish/guide-for-authors
2. https://www.elsevier.com/journals/medical-image-analysis/1361-8415/guide-for-authors
3. https://www.elsevier.com/locate/media

因此，以下事项保持待作者在投稿系统或可访问的官方指南核实：摘要和主文字数、关键词数、Highlights具体要求、图文摘要是否必需、匿名评审类型、单独标题页要求、参考文献格式和上传文件类别。当前稿件中的摘要、关键词、Highlights、声明及作者占位用于继续编辑，不意味着MIA专属条款已逐条核实。源码ZIP和独立编译验证由最终构建流程生成，实际文件与校验值见`notes/build_verification.json`。

旧 Elsevier 模板 URL `https://www.elsevier.com/__data/assets/file/0011/56846/elsarticle-template.zip` 跳转至已拒绝访问的旧文件存储，未用于稿件。CTAN 包也曾测试可达，但最终实际采用的是上述官方页面当前发布的 ZIP。

## 已核实的出版社 AI 使用声明政策

- 官方政策：https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals
- 当前访问 HTTP 200；源网页和文本保存在 `sources/official/elsevier_ai_policy.*`。
- 政策要求投稿时单列 **Declaration of generative AI and AI-assisted technologies in the manuscript preparation process**，说明工具名称、使用目的和作者监督；作者仍对准确性、引用和最终稿承担责任。
- 本次实际使用 OpenAI Codex 协助稿件初稿、LaTeX 整理和结果呈现。作者最终审阅、修改和责任确认尚需由作者本人完成；不能在尚未审阅时提前声称全部已由作者检查。
- 此政策是 Elsevier 通用政策；上述 MIA 专属指南仍待核实。

## 文献来源和稳定 BibTeX keys

`references.bib` 中 21 条文献均有实时核实过的元数据。`sources/official/bibliography_verification.json` 逐条记录来源；Europe PMC core / Crossref JSON、TCIA 官方页面、arXiv 官方元数据和 CVF 官方 BibTeX 均保留在 `sources/official/`。历史脚本 `build_verified_references.py` 仅重建最初19条；v2另据已保存的精确DOI Crossref元数据加入MCC及DCA两条，最终21条以根目录BibTeX为准。作者姓名的重音符转换为可移植的 TeX 记号，原始 Unicode 元数据保持原样。

| Key | 用途与核验要点 |
|---|---|
| `ctneobc2014` | Cortazar et al., Lancet 2014；pCR 与长期结局的临床背景。DOI 10.1016/S0140-6736(13)62422-8。 |
| `hylton2012mri` | Hylton et al., Radiology 2012，ACRIN6657/I-SPY1 的 MRI 应答研究；不能把它写作本稿 I-SPY2 数据集来源。 |
| `hylton2016ftv` | Hylton et al., Radiology 2016，功能肿瘤体积与无复发生存。 |
| `ispy2trial2016` | **Park et al.** 的 NEJM2016 neratinib 适应随机试验；Esserman 是合著者，不能写成 Esserman 第一作者。 |
| `rugo2016ispy2` | Rugo et al., NEJM2016 veliparib-carboplatin 的 I-SPY2 试验。 |
| `tciaispy2_2022` | TCIA 官方 ISPY2 Version1 数据集引用，DOI 10.7937/TCIA.D8Z0-9T85。 |
| `tciaacrin6698_2021` | TCIA 官方 ACRIN6698/I-SPY2 Breast DWI 数据集引用，DOI 10.7937/TCIA.KK02-6D95。 |
| `ACRIN6698diffusion2018` | Partridge et al., Radiology289(3):618–627，DOI 10.1148/radiol.2018180273。 |
| `li2020ispy2` | Li et al., npj Breast Cancer 2020，I-SPY2 多特征 MRI 预测；TCIA 页面也推荐该论文。 |
| `mamamia2025` | Garrucho et al., Scientific Data12:453，DOI 10.1038/s41597-025-04707-4。准确标题：A large-scale multicenter breast cancer DCE-MRI benchmark dataset with expert segmentations。 |
| `han2024synthesis` | Han et al., Medical Image Analysis2024，DOI 10.1016/j.media.2023.103044；MRI 跨序列合成表征，不应称作已经实现的纵向未来状态预测对照。 |
| `du2025early` | Du et al., Breast Cancer Research2025，ACRIN6698及前瞻性队列的早期应答预测相关工作。 |
| `tang2025spatiotemporal` | Tang et al., eClinicalMedicine2025，时空 MRI 应答预测相关工作。 |
| `peng2022pretreatment` | Peng et al., Frontiers in Oncology2022，治疗前 DCE-MRI 的深度学习 pCR 预测。 |
| `khan2022review` | Khan et al., Tomography2022，MRI 与临床信息预测 pCR 的系统综述。 |
| `jepa2023` | Assran et al., CVPR2023 的 I-JEPA；DOI 10.1109/CVPR52729.2023.01499。只能说明预测表征思想来源，不能暗示本实现有非零 I-JEPA 掩码损失。 |
| `vqgan2021` | Esser et al., CVPR2021；使用 CVF 官方 BibTeX，未添未核实的 DOI。 |
| `vaswani2017` | Vaswani et al., NeurIPS2017 Transformer；作者/标题来自 arXiv 官方，会议页面也实时核实。 |
| `adamw2019` | Loshchilov and Hutter, ICLR2019；arXiv 官方注明会议版本。 |
| `chicco2020mcc` | Chicco and Jurman, BMC Genomics2020；MCC与F1/accuracy比较，DOI 10.1186/s12864-019-6413-7。 |
| `vickers2006dca` | Vickers and Elkin, Medical Decision Making2006；决策曲线，DOI 10.1177/0272989X06295361。 |

## TCIA 队列关系

TCIA 官方 ISPY2 页面明确说明：ISPY2 collection 和 ACRIN6698 collection 是同一 I-SPY2 Imaging Cohort1 的公开影像来源，应使用包含两者的 manifest 获取完整 cohort。不能仅凭 collection 名称将二者写成互相独立的内部/外部验证队列。具体本地 976 人构建和筛选仍以本地已冻结 manifest 为准，不能直接套用官网 collection 总人数。

## v3结果、源码和可重复范围

主三模型为NoJEPA、DirectReadout、ForecastAttention，ForecastMean为补充池化对照。当前仅使用seed44：对每位患者、时点先将五个折模型的概率算术平均，再计算指标；Macro等权平均三个landmark指标。共20个拟合模型，主比较15个、补充5个。相同175人内部holdout不能因五折或三个时点被视为额外独立患者。seed44在其他种子和结果已可见后回顾性选择，仍须披露。

v3提供14项指标、验证集阈值对照、八组图、临床亚组、概率校准、假设性效用及归档训练probe分析。替代阈值从互不重叠验证折的单模型OOF预测选择，再应用于测试五模型概率均值；验证与测试分数分布可能不同，因此它是阈值敏感性分析，不是集成匹配的验证或校准。患者bootstrap对固定集成概率重算指标，不包含重训、事后选择或阈值重估不确定性。训练probe机制指标仍为seed44折snapshot诊断值的描述性均值，不是潜在状态集成或留出未来预测验证。原始患者级预测和MRI不随源码包发布；CSV为聚合数据，图中的MRI是一个匿名开发病例的渲染示例。外部接收者可独立编译已生成的图表，但重算患者配对bootstrap、亚组和图像示例需要原本地评估与manifest。`requirements_analysis.txt`和`regenerate.sh`描述脚本依赖与重建顺序；scikit-learn仅用于可选的独立数值核验，运行脚本使用NumPy和Matplotlib。

历史方法与队列证据记录建立的是冻结实现和数据来源事实；其旧标题或旧聚合说明不定义本v3的概率优先口径。v1/v2结果与本版不能混用。当前指标与报告身份以`data/extended_provenance.json`、`data/subgroup_provenance.json`及最终构建回执为准。

## 尚未完成或不作出的断言

没有核实 MIA 专属硬性格式条款；没有声称本研究已经运行所引外部算法基线；没有把元数据查询当作已系统阅读全文；没有对尚未完成的外部验证、前瞻性验证或统计显著性作出结论。
