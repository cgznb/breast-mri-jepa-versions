# v8 官方模板、投稿要求与文献核验

官网和文献元数据原始核查日期：2026-11-02，沿用当时已下载的官方模板与凭据。本版v8总体模型为ForecastMean，使用seed42、44、71、2026；每seed先作五折患者概率集成，再计算指标，最后等权平均四seed指标并报告均值±样本SD。总体模型及seed集合为回顾性变更。历史版本保留；本文件仅记录模板和来源核查范围，实际页数以最终PDF为准。

## 已从出版社官网实际下载的模板

- 官方入口：https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions
- 此页面上实际的 **Download the journal article template package** 链接：
  https://assets.ctfassets.net/o78em1y1w4i4/4MpsJHO0MOJ2xZuwGTAbOZ/7bc64af36477c5d6cfce335a1f872363/elsarticle.zip
- 下载成功（HTTP 200），原始压缩包：`vendor/elsarticle-official.zip`，1,430,199 bytes。
- SHA256：`0b093093e84db49f99bcc9a7c3f69ed1fb61b0147c6d296427aff7963e7f50f6`。
- 完整下载凭据：`sources/official/template_download_receipt.json`。
- 官方原包完整保留并解压至 `vendor/elsarticle/elsarticle/`。它包含 `elsarticle.dtx`、`elsarticle.ins`、Harvard/数字参考文献样式、示例 `.tex`、文档和manifest。原包未预生成`.cls`；通过官方`elsarticle.ins`的docstrip步骤生成根目录使用的类文件。
- 当前v8使用`\documentclass[5p,twocolumn,authoryear]{elsarticle}`，采用双栏全文与跨栏图表，并使用Harvard作者年份参考文献样式。v1曾使用`preprint,12pt,authoryear`单栏审阅格式。这些均来自出版社通用类的合法选项，不能据此称作已经核对过全部MIA专属格式要求。
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

`references.bib` 当前保留25条文献。最初21条的核查来源见`sources/official/bibliography_verification.json`；后加的Davison、DeLong、Holm及Phipson四条统计文献凭据见`sources/official/statistical_references_verified.json`，其可核实范围以各记录为准。当前检验引用Davison与Holm；保留DeLong/Phipson条目不代表本版使用DeLong或概率互换置换检验。Europe PMC/Crossref、TCIA、arXiv和CVF来源凭据保持原样。历史脚本`build_verified_references.py`仅重建最初19条，不能覆盖当前25条书目；最终条目以根目录BibTeX为准。作者姓名的重音符转换为可移植的TeX记号，原始Unicode元数据保持原样。

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
| `davison1997` | Davison与Hinkley的bootstrap教材，Cambridge，DOI 10.1017/CBO9780511802843。 |
| `holm1979` | Holm的逐步多重校正方法；核验范围见统计文献凭据。 |
| `delong1988` | 保留的相关AUROC检验文献，本版未使用该检验。 |
| `phipson2010` | 保留的Monte Carlo置换p文献；本版零中心bootstrap不以它主张有限样本精确性。 |

## TCIA 队列关系

TCIA 官方 ISPY2 页面明确说明：ISPY2 collection 和 ACRIN6698 collection 是同一 I-SPY2 Imaging Cohort1 的公开影像来源，应使用包含两者的 manifest 获取完整 cohort。不能仅凭 collection 名称将二者写成互相独立的内部/外部验证队列。具体本地 976 人构建和筛选仍以本地已冻结 manifest 为准，不能直接套用官网 collection 总人数。

## v8结果、源码和可重复范围

NoJEPA、DirectReadout、ForecastMean均为主模型。使用seed42、44、71、2026，每seed五fold，共60个拟合模型。先在每seed内逐患者平均五fold概率，再计算指标与T0/T1/T2 Macro，最后平均四seed指标并计算样本SD；不跨seed平均概率。60个模型及180份时点报告重复评价同一175位患者，不增加独立患者数。

本版提供14项指标、阈值敏感性、八组图、亚组、假设性效用和归档训练probe。患者bootstrap在每seed内重算指标后再平均，固定四seed，不重训或重抽seed。全部指标使用99,999次零中心配对患者bootstrap近似检验，主要9项和次要69项分别Holm校正，另报全部78项共同校正。汇总方式、总体模型和seed选择均为事后政策，不能称验证性设计。T2 AUROC另作999,999次独立数值敏感性检查，只替换该项时Holm78从0.04408变为0.054872；该阈值决定对模拟精度敏感，主表统一99,999次结果保留。

OOF单模型分数与测试五模型概率均值的分布不同，阈值分析不是独立集成校准。训练probe在各snapshot自身EMA坐标内计算，先fold内/seed内汇总，再汇总seed；不集成潜在状态，不替代最佳pCR检查点的独立未来预测验证。

源码包只发布聚合CSV、匿名开发病例渲染图及复算脚本，不包含患者标识、逐患者概率、MRI原档或checkpoint。预生成图表可独立编译；重算患者bootstrap和亚组需要本地评估报告及manifest。
