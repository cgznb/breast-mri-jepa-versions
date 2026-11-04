# v3 官方模板、投稿要求与文献核验

官网和文献元数据原始核查日期：2026-11-02，沿用已下载官方模板与来源凭据。本版v5仅seed44，每折模型独立计算指标后汇总五折均值±样本SD，不平均概率。历史版本保留。本文件仅记录模板和已核查政策范围，最终页数以PDF为准。

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

主三模型为NoJEPA、DirectReadout、ForecastAttention，ForecastMean为补充池化对照。本版只使用seed44，各fold模型分别在同一175人/56阳性内部留出集计算指标，随后汇总五折指标均值和样本SD。共20个模型，主15个、补充5个；不集成预测概率、不把fold或时点当独立患者。选seed44及报告政策为事后选择，须披露。

v5提供14项指标、逐折验证Youden阈值敏感性、八组图、亚组、假设性效用和归档训练probe。每fold在自身验证分数上选阈值并应用同fold模型test；验证也参与checkpoint选择，因此不是嵌套/独立校准。患者bootstrap对每fold独立重算metric再平均，固定五个模型、共享患者draw，不重抽fold也不重新训练。所有指标的p采用99,999次零中心配对患者bootstrap近似检验，主要9和次要81分别Holm，另报全部90共同校正。当前结果以data文件为准，不沿用历史单seed概率集成或三seed数值。

训练probe在各snapshot的EMA坐标内计算诊断后取五fold描述性均值和SD，不集成潜在状态，也不是最佳pCR checkpoint的独立未来状态评价。源码包只有聚合CSV、匿名开发病例渲染图和生成脚本，无患者ID、患者级预测、MRI原档或checkpoint。收件者可独立编译已生成图表，重算患者统计需本地原始评估/manifest。生成依赖NumPy与Matplotlib，可选独立验证需scikit-learn/scipy。
