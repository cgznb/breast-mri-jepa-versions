# 显著性补充：seed44五折概率集成

本次仅增加统计分析和稿件表述，保留既有全部点估计与2,000次患者bootstrap百分位95%区间。每个模型均先将五折概率逐患者、逐时点平均；175位内部留出患者，56位阳性，固定阈值0.5。患者是独立分析单位，五折和重复时点不增加样本量。

## 方法与校正

AUROC采用双侧配对DeLong渐近检验。Macro首先分别对每位阳性/阴性患者三个时点的placement值取均值，再用配对差值估计方差，保留时点相关性。

其余13指标采用零中心配对患者bootstrap近似检验，共99,999次、RNG20261103，所有模型/时点共享患者抽样。检验量为有符号指标差；p按 `1 + count(abs(delta_boot - delta_observed) >= abs(delta_observed))` 除以100,000计算，包含并列。加一约定避免Monte Carlo零值，不赋予有限样本精确性。非光滑指标及单种子、事后选择均限制解释。

主要比较组：三个主模型两两比较×Macro AUROC/AP/F1，共9项。次要比较组合计81项：逐时点主指标27项、其余Macro指标33项、Attention对Mean池化比较12项、三个主模型T2−T0变化9项。组内分别Holm校正，另导出全部90项共同Holm校正的敏感性检查。各表同时保留原始p与校正p，显著标记依据组内校正p，ns表示校正p≥0.05。

## 主要结果

| Macro比较 | 指标 | 差值 | 原始p | 主要9项Holm p |
|---|---|---:|---:|---:|
| DirectReadout − NoJEPA | AUROC | +0.0121 | 0.1401 | 0.8404 |
| DirectReadout − NoJEPA | AP | +0.0107 | 0.3286 | 1.0000 |
| DirectReadout − NoJEPA | F1 | +0.0261 | 0.1150 | 0.8048 |
| ForecastAttention − NoJEPA | AUROC | +0.0228 | 0.0462 | 0.4161 |
| ForecastAttention − NoJEPA | AP | +0.0315 | 0.0899 | 0.7190 |
| ForecastAttention − NoJEPA | F1 | +0.0188 | 0.4471 | 1.0000 |
| ForecastAttention − DirectReadout | AUROC | +0.0108 | 0.3055 | 1.0000 |
| ForecastAttention − DirectReadout | AP | +0.0208 | 0.2327 | 1.0000 |
| ForecastAttention − DirectReadout | F1 | −0.0074 | 0.6995 | 1.0000 |

主要9项及次要81项均无组内Holm校正后显著结果；全部90项共同校正同样无显著结果。没有显著星号，表内如实标为ns。不能把“未显著”解释为模型等效或没有效应。

ForecastAttention的T2−T0 AUROC变化+0.0099，原始p=0.01452，次要组校正p=1。Attention相对Direct的Macro特异性差−0.0588，原始p=0.00175，次要组校正p=0.14175。未经多重校正的95%区间与校正后p并非同一推断，不能由区间是否跨零反推p。

## 完整保留的方法修订

最先运行的患者内完整概率轨迹交换检验针对更强的“给定结局后概率分数可交换”零假设。由于不同模型的概率尺度可以不同，而AUROC/AP只关心排序，该交换零假设与指标差为零的目标并不相同。检查初始结果后，主要方法改为配对DeLong与零中心bootstrap，方法选择为事后，未声称预先规定。

原始90项轨迹交换结果完整保留于 `data/significance_score_swap_sensitivity.csv`，其方法凭据为 `data/significance_score_swap_provenance.json`，生成脚本为 `scripts/significance_score_swap.py`。它们不用于主要表格的显著性判断。bootstrap对AUROC的近似p也保留在主要CSV的 `p_centered_bootstrap` 列中，主要AUROC的 `p_raw` 始终是DeLong p。保留全部结果便于审查方法选择，不择优挑选显著p。

## 独立数值核验

`scripts/verify_significance.py` 使用独立midrank联合协方差实现核对19项DeLong，并通过显式重复患者行与独立秩公式逐次重算主要9项的99,999次bootstrap计数，计数全部一致；另核对全部90项p公式、两组与全90项Holm、标记、源文件哈希。凭据在 `notes/significance_verification.json`。可选验证环境为本仓库 `versions/v1_evolution/.venv/bin/python`，需scipy/scikit-learn；正文生成不增加这些依赖。

所有检验条件于当前固定已训练集成，不能反映重新训练波动、事后选seed44、集成策略或长期反复使用holdout带来的不确定性。未新增OOF阈值、临床亚组、DCA阈值网格或训练probe的显著性检验。
