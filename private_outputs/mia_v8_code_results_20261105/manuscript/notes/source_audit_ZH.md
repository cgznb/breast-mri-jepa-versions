# ForecastMean 四种子来源与读出审计

已实时重新核验 seed42、44、71、2026 下 NoJEPA、DirectReadout、ForecastMean 三模型，每模型五 fold，共 60 个任务（每种子 15 个、每模型 20 个）。60 个 matrix 状态、正式评估 receipt 和 evaluate worker receipt 全部完成，180 份注册报告的路径、字节数与纳秒修改时间全部相符。未重训，未读取 checkpoint 权重。

与 seed44 配方逐模型×fold 比较的 45 组配置差异仅为 seed。四个 suite 的 92 个冻结 runtime 文件按相对路径 SHA256 完全相同，共核验 5520 项文件身份（368 个不同路径文件）。输入配置、manifest、codec、training_done 和 run_identity 各 60 项身份核验通过，五个 fold 的 manifest SHA256 一致。

全部 180 个模型×种子×fold×时点 test ID/标签向量严格一致，且与 manifest test 分组及 pCR 标签吻合：T0、T1、T2 均为同一内部固定留出集 175 人、56 阳性，independent_test=false。180 个 validation 时点向量均属于相应 fold 的 validation 分组且标签匹配。seed71 新选定的 15 个任务已于北京时间 2026-11-04 16:22:38 至 16:27:17 完成正式评估。

汇总按同一模型、同一种子内逐患者平均五 fold 概率后计算指标，再等权平均四个种子指标；标准差为四个选定种子指标的样本 SD（ddof=1）。四个种子、三个模型、三个时点共有 36 个 OOF Youden 阈值。不能跨种子混合概率，种子 SD 不代表独立临床队列不确定性。

ForecastMean 冻结配置为 pcr_readout=forecast、spatial_readout=mean、pcr_delta_reference=current、dim=768、token_grid=[2,4,4]、pcr_queries=4。实际代码将四个 pCR query 特征求均值得到 768 维向量；将预测 T3 状态的 32 个空间 token 求均值得到 768 维向量；将 T3 预测减去最新观测状态得到的 32 个变化 token 求均值得到 768 维向量。三向量拼接为 2304 维，经 LayerNorm 和 Linear(2304,1) 输出 logit。直接读出模型仅使用 768 维 query 均值。最新观测状态由 no_grad EMA teacher 编码，自主 rollout 从该状态预测至 T3，分类读出中的 terminal 未 detach。对应源码文件、SHA256 和行号记于 source_audit.json 的 mean_readout 字段。

本审计不重新验证原始 MRI 几何、预训练患者重叠或外部独立性。种子及模型集合在查看既有结果后调整，正文应保留探索性与选择不确定性说明。输出不含患者标识或患者级预测。
