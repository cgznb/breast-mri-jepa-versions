# 三种子数据来源审计

已实时重新核验 seed44、2026、3407 对应的 60 个模型×种子×fold 任务。60 个 matrix 状态、正式评估 receipt 和 evaluate worker receipt 全部完成，180 份注册报告的路径、文件字节数与纳秒修改时间全部相符。未启动训练，未读取 checkpoint 权重。

每种子包含 NoJEPA、DirectReadout、ForecastAttention、ForecastMean 四模型，每模型五 fold；正文前三模型共 45 个任务，补充 ForecastMean 共 15 个任务。汇总口径为：同一模型、同一种子内逐患者平均五 fold 概率后计算指标，再等权平均三个种子的指标。不能跨种子混合概率。标准差是这三个选定种子指标的样本标准差（ddof=1），不代表三个独立临床队列。

与 seed44 配方逐模型×fold 比较的 40 组配置差异仅为 seed。三个 suite 的 92 个冻结 runtime 文件按相对路径 SHA256 完全相同，合计核验 5520 项 runtime 文件身份（276 个不同路径文件）。输入配置、manifest、codec、training_done 与 run_identity 各 60 项身份核验通过，五个 fold 的 manifest SHA256 一致。

全部 180 个模型×种子×fold×时点 test ID/标签向量严格一致，并与 manifest test 分组及 pCR 标签吻合：T0、T1、T2 均为同一内部固定留出集 175 人、56 阳性；independent_test=false。180 个 validation 时点向量均属于相应 fold 的 validation 分组且标签匹配。审计输出不含患者标识或患者级预测。

seed3407 的 20 个正式评估于北京时间 2026-11-04 16:27:49 至 16:32:12 完成。三个来源 suite 目前均已全部完成训练与正式评估；本稿在每个 suite 内仅选择指定种子的 20 个任务。

本审计不重新验证原始 MRI 几何、预训练患者重叠或外部独立性。种子集合在查看既有结果后进行了回顾性调整，应保留探索性与种子选择不确定性的说明。完整路径、状态计数、五fold manifest SHA256及完成时间见 source_audit.json。
