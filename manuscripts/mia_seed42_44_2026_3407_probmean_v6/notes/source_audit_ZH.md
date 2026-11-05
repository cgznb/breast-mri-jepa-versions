# 四种子数据来源审计

已重新核验 seed42、44、2026、3407 对应的 80 个模型×种子×fold 任务，80 个 matrix 状态及正式评估 receipt 均完成，240 份注册报告的路径、文件字节数与纳秒修改时间全部相符。未启动训练，未读取任何 checkpoint 权重。

每个种子含 NoJEPA、DirectReadout、ForecastAttention、ForecastMean 四模型，每模型五 fold。当前结果应按“同一模型、同一种子内，逐患者平均五 fold 概率后计算指标，再等权平均四个种子的指标”汇总；不能跨种子混合概率。标准差是四个选定种子指标的样本标准差（ddof=1），不代表四个独立临床队列。

与 seed44 配方逐模型×fold 比较的 60 组配置差异仅为 seed。四 suite 的 92 个冻结 runtime 文件按相对路径 SHA256 完全相同，合计核验 7360 项 runtime 文件身份（368 个不同路径文件）。输入配置、manifest、codec、training_done 与 run_identity 各 80 项身份核验通过。五个 fold 的 manifest SHA256 与原三种子稿一致。

全部 240 个模型×种子×fold×时点 test ID/标签向量严格一致，并与 manifest test 分组及 pCR 标签吻合：T0、T1、T2 均为同一内部固定留出集 175 人，56 阳性；independent_test=false。validation 行均属于相应 fold 的 validation 分组且标签匹配。审计输出不含患者标识或患者级预测。

新增 seed3407 的 20 个正式评估于北京时间 2026-11-04 16:27:49 至 16:32:12 完成。所在新增五种子 suite 当前总计 100/100 训练与正式评估完成；本稿仅选择其中 seed3407 的 20 个任务。

本审计不重新验证原始 MRI 几何、预训练患者重叠或外部独立性。新增种子属于既有结果查看后的回顾性扩展，应在论文中保留探索性与种子选择不确定性的说明。完整路径、状态计数、五fold manifest SHA256及完成时间见 source_audit.json。
