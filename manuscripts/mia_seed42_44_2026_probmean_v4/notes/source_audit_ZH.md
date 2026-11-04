# 三种子预测来源审计

已实时核验 seed42、seed44、seed2026，四模型、五折，共60个目标运行。全部目标矩阵任务为 completed，60个 evaluation_done.json 均 completed=true。完整suite矩阵分别为60/60、60/60、80/80正式评估完成；这里仅选取每suite对应目标seed的20个运行。

| Seed | 来源suite | 目标运行 | 正式评估完成（北京时间） |
|---|---|---:|---|
| 42 | medium976_t0_t2_no_tf_add3seed_17_29_42_v1 | 20 | 2026-10-31 00:55–00:59 |
| 44 | medium976_t0_t2_no_tf_cv5_multiseed_v1 | 20 | 2026-10-29 22:32–22:37 |
| 2026 | medium976_t0_t2_no_tf_add4seed_30_43_809_2026_v1 | 20 | 2026-11-03 12:19–12:23 |

三个suite均位于 `/projects/xts0030/public/cxx/`。准确路径、完成时刻与共享manifest哈希见 source_audit.json。

核验通过：180份登记评估报告的文件大小和mtime_ns身份；60个manifest、60个输入配置、60个training_done、60个run_identity及60个codec资产身份；5,520次记录内runtime文件身份检查。每次运行记录92份runtime文件，三个suite共276份独立文件按相对路径计算SHA256，内容全部一致。

同模型、同fold下，seed42/2026相对seed44的40次训练config.json比较严格只有顶层seed不同。模型设置、优化器、训练预算、诊断设置、输入数据、fold划分、codec路径均保持匹配。五折manifest为同路径同哈希，均为976人队列；codec资产为同路径、大小与时间身份。没有重新训练，也没有加载检查点payload。

180个模型×种子×fold×时点测试向量均为同一批175人、56位阳性；患者排序、标签、manifest中test成员及pCR标签逐项匹配。测试标记均为内部downstream holdout、independent_test=false，不能称为外部验证。三个重复时点及不同种子不会增加独立患者数。

本审计只读已保留的预测与冻结身份；未重做MRI、预训练重叠或配准解剖核查。本文档和JSON仅记录聚合计数、公共路径与文件哈希，不含患者ID或患者级预测。
