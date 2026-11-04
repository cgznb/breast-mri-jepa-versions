# CoRe-WM 论文对比实验复现审查

审查对象：`paper/2610.25333v1.pdf`。核查日为2026-11-02；arXiv abs、HTML和API均实时获取成功，正文和关键PDF页文本保存在本目录，网络来源与哈希见JSON凭据。

## 结论

可以借鉴其比较层次，并在我们的协议下重实现多数比较方法；目前不能承诺精确复现其表内数值。CoRe-WM本体的完整官方代码尚未被核实公开。论文p9写的是“Code for preprocessing, training, evaluation, and figure generation will be made publicly available upon publication.”；arXiv正文/abs无作者代码链接，针对论文全名、arXiv号及“CoRe-WM breast”的GitHub仓库搜索没有定位到官方代码。这是“未核实”，不证明互联网上绝不存在代码。

arXiv API显示v1为2026-10-18，未给journal_ref或DOI。应称arXiv预印本，不能凭IEEE Transactions on Medical Imaging页眉称已被TMI录用。

## 它的对比逻辑值得借鉴什么

1. **输入时间轴分层（p6、Table I/p7）**：T0静态方法、T0-T1早期方法、T0-T2纵向方法分组，并显式写Available MRI。不能让一个方法用T3来与另一个方法T0作同条件胜负比较。训练期T3目标与推理期未来输入也分别标明。
2. **基础结构和已发表方法同时比较（p6）**：ViT及完整两阶段融合、matched 3DResNet和PrefixGRU、三种文献纵向方法。我们不能把NoJEPA/DirectReadout/ForecastMean当作外部SOTA方法。
3. **同患者划分与验证选择（p5-6）**：808例I-SPY2、275阳性；五个互斥外层test folds，各约525训练/121验证/162测试。阈值、模型和超参数只用各fold验证，报告fold mean±SD。
4. **独立跨队列迁移（Table II/p7）**：156例I-SPY1、41阳性；五个来源fold模型预测概率逐患者平均，阈值由来源验证确定，目标队列不拟合。内部CV与外部集成使用不同估计对象，论文有区分。
5. **主结果之外检查概率、亚型与机制（p7-9）**：验证集Platt校准、non-pCR为事件的DCA、受体亚型分层、条件置换、冻结状态对实际体积变化的probe。

## 它的模型和我们并不相同

- p5§III-F：先不使用pCR标签预训练，再冻结所有模型权重，pCR仅在class-balanced logistic regression阶段出现；一个readout汇总所有患者/所有决策点的prefix摘要。
- p4-5：最终FLR使用FRS未来响应状态的initial/current/prefix mean/recent revision/cumulative displacement。FRS显式输入9维ROI几何以及患者/治疗条件，不能把性能增益简单归为MRI latent forecasting。
- p5预处理：8通道32×96×96，包含pre/early/late、两种subtraction、peak enhancement、washout和binary lesion region；T0病灶中心裁剪，区域来自FTV mask，缺失时enhancement定位，另取9维几何。
- p6：3D残差网络16/32/64/128通道，192维visit表示，三层4头Transformer，64维响应分支，EMA0.996；AdamW5e-5、batch32、最多100epoch、patience4；未来目标权重2/1/.5。
- 我们现有结果是976患者中的801开发、175固定内部holdout；每个模型seed44的五折概率集成后计算指标。不能拿自己的集成0.7507直接与其CV折均值0.7620作胜负判断。

## 对比清单及复现边界

| 方法 | 论文输入/时点 | 我们如何使用 | 边界 |
|---|---|---|---|
| 3DResNet | 原生8通道，T0 | 当前ROI下建立影像单独及影像+clinical两个matched版本 | 结构可实现，当前3通道版本是适配，不能称原论文数值复现 |
| PrefixGRU | shared3D encoder+ordered GRU，T0-T2 | 当前已观察prefix聚合，补GRU+相同clinical条件 | 必须前缀mask，保持checkpoint选择与输入字段一致 |
| BreastDCEDL ViT及两阶段融合[23] | MinCrop，MAE初始化，T0；完整clinical+geometry | 保留官方路径的独立复现，并另做matched-input适配 | MAE来源、MinCrop几何、两阶段特征拟合需代码核对 |
| Jing/Wang dual-task[33] | 三通道early-subtraction+clinical，T0-T1 | 机制最接近，应优先纳入正式文献对比 | T0-T2扩展属于适配；数据和未来目标限制要一致 |
| BSTNet[32] | peak-postcontrast四visit temporal-order预训练，再改prefix pCR head | 有官方代码时优先使用；预训练只用开发/训练患者 | CoRe-WM作者已经改了其head，不能将此等同原作原封复现 |
| Janickova temporal representation learning[34] | MIP、重建/时间连续/应答对齐 | 有训练代码和完整损失时纳入 | 不直接用三通道体积替换MIP却保留原名；监督定义需要原文审核 |
| CoRe-WM本体 | 8通道+geometry+profile/therapy，outcome-free→frozen LR | 可按文稿重实现，需先明确缺失细节 | 尚无核实官方repo、原808队列/划分、seed标识与完整几何/IRG定义 |

各方法精确清单保存在`comparison_inventory.json`；文献[32]-[34]题目及arXiv号见PDF p10。

## 不应照搬的地方

- **一定补强clinical-only和clinical+geometry基线。** Table III/p8中image-only JEPA .5816，加profile至.7135，是最大增量+.1319；treatment再加后.7115，增量为负；geometry后.7380。FRS显式依赖几何，故Logistic/XGBoost(current/Δgeometry)+clinical及geometry-GRU是必要参照。p6并未明确裸ResNet/PrefixGRU接入全部相同条件，不能假定所有方法具有同等临床/几何输入。
- **不要选择弱对照来保证领先。** Table I中3DResNet/PrefixGRU约.54/.58提示应检查预训练、clinical输入及验证超参预算。不能据数值本身断定论文方法有错，也不能据此照搬一个可能欠拟合的实现。
- **累计消融只给有顺序的边际差。** Table III后一行相对前一行，加入顺序会影响归因；应保留我们已经匹配的NoJEPA→DirectReadout，并另加classifier-width-matched、copy-state、forecast-detach等控制。
- **外部并非全部领先。** Table II中two-stage dual-task在T0-T1 AUROC .7635，高于CoRe-WM .7321；其差−.0314的CI跨0、p=.5167。CoRe-WM在I-SPY1的AUROC还从T0 .7396降至T2 .7082。应完整保留这种反例。
- **统计口径不能搬混。** 文中fold SD衡量分区波动，不是seed SD；其pooledOOF、foldmean、校准后OOF、外部五模型概率集成是不同对象。我们五fold在同175人上评估，不能当五个独立test folds。
- **不要用多项未校正p当总体显著性。** 论文有若干paired AUROC p，但正文没有完整命名检验/多重校正计划；我们的主要9项/次要81项Holm与患者配对必须继续遵守。其TNBC longitudinal claim明确标注未多重校正、探索性（p8）。
- **校准必须与待部署打分形式匹配。** 原文对各fold单模型在该fold验证集拟合Platt再应用互斥test；这不能直接套成用单模型OOF给五模型均值校准。当前我们更应独立设计ensemble校准集或诚实作为敏感性分析。
- **条件置换是模型依赖，不是因果治疗效应。** 原文p9也明确不估计未分配疗法结局。T0 eventual assigned-arm是否当时已知需审计。
- **体积probe是预训练目标一致性检查。** p9承认体积变化就是IRG一部分，不是独立发现新生物标志物。我们若用forecast error/copy baseline，应独立holdout、同EMA坐标比较，不混不同训练空间。

## 当前最合适的落地顺序

先锁定本研究患者/seed44/folds/prefix/概率集成/阈值规则，再形成：clinical-only → strong 3DCNN(+clinical) → GRU/Transformer(+clinical) → 两到三种真实文献纵向方法 → 我们方法与内部消融。优先dual-task、BSTNet及一项有完整代码的2024-2026影像/临床强基线。CoRe-WM本体列为等待代码或明确标注paper-based reimplementation，不先许诺可复现其原始数值。

原生输入和matched输入可以作为两个互补比较面板，但每行都要列通道、临床字段、geometry、预训练患者/权重、可见时点以及作者模型/适配区别。全流程不得把holdout pCR用于生成条件、阈值选择或任何额外训练目标。若未来重建8通道/FTV裁剪，必须使用配套MRI JSON、mask header、T0网格和变换，不能仅换metadata。
