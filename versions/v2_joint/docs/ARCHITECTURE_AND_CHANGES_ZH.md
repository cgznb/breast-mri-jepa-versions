# 架构、源码映射与修改说明

## 1. 交付范围与来源

本包参照 `cgznb/breast-mri-vla-jepa` 在 `[commit reference omitted]` 的公开数据合同和网络结构，提供独立 `response_jepa` 名称空间。它不是上游的完整 checkout；不覆盖原 `train.py`，不兼容原 checkpoint，没有打包其真实训练数据或实验权重。为严格历史复现保留原仓库并单独运行。

网络组件基于 PyTorch 原语独立实现。VLA-JEPA 的共享转移/任务查询、JEPA 的无梯度目标、VICReg 的方差协方差公式和残差动态是方法参考，不应写成“完整复制了原论文官方代码”或“复现了原论文全部设置”。没有下载任何第三方基础模型权重。

参考入口（方法来源，不作为本包性能证明）：

- 上游项目：https://github.com/cgznb/breast-mri-vla-jepa
- VLA-JEPA：https://github.com/ginwind/VLA-JEPA
- VICReg：https://github.com/facebookresearch/vicreg
- V-JEPA 2：https://github.com/facebookresearch/vjepa2
- PyTorch：https://pytorch.org/docs/stable/

## 2. 总体数据流

```text
合法已观察 MRI                    合法临床字段/Arm
      │                                 │
多尺度 3D Encoder                 逐字段 token + Arm token
      │                                 │
每检查 256×128 空间 tokens                │
      │                                 │
32-query attention pooling ──────────────┘
      │
观察上下文 + 三组内部转移 queries + pCR queries
      │
两层共享 Fusion（观察context双向；query有序因果）
      ├──────── pCR queries → direct head → p_direct
      │
      └──────── 转移条件 z01,z12,z23
                         │
最后真实观察 ─冻结目标encoder→ S_anchor
                         │
         四层时间因果残差世界预测器
                         │
            Ŝ_anchor+1 → ... → Ŝ3
                                 │
                      terminal attention pool
                      +临床/Arm/landmark
                                 │
                            p_forecast
                                 │
         p_fused=(1-g)*p_direct+g*p_forecast

真实未来 MRI → 固定目标 encoder → 仅未来损失目标
真实前驱状态 → TF辅助路径 → 仅TF损失
```

输出请求 query_mask 不参与内部患者状态推断，不能改变 pCR。稀疏请求只改变最终返回的 latent 掩码；预测路径内部必要的中间阶段仍会计算。

## 3. 默认编码器层级与尺寸

下面 V 表示当前编码块中真实检查的数量，不是时间维；每次检查独立编码，没有把不同阶段放入会混合统计的 BatchNorm。

| 位置 | 结构 | 输出 |
|---|---|---|
| 输入 | 三个DCE相位 | V×3×32×128×128 |
| stem | Conv3D(3→24,k3×5×5,stride1×2×2)+GroupNorm+GELU | V×24×32×64×64 |
| stage0 | 2个depthwise3D/pointwise残差块 | V×24×32×64×64 |
| stage1 | stride2降采样→48，再2个残差块 | V×48×16×32×32 |
| stage2 | stride2降采样→96，再2个残差块 | V×96×8×16×16 |
| stage3 | stride2降采样→128，再2个残差块 | V×128×4×8×8 |
| lateral | stage2/3分别1×1投影128，汇聚至4×8×8，相加/√2 | V×128×4×8×8 |
| 空间建模 | flatten+learned position+2层PreLN attention+LN | V×256×128 |
| context resampler | 32个可学习query交叉注意力 | V×32×128 |

world目标保留256个空间token，只有共享历史fusion用较少的32个context token，避免全历史高分辨率attention开销同时爆炸。空间提升不是病灶分割，也没有被证明捕获特定生物机制。

每个 PreLN attention block 的FFN宽度为4×dim；各层单独构造初始化，不共用同一层参数。实际默认模型包含教师和A阶段辅助头共4,564,068参数；C full配置可训练参数1,684,355。见 `validation/default_shape_check.json`。总参数数不代表全部都在C训练或部署执行。

## 4. 共享主干的有效长度和掩码

每患者context包括root1、四个visit的32个tokens、4个临床字段token和1个Arm，共134位置。再加入三组×8转移queries和4个pCR queries，共162位置。

缺失/尚未可见检查作为被屏蔽的key，不携带未来图像。context内允许双向整合已经合法获得的信息，但context不能读取后面的query。查询按预定顺序有序读取；pCR可以利用共享转移特征。

临床字段不再压成单个四维数值向量token，每个字段有独立身份和值/可见性编码。没有语言模型参与，不需要LLM将4个结构化临床字段改写成文本。

内部转移queries始终三组存在，独立于请求返回哪些未来阶段；输出层的query_mask只控制返回。

## 5. 世界模型：时间因果、空间双向、残差状态

一个时间块为 `[8个z token,256个源状态token]`；块内双向，块间因果，root始终合法。完整三个块为793个位置。输出对应源状态的下一阶段增量：

`S_hat[j+1] = S_source[j] + delta[j]`。

残差模式的delta最终投影权重/偏置零初始化，所有初始预测严格等于源copy；学习到的提升应针对这个明确起点评价。absolute模式去掉残差，作为独立消融，不能用残差B checkpoint直接切换语义。

free rollout从当前最后真实检查的固定教师状态开始，多步调用同一世界预测器，将预测状态接回后续时间块，不重新编码不存在的未来MRI。当前实现是阶段索引0–3，没有连续物理时间或逐周期真实药物动作。

TF路径独立取得真实源/目标，全部no_grad，不传入pCR。TF与free共享预测器，只有loss路径的数据来源不同。

## 6. pCR读出与梯度

直接分支是4个pCR query均值→LN→Linear。未来分支对预测末态做attention pooling，融合临床、Arm和当前landmark token，再得到二分类logit。

full中pCR损失可通过未来读出反传到预测器和共享转移queries；不会更新固定目标encoder。`stopgrad_forecast`只对这条任务梯度使用detach，世界损失仍正常训练。同stage临床字段mask保持合法。

`copy_readout`只把未来分类分支的输入替换为当前anchor。它仍保留全套free rollout、world losses、同一B初始化和same-size heads，因此是读出侧的预测变化对照，不是假装训练过的copy世界模型。

`full_encoder_ft`仅更新在线观察encoder，并用观测状态对齐固定目标；目标encoder不EMA漂移。默认full冻结在线encoder，先获得清楚的固定坐标结论。

## 7. A阶段为什么不再做粗糙像素重构

A使用同次影像的mask latent prediction。没有从2×4×4经过1×1卷积再插值到整幅MRI的重构目标，也不要求网络从96个低分辨率强度参数恢复全部影像。

方差项从物理患者轴、同阶段/同位置检查；pooled covariance在autocast禁用上下文计算FP32矩阵乘法。方差不充分时可能表征失效，但有非零方差也不证明医学有效，应使用冻结探针和外部任务检验。

## 8. 与原仓库的逐模块映射

| 原位置 | v2位置 | 实质变化 |
|---|---|---|
| encoder.py | model.py / MultiScaleEncoder、LegacySmallEncoder | 多尺度局部编码、256token、学习式context汇聚 |
| model.py / RawVLAJEPA | model.py / ResponseJEPA | 请求掩码解耦，直接+预测末态读出 |
| TimeCausalWorldPredictor | model.py / WorldPredictor、LinearWorldPredictor | 零初始化残差、同一预测器自主多步 |
| compute_loss/teacher_forcing_loss | objectives.py | A同次mask，B固定坐标，C预测状态任务梯度，free rollout监督 |
| data.py/contracts.py | data.py/contracts.py | 保留原ROI32合同；输入/目标严格分离；独立实现兼容读取 |
| training.py | engine.py | A/B/C、严格warm start、epoch恢复、数据/源码/环境指纹 |
| diagnostics.py | diagnostics.py/metrics.py | 梯度探针、fixed-space copy、patient-paired bootstrap |
| scripts与配置 | experiments.py、scripts、configs | 18条件、51模板、A/B依赖复用、明确实验计划 |

完整历史模型不在本包内。`LegacySmallEncoder`不是与历史torch state_dict兼容的“旧checkpoint加载器”。

## 9. 原则性单元测试

测试直接验证而非只靠文档说明：未来请求不改变pCR；未来真实目标不改变合法forward；teacher/future像素无反向梯度；time-block因果掩码；未激活NaN源隔离；exactcopy初始化；未来task梯度到world以及detach对照；缺失中间阶段free supervision；T3零world loss；物理患者轴方差；缓存逐值一致；训练集标准化；伪造未来路径不被推理读取；严格CPU恢复；A/B/C目标hash不变。

具体测试数量、运行环境与日志在 `validation/`。这些都是工程正确性检查，不是临床泛化证据。

## 10. 尚未实现且不应写进方法结论的内容

没有病灶分割/FTV监督、没有真实解剖/治疗反应语义解耦、没有未来MRI像素生成、没有随机多未来分布或置信度校准、没有天数/剂量输入、没有干预识别。模型复杂度和来源参考也不自动构成CVPR创新。下一步贡献应由实际消融、固定坐标下的预测质量和独立临床任务增益支撑。
