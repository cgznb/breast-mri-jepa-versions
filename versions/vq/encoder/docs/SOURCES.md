# 来源与许可范围

本包从本地乳腺 MRI / I-SPY2 VLA-JEPA 实验源码整理。基础 raw VLA-JEPA 医学适配来自 [cgznb/breast-world-model-v3-pcr](https://github.com/cgznb/breast-world-model-v3-pcr) 的本地开发版本，模型思想参考 [ginwind/VLA-JEPA](https://github.com/ginwind/VLA-JEPA)。这些链接说明来源，不指定可逐值复现本次本地版本的上游 commit；本包不含 Qwen3-VL/V-JEPA2 预训练模型或其权重。

VQ 连续前端保留 `MeWM → cgznb/symm-fm → 本地 codec` 的来源链，实际派生文件为 [`src/mri_vla_jepa/vq_encoder.py`](../src/mri_vla_jepa/vq_encoder.py)，保留 CC BY-NC 4.0，完整归属和改动见 [CODEC_NOTICE](../licenses/CODEC_NOTICE.md)。代码只用 encoder 与 pre_quant，未接入量化器、码本、解码器或 GAN 判别器。

其他原创新增代码保留现有 MIT 及版权声明，见 [LICENSE](../LICENSE)。整体许可表达式为 `MIT AND CC-BY-NC-4.0`，不能把整个包视为纯 MIT。依赖库各自保留其许可。代码许可不授予 I-SPY2 影像、临床表或外部预训练权重的使用权；正式 codec 权重需另行取得。

本次整理保留训练模型与损失实现、已有配置及测试。历史机器路径与实验说明是来源记录；新机器的数据、缓存和初始化权重路径应在新生成的配置中替换。发布包不附患者数据、正式权重及逐患者预测行；聚合结果的位置见本包 README。
