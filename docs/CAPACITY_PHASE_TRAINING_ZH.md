# VLA-PCRV1 容量、相位与训练方式实验

本轮固定原来的764人训练、102人验证、0人测试。读取现有ROI32 NPZ，沿用配套VQ权重与标准化，不重新划分患者。结果均属于验证集结果。

## 配置与训练机制

训练配置新增：

```yaml
prediction_training: teacher_forcing  # 默认hybrid，兼容原配置
selection_metric: f1                 # 默认nll，兼容原配置
f1_threshold: 0.5
model:
  model_size: medium
  mri_phases: first_post_aqc1
```

teacher_forcing训练用真实状态作前驱，一次并行世界模型前向预测三个转移，预测同时用于JEPA和pCR。pCR读取真实前驱有效的最远预测；缺目标影像只取消该目标的JEPA监督。无有效转移则回退最新观察状态。验证和公开预测始终自主递推。

hybrid保留原有自主分类路径和teacher-forcing辅助损失。正式TF实验的rollout权重为0，hybrid为0.5，因此两者是完整训练方式的对照。两者JEPA权重均为0.5，pCR权重为1。

|规模|token维度|编码/融合/世界模型层数|头数×每头维度|FFN|三相可训练参数|
|---|---:|---|---|---:|---:|
|small|128|2/2/2|4×32，原Transformer|512|1,301,252|
|medium|768|8/8/8|16×64|3072|191,637,508|
|large|1024|12/12/12|16×64|4096|458,027,012|

未指定model_size的旧配置保留自己的结构；指定预设后结构字段必须与预设一致。独立attention_head_dim与ffn_dim用于扩展注意力内部宽度和前馈层。

相位可选all、pre_aqc0（增强前）、first_post_aqc1（首次增强后）、metadata_late（指定晚期相位）。NPZ仍为三相，encoder内部统一选相位。单相生成的C=1，三相C=3；导出记录phase_order。

## 48次训练与F1

矩阵为3种规模×4种相位×TF/hybrid×种子17/43。所有组独立初始化VLA，冻结相同VQ前端；保持相同患者划分和训练batch。

每次最多200 epoch，连续50 epoch验证F1没有严格提升则停止。pCR=1为正类，概率>=0.5判阳性。all_observed验证中，每名患者有n条有效前缀，每条权重1/n，再累计TP/FP/FN计算F1。并列保留较早best，首次F1=0也会保存best。

## 双卡队列

从项目根目录运行，使用已有环境：

```bash
PYTHONPATH=versions/v1_evolution/src \
versions/v1_evolution/.venv/bin/python \
versions/v1_evolution/scripts/run_capacity_phase_matrix.py \
  --output /personal_fast_storage/VLA-PCRV1/private_outputs/capacity_phase_tf_hybrid_20261018 \
  --gpus 0,1 --detach
```

首次运行会生成48组配置，双卡测量24类结构与训练组合的真实前向、反向、优化器、验证和诊断峰值，并比较同类及混合任务并发吞吐。每卡预留至少8GiB和10%显存。默认batch16；若单任务预检OOM，正式冻结前统一回退batch。正式训练仅减少同卡并发，不自动改变模型或batch。

预检用独立模型，不给正式训练传递参数或优化器状态。预检完成后复制runtime源码与脚本，冻结配置、manifest、codec及数据身份。子进程通过CUDA_VISIBLE_DEVICES绑定GPU UUID，内部使用cuda:0。原工作区后续编辑不会改变运行中的冻结版本。

状态与恢复：

```bash
PYTHONPATH=versions/v1_evolution/src \
versions/v1_evolution/.venv/bin/python \
versions/v1_evolution/scripts/run_capacity_phase_matrix.py \
  --output /personal_fast_storage/VLA-PCRV1/private_outputs/capacity_phase_tf_hybrid_20261018 --status
```

用同一命令的--resume --detach替换--status恢复队列；已有存活任务按PID和进程起始时间接管，避免重复启动。训练从last.pt恢复优化器、EMA、随机状态、患者采样位置与早停计数。

输出目录中的matrix.json记录每个任务及GPU状态，profiles.json和throughput.json记录准入依据，summary.json和summary.md汇总两个种子与各时点指标。每次训练正式结束后评价best/last分类及世界模型表现，重新核对best F1。48次均完成且评价通过才将矩阵标为completed。
