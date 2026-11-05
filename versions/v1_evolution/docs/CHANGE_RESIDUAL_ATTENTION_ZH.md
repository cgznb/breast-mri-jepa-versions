# 变化条件的残差 attention 分类读出

这是新增的可选 pCR 读出，实验效果尚未验证。入口为 `model.spatial_readout: change_residual_attention`。

## 输入与计算

以 Medium 为例，当前状态 `current`、预测终态 `terminal` 和预测变化 `change` 都是 `[B,32,768]`；四个 pCR-query 的均值 `query` 是 `[B,768]`。默认 `change = terminal - current`。

对每个空间 token 计算：

```text
h_i = tanh(W_current LN(current_i)
         + W_change LN(change_i)
         + W_query LN(query))                 # 32 维
a_i = softmax_i(w^T h_i)
lambda = sigmoid(mix_logit)                   # 全局可学习标量
weight_i = (1 - lambda) / N + lambda * a_i
```

`query` 必须在 `tanh` 之前参与交互。若把一个 query 标量直接加到所有位置的最终分数上，该标量会被 softmax 抵消，无法改变空间权重。`LN(change)` 归一化变化特征，因此 scorer 主要读取变化模式；它本身不是变化幅度测量器。

使用同一组 `weight` 汇总原始终态和变化：

```text
terminal_summary = sum_i weight_i * terminal_i
change_summary   = sum_i weight_i * change_i
features = concat(query, terminal_summary, change_summary)
logit = Linear(LayerNorm(features))
```

Medium 的分类器输入仍为 2304 维。混合池化等价于 `mean(value) + lambda * (attention(value) - mean(value))`，即在 mean 基础上学习空间修正。

## 初始化与梯度

`w` 零初始化，`lambda` 默认初始化为 0.1：初始 `a` 为均匀分布，读出在数值容差内等同 mean；score 层仍能收到梯度。第一步投影层和 gate 梯度为零，score 更新后它们才开始学习。

不能同时把残差系数设为 0 并使 attention 为均匀分布，否则 residual 和 scorer 两侧都可能没有梯度。配置拒绝 0 或 1 的混合初值。`lambda` 可在训练中接近 1，因此这种设计不保证始终保留固定比例的 mean。

scorer 只使用合法 observed-prefix 的 query、当前 EMA 状态和自主预测终态。未来真实 MRI 只作原有监督目标。`forecast_detach` 仍切断分类损失到预测终态的梯度；当前 EMA 参考仍不参与反传。若配置 `pcr_delta_reference: t0`，变化参考改为 T0，但 scorer 的 `current` 输入仍是最新观测状态。

## 配置与运行范围

将下面字段合并进一份复制出来的、已核验的 forecast 训练配置：

```yaml
model:
  fixed_transition_queries: true
  pcr_readout: forecast
  spatial_readout: change_residual_attention
  pcr_delta_reference: current
  pcr_attention_dim: 32
  pcr_attention_mix_init: 0.1
```

这是配置片段，完整模型结构、数据划分、预处理、训练损失和选模协议沿用对照配置。新增字段会随模型配置保存，旧 `mean` / `attention` 模式保留原来的参数结构。新模式需要 state-based pCR readout。

当前实现位于 `versions/v1_evolution`，不会自动更新 `planned_tasks` 下既有的隔离 runtime。正式实验应复制为一个新的独立 runtime 和输出目录，以匹配的 mean、原 attention 对照验证；不能把此处代码修改视为旧实验已使用新分类头。这里没有启动训练。

已为 Medium976 准备独立的六种子五折任务：`/projects/xts0030/public/cxx/planned_tasks/medium976_t0_t2_change_residual_attention_6seed_v1`。种子为44、2026、42、7、21、3407，共30次训练。该任务从原实验runtime移植此分类头，其他源文件与匹配配置保持一致，四卡命令与验证记录见该目录 `README.md`。

从本版本目录运行 CPU 检查：

```bash
.venv/bin/python ../../tools/run_cpu_tests.py --project . -- -q tests/test_pcr_spatial_readout.py
```

诊断中的 `pcr_spatial_attention` 是实际用于池化的混合权重；`pcr_spatial_learned_attention` 是混合前的 softmax；`pcr_spatial_attention_mix` 是当前混合系数。它们均已 detach，不能直接视为解剖定位或预测不确定性。

## 对照要回答的问题

先比较 mean、原 attention 和此模式。在此基础上，用“相同混合、相近容量但仅看终态”的 scorer 区分残差混合与变化条件的贡献，再去掉 query 判断患者条件是否有额外价值。固定种子、划分、训练预算和验证集选模方式，分别检查 T0/T1/T2 的区分度与概率质量。
