# 在本机逐行调试 VLA-PCRV1

本机当前使用 `versions/v1_evolution`，参考方案为
`vq_forecast_history_residual_free_encoder64`。本次学习目标是训练阶段，
先从正式入口跟完初始化、训练数据、损失、反向传播、AdamW 和 EMA 更新。

## 1. 正式训练入口：只执行一次更新

在 VS Code“运行和调试”中选择 **`[训练] VLA-PCRV1：train.py 单步`**，按 F5。
启动时输入本次的新输出目录名（如 `run01`，再次运行改为 `run02`），
结果位于 `private_outputs/training_debug/` 下。程序会在入口暂停，
可以先设好下面的断点，再按 F5 运行。

调用链是：

```text
versions/v1_evolution/train.py
  → mri_vla_jepa.cli.main
  → training.train
  → training._train
  → model.compute_loss
  → backward → 梯度裁剪 → AdamW → EMA
```

这个入口从本机配置与预训练 VQ codec 初始化，其他网络部分重新初始化。
本地调试配置为 `versions/v1_evolution/local_configs/encoder64_seed17_train_debug.yaml`：
batch=1、accumulation=1、FP32、同步加载、关闭诊断探针、最多一个 epoch；
启动参数 `--stop-after 1` 将运行限制为一次优化器更新。
保留 `all_observed` 采样协议和原模型、损失配置，训练时每位患者的 landmark
由采样器选择；某个样本没有未来检查时，相应辅助损失可为零。

正式训练路径会保存调试 checkpoint 和日志。已有 `last.pt` 的输出目录会拒绝
作为新的训练运行启动，故重复调试需要换目录名。原正式训练 YAML 保留原值。

| 断点 | 观察内容 |
| --- | --- |
| [cli.py:69](../versions/v1_evolution/src/mri_vla_jepa/cli.py#L69) | 进入 `train-vla-jepa` 分支，读取配置与数据清单 |
| [training.py:325](../versions/v1_evolution/src/mri_vla_jepa/training.py#L325) | 创建模型，随后加载预训练 VQ codec |
| [training.py:342](../versions/v1_evolution/src/mri_vla_jepa/training.py#L342) | 冻结策略已设置，创建 AdamW |
| [training.py:474](../versions/v1_evolution/src/mri_vla_jepa/training.py#L474) | 生成患者及 landmark 任务，读取输入与监督 |
| [training.py:479](../versions/v1_evolution/src/mri_vla_jepa/training.py#L479) | 进入 `compute_loss`，同时观察 `inp` 和 `sup` |
| [training.py:485](../versions/v1_evolution/src/mri_vla_jepa/training.py#L485) | 反向传播，随后检查参数的 `.grad` |
| [training.py:490](../versions/v1_evolution/src/mri_vla_jepa/training.py#L490) | 梯度裁剪 |
| [training.py:499](../versions/v1_evolution/src/mri_vla_jepa/training.py#L499) | AdamW 更新训练参数 |
| [training.py:504](../versions/v1_evolution/src/mri_vla_jepa/training.py#L504) | EMA 更新 teacher |

`scripts/debug_vla_jepa.py` 的 `main()` 是辅助单 batch 演示入口。
正式训练使用上面的调用链，包含采样、梯度累积和保存等完整训练逻辑。

## 2. 辅助脚本：单独观察一批数据

已分别为打开 `/personal_fast_storage` 和打开 `/personal_fast_storage/VLA-PCRV1`
配置 `.vscode/launch.json`。在“运行和调试”中选择：

| 配置 | 执行内容 |
| --- | --- |
| `[1] VLA-PCRV1：T0 单例推理` | 使用 seed17 已训练权重，读取一位验证患者，仅做前向 |
| `[2] VLA-PCRV1：单 batch 损失` | 使用同一权重，读取一位训练患者，计算完整损失 |
| `[3] VLA-PCRV1：一次 backward / AdamW / EMA` | 在内存中更新一次模型，不写模型文件 |

选择上表某项，按 F5。程序会在 `debug_vla_jepa.py` 内的 `breakpoint()` 暂停，
此时模型和数据对象已准备好，下一步开始装配 batch。

配置明确指定现有 `.venv/bin/python`、本版本 `src`、真实 manifest 和参考权重；
默认 `batch_size=1`、T0、CUDA、FP32。FP32 方便观察数值，正式实验使用 BF16，
因此数值可能略有不同。VS Code Python Debugger 扩展提供调试器，无须为了 F5
另向项目环境安装 debugpy。

常用操作：F9 添加断点；F10 执行当前行；F11 进入函数；Shift+F11 返回调用者；
F5 运行到下一个断点；Shift+F5 停止。建议直接在项目函数里设断点，
在两个断点之间按 F5；F11 可能先进入 PyTorch 的调用包装。

## 3. 可选：跟着 T0 推理理解数据流

入口为 [debug_vla_jepa.py](../versions/v1_evolution/scripts/debug_vla_jepa.py)。
先在 `inp = inp.to(args.device)` 这一行设断点，此时 batch 已装配，尚未搬到 GPU。

在“监视”或“调试控制台”中检查：

```python
inp.images.shape
inp.observed_mask
inp.query_mask
inp.clinical.shape
inp.clinical_mask
inp.arm_id
inp.arm_mask
sup
```

T0 单例应看到：

```text
inp.images.shape  = [1, 4, 3, 32, 128, 128]
inp.observed_mask = [[True, False, False, False]]
inp.query_mask    = [[False, True, True, True]]
inp.clinical.shape = [1, 4]
sup = None
```

四个纵向槽位表示 T0–T3，三个影像通道表示 DCE 相位。预测时只读合法可见的
真实影像；未来槽位为零。现有 NPZ 已预处理，`data.py` 的 ROI32 分支直接返回
原数组，不再次标准化或缩放。

按以下顺序设断点并继续：

| 位置 | 要理解的问题 |
| --- | --- |
| [data.py:355](../versions/v1_evolution/src/mri_vla_jepa/data.py#L355) `RawMRIStore.batch` | 如何分开合法输入 `inp` 和训练监督 `sup`？ |
| [model.py:153](../versions/v1_evolution/src/mri_vla_jepa/model.py#L153) | 为什么只编码 `images[observed_mask]`？ |
| [encoder.py:161](../versions/v1_evolution/src/mri_vla_jepa/encoder.py#L161) `VQVisitMRIEncoder.forward` | 三个相位如何复用单通道 VQ 编码器？ |
| [encoder.py:176](../versions/v1_evolution/src/mri_vla_jepa/encoder.py#L176) | 相位特征如何拼接、池化、投影成 tokens？ |
| [model.py:167](../versions/v1_evolution/src/mri_vla_jepa/model.py#L167) | MRI、临床、Arm 和 query 如何拼接？ |
| [model.py:176](../versions/v1_evolution/src/mri_vla_jepa/model.py#L176) | padding mask 和 causal mask 控制哪些信息可见？ |
| [model.py:194](../versions/v1_evolution/src/mri_vla_jepa/model.py#L194) | 如何从当前状态自主预测至 T3？ |
| [model.py:340](../versions/v1_evolution/src/mri_vla_jepa/model.py#L340) | 预测的 T1、T2 如何继续作为下一步输入？ |
| [model.py:60](../versions/v1_evolution/src/mri_vla_jepa/model.py#L60) `TimeCausalWorldPredictor.forward` | 历史差分和残差转移在哪里计算？ |
| [model.py:200](../versions/v1_evolution/src/mri_vla_jepa/model.py#L200) | 如何组合 pCR query、预测 T3、相对当前状态的变化？ |
| [model.py:203](../versions/v1_evolution/src/mri_vla_jepa/model.py#L203) | 分类特征如何变成 logit？ |

每处断点先看当前行的输入，按一次 F10 后再检查这一行产生的变量。
例如停在 `features = torch.cat(...)` 前，`features` 还未完成赋值。

该配置的关键形状如下；`O` 是 batch 中所有可见检查的总数，T0 单例时为 1：

```text
选中的 MRI                    [O,3,32,128,128]
每相 encoder64 特征           [O,64,8,32,32]
三相特征拼接                  [O,192,8,32,32]
池化                          [O,192,2,4,4]
MRI tokens                    [O,32,128]
融合前 packed                 [1,166,128]
dynamics_features             [1,4,8,128]
pcr_features                  [1,4,128]
自主状态 states               [1,4,32,128]
pcr_state_features            [1,384]
pcr_logit                     [1]
pcr_probability               [1] = sigmoid(pcr_logit)
```

`pcr_features` 中的 4 是四个分类 query。`states` 中的 4 才是纵向阶段。
预测的是未来潜状态，并非未来 MRI 像素。

`model(inp)` 返回前会移除内部的 `autonomous_states` 缓存，
因此应在 `_backbone` 内查看 `states`。需要公开的未来潜状态时，可在前向结束后
于调试控制台执行 `forecast = model.forecast(inp)`，检查
`forecast['state_prediction'].shape`；未请求的阶段会置零。

## 4. 辅助脚本：看完整损失

停止第一轮，选择第二个启动配置。数据装配时 `sup` 含 pCR 标签、真实未来 MRI
及相应 mask，供损失计算使用。它与模型推理输入分开：真实未来 MRI
不参与分类分支的自主预测。

在 [model.py:433](../versions/v1_evolution/src/mri_vla_jepa/model.py#L433)
的 `compute_loss` 内设断点，重点看：

```text
task     = pCR 的 BCE
world    = teacher forcing 的潜状态 L1
rollout  = 自主多步预测的潜状态 L1
loss     = task + 0.5 * world + 0.5 * rollout
```

`world` 是函数内的变量名，返回的 `terms` 中对应 `jepa` / `world_model` 键。

teacher forcing 使用真实的前驱状态；自主预测逐步使用自己生成的前驱状态。
缺少真实目标的阶段由监督 mask 排除，不应仅凭张量有四个槽位就认为都有监督。
`compute_loss` 返回后，检查 `terms`、`loss.item()`、`loss.requires_grad`，
并确认 `rollout` 对应的权重为 0.5。

## 5. 辅助脚本：看一次梯度更新

选择第三个配置，在入口脚本的下列行设断点：

```python
loss.backward()
gradient_norm = torch.nn.utils.clip_grad_norm_(...)
optimizer.step()
model.update_teacher()
```

执行 backward 后、optimizer.step 前检查：

```python
model.pcr_head[1].weight.grad.norm()
model.encoder.projection.weight.grad.norm()
next(model.encoder.frontend.parameters()).requires_grad
next(model.encoder.frontend.parameters()).grad
next(model.teacher_encoder.parameters()).grad
```

该方案冻结 VQ 前端，因此前端无梯度；后续 projection 和 spatial adapter 仍训练。
EMA teacher 不通过 backward 更新，而在 optimizer.step 后由 `update_teacher()` 更新。

这里的 train-step 仅用全新 AdamW 做一次内存演示，不恢复旧实验的优化器、
学习率调度或 RNG 进度，也不保存更新后的权重。每次重新启动仍从原参考权重加载。

之后阅读 [training.py](../versions/v1_evolution/src/mri_vla_jepa/training.py)
的 `_train`，理解批次采样、梯度累积、验证、早停和保存。完整训练应另用
[本机运行说明](LOCAL_RUN_ZH.md) 中的训练入口。

## 6. 使用终端和 pdb

不使用 VS Code 时，从仓库根目录运行：

```bash
cd /personal_fast_storage/VLA-PCRV1
source versions/v1_evolution/.venv/bin/activate
PYTHONPATH=versions/v1_evolution/src python versions/v1_evolution/scripts/debug_vla_jepa.py \
  --checkpoint data/reference_runs/encoder64_seed17/best.pt \
  --manifest data/registered_roi32/manifest.json \
  --split val --landmark 0 --batch-size 1 \
  --mode predict --device cuda --precision fp32 --break
```

到 `(Pdb)` 后：`n` 执行下一行，`s` 进入函数，`r` 返回，`c` 继续，
`p 表达式` 查看变量，`b 文件:行号` 设置断点，`q` 退出。
例如先设置 `b versions/v1_evolution/src/mri_vla_jepa/model.py:153` 再输入 `c`。

查看损失时改为 `--mode loss --split train`；单步更新时改为
`--mode train-step --split train`。显式指定本机 manifest，避免入口脚本的默认路径
指向版本目录下不存在的数据。

这些启动方式不生成患者级输出文件。若把调试控制台内容另存到文件，
使用已被 Git 忽略的 `private_outputs/`。
