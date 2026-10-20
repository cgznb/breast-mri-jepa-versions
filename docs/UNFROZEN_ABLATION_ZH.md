# large / all 解冻优化实验：纵向与 T0 对照

当前比较计划包含 **10 个配置 × seed 17、43，共 20 次训练**：原七项纵向优化、两项仅 T0 输入的对照，以及 2026-10-20 按用户要求补充的公共模板解冻基线。原 18 个任务保留既有冻结运行快照；两份基线已在独立目录 `private_outputs/large_all_tf_unfrozen_baseline_20261020` 中加入等待队列，复用原种子的 manifest 与 codec，等待原队列完成后自动接续运行。所有任务均从原 `data/pretrained/codec.pt` 初始化，使用 `encoder_training=finetune` 解冻参与前向的 VQ 前端；不加载旧实验训练后的权重。前端和其余网络的公共学习率均为 `1e-4`。

## 固定设置与实验配置

公共模板为 large / all / teacher forcing，32 个空间 token，固定 transition queries、world residual 和 history delta。公共参数为 WD `0.01`、dropout `0.3`、EMA `0.995`、均值池化、`terminal-current`。batch 16，BCE + `0.5 × JEPA`，最多 200 轮、早停 patience 50。checkpoint 按每轮 38 个 optimizer updates 保存。

| 配置 | 相对公共模板的唯一改动 |
|---|---|
| `baseline` | 无改动：固定 LR `1e-4`、WD `0.01`、dropout `0.3`、EMA `0.995`、均值池化、`terminal-current` |
| `warmup_cosine` | 前 10 轮 `1e-5 → 1e-4`；第 100 轮余弦下降至 `1e-6`，随后保持 |
| `plateau` | 验证 F1 停滞时 LR 乘 0.5；patience 5，最低 `1e-6` |
| `weight_decay` | WD `0.01 → 0.05` |
| `dropout` | dropout `0.3 → 0.1` |
| `ema` | EMA `0.995 → 0.999` |
| `attention` | 32-token 均值改为可学习注意力池化 |
| `t0_delta` | 分类差值改为 `terminal-T0` |
| `t0_forecast` | 沿用原 TF 和 forecast 读出，仅将可观测输入固定为 T0，BCE + `0.5 × JEPA` |
| `t0_direct` | 始终只输入 T0；去掉 world/生成分支，分类输入为 `concat(mean(pCR queries), zero, zero)`，保留原 `3×dim` 分类头，BCE-only |

新接口字段为 `lr_scheduler=warmup_cosine`、`warmup_epochs=10`、`warmup_start_factor=0.1`、`cosine_end_epoch=100`、`cosine_min_lr=1e-6`；模型接口为 `spatial_readout=mean|attention`、`pcr_delta_reference=current|t0`、`enable_world_model=true|false`，默认分别为 `mean`、`current`、`true`。新增两项均为 `landmarks=t0`；direct 额外设 `enable_world_model=false`、`pcr_readout=zero_state`、所有辅助损失为零，不实例化 world predictor、EMA teacher、重建或生成模块。分类头仍为原 `3×dim`，仅将 mean(terminal) 和 mean(delta) 两段替换为同维度零向量，保留 mean(pCR queries)。forecast 按用户确认沿用 `prediction_training=teacher_forcing`，相对公共模板只改 `landmarks=t0`；训练中的真实未来 teacher 状态按既有 TF 路径参与监督与读出，不能将其解释为 BCE 仅由 T0 自主 rollout 产生。推断时只提供 T0，不提供未来 MRI。原七项继续TF、不变。

注意力通过 `LayerNorm → Linear → softmax` 生成 terminal 的空间权重；评分层零初始化，同一组权重池化 terminal 及差值，pCR queries 仍取均值。T0 差值使用已有 EMA teacher 的观测 T0 表示；不改变 world 内部历史差值或未来信息的可见范围。注意力权重仅描述现有 32 格表示的读取，不能直接视为病灶定位。

## 患者划分与评估边界

866 人按 seed 17、43 **分别**划为 **606 训练 / 87 验证 / 173 测试**。先按规范化患者 key 排序，再以 pCR 标签 × 四访视可用模式分层，使用同 seed 的 sklearn `train_test_split` 先划出测试，再划验证。同一 seed 下十配置共用已经冻结的患者划分，新增基线直接引用原 manifest 路径与哈希，不重新抽样；同一患者所有时点始终在同一集合，两 seed 患者分配不同。

派生 manifest 将 NPZ 引用转为原资源的绝对路径，保留原 manifest、NPZ 和 codec。临床标准化由 `RawMRIStore` 仅在各自新训练集拟合；影像数组的历史标准化与 codec 的历史预训练保持原来源。因此 `independence_scope=downstream_only`：测试集是本轮下游训练与选模的留出集，不能宣称整个上游流程都独立于测试患者。

原七项及补充基线的验证选模与早停使用患者等权整体 F1，两项 T0 对照使用 **T0 验证 F1**，阈值均固定为 0.5。测试成绩不参与训练、调度或早停。每个 run 训练后立即评估并保存结果；2026-10-20 用户明确要求提前查看已有测试成绩，独立测试报告已按此请求公开可用结果，原冻结报告器的历史展示规则保持原样。补充基线使用此前已固定的公共参数，配置差异审计保留在新目录。原七项及基线的 T0–T3 指输入可见至对应纵向时点并预测相同最终 pCR 标签，缺失对应访视的患者不计入该时点。两项 T0 对照训练、推断都只输入 T0；其 T1–T3 明确标为 `not_applicable`，不输入后续访视制造重复分数。

## 命令

在项目根目录设置路径：

```bash
project_root=/personal_fast_storage/VLA-PCRV1
suite_root="$project_root/private_outputs/large_all_tf_unfrozen_t0_dynamic_20261019"
train_python="$project_root/versions/v1_evolution/.venv/bin/python"
suite_script="$project_root/versions/v1_evolution/scripts/run_unfrozen_ablation.py"
base_config="$project_root/private_outputs/capacity_phase_tf_hybrid_20261018/input_configs/large_all_teacher_forcing_seed17.yaml"
```

只准备划分、配置、冻结运行快照和初始报告，不启动训练：

```bash
PYTHONPATH="$project_root/versions/v1_evolution/src" "$train_python" "$suite_script" \
  --output "$suite_root" --base-config "$base_config" \
  --manifest "$project_root/data/registered_roi32/manifest.json" \
  --codec "$project_root/data/pretrained/codec.pt" --gpu 1 --deferred-gpu 0 \
  --old-suite "$project_root/private_outputs/capacity_phase_tf_hybrid_20261018" \
  --prepare-only
```

首次直接启动：对尚未准备的新目录执行上述命令，去掉 `--prepare-only`。当前原 18 任务已有运行中的控制器，不重复启动。该目录发生过配额中断，恢复时使用已封存的串行 checkpoint 保存执行层：

```bash
PYTHONDONTWRITEBYTECODE=1 "$train_python" \
  "$suite_root/control_overrides/checkpoint_serialization_20261020/controller_harness.py" \
  --resume --manifest-sha256 a88904d17b7b7d0928afaf2a7897a2823f7ef9117bd9a9db14c823b0380ba2a7
```

只重建报告，不训练、不重新评估权重：

```bash
PYTHONPATH="$project_root/versions/v1_evolution/src" "$train_python" "$suite_script" \
  --output "$suite_root" --report-only
```

同一 suite 不重复启动控制器；恢复使用已冻结的 runtime，代码、配置、manifest 和输入身份不匹配时停止。运行前测量各配置训练、验证及诊断的显存需求；调度按可用显存动态填充，**不人为设置并发任务数上限**，保持安全显存余量，batch 始终为 16、不因并发缩小。GPU1 先运行，GPU0 的旧任务自然结束并完成评估后加入新 suite。

## 输出与权重删除

所有输出均位于 `$suite_root`：

- `matrix.json`、`controller.log`：队列状态、进度及控制器日志。
- `suite_protocol.json`、`splits/seed17/`、`splits/seed43/`、`input_configs/`：固定方案、患者列表、分层审计和 18 份配置；扩展已有 suite 时记录 revision 并重新冻结 runtime，保留原分割。
- `runs/<variant>_seed<seed>/`：训练日志、诊断、评估指标及逐患者逐时点预测。
- `report.html`、`report.json`、`overall.csv`、`per_config_seed_timepoint.csv`、`seed_summary.csv`：公开汇总。时间点主表共 **144 行，其中120行适用、24行明确N/A**：原七项112行，加新增两项8行真实T0结果；`applicable_timepoints.csv`只保留120行适用记录。未完成结果留空，整轮完成前适用测试行标为 `withheld`。
- `t0_comparison.csv` 和 HTML 的 T0 比较表：在各模型原选模 checkpoint 上对比同一 T0 输入；不为纵向模型另选一个最优 T0 checkpoint。

每个配置、种子及适用时间点报告 F1、AUROC、AUPRC、NLL、Brier、10 等宽分箱 ECE、样本数、阳性数及 TP/FP/FN。主排序按选模范围分为 `all_observed` 与 `t0` 两组，各组内按两 seed 验证 F1 均值排序，测试沿用该顺序；不混合两种选模范围的 overall 排名。另展示样本标准差、world teacher forcing / rollout / copy 相对诊断，以及 attention 的权重熵和有效 token 数；direct 的 world 诊断写明确N/A记录。种子差异同时包含划分和训练随机性。独立合并报告包含补充基线，预定为 160 条时点记录，其中 136 条适用、24 条 N/A；基线结果尚未完成时不计算净提升。完成后按相同种子、相同 split、相同时点计算原七项各自相对基线的差值；T0 对照仅在 T0 表中比较并注明选模范围不同。

权重保存在 `/personal_fast_storage`。流程为 **训练完成 → 验证、测试、T0–T3及预定 world 诊断 → 重读指标和逐例预测、核对人数与标签 → 写入带 checkpoint SHA256 的评估完成 receipt → 立即删除本 run 生成的 best、last 及临时 checkpoint**。`evaluation_complete.json` 与 `checkpoint_cleanup.json` 保留评估依据和删除清单；中断恢复时可幂等收尾。活动、待恢复或评估未完整通过的任务保留 checkpoint。原 codec、旧实验权重、配置、划分、日志和评估结果不属于此次清理范围；权重删除后不再支持新增需要该权重的推断。

旧 suite 原有 drain 策略及 `paused` 队列保持不变，不自动恢复暂停任务。只交接旧控制器和 GPU 可派发列表，已有 worker 自然结束；GPU1 完全空闲后先用于新 suite，GPU0 旧任务清空后加入，结束后恢复原 GPU 列表。存储配额或写入检查失败时暂停新任务派发。

## 2026-10-20 补充公共模板解冻基线

两份配置为 `private_outputs/large_all_tf_unfrozen_baseline_20261020/input_configs/baseline_seed17.yaml` 和 `baseline_seed43.yaml`。它们从对应原 `weight_decay` 配置仅将 WD `0.05` 恢复为 `0.01`；`baseline_config_audit.json` 逐项验证了相同种子下七个纵向变体均仅有原定单因素差异，且 codec、manifest、标准化来源和科学源码快照一致。

补充队列在独立执行层等待原 18 任务全部训练、验证、测试、诊断和权重清理完成，并等待原 controller/worker 退出、GPU allocation 锁自然释放。启动前用实际写入、fsync 与块分配检查确认空间，失败时不派发任务；串行 checkpoint 锁与原队列共享。恢复时完整既存权重计入预算，活动临时文件不作为可删除文件或可抵扣空间。基线仍使用原动态显存调度和实测 preflight、batch 16，无人为并发数量上限。

2026-10-20 08:52（北京时间）已核验入队：等待进程 PID `283807`、start ticks `2571920987`，状态为 `waiting_for_predecessor`，两份基线均为 `queued`；没有基线 GPU worker，32 GiB 空间探针尚未执行。原控制器及四个活动任务的进程身份保持一致。执行层封存 SHA256 为 `26079449f3aac01abfc4a6342137e7e0c3d60cc3cec161754eb9c5aacd983b45`，20 项 CPU 检查和生产身份验证通过。入队复核凭据为补充目录中的 `control_overrides/baseline_launch_verification.json`；此处为入队时快照，实时状态以 `matrix.json` 和 `control_overrides/baseline_execution/supervisor_state.json` 为准。

合并报告位于补充目录的 `comparison/report.html`，数据为 `comparison/overall.csv`、`timepoints.csv`、`seed_summary.csv`、`baseline_deltas.csv` 和 `comparison.json`。只读重建命令如下，不启动训练或评估：

```bash
baseline_root="$project_root/private_outputs/large_all_tf_unfrozen_baseline_20261020"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$baseline_root/runtime/src" \
  "$train_python" "$baseline_root/comparison/build_comparison.py"
```

基线结果未完成时差值留空；已有测试成绩按用户明确请求展示，仍沿用验证集选定的 checkpoint。
