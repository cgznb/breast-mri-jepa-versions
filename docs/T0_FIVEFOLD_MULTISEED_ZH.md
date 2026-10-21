# 八卡五折：增加训练随机种子

新配置为 `versions/v1_evolution/configs/t0_fivefold_8gpu_multiseed.yaml`。
新配置只运行六个新增种子：**43、44、809、203、30、21**，排除原训练种子 **17**，
共 **6 个训练种子 × 3 个任务 × 5 折 = 90 次新训练**。
原 seed17 单独仍为 15 次训练，合计七个种子共 105 次。

原单种子 YAML 与原输出目录保持原样。新配置使用独立目录，只追加其余六个种子，不重复训练 seed17。
准备之后不应修改已冻结的 YAML；变更种子列表需使用新的 suite 名称。

## 哪些随机性发生变化

`cv.train_seeds` 控制各 run 的网络初始化、训练采样及 dropout 等训练随机性。
每个任务在每个种子下均从原 codec 初始化，独立运行五折，绝不继承其他种子或其他折的权重。
同一个种子下三个任务继续使用配对的共享网络初始化。

固定测试集仍为 173 人、55 阳性，`holdout_seed=17`；开发集五折仍使用 `split_seed=20261020`。
全部训练种子共用同样五份 manifest 与对应的训练集临床标准化，保持训练/验证人数及阳性数不变。
因此本轮种子间差异反映训练随机性，不包含重新抽取测试集或重新划分五折的差异。
`cv.train_seed: 43` 和 `training.seed: 43` 是公共模板；生成的每份 run 配置使用其实际训练种子。
配置中保留的 `holdout_seed: 17` 仅标识原固定测试划分，不会生成训练 seed17 的任务。

三个任务的读出、JEPA 权重、warmup/cosine、Attention、WD、EMA、batch16、选模和清理规则与原方案相同。

## 平台提交

选择 **custom、1 台机器、8 张 GPU**，仅启动一个调度器，填写：

```bash
/personal_fast_storage/VLA-PCRV1/versions/v1_evolution/.venv/bin/python \
  /personal_fast_storage/VLA-PCRV1/versions/v1_evolution/scripts/run_t0_fivefold.py \
  --config /personal_fast_storage/VLA-PCRV1/versions/v1_evolution/configs/t0_fivefold_8gpu_multiseed.yaml
```

新输出目录：

```text
/projects/xts0030/public/cxx/t0_cv5_combined_multiseed_v1/
```

支持相同的 `--prepare-only`、`--preflight-only`、`--report-only` 和同命令断点恢复。
同硬件规格、同任务共享一次显存测量，不因为训练种子不同重复预检；预检记录实际测量种子与共享范围。
并发由八卡显存和主机资源预算动态决定，batch 固定 16，不限制每卡只能运行一个任务。
验证、固定测试及全部预定诊断成功并重新读取校验后，才删除该 run 的权重。

## 结果解释

每个训练种子分别保留五折验证、五折测试、693 人 OOF 和 173 人测试概率集成结果。
每个种子的测试集成只平均该种子的五折概率，然后重新计算全部指标及分层 bootstrap 95% CI。
不会把不同种子的模型默认融合成一个 30 模型的测试结果。

`reports/seed43/`、`reports/seed44/` 等目录为逐种子完整报告；顶层 `reports/report.html`、JSON 和 CSV
汇总逐种子结果及跨种子的均值、样本标准差。跨种子统计在各种子的同定义指标上计算，
不会把 30 个折当作 30 个独立测试集。新目录的汇总只覆盖这六个种子，不自动合并旧 seed17 结果。
主排序使用各训练种子五折验证 F1 均值的再平均，测试沿用这一顺序。
所有患者级预测包含实际训练种子，以区分同一患者在多个模型下的预测。

测试报告统一在整个 suite 完成后公开。测试集历史上已查看，报告继续注明独立性限于本轮下游训练。
原方案详细公共设置与资源要求见 [八卡单种子说明](T0_FIVEFOLD_8GPU_ZH.md)。

本次通过 126 项本地测试，并以真实 866 人队列完成临时 CPU 入口演练：生成 90 份配置、确认没有训练 seed17、
核对原方案的患者划分与训练集标准化一致，验证重复准备、报告并行读取及六个种子的冻结 worker 加载。
临时演练文件已移除。八卡显存与并发能力仍由平台实际 worker 预检确认；本次未提交或启动正式训练。
