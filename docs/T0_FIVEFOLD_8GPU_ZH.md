# T0 三任务五折：单机八卡交付与运行

本文对应原 seed17 单种子方案。新增六个训练种子的配置与提交命令见
[多种子运行说明](T0_FIVEFOLD_MULTISEED_ZH.md)，它使用独立 suite，不覆盖本文目录。

运行入口为 `versions/v1_evolution/scripts/run_t0_fivefold.py`，正式配置为
`versions/v1_evolution/configs/t0_fivefold_8gpu.yaml`。平台选择 **custom、自定义启动命令、1 台机器、8 张完整 GPU**。
只启动一个 controller；它把独立训练进程绑定到平台分配的 GPU UUID，每个进程内部使用 `cuda:0`。
不使用 DDP、`torchrun` 或八个相同 controller。GPU 可以承载几个任务，由实际显存、CPU、RAM 测量决定。

本方案复用平台已经挂载的 `/personal_fast_storage` 全部内容，额外挂载
`/projects/xts0030/public`。现有 NPZ、原 codec、Python 环境继续从 personal 读取，无需搬迁影像或复制旧实验目录。
**所有新实验配置、划分、冻结代码、日志、预测、指标和临时 checkpoint 都写到：**

```text
/projects/xts0030/public/cxx/t0_cv5_combined_v1
```

当前开发机器没有 `/projects` 挂载。本次交付没有在本机伪造该目录，也没有向平台提交训练。
目标平台 worker 的八卡并发、显存、吞吐和完整恢复流程，仍需在实际分配的资源上执行预检和验证；CPU 测试通过不等于八卡实测通过。

## 三个任务与公共设置

每个任务运行五折，合计 **15 次训练**。每折训练随机种子均为 17，不再增加第二个训练种子。

| 任务 | pCR 读出 | JEPA 权重 | 空间读出 |
|---|---|---:|---|
| `t0_forecast` | Forecast，含预测终态及其差值 | 0.5 | Attention |
| `t0_direct_no_jepa` | Literal Direct，直接分类当前 T0 表征 | 0 | Mean |
| `t0_direct_readout` | Literal Direct，直接分类当前 T0 表征 | 0.5 | Mean |

三个任务都保留 `enable_world_model=true` 的共同模块结构，使用配对初始化；Direct 的分类头输入为 `dim`。
Direct JEPA=0 的 predictor 没有训练损失路径，world 指标明确记为不适用。
它与历史 `zero_state` 配置的三段特征分类头不同。只有 Forecast 使用空间注意力，Direct 保持原定义。

公共设置为 large / all / T0 输入、解冻实际参与前向计算的 VQ 前端、原 codec 初始化、EMA teacher 无梯度。
组合改动为 warmup + cosine、WD `0.05`、EMA `0.999`，并仅给 Forecast 使用 Attention；dropout 保持 `0.3`。
前端与其余网络基础 LR 都为 `1e-4`，前 10 轮从 `1e-5` 升至 `1e-4`，第 100 轮末降至 `1e-6` 后保持。
训练使用 BF16、activation checkpointing、batch 16，最多 200 轮，验证 F1 早停 patience 50，允许在余弦周期结束前停止。
损失为 BCE 加表中的 JEPA 权重，其余辅助损失权重为零。选模阈值固定为 `0.5`。

## 固定测试集与五折划分

测试集沿用已存在 seed17 划分的 **173 名患者，其中 55 名阳性**，以原 manifest 和分割审计中的明确患者名单为准。
不重新抽取测试集。剩余 **693 名患者、221 名阳性**组成开发集。

开发集按规范化患者 ID 固定排序，以 pCR 标签和四时点访视可用模式联合分层，使用
`StratifiedKFold(n_splits=5, shuffle=True, random_state=20261020)`。
`holdout_seed=17`、`cv_split_seed=20261020`、`train_seed=17`分别记录。
三个任务共享同一折的训练/验证/测试名单；同一患者的所有访视归属一致。

| Fold | 训练人数 | 训练阳性 | 验证人数 | 验证阳性 |
|---|---:|---:|---:|---:|
| 1 | 554 | 177 | 139 | 44 |
| 2 | 554 | 176 | 139 | 45 |
| 3 | 554 | 176 | 139 | 45 |
| 4 | 555 | 178 | 138 | 43 |
| 5 | 555 | 177 | 138 | 44 |

每折临床标准化仅用该折训练患者拟合；影像数组及历史影像标准化保持原样。
Forecast 和 JEPA 训练需要未来 teacher 目标，因此仍保留全部 3039 个已有访视资产。
既往已查看过该测试集成绩，codec 与影像预处理也有历史来源；本轮报告仅声明下游训练独立性，不将其描述为未接触过的外部测试。

每折按该折验证 F1 选择一个最佳 checkpoint。主排序使用五折验证 F1 的算术均值，同时展示样本标准差。
OOF 把五个互斥验证集逐例拼接，每个任务恰有 693 条预测；OOF 的整体指标作为辅助结果。
主测试结果对同一患者的五折概率先取算术均值，再计算指标，每个任务恰有 173 条集成预测。
五折测试指标的均值和标准差仅为辅助结果，不能代替概率集成，也不是五个独立测试集。
没有在全部 693 名开发患者上重新训练的第六个模型。

## 平台填写与绝对启动命令

平台应挂载 personal 原路径和目标 projects 路径，提供八张可见、支持配置所需 BF16 的完整 GPU。
当前入口不支持 MIG 配额。保留平台设置的 `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES`，程序据此限制可用设备。
平台运行时限应覆盖目标预检及训练；若到时中断，可使用同一配置继续恢复，不需要重新生成划分。

custom 启动命令填写为一条命令：

```bash
/personal_fast_storage/VLA-PCRV1/versions/v1_evolution/.venv/bin/python /personal_fast_storage/VLA-PCRV1/versions/v1_evolution/scripts/run_t0_fivefold.py --config /personal_fast_storage/VLA-PCRV1/versions/v1_evolution/configs/t0_fivefold_8gpu.yaml
```

该命令先准备并冻结输入，再执行目标 GPU 预检，随后派发正式任务。
Python 环境和数据路径都是绝对路径，不依赖平台工作目录。现成 NPZ 训练无需重新安装 NIfTI/Excel 预处理依赖。
当前环境主要版本为 Python `3.12.1`、PyTorch `2.5.1+cu121`、NumPy `1.26.4`、PyYAML `6.0.3`、scikit-learn `1.9.1`；
目标 worker 仍需确认该既有环境及 GPU 驱动可用。

可在同一命令末尾选择以下互斥参数：

| 参数 | 行为 |
|---|---|
| `--prepare-only` | 仅生成并核验五折划分、十五份配置及冻结 runtime，生成待运行报告；不启动 GPU 工作。 |
| `--preflight-only` | 准备后在目标 GPU 测量三类任务的训练、验证及诊断资源，保存预检结果，不派发十五次正式训练。 |
| `--report-only` | 读取已有原子状态快照并重新生成报告，不启动训练；使用独立报告锁，可在 controller 运行时调用。 |
| 不附加上述参数 | 完整执行，或恢复已存在的同名 suite。 |

例如，先做目标平台预检：

```bash
/personal_fast_storage/VLA-PCRV1/versions/v1_evolution/.venv/bin/python /personal_fast_storage/VLA-PCRV1/versions/v1_evolution/scripts/run_t0_fivefold.py --config /personal_fast_storage/VLA-PCRV1/versions/v1_evolution/configs/t0_fivefold_8gpu.yaml --preflight-only
```

预检包含实际前向、反向、优化器及评估计算，测量结果只用于资源准入。正式训练仍从原 codec 和固定种子重新初始化。
同型号、同显存规格使用代表 GPU 测量；异构 GPU 分别测量。预检不保证目标机器上所有并发组合已经实测。

## 显存、CPU、RAM 与存储准入

显存使用测得的训练、验证、训练诊断和 FP32 world 诊断峰值，并增加每任务 `768 MiB` 余量。
默认显存预算为每卡 `90%`；调度器计入既有和无法映射的外部占用。
任务数量受实测显存与主机资源共同限制，batch 始终保持 16。OOM 允许一次保留 batch 的独占重试，再失败会保留 checkpoint 并停止。

每任务默认四个计算线程、两个加载线程；八任务同时运行需要约 48 个可用 CPU 线程。
程序读取 CPU affinity、cgroup CPU/RAM 上限及当前可用 RAM；主机内存默认预算为总限额的 `85%`。
RAM 预留同时考虑实测峰值和 checkpoint 保存/加载时的 CPU 副本。
可将 128 GiB RAM 作为资源申请起点，256 GiB 更宽裕；最终并发取决于目标 worker 的实际测量，不能据此保证八个任务同时运行。

存储预算覆盖活动任务的 best、last 及一个原子保存副本，并乘默认 `1.25` 余量。
程序检查可用容量并执行真实空间分配/写入与同步，以检查用户或项目配额。
写入或配额检查失败时停止新训练派发；可执行的待评估任务优先处理，以完成评估并释放权重。
预留量会随目标预检测得的模型状态大小计算，不以影像数据体积代替权重预算。

## 冻结、恢复与评估后删除

首次准备生成固定的 suite protocol、每折 manifest/审计/标准化、逐任务配置、数据文件身份和冻结 runtime。
source manifest、参考 seed17 manifest/审计、codec、suite YAML 均固定 SHA256；影像资产固定路径、大小和修改时间身份。
恢复时核验这些输入与冻结实现。已经准备的 suite 应保持原 YAML 不变；新方案另存 YAML 并使用新的 `suite_name`。
入口会转入该 suite 的冻结 controller，worker 也从冻结 runtime 导入训练、预检和评估代码。

同一条正式命令即可恢复。controller 使用进程 PID 和 start ticks 识别存活 worker，避免重复派发；
每个 run 还有独占 worker 锁。已完成训练进入评估，未完成训练从 last checkpoint 一致恢复。
若中断发生在 best 已完整保存、last 尚未保存之间，程序校验模型、优化器、RNG、调度器及分割身份后，
将 best 原子复制为 last 并记录来源 SHA256；缺失的验证记录只用验证集恢复。
中断不会主动删除活动或待恢复权重。若已有 controller 存活，第二个训练 controller 会拒绝运行；报告入口可单独使用。

每个实验遵循以下顺序：

1. 训练完成，校验 best/last、选模得分、epoch、来源和分割身份。
2. 写入 `evaluation_best_val.json`、`evaluation_best_test.json`、`evaluation_last_val.json`，并保存对应 world 诊断和计划内训练诊断。
3. 重新读取预测，核验患者覆盖、标签、所有指标及 checkpoint 标识，持久化评估完成记录。
4. 立即删除该 run 已收据化的全部生成 checkpoint，保留删除记录；重试可完成中断的删除。

原 codec、原数据、旧实验和其他运行目录不属于本轮清理范围。
权重删除后可用保存的逐例预测重新计算和审计指标，不能再用该权重执行新的推断任务。
若要增加原计划以外的评估，应在正式启动前确定。

## 结果位置与公开时机

正式输出目录主要结构如下：

```text
t0_cv5_combined_v1/
  matrix.json
  suite_protocol.json
  runtime/
  input_configs/
  splits/fold1/ ... fold5/
  preflight/
  runs/<variant>_fold<1-5>/
  reports/
```

运行中可查看 `matrix.json`、controller 标准输出和各 run 的训练/评估日志。
**全部十五个任务完成前，公开报告不读取或展示测试成绩。** 测试不参与训练、选模、早停、调度或剩余方案修改。

`reports/report.html` 为入口，`report.json` 保留结构化内容，主要表格和预测文件为：

| 文件 | 内容 |
|---|---|
| `fold_metrics.csv` | 各任务、各折验证与测试的最佳 checkpoint 结果。 |
| `fold_summary.csv` | 五折各指标算术均值和样本标准差。 |
| `supplementary_last_val.csv` | last checkpoint 验证结果，辅助检查后期变化。 |
| `oof_predictions.csv/json` | 三任务合计 `3 × 693 = 2079` 条 OOF 预测。 |
| `test_ensemble_predictions.csv/json` | 三任务合计 `3 × 173 = 519` 条概率集成预测。 |
| `fold_test_predictions.csv/json` | 完成后公开的各折逐例测试预测，共 `3 × 5 × 173 = 2595` 条。 |
| `aggregate_metrics.csv` | OOF 和测试概率集成指标及置信区间。 |

报告包含 F1、AUROC、AUPRC、NLL、Brier、10 等宽分箱 ECE、阳性预测比例，以及人数、阳性数、TP/FP/FN。
默认 bootstrap 为 2000 次，按标签分层、患者为抽样单位，使用 seed20261020；各任务复用相同患者抽样，报告配对差值。
置信区间以当前已训练的模型为条件，不覆盖重新训练的不确定性。
各 run 的原始分类结果、TF/自由 rollout/相对 copy 的 world 诊断和 Attention 诊断保留在原位，并由 HTML 链接。

本次交付通过了 108 项 CPU 自动化测试，以及真实 866 人队列的完整入口准备、重复恢复和并行报告核验。
当前未启动正式训练；目标八卡平台的显存与并发能力由实际 worker 预检确认。
目标平台正式启动后，以该目录中的预检、状态、逐例预测、评估和清理收据作为完成证据。
