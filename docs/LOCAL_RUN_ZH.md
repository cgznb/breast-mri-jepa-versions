# 本机运行说明

项目根目录为 `/personal_fast_storage/VLA-PCRV1`，运行版本为 `versions/v1_evolution`。
本次迁移原实验处理好的 ROI32 数据，保留原患者划分和数值，准备独立环境及本机配置。
正式训练由你后续手动启动；环境检查不会启动训练。准备阶段的少量工程验证结果记录在 `private_outputs/setup_runtime/`。
本次完整迁移、校验和真实 GPU 推理均已通过，记录见 [LOCAL_VALIDATION_ZH.md](LOCAL_VALIDATION_ZH.md)。

## 本机路径

| 内容 | 项目根目录下的相对路径 |
| --- | --- |
| 独立 Python 环境 | `versions/v1_evolution/.venv/` |
| 原实验处理后的影像 NPZ | `data/registered_roi32/views/` |
| 本机运行清单 | `data/registered_roi32/manifest.json` |
| 原服务器清单快照 | `data/registered_roi32/manifest.original.json` |
| 原服务器文件大小及清单 / codec SHA256 | `data/registered_roi32/migration-source-inventory.json` |
| 预训练 VQ codec | `data/pretrained/codec.pt` |
| 旧实验参考权重（只用于验证推理） | `data/reference_runs/encoder64_seed17/best.pt`、`encoder64_seed43/best.pt` |
| seed17 本机配置 | `versions/v1_evolution/local_configs/encoder64_seed17.yaml` |
| seed43 本机配置 | `versions/v1_evolution/local_configs/encoder64_seed43.yaml` |
| 原配置 | `versions/v1_evolution/configs/evolution_vq_forecast_history_residual_free_encoder64_seed17.yaml`、同名 `seed43.yaml` |
| 汇总检查报告 | `private_outputs/setup_runtime/runtime_check.json` |

866 名患者、3039 次 MRI、训练 / 开发验证 764 / 102 人；无独立测试集。
每个 NPZ 的 `image` 是已经标准化的 `float32 [3,32,128,128]`，不能再次 z-score 或调整尺寸。
本机 YAML 相对原 YAML 只改 `encoder_checkpoint` 的绝对路径和 `image_cache: null`，直接读原 NPZ。
本次复制输入数据、预训练 codec 和两份旧实验推理参考权重；重新训练使用新输出目录。数据来源与处理过程见 [DATA_PROVENANCE_ZH.md](DATA_PROVENANCE_ZH.md)。

## 激活环境与完整检查

```bash
cd /personal_fast_storage/VLA-PCRV1
source versions/v1_evolution/.venv/bin/activate
python -m mri_vla_jepa --help
python -m mri_vla_jepa train-vla-jepa --help
python tools/check_local_runtime.py
```

检查脚本逐一核对 3039 个 NPZ 的存在、大小、`image` 及可选 `roi/support` 数组头，
核对原清单和 codec 的 SHA256、患者划分、本机清单是否只改变影像路径、本机配置及 CUDA BF16 执行。
默认只读取 NPZ 数组头，不完整解压影像、不验证全部 ZIP CRC。
检查失败会返回非零退出码；报告只有汇总，不输出患者 ID 或临床记录。

首次迁移后还应完整读取 17.6 GB 文件，核对原服务器逐影像 SHA256：

```bash
python tools/check_local_runtime.py --full-image-sha256
```

原服务器校验值保存在 `data/registered_roi32/migration-source-sha256.json`。
环境版本、依赖锁文件及重新安装方式见
[LOCAL_ENVIRONMENT_ZH.md](../versions/v1_evolution/docs/LOCAL_ENVIRONMENT_ZH.md)。

只核对数据、无需安装依赖时，可使用基础 Python：

```bash
python tools/check_local_runtime.py --data-only \
  --output /personal_fast_storage/VLA-PCRV1/private_outputs/setup_runtime/data_check.json
```

默认检查全部数组头；`--skip-image-headers` 可加速重复检查，但报告会明确标记跳过。
迁移未完成时，缺失数据会导致检查失败，不能将部分文件当作完整数据集。

## 直接运行一次真实 MRI 推理

```bash
cd /personal_fast_storage/VLA-PCRV1
source versions/v1_evolution/.venv/bin/activate
python tools/run_reference_inference.py
```

此命令读取 seed17 的旧实验权重，选择一位影像已到位的验证患者，
使用 T0 在 GPU 上输出 pCR 概率与未来潜状态，不更新模型。
终端只显示工程验证汇总，患者级结果保存到 `private_outputs/reference_inference/` 下的新时间目录。
`--seed 43` 可切换另一份参考权重。单例运行成功只证明迁移后的推理链路可用。

## 可选的小型合成验证

下面命令由你手动执行，使用小型合成数据、CPU 和基础 CNN，检查保存、验证、推理的工程链路。
它会做短暂的合成训练，不使用真实患者数据，也不代表 encoder64 正式实验已复现。
先把 `CHANGE_ME_TO_UNUSED_DIRECTORY` 替换成从未使用过的新目录名：

```bash
cd /personal_fast_storage/VLA-PCRV1
source versions/v1_evolution/.venv/bin/activate
VLA_SMOKE_OUT="/personal_fast_storage/VLA-PCRV1/private_outputs/CHANGE_ME_TO_UNUSED_DIRECTORY"
if [ -e "$VLA_SMOKE_OUT" ]; then
  printf '输出目录已存在，请选择新目录。\n' >&2
else
  python -m mri_vla_jepa smoke-vla-jepa --output "$VLA_SMOKE_OUT"
fi
```

真实数据的 GPU 工程验证应使用另建的小清单和临时运行配置，保留原 manifest、本机正式配置及 3039 个 NPZ。
`train-vla-jepa --stop-after` 的单位和训练循环有关，不能仅凭参数名假设它必然只做一次优化器更新。

## 日后手动启动原最优方案

以下是原 V1 最优方案 seed17 的完整训练入口，最多 200 epoch、patience 50，读取全部原训练 / 开发验证清单。
本次环境准备不会执行这条命令。先确保完整检查通过，并把输出变量改为从未使用过的新目录。

```bash
cd /personal_fast_storage/VLA-PCRV1
source versions/v1_evolution/.venv/bin/activate
VLA_RUN_OUT="/personal_fast_storage/VLA-PCRV1/private_outputs/CHANGE_ME_encoder64_seed17_new_run"
if [ -e "$VLA_RUN_OUT" ]; then
  printf '输出目录已存在，请选择新目录。\n' >&2
else
  python -m mri_vla_jepa train-vla-jepa \
    --manifest /personal_fast_storage/VLA-PCRV1/data/registered_roi32/manifest.json \
    --config /personal_fast_storage/VLA-PCRV1/versions/v1_evolution/local_configs/encoder64_seed17.yaml \
    --output "$VLA_RUN_OUT"
fi
```

seed43 使用 `local_configs/encoder64_seed43.yaml` 和另一个新输出目录。
`image_cache: null` 节约一次额外缓存生成及其空间；首次读压缩 NPZ 时会有解压开销。
若需后续加速，可按版本文档生成新缓存，并另存配置副本。

所有运行输出放入 `private_outputs/` 下的新目录；不要将输出指向 `data/`、`preprocessing/`、历史运行记录或已有实验目录。
数据、患者清单、模型权重、独立环境和逐患者输出均不进入 Git；提交源码、说明及不含患者信息的配置。

## 生成数据的代码

生成代码的入口和依赖见 [PREPROCESSING_SOURCES_ZH.md](PREPROCESSING_SOURCES_ZH.md)，
实际源码归档在 `preprocessing/upstream/`，逐文件 SHA256 记录在 `preprocessing/SOURCE_MANIFEST.json`。
原服务器的只读来源快照另保留在 `data/provenance/source_preprocessing/`。
目前训练直接读取迁移的 NPZ，不需要先重跑数据生成。
版本内的 `scripts/prepare_registered_roi32.py` 是已有 NPZ 的审计及临床 manifest 构建步骤；
从配准 NIfTI、T0 预测病灶、几何 / 相位元数据开始裁剪并标准化，属于上游处理代码。
完整重建还需要原清单、T0 预测病灶及元数据等输入，不能只运行一个脚本并用任意肿瘤分割替换。
归档配置保留原服务器绝对路径；本次独立环境配置的是读取现成 NPZ 的 V1 运行环境，
配准和 nnUNet 等上游工具的完整环境需在实际重新生成数据时单独准备。
