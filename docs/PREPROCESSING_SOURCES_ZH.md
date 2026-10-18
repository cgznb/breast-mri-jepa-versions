# 数据生成代码与重建输入

归档日期：2026-10-18。代码位于项目内的 `preprocessing/upstream/`，可以纳入本地 Git 版本管理。
本次迁移现成 NPZ，并保存产生这批数据的原处理代码；没有在本机重新生成影像或启动正式训练。
已迁移数据的使用方法见 [本机运行说明](LOCAL_RUN_ZH.md)，患者批次和处理参数见
[数据来源说明](DATA_PROVENANCE_ZH.md)。

## 每个阶段对应什么代码

| 阶段 | 本项目保存的主要入口 | 作用 |
| --- | --- | --- |
| DICOM 准备与配准 | `preprocessing/upstream/registration/src/ispy2_registration/{dicom_prepare,phase_alignment,pipeline,transforms}.py` | 准备 NIfTI 与真实采集几何；各相位对齐到增强前，各期再对齐到 T0，保留刚性回退 |
| 上游固定队列 | `preprocessing/upstream/MOTFM-tumor-first/scripts/{build_ispy2_registered_strict_a_bundle,expand_ispy2_registered_strict_a_bundle,resplit_ispy2_registered_validation_holdout}.py` | 原 bundle 的筛选、增量入组、固定验证集划分；对应 `utils/` 和原配置一同保留 |
| 预测病灶 | `preprocessing/upstream/MeWM-ISPY2/mewm_ispy2/first_post_segmentation.py`、`scripts/launch_first_post_segmentation.py` | 用未配准首次增强后影像做 MAMA-MIA nnUNet 预测；原作者分支保存在 `nnUNet-MAMA-MIA/` |
| T0 几何与中心审计 | `preprocessing/upstream/geometry_audit/{registered_audit,largest_component_crop_audit}.py` | 保存 T0 病灶到原空间的映射、最大连通区域中心及裁剪覆盖信息；几何依赖位于 `longitudinal_temporal_pillar/src/` |
| 固定 ROI32 | `preprocessing/upstream/MeWM-ISPY2/mewm_ispy2/registered_roi32_data.py` | 读取原 split、visits、transitions、T0 预测 mask 和几何审计结果，固定裁剪为 `32×128×128`，拟合训练集 DCE0 统计量 |
| 三相选择与标准化 | `preprocessing/upstream/MeWM-ISPY2/mewm_ispy2/registered_three_phase_data.py` | 取 `aqc_0 / aqc_1 / metadata_late`，共用同一 T0 裁剪网格与训练集 DCE0 标准化；不会把第三相固定成 `aqc_2` |
| 原 NPZ 打包 | `preprocessing/upstream/symmv4/symm-fm-trjepa/scripts/prepare_registered_roi32.py` | 生成 `image / roi / support / latent`，校验相位、几何、标签、训练统计与原 codec 绑定 |
| 后续 NPZ 复用 | `preprocessing/upstream/symm-fm-trjepa-v2/scripts/prepare_registered_roi32.py` | 核验原 NPZ 后硬链接，不重新裁剪、缩放或标准化 |
| 当前 VLA manifest | `versions/v1_evolution/scripts/prepare_registered_roi32.py` | 审计已有 NPZ，联结官方临床、Arm 与 pCR，生成当前版本所读的 manifest |

MeWM 的完整 `mewm_ispy2/` 包、原 `scripts/`、`configs/`、`requirements.txt` 和许可证都已保存。
两版 TRJEPA 的 `trjepa/` 包也完整保存，避免只留一个脚本而缺少被导入的模块。
原上游的其他肿瘤合成、训练实验、输出、权重、虚拟环境和 `.git` 没有混入归档。

关键配置是 MeWM 的 `configs/registered_dce0_roi32_firstpostmask_v1.yaml`、
`configs/registered_three_phase_roi32_v1.yaml` 和 `configs/ispy2_first_post_mamamia_segmentation_v1.yaml`。
MeWM 裁剪模块还提供 `python -m mewm_ispy2.registered_roi32_data --config ...` 入口；
这里的 `...` 必须指向完成本机路径适配后的独立配置。三相控制器同时支持训练阶段，查看原脚本时应按具体阶段使用。

## 已保存的小型原输入

`data/provenance/rebuild_inputs/` 保存 7423 个原输入或收据，合计 134,787,285 bytes，
逐文件 SHA256 核对通过，选定文件无缺失。它处于 Git 忽略范围内，包括：

- `bundle/`：原始 **867 人、3043 次检查**的 `split.json`、`visits.csv`、`transitions.csv`、筛选和空间契约。此处保留原划分，不重新随机划分患者。
- `geometry_audit/`：原 T0 身份映射、几何映射、中心和覆盖审计 JSON/CSV。
- `segmentation/`：原 inventory、执行设置、**867 个 T0 预测 mask** 和对应 case reports。包含后续因 T0 预测为空而被排除的一人，原裁剪代码会核验该空 mask；最终队列仍为 **866 人、3039 次检查**。
- `referenced_files/`：上述记录引用的原 native / registered 元数据与 T0 相位变换，按照原绝对路径去掉开头 `/` 后保存。
- `official_metadata/`：原 `metadata_enriched.csv`、官方患者身份表与临床表。
- `intermediates/`：单相与三相 inventory、原标准化统计、crop JSON、latent 清单和完成收据。

`INPUT_MANIFEST.json` 记录原绝对路径、本地相对路径、大小和 SHA256。原输入文件内容没有改写；
记录里的绝对路径仍属于旧服务器。迁移到本机后应使用清单对应关系适配，而不是把旧收据误认为本机缓存已重新验证。

这里保存的是原裁剪和核验所需的小型来源材料；完整已配准 NIfTI 可从本机
`/public_fast_storage/cxx26/organized_v1/03_registered_nifti/` 按官方身份映射对应。
上述重建输入目录没有另外复制全部三相 NIfTI、原 VQ-GAN 完整训练恢复检查点或独立 latent 数组。
当前 VLA 所需的 VQ 权重导出已经另存于 `data/pretrained/codec.pt`，具体内容见下节。

## VQ-GAN 的代码、权重和当前用法

已复制当前 V1 运行所需的 VQ 代码与预训练权重：

| 文件 | 内容 |
| --- | --- |
| `versions/v1_evolution/src/mri_vla_jepa/vq_encoder.py` | 当前 VLA 的单相编码器与连续特征前端，严格加载原 encoder / pre_quant 权重 |
| `versions/v1_evolution/src/mri_vla_jepa/encoder.py` | 三相共享同一前端，拼接后做网格池化、投影与空间 Transformer |
| `preprocessing/upstream/MeWM-ISPY2/mewm_ispy2/vqgan.py` | 原 VQ-GAN 结构：3D encoder、pre_quant、EMA 码本、post_quant、decoder 与判别器实现 |
| `preprocessing/upstream/MeWM-ISPY2/mewm_ispy2/registered_roi32_vq.py` | 原 ROI32 VQ-GAN 训练逻辑，含重建、感知、码本承诺、2D/3D 对抗和特征匹配损失 |
| `preprocessing/upstream/symm-fm-trjepa-v2/trjepa/codec.py` | 可独立加载安全权重导出的完整自编码器实现 |
| `data/pretrained/codec.pt` | 当前模型使用的安全权重导出，8,479,721 bytes，SHA256 与原服务器一致 |

`codec.pt` 的格式为 `symm_world_codec_v2`，保存 87 个状态张量，
包含 encoder、pre_quant、quantizer、post_quant 和 decoder。
它记录的来源是原服务器 `MeWM-ISPY2/runs/registered_dce0_roi32_firstpostmask_v1/vq/best.pt`。
本机已分别验证当前 VLA 前端和完整自编码器均能严格加载该导出。
这份导出不含 GAN 判别器权重、优化器或训练进度；原 VQ-GAN 完整续训检查点没有迁移。

完整自编码器的单相路径是：

```text
MRI [1,32,128,128]
  → 3D encoder [64,8,32,32]
  → pre_quant [8,8,32,32]
  → 16384 项 EMA 码本
  → post_quant + decoder
  → MRI 重建 [1,32,128,128]
```

当前 seed17 / seed43 的设置为 `encoder_type: vq_encoder`、`encoder_training: frozen`。
实际读取 encoder 输出的每相 64 通道特征，跳过 8 通道压缩和码本量化，
也不调用重建 decoder 或 GAN 判别器。
三相共享这套单通道编码权重，拼接成 192 通道，再池化、投影为 JEPA 使用的空间 tokens。
当前前端加载 34 个 encoder / pre_quant 状态张量，保留 pre_quant 的检查点兼容性，
但 encoder64 模式不执行 pre_quant 的前向计算。

旧 NPZ 的 `latent: float16 [24,8,32,32]` 是上游三相 prequant 历史缓存。
当前 VLA 读取 `image` 并在运行时提取上述 64 通道特征，不读取该历史 latent 字段。

## 复现时需要适配的内容

这批原代码保留原算法和参数，配置中仍有 `/root/autodl-tmp/...`、`/home/cxx/...`、
`qingyuan:` 等旧路径。原脚本还会校验路径、文件大小、mtime、代码来源和 cache binding。
因此，未来在本机从已配准影像重建时，需要先复制配置到本机配置目录，再逐阶段适配影像路径、
metadata 引用、原 split、mask 和几何记录，并让各阶段生成本机自己的 inventory 与收据。
不能仅替换配置中的一处目录就认为整个历史契约已完成迁移。

原 NPZ 打包脚本还需要旧 VQ codec、连续 latent 缓存及统计，
这是历史生成任务附带的字段，不等同于当前 VLA 模型必须重新训练 VQ。
这次保存源码与来源输入，没有把历史 VQ 训练、latent 生成和所有上游实验再执行一遍。

运行现成 NPZ 的 Python 环境与从 DICOM 开始的重建环境是两件独立的工作：
registration 原 `pyproject.toml` 要求 Python ≥3.12、ITK 5.4.6、itk-elastix 0.25.3；
MAMA-MIA 作者 nnUNet 的依赖在其 `pyproject.toml`；MeWM 原 `requirements.txt` 是历史完整环境快照，
包括与当前运行无关的软件，不能直接据此覆盖本机运行环境。
`geometry_audit/verify_shapes.py` 以及 MeWM 的旧 FM 训练桥接还会引用外部 `ispy2_symmflow`，
这些历史架构检查和 FM 训练扩展不属于本次 ROI 裁剪/NPZ 源码闭包的可运行范围。

## 来源与完整性

`preprocessing/SOURCE_MANIFEST.json` 共记录 515 个来源文件，保留包的原服务器路径、复制时 Git HEAD、
`source_sha256` 和 `archived_sha256`。除以下 4 个文件的真实患者示例脱敏外，两个 hash 相同：

- `MeWM-ISPY2/README_ISPY2.md`
- `registration/README.md`
- `registration/src/ispy2_registration/fallback_visualization.py`
- `registration/tests/test_fallback_visualization.py`

这四份原件保存在被 Git 忽略的 `data/provenance/source_preprocessing/` 对应目录，
归档清单对改动逐项标注原因。脱敏只替换示例标识及相应测试；患者源数据和清单存放在忽略目录。
Git HEAD 是复制时 checkout 的信息；已有未提交上游修改，因此每个文件的 SHA256 才是此次快照的精确标识，
不能把复制时 revision 当作历史缓存生成当时的精确 revision。

可以在项目根目录执行纯文件校验，它不导入原处理模块、不写缓存、不运行影像处理：

```bash
python3 preprocessing/verify_sources.py --check-syntax
```

原各包的 LICENSE 和第三方 notices 随代码保留；两版 TRJEPA 的 `docs/SOURCES.md` 也保留。
归档没有改变这些文件原有的授权范围。
