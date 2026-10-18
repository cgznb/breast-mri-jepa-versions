# 原实验的数据批次与处理方法

核对日期：2026-10-18。依据原服务器的 manifest、上游 inventory / bundle，以及实际处理源码核对。本次只追溯数据来源；未启动训练，影像迁移已暂停。

## 原实验实际使用哪批数据

这是 I-SPY2 乳腺动态增强 MRI 纵向数据。临床信息来自官方
`ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx`；ACRIN / TCIA 标识与 I-SPY2 Research ID 通过
`Full-Collection-Ancillary-Patient-Information-file.xlsx` 对应。

最终数据不是本机公共目录里的全部患者，而是旧实验固定清单中的以下子集：

| 内容 | 数量 |
| --- | ---: |
| 患者 | 866 |
| 训练 / 开发验证 / 测试患者 | 764 / 102 / 0 |
| MRI 检查 | 3039 |
| T0 / T1 / T2 / T3 检查 | 866 / 836 / 727 / 610 |
| 具有全部四期检查的患者 | 580 |
| pCR 阴性 / 阳性 | 590 / 276 |

缺少某次随访的患者仍然保留，manifest 对应槽位为 `null`，没有补造影像。

原服务器的影像输入分为两个目录；这是存放位置的统计，不等同于两个互斥的官方临床队列名称：

| 原服务器影像目录 | 患者 | 检查 |
| --- | ---: | ---: |
| `/root/autodl-tmp/Datasets/manifest-1781750940386` | 575 | 1996 |
| `/root/autodl-tmp/Datasets/qingyuan_acrin_registered_minimal_20260804` | 291 | 1043 |

这两个输入目录的患者在最终清单中没有重叠。

更早的数据包是 `MOTFM-tumor-first/runs/registered_strict_a_z96_target_oracle/data_bundle.qingyuan_20260804.val102_high_transition`，含 867 人、3043 次检查，训练 765 / 验证 102。其 `split_policy` 为 `frozen_base_val_new_patient_holdout_v1`。ROI32 步骤继承该划分，核验并排除一名 T0 预测病灶为空的训练患者及其四次检查，得到最终 866 / 3039。

上述数据包已有更早的质量筛选、增量入组和转移排除记录；不能把 866 人解释为对公共目录全部患者重新随机划分。完全复用原实验时，应以原 manifest 的患者、时点和 split 为准。

## 真实处理链

1. **转换与配准。** 上游先准备 NIfTI、采集元数据及变换。同一次检查内，首次增强后和其他相位对齐到增强前 `aqc_0`；各期检查再对齐到患者 T0。ROI32 代码要求这些配准已经完成。原有策略保留刚性回退，并在涉及回退的影像对上记录空间有效性。
2. **预测 T0 病灶。** 使用 MAMA-MIA nnUNet，在未配准的首次增强后 `aqc_1` 上产生分割预测。ROI32 定位只使用 T0 的预测病灶，不用未来检查的病灶决定裁剪位置。
3. **建立固定裁剪网格。** 利用原始采集的方向、位置、间距及已保存的相位变换，把 T0 预测病灶映射到 RAS 网格。取最大 26 连通分量的包围盒中心，固定裁剪 `32×128×128`，ZYX 间距为 `2.0×0.7032×0.7032 mm`，约覆盖 `64×90×90 mm`。所有时点复用同一裁剪位置。影像用线性插值，mask / 采集覆盖用最近邻，网格外填 0。尺寸不会按病灶大小自适应扩大，部分病灶可能被截断。
4. **选取三相。** 按顺序取增强前 `aqc_0`、首次增强后 `aqc_1`、元数据 `post_late_Tk` 明确指定的晚期。第三相不是固定的 `aqc_2`，不能随意取第几个文件代替。
5. **标准化。** 只用训练患者的全部唯一检查的增强前 DCE0 裁剪，在有效采集覆盖内的非零前景计算全局统计量：`mean=201.38859897911084`，`std=252.09299166053188`。这套统计量共同用于三个相位的有效非零体素，执行 `(x-mean)/std`，背景和填充保持 0。这里不是每位患者单独做 z-score，也不是每相分别拟合统计量。
6. **打包与联结临床。** 保存 NPZ，主要包含 `image: float32 [3,32,128,128]`、`roi: bool [1,32,128,128]`、`support: bool [3,32,128,128]`。随后联结年龄、HR、HER2、MammaPrint、分配的治疗 Arm 和 pCR 标签，生成 VLA manifest。旧 NPZ 还可能含其他上游任务的字段；本项目主要读取上述 MRI / mask 数据。

当前仓库的 `versions/v1_evolution/scripts/prepare_registered_roi32.py` 是最后的审计和 manifest 构建步骤：它读取已经做好的 NPZ，不执行从 DICOM 开始的整套处理，也不重新归一化或调整影像大小。

公共目录保存的配准归档进一步说明了上游实现：

- DICOM 按时间相位分组，按采集方向和切片位置排序，DCE 保留原存储像素值并转 float32。NIfTI 写入的是单位占位 affine，真实空间信息保存在配套元数据中；不能只读取 NIfTI header 就推断原实验几何。
- 期内采用 Euler3D 刚性配准，所有相位对齐到 `aqc_0`。跨期将 T1–T3 对齐到 T0，先刚性，再 Elastix B-spline；形变或质量检查不通过时回退到刚性。
- 用于估计变换的均值图、CLAHE 和肿瘤加权图是配准辅助图，不是最终三相 MRI 的标准化方式。最终对原始相位组合期内和纵向变换后只重采样一次。

归档位于 `06_software_and_cache/workflows/ispy2_acrin_t0_registration_v1/code/registration/`。其输出目录与原服务器恢复脚本记录的 `/data1/cxx/organized_v1/03_registered_nifti` 一致，几何契约也吻合；本次未核验缓存生成当时的精确源码 revision 或跨服务器逐文件字节一致性。

实际来源链为：

```text
已配准 NIfTI / 元数据 / 变换 + T0 预测病灶 + 固定患者划分
  → MeWM-ISPY2 的固定 ROI32 裁剪与三相标准化
  → symmv4/symm-fm-trjepa 打包 NPZ
  → symm-fm-trjepa-v2 核验并硬链接
  → breast-mri-vla-jepa 构建临床 manifest
  → v1_evolution 复用这批影像和划分
```

## 本机可以复用什么

已找到的公共数据根目录是 `/public_fast_storage/cxx26/organized_v1`。其中有：

- `02_preprocessed_nifti/`：预处理影像。
- `03_registered_nifti/`：已配准影像。
- `05_manifests_and_logs/active/portable/`：可迁移的患者、检查、影像和路径清单。
- `06_software_and_cache/workflows/ispy2_acrin_t0_registration_v1/`：原配准工作流归档。
- `04_research_runs/pro61/`：既有研究代码、数据包与来源记录。

使用官方 TCIA → I-SPY2 Research ID 映射核对后，旧实验的 **866 名患者和 3039 次检查全部在本机预处理、配准清单中找到**，均无缺失或重复患者时点。按原代码实际选择 `aqc_0 / aqc_1 / metadata_late` 后，两套目录各有 **9117 / 9117 个相位文件实际存在**。晚期索引并不相同：6 / 5 / 4 / 3 / 7 分别对应 1499 / 1497 / 33 / 8 / 2 次检查。

公共清单整体包含 1095 名已配准患者、3767 次检查，大于这次要使用的固定子集。3039 个参考分割文件也存在，但不能据此判定原实验的 T0 模型预测分割已经存在。公共 `pro61` 中找到的旧 `data_bundle` 是 524 / 52 划分，与本次 764 / 102 不同，不能直接拿来替换原划分。

原始 DICOM 虽然列在公共清单中，其记录的 1104 个患者根目录在当前挂载下均不存在；当前确认完整的是本次所需的预处理和已配准 NIfTI。因此，可以复用本机已配准 NIfTI，接着做 ROI32 部分，不需要为此从 DICOM 重做前面的转换与配准。

逐项比对的无患者明细汇总保存在 `data/provenance/public_data_comparison.json`。

要重做完全相同的裁剪，还需要同一版 T0 预测 mask、物理几何 / 相位变换、晚期相位表、临床身份映射和原划分。普通肿瘤标注或 FTV mask 不能直接假定就是原实验的 MAMA-MIA 预测 mask。

原服务器这 866 个 T0 预测 mask 全部存在，压缩后合计约 **10.1 MB**；相位变换约 **0.24 MB**。如果本机影像完整而这些文件缺失，可以只补这些小型资源及几何记录，再在本机生成 NPZ，无需优先搬运整套约 **17.6 GB** 的旧 NPZ。

## 已保存的核对材料

本机 `data/provenance/` 保存了只读来源快照：

- `metadata/`：官方身份映射表、临床表和相位元数据。
- `source_preprocessing/`：原处理源码与相关配置。
- `data_bundle/`：更早数据包的配置、统计及划分策略元信息。

`data/registered_roi32/manifest.original.json` 是原实验清单；同目录 `manifest.json` 只改了影像路径。此次影像复制已经暂停，`views/` 只有部分文件，不能当作完整数据集使用。

数据、患者清单、原始来源快照和模型权重均在 Git 忽略范围内。本说明只记录汇总和方法。
