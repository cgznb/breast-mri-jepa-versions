# 间距、JSON 原点与 mask 映射核查

核查日期：2026-10-18。范围为当前 866 名患者、3039 次检查，以及保留的 867 个原始 T0 预测（包含已排除的一个空预测）。本次只读源数据，没有重做分割、配准、裁剪或训练。

**结论：配准和 mask 映射均采用每例 JSON 记录的毫米间距，而不是 NIfTI 占位头中的 1 mm。本批没有发现间距在分割、配准输出、裁剪之间传错；但旧 JSON 确实存在原点错误和坐标语义混用。原点问题不能等同于 spacing 错误，也不能仅凭材料内部一致就保证所有历史 DICOM 或配准解剖结果完全正确。**

## 1. 间距如何参与计算

原生体素索引与物理坐标的关系为：

```text
物理坐标(mm) = origin + direction × diag(spacing_xyz) × 体素索引_xyz
spacing_xyz = [pixel_spacing[1], pixel_spacing[0], spacing_between_slices]
```

`pixel_spacing` 按 DICOM 的行、列顺序保存；转换到 XYZ 时交换前两项。层间距使用 `spacing_between_slices`，不能把 `slice_thickness` 的名字直接当作切片中心间距。

间距来源存在两条历史路径：旧 ISPY2 转换器直接读取 DICOM 的 `PixelSpacing`、`SpacingBetweenSlices` 和 `SliceThickness`；另一条 [dicom_prepare.py](../preprocessing/upstream/registration/src/ispy2_registration/dicom_prepare.py#L294) 路径按切片 IPP 在法向上的相邻距离中位数确定 Z 间距，取不到时才回退到上述标签。不能把后一脚本的算法泛化为所有旧病例都曾逐层重算间距。该中位数算法本身也不能证明不存在不规则切片间距；具体真实性需要原始切片位置证据。

- 分割输入由 [first_post_segmentation.py](../preprocessing/upstream/MeWM-ISPY2/mewm_ispy2/first_post_segmentation.py#L95) 从保存字段恢复 spacing、direction、origin，再转成 LPS 轴顺序。nnUNet 内部的 1 mm 模型重采样是后续操作。
- 配准由 [manifest.py](../preprocessing/upstream/registration/src/ispy2_registration/manifest.py#L286) 从 JSON 建立 Geometry；仅缺少 `spacing_between_slices` 时才回退到 `slice_thickness`。SITK/ITK 图像均明确调用 `SetSpacing` 和 `SetDirection`，见 [geometry.py](../preprocessing/upstream/registration/src/ispy2_registration/geometry.py#L44)。
- 配准计算把体积中心设为原点：`origin = -direction @ ((size_xyz - 1) * spacing_xyz / 2)`。这是毫米坐标的平移，不会把体素变成 1 mm，也没有取消不同检查之间的间距差异。
- 已配准各期输出采用各自患者 T0 的目标网格；其 source spacing 与 target spacing 可以不同。
- MRI NIfTI 的 identity affine 是归档存储约定，见 [io.py](../preprocessing/upstream/registration/src/ispy2_registration/io.py#L30)。这些 MRI 数组存为 ZYX，不能交给默认 NIfTI XYZ/affine 解释后直接与 mask 叠加。

## 2. T0 mask 怎么映射到裁剪网格

核心实现为 [registered_roi32_data.py](../preprocessing/upstream/MeWM-ISPY2/mewm_ispy2/registered_roi32_data.py#L312)：

1. 从原生首次增强后 mask 的 header 取得其物理网格，结合原生 JSON 的 spacing、direction、origin，将 mask 索引与原生 LPS 毫米坐标对应。
2. 转入原相位配准使用的中心坐标；利用保存的 T0 相位刚体变换，找到增强前参考相位 `aqc0` 中的对应位置。保存变换是“输出参考点 → 输入相位点”，因此从原生 mask 往目标网格走时使用其逆向关系。
3. 转为统一 RAS 网格中的坐标，取原生 mask 最大 26 连通域映射后包围盒的中心。
4. 以该中心建立 ZYX `32×128×128`、`2.0×0.7032×0.7032 mm` 的固定裁剪网格。这里的 spacing 是人为选择的输出分辨率；原生间距仍用于计算正确的采样比例和位置。
5. T0–T3 的 MRI 已经配准到 T0，按同一裁剪网格采样。MRI 用线性插值，mask 用最近邻；不会把原生 mask 的数组下标直接套在配准 MRI 上。

代码还检查 mask header 是否与原 JSON 重建的 LPS 网格一致、映射矩阵是否符合原几何审计、各期是否共享 T0 目标网格。这些检查可以防止混用不同版本的几何。

## 3. 旧 JSON 确实有哪些问题

### 原生影像的第 0 层原点可能写错

public 保存的旧转换器 `ispy2_dicom_to_breastdcedl_nii.py` 先在第 392–400 行从 `files[0]` 读取 IPP、PixelSpacing 等字段，随后第 477 行才按切片位置排序输出数组。因此所写 IPP 未必对应最终数组第 0 层。方向和 spacing 正确，也可能有错误的 origin。本轮未恢复历史运行时的精确源码 revision；具体修正病例以保存的原点恢复证据为据。

2026-10-11 的原点恢复记录覆盖 2452 次旧转换检查，其中 1444 次实际修正 origin。映射到本批后：

| 范围 | 原点修正 | spacing / shape / direction 与旧记录比较 |
| --- | ---: | --- |
| 866 个使用中的 T0 | 336 | 全部一致，容差 `1e-5` |
| 3039 次检查的 public 原生元数据 | 1143 | 全部一致，容差 `1e-5` |

### 配准输出的原点语义曾不明确

旧 registered JSON 顶层 IPP 保留 T0 的采集原点，但实际配准变换运行在中心坐标中。public 的规范化版本将 native 与 registered 几何分开，新增 `output_geometry`，将输出原点明确设为 `registration_centered`。

全量 public 3767 次检查中，旧 `target_geometry` 与规范化 `output_geometry` 的 shape、spacing、direction 逐值完全相同；顶层原点全部改变。规范化脚本和完成记录说明这一步只改元数据，没有重新配准或改变影像像素。这个记录只描述那次规范化操作，不能推导出 public 当前文件与另一个历史服务器快照逐字节相同。

### 是否改变了当前裁剪

对当前 866 人进行了两类数值验证：

- 重新读取实际 mask header、相位变换和元数据，重算映射与裁剪矩阵：866/866 与原 inventory、crop 收据一致，独立逐点换算最大误差约 `2.0e-13 mm`。
- 只将旧 JSON 原点替换为 public 修正值，并将 mask header 同步平移到对应原点：866/866 映射保持一致，矩阵最大差约 `2.3e-13`。再使用完整 public 几何字段、保持原历史相位变换时，探测点差异最大约 `1.45e-6 mm`；微小残差来自方向余弦浮点规范化。

原因是原流程进入配准中心坐标时会去掉采集原点；mask 与其配套 native JSON 使用相同原点，T0 参考网格也使用配套的原点。平移在完整的索引转换中抵消。因此此次发现的原点错误不会单独使现有裁剪发生同等毫米偏移。

**不能只把新 public native JSON 换进去而沿用旧 mask header。** 本次 336 个原点修正病例都会形成不配套的坐标声明，原代码的 mask 几何校验也会拒绝。以后重建必须同步适配输入几何，并明确使用哪一版变换。

## 4. 本批逐项核验结果与边界

| 检查 | 结果 |
| --- | --- |
| 867 个 T0 分割记录 vs 原 native JSON | 尺寸、spacing、direction、origin 全部一致 |
| 867 个 T0 mask header vs JSON 重建的 LPS 网格 | 全部通过；spacing 最大误差 `9.54e-8 mm`，origin 最大误差 `1.52e-5 mm` |
| 3039 次 registered 顶层间距 vs target_geometry | 全部一致 |
| 3039 次 target_geometry vs 同患者 T0 目标网格 | 全部一致 |
| 3039 次 public 新旧 source / target 几何 | shape、spacing、direction 全部在 `1e-5` 容差内一致 |
| 866 人 mask 到裁剪坐标链及原收据 | 全部通过 |

这些结果支持“本批按记录的毫米尺度一致处理”，不能独自证明每个历史 spacing 都来自无误的原始 DICOM。现存 public 数据还保存了历史 DICOM 头审计证据：对 2452 次旧转换检查，每例选一个完整 SER/PE 序列逐层核对 PixelSpacing 和 IPP，要求 IPP 符合所记层间距的规则网格；DCE 本体通过有限头探查和原转换器的跨相位/派生图一致性契约验证，不是全 DCE 逐层重查。当前挂载下未找到清单所列原始 DICOM，不能声称本轮已从全部原片重新确认。

另有两个独立限制：

- 本批 public 有 **1 次非 T0 检查**标记历史 FTV mask 的 X/Y 转置导致其物理网格与 DCE 不同，需要解剖对齐审阅。它是原配准使用的 FTV 引导 mask，**不是这 867 个 MAMA-MIA 预测 mask**。原点抵消证明不能排除该例配准引导存在问题，本次没有替换或重做其配准。
- public 的 866 个 T0 相位变换中，852 个与迁移原件 SHA256 相同，14 个有很小数值差异，原视野角点最大映射差约 `0.001388 mm`。其对应 28 个后增强 MRI 文件压缩大小也不同；未取得两份完整原 MRI 做像素比较，因此从当前 public 重建不保证逐字节复现旧 NPZ。当前迁移 NPZ 已在迁移时全量 SHA256 对照原服务器通过，本轮没有改变它们。

## 5. 核查材料

匿名汇总、可重复审计脚本、原规范化源码与 SHA256，以及需审阅检查的私有明细保存在 Git 忽略目录 `private_outputs/geometry_spacing_audit/`：

- `segmentation_metadata_summary.json` / `audit_segmentation_metadata.py`
- `mapping_summary.json` / `audit_mapping.py`
- `origin_correction_summary.json` / `audit_origin_correction.py`
- `public_metadata_audit.summary.json` / `public_metadata_audit.py`
- `public_metadata_audit.differences.json` / `public_metadata_audit.differences.py`
- `normalization_history_summary.json` / `normalization_history_sources/`

原 public 规范化证据根目录为 `/public_fast_storage/cxx26/organized_v1/05_manifests_and_logs/normalization/20261011_portable_v1/`；其发布脚本为 `06_software_and_cache/tools/organized_dataset/normalize_metadata.py`。患者级细节不纳入本说明或 Git。
