# 项目协作约束

## MRI 元数据与 mask 的几何必须配套

用户明确要求长期记住并在相关操作时提醒：**不能直接换成新 public JSON，同时继续使用旧 MAMA-MIA mask header。**

- 当任务涉及替换或升级 metadata、把原生 MRI 与旧 mask 组合、从 public 数据重建裁剪/NPZ，或重做上游分割与配准时，在依赖这些输入进行处理前，用中文提醒这条约束，并检查它们是否来自配套版本。
- 新 public native JSON 修正了部分采集原点；旧 mask header 仍记录旧原点。registered JSON 的 `output_geometry` 则使用配准中心坐标。必须区分 native LPS 和 registration-centered LPS，不能把配准输出几何直接套给原生 mask。
- 联合核验尺寸、数组轴序、spacing、direction、origin、同患者 T0 目标网格，以及相位/纵向变换的方向和来源版本。仅修改路径或确认 spacing 相同，不足以证明配套。
- 若需适配，应保留原始 JSON、mask 和变换，在独立派生文件中完成有依据的坐标转换或重采样，并重新验证映射与裁剪采样位置。不能为了通过校验而盲目覆盖 mask header、跳过几何检查或修改原档。
- 2026-10-18 核查确认：当前 866 个 T0 中有 336 个 native 原点被 public 修正。仅当 JSON、mask 与参考网格同步处理、且其他几何和变换保持匹配时，该原点变化在原裁剪坐标链中抵消；这不是允许任意混用的依据，也不等于逐例证明配准解剖准确。
- 直接读取已迁移、已核验的现成 NPZ 不涉及重新组合 JSON 与 mask，无需为此反复请求确认或重做预处理。若操作不受该约束影响，正常继续。

核查依据：[间距、JSON 原点与 mask 映射核查](docs/GEOMETRY_SPACING_AUDIT_ZH.md)。可重复审计脚本与详细结果位于 Git 忽略目录 `private_outputs/geometry_spacing_audit/`；患者级数据不得写入本文件。
