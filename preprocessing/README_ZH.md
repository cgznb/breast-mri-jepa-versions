# 原数据生成代码

这里保存原服务器的数据生成代码、配置、依赖声明和许可证，共 515 个文件。
处理方法沿用原实现；4 个文件中的真实患者示例已替换为中性示例，其原件保存在被 Git 忽略的
`data/provenance/source_preprocessing/`。其余文件与复制时的原服务器字节一致。

| 目录 | 用途 |
| --- | --- |
| `upstream/registration/` | DICOM 准备、期内相位对齐、纵向配准到 T0 |
| `upstream/MOTFM-tumor-first/` | 固定患者筛选、bundle 构建、扩充及划分所用的原脚本和工具 |
| `upstream/nnUNet-MAMA-MIA/` | 原服务器使用的 MAMA-MIA 作者 nnUNet 分支 |
| `upstream/MeWM-ISPY2/` | 预测病灶、固定 ROI32 裁剪、三相选择与标准化 |
| `upstream/geometry_audit/` | 原裁剪几何审计和中心计算的辅助脚本 |
| `upstream/longitudinal_temporal_pillar/` | 几何审计所需模块及历史元数据构建脚本 |
| `upstream/symmv4/symm-fm-trjepa/` | 将三相影像、ROI、support 与旧 VQ latent 打包为 NPZ |
| `upstream/symm-fm-trjepa-v2/` | 核验并硬链接同一批 NPZ |

完整阶段、输入、环境边界见 [数据生成代码说明](../docs/PREPROCESSING_SOURCES_ZH.md)。
现在运行模型使用已迁移的 `data/registered_roi32/`，参见 [本机运行](../docs/LOCAL_RUN_ZH.md)。

原配置含旧服务器绝对路径和旧缓存身份绑定，不能直接当作本机重建命令使用。
本次没有运行预处理或训练；重新生成数据前需适配路径并逐阶段检查输入与缓存契约。

`SOURCE_MANIFEST.json` 记录每个包的原路径、复制时 Git revision、原始与归档 SHA256。
revision 反映复制时的 checkout，不能单凭它证明历史缓存生成时的精确源码版本。
可在项目根目录验证归档：

```bash
python3 preprocessing/verify_sources.py --check-syntax
```
