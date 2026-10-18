# 已配准三相 ROI32 适配

本适配保留固定 T0 和动态 T0–T3 的 pCR 预测及 JEPA 未来状态学习。独立 3D 编码器、Transformer、查询、EMA 教师和损失结构沿用原包；MRI flow 关闭。

## 数据与预处理

准备脚本读取现有 inventory 的 `visits`，按 `patient_id` 分组，解析 `visit_id` 的阶段后缀，按 inventory 所在目录解析 `file`。每名患者保留四槽，缺失检查写 `null`，不会将 T0/T2/T3 压缩成连续三次检查。

输入声明为：

```json
{
  "image_preprocessing": {"mode": "preprocessed_roi32"},
  "phase_order": ["pre_aqc0", "first_post_aqc1", "metadata_late"]
}
```

每个非空 visit 的 `image` 指向一个 NPZ。运行时只读取其中的 float32 `image[3,32,128,128]`，不读取 `latent`、`roi` 或 `support`。缓存已经采用训练集 DCE0 共同均值/标准差，背景与填充保持零，因此此模式不再次标准化或插值。数据模式和白名单提取的 `image_normalization` 随 manifest、normalization 与 checkpoint 保存，推理时必须匹配。

全量准备检查图像形状、有限值、support/roi 类型及维度、采集范围之外零背景、重复资产、阶段槽位、患者分割和临床联结。support 是采集覆盖，不能等同于肿瘤或前景 valid mask；它仅用于审核，不改变首版重构损失。

原 NPY/NIfTI manifest 缺省使用 `raw_t0_zscore`，保留原 T0 相位标准化和 resize 行为。两个输入模式不能在同一 manifest 中混用。

## 临床与信息边界

官方表按规范化患者 ID 联结，检查现有配对记录中的 pCR 和分配 Arm 是否冲突。保留年龄、HR、HER2、MP，缺失值为 `null`；四项基线字段默认 T0 已知。临床年龄缩放只从训练患者拟合。

Arm 采用 `assigned_arm_scenario`，将官方分配类别视为 T0 已知的实验场景。该设定没有补充剂量、周期、实际治疗变化或分配时间证明。未知 Arm 保留未知类别和 mask。

模型输入只有当前合法历史，未来影像位于独立监督对象。普通 pCR 前向不读取 EMA 未来状态或未来预测结果；请求 `--forecast-states` 才运行自主状态推演，输出是特征状态而非 MRI 图像。

固定划分为 764 train / 102 val / 0 test，共 3039 次检查。102 名患者用于开发验证，不改名为独立测试集。

## 训练与评估

在代码包根目录按 README 准备数据，再运行两份 `configs/registered_roi32_{t0,dynamic}.yaml`。两份配置均为一阶段 6000 次更新、学习率 1e-4、batch 1、梯度累积 4、CUDA BF16；只在合法历史范围上不同。

损失为 `BCE + 0.5*WM_L1 + 0.1*observed_reconstruction + 0.01*variance`。固定 T0 以验证 NLL 选模；动态版先对每名患者的合法阶段平均 NLL，再对患者平均。

```bash
PYTHONPATH=src python train.py evaluate-vla-jepa \
  --checkpoint runs/registered_roi32_dynamic/best.pt \
  --manifest data/registered_roi32/manifest.json --split val \
  --device cuda --output results/registered_roi32_dynamic_val.json
```

报告分别提供 T0–T3 的 AUROC、AUPRC、NLL 和 Brier。固定 T0 的训练命令加 `--resume` 可恢复对应 run；动态版同理，配置和数据保持原样。

新 checkpoint 为 `responsewm_raw_vla_jepa_checkpoint_v2`，保存配置、规范化状态、私有患者分区、manifest 结构、源代码文本快照及输入文件路径/大小/修改时间。恢复时直接比较，未记录 checksum 值。文件元数据不能证明影像字节完全不变。旧 v1 checkpoint 支持读取、评估和预测，续训要求新 v2 run。

## 工程验证

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python train.py smoke-vla-jepa \
  --config configs/raw_vla_jepa_smoke_t0.yaml --output runs/smoke_t0
PYTHONPATH=src python train.py smoke-vla-jepa \
  --config configs/raw_vla_jepa_smoke_dynamic.yaml --output runs/smoke_dynamic
PYTHONPATH=src python scripts/verify_registered_roi32.py \
  --manifest data/registered_roi32/manifest.json \
  --output runs/registered_roi32_verification --steps 4
```

真实短程验证使用完整尺寸输入，两版各做四次更新，并在原验证患者上检查前向与逐阶段评估。验证目录中的 checkpoint 是工程 smoke 权重，正式训练应使用新的 run 目录。短程 pCR 指标不作为临床或算法效果结论。

准备 manifest 和带患者行的 run 产物保存在私有目录，公开预检报告只含汇总。原 NPZ、配准图像、已有模型和压缩包保持原位。
