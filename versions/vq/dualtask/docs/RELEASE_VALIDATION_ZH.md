# VQ dualtask 发布副本验证（2026-10-18）

本次只验证整理后的代码与工程适配；没有运行真实患者训练/推理，没有新增开发验证选择或测试集评估，也未覆盖历史结果。

## 最终 CPU 测试

独立进程使用 Python 3.12.13、PyTorch 2.12.1+cu130，根目录 CPU runner 隐藏 CUDA 并关闭 fused MHA/oneDNN。结果为 **311 passed（30.36 秒）**，包含全部既有模型、数据、信息边界、梯度、冻结/微调、断点恢复和队列测试，以及新增发布资源路径合同测试。没有跳过 CPU BF16 项。命令从仓库根目录运行：

```bash
python tools/run_cpu_tests.py --project versions/vq/dualtask -- -q -p no:cacheprovider
```

虽然 PyTorch 构建名含 CUDA，测试的设备是 CPU；本次没有使用 GPU 或真实 MRI。各包单独进程执行，避免同名导入包互相覆盖。

直接使用本机默认后端时，PyTorch 2.5.1 的两个 BF16 项在 Transformer eval fastpath 中出现 BF16/FP32 类型不匹配；PyTorch 2.12.1 的默认 oneDNN 则不支持本机 AVX2 BF16 backward。CPU runner 关闭这两个后端后全部通过。模型、损失与训练 `src/` 没有为测试做更改；该结果不表示默认 CPU BF16 后端支持正式训练。

## CLI 和发布工程适配

`train.py --help`、其 train/evaluate/predict 子命令帮助、`scripts/prepare_registered_roi32.py --help`、`scripts/evaluate_registered_roi32_world.py --help` 及 `scripts/run_vq_dualtask_experiment.py --help` 均正常退出（0），未读取患者数据。

工程适配将旧兄弟工程路径解析到当前版本目录；启动前在新输出 `input_configs/` 生成资源路径副本，默认 codec 为包内 `weights/roi32_codec.pt`，支持显式覆盖 codec/缓存和自定义输出。原始 `configs/`、包内资源配置及科研 `src/` 保持与来源快照一致。新增测试验证源 YAML 字节不变、非资源科学参数不变、阶段 B 指向同一新队列 A best、只读计划不创建目录、恢复时拒绝静默更换资源。

原队列的调度测试使用虚构配置路径，因此在这些纯调度 fixture 中隔离资源准备层；真实资源准备由新增合同测试覆盖，生产仍拒绝缺失配置。准备 ROI32 脚本只把历史机器默认 inventory/clinical 改为必传参数。其余发布调整为导航、输入输出、来源/许可说明与缺失历史工件的链接说明。

基础代码保留 MIT，派生 VQ 连续前端继续保留 CC BY-NC 4.0 及归属，没有重授许可。正式外部权重和临床数据不随源码发布。测试验证工程实现，不替代独立患者测试或历史成绩重现。
