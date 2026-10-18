# 本机 CUDA 环境

本机使用独立的 `versions/v1_evolution/.venv`，不修改系统 Python 或其他版本的环境。Python 为 **3.12.1**，PyTorch 为 **2.5.1+cu121**，NumPy 为 **1.26.4**。CUDA 运行库随 PyTorch wheel 安装，不需要另装系统 CUDA Toolkit。

机器为 NVIDIA A100-SXM4-80GB，驱动 535.129.03（`nvidia-smi` 显示最高 CUDA 12.2）。因此选用 CUDA 12.1 wheel；系统原有的 CUDA 13.0 PyTorch 不适合当前驱动。`nvidia-nvjitlink-cu12` 同时固定为 12.1.105。

## 使用现有环境

从仓库根目录执行：

```bash
source versions/v1_evolution/.venv/bin/activate
cd versions/v1_evolution
python train.py --help
```

以上命令只显示帮助。数据和配置路径由本机运行配置提供，避免修改历史实验 YAML。

## 重建环境

以下命令从仓库根目录执行，使用 Python 3.12 创建环境。依赖锁不包含本机绝对路径，项目源码另以 editable 方式安装：

```bash
python3.12 -m venv versions/v1_evolution/.venv
source versions/v1_evolution/.venv/bin/activate
python -m pip install -r versions/v1_evolution/requirements/runtime-cu121.lock.txt
python -m pip install --index-url https://pypi.org/simple --no-build-isolation --no-deps -e 'versions/v1_evolution[raw,test]'
python -m pip check
```

锁文件明确使用官方 PyPI 和 `https://download.pytorch.org/whl/cu121` 额外索引，以取得 `torch==2.5.1+cu121`。当前机器默认的第三方 PyPI 镜像曾返回 SSL EOF，重建时以上索引覆盖该默认设置。首次安装 wheel 需要约数 GB 下载和约 6 GB 环境空间，可把 `TMPDIR` 和 `PIP_CACHE_DIR` 指向 personal storage 上自己的临时目录。

已安装 `raw` / `test` 依赖及 SimpleITK、scikit-image，可以读取迁移的 NPZ、NIfTI 和临床表格，并导入基本 ROI32 处理依赖。完整 nnUNet / MeWM 分割再生成流程需要其原始专用环境；这份 VLA 环境没有安装那套训练框架。

## 验证与历史环境边界

本机通过现有 `tools/run_cpu_tests.py` 检查 evolution 模型、数据特征与训练/精确恢复三组测试，**45 项通过，14.84 秒**；CLI 帮助、`pip check`、基础处理依赖导入，以及 A100 上 BF16 矩阵乘法前向和反向均通过。具体结果、完整 `pip freeze` 和 GPU 信息保存到仓库根目录的 `private_outputs/setup_runtime/`，不含正式训练结果。

仓库的 [发布验证记录](RELEASE_VALIDATION_ZH.md) 使用 Python 3.12 / PyTorch 2.12.1+cu130 执行 CPU 测试；本机采用上述兼容驱动的 PyTorch 版本。原正式训练服务器的全部软件版本尚未逐项锁定，当前环境验证代表功能可运行，不等同于承诺历史训练逐位一致。科学配置保持原数值，软件、GPU 和浮点算子版本差异仍可能改变数值结果、速度和显存需求。
