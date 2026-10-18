# 本发布副本的 CPU 核验

核验日期：2026-10-18。在本版本目录隔离 `PYTHONPATH=src`，使用已有 CPU Python 环境；没有读取真实患者行、影像或启动真实训练/评估。

| 检查 | 结果 |
|---|---|
| 现有完整 pytest | 251 passed，23.11 秒 |
| 模块 CLI / predict --help | 通过 |
| 外部更新制训练入口 scripts/train_overfit.py --help | 通过 |
| 已有 CLI smoke | 合成 A→B→C 两轮集成通过 |
| 合成 checkpoint 的 CLI predict + states-out | 通过；未来监督读取标记 false；小模型 states 为 [1,4,4,32]，未请求 T0 为零 |
| scripts/check_default_shape.py | 随机合成 [1,4,3,32,128,128] 前向通过，编码器 [1,256,128]，rollout [1,4,256,128]，输出均有限 |
| 15 个训练源码 Python 模块与整理前快照比对 | 完全相同 |
| LICENSE / pyproject.toml / requirements.txt 与快照比对 | 完全相同，保留原版权及依赖 |

核验环境：Python 3.12.13、PyTorch 2.5.1+cpu、NumPy 2.5.3、PyYAML 6.0.3、scikit-learn 1.9.1、pytest 9.1.1。合成测试及临时 checkpoint 保存在包外临时目录，不随仓库发布。该检查不重新验证历史 GPU 数值复现或临床性能。
