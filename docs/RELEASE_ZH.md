# 来源、整理范围与工程验证

整理日期：2026-10-18。本次只处理独立发布副本，原实验代码、权重、数据和结果保持冻结；没有真实训练或新的患者级评价。

## 源工程与发布目录

源工程在既有研究服务器 `seetacloud-bjb2` 的 `/root/autodl-tmp/line b/` 下：

| 源工程名 | 本仓库位置 |
| --- | --- |
| `breast-mri-vla-jepa` | `versions/v1` |
| `breast-mri-response-jepa-v2-joint` | `versions/v2_joint` |
| `breast-mri-vla-jepa-vq-encoder` | `versions/vq/encoder` |
| `breast-mri-vla-jepa-vq-dualtask` | `versions/vq/dualtask` |
| `breast-mri-vla-jepa-v1-evolution` | `versions/v1_evolution` |

本地未改动源快照在 `/home/cxx/work/breast_mri_jepa_release_20261018/source_snapshot/`；本仓库工作副本在同级 `breast-mri-jepa-versions/`。所有取回的源码、配置、测试、文档和许可证都映射到 [逐文件来源索引](SOURCE_INVENTORY.tsv)，包含原入口 `train.py` / V2 `train_response.py`，未只抽取最优模型代码。

V1 额外保留了此前本地已整理发布副本的聚合 CSV 与历史说明。原 `reports/` 内容移动到 `results_summary/`；选取的 `runs/` 中冻结配置和聚合 Markdown 移动到 `historical_runs/`。因此后者只是历史报告/配置，**不是完整运行备份**。原始工件位置仍可从历史文档查到。

## 发布适配

各项目的训练模型、数据合同和历史科学配置保留。发布适配集中在以下方面：

- 重写导航及输入输出文档，补充许可证范围和实际依赖；V1 LICENSE 与此前已核验发布正文一致，移除源副本中指向不存在 legacy 文件的错误例外。
- V1/VQ 的 ROI32 准备脚本改为显式传入 inventory / clinical 路径，消除原机器默认输入；没有新增预处理算法。
- VQ/evolution 历史队列、分析和 CUDA 检查脚本兼容本仓库分组目录，同时保留旧独立目录布局兼容。历史 YAML 不覆盖，运行前生成新的路径配置；原科学参数偏移仍拒绝，暖启动指向对应输出中的冻结父模型。
- 新增顶层 `tools/prepare_config.py`，只替换已有资源路径；新输出禁止覆盖，不能改变 fresh-start / warm-start 设计。它不更改网络、损失、种子、划分或验证定义。
- 新增 CPU 测试包装器和五项目隔离的 GitHub Actions。相同 Python 包名的版本不在一个环境中混装。

正式训练仍需私有数据与外部 codec；只改绝对路径不能保证任意新数据符合原数值与几何合同。直接单配置入口及历史完整队列的依赖见各版本文档。

## 验证方式

本地使用 Python 3.12 的既有环境验证，不修改其它研究项目环境。V1 / V2 使用 PyTorch 2.5.1 CPU；VQ / evolution 使用 PyTorch 2.12.1+cu130，但测试进程隐藏 CUDA、全部在 CPU 上运行。每个项目在独立进程内优先加载自己的 `src/`，执行原有完整 pytest；有合成 MRI/codec 的测试不读取患者数据。

CPU BF16 的 fused MHA / oneDNN 在部分处理器及 PyTorch 组合上不支持所需混合类型或反向操作。`tools/run_cpu_tests.py` 仅在测试进程内隐藏 CUDA、关闭这两类融合后端，保留全部测试和 BF16 路径；不改模型或历史 CUDA 配置。CI 使用 Python 3.11 + PyTorch 2.5.1 CPU、相同包装器，并分别构建 wheel/sdist。

完整套件检查时 V1 208、V2 251、VQ encoder 256、VQ dualtask 311、evolution 373 项通过。之后仅对三包队列修复了相对 cache 路径在后台进程切换 cwd 后的解析问题，各新增一项跨 cwd 回归；最终相关测试分别为 encoder 54、dualtask 70、evolution 87 项通过，未无依据重复整套。五个项目均成功构建 wheel 和 sdist，许可证及预期配置资源已检查。

各包完整检查记录：

- [V1](../versions/v1/docs/RELEASE_VALIDATION_ZH.md)
- [V2](../versions/v2_joint/docs/RELEASE_VALIDATION_ZH.md)
- [VQ encoder](../versions/vq/encoder/docs/RELEASE_VALIDATION_ZH.md)
- [VQ dualtask](../versions/vq/dualtask/docs/RELEASE_VALIDATION_ZH.md)
- [V1-evolution](../versions/v1_evolution/docs/RELEASE_VALIDATION_ZH.md)

顶层路径工具已逐一用五项目的真实历史配置检查：只改资源键，其余解析后的 YAML 值全部相同；拒绝覆盖既有输出、拒绝将原本关闭的初始化改为暖启动。发布前另检查相对链接、配置资源副本一致性、源码清单、依赖/许可证打包，以及待提交内容。

这些验证说明源码可安装、测试和进行合成工程运行；没有在新机器重新复现临床训练成绩。历史开发验证没有独立测试支持，性能解释见 [实验结果说明](EXPERIMENTS_ZH.md)。

## 不上传的研究工件

发布范围不含患者 MRI / 临床表 / manifest、image cache、外部 codec、正式训练 checkpoint、逐患者预测、拟合患者清单或原运行日志。已保留的 CSV 都是实验/种子/阶段/转移层级的聚合指标。测试里的虚构样本和临床小表仅用于合成测试。

`.gitignore` 排除数据目录、运行输出、常见影像与权重格式、私有配置、环境和缓存。新生成患者级产物应放在 `private_outputs/` 等被忽略目录；历史报告和配置保留在明确的 `historical_runs/` / `results_summary/`，便于与新运行区分。
