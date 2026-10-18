# 2026-10-18 源码整理验证

本次只验证发布工程、现有合成测试及输入输出接口，没有训练或评价真实患者，没有重新选择候选、种子、权重或阈值。

| 项目 | 结果 |
| --- | --- |
| 完整 CPU 测试 | **373 passed，36.57 秒** |
| 运行环境 | Python 3.12、PyTorch `2.12.1+cu130`；`CUDA_VISIBLE_DEVICES=''`，实际全部使用 CPU |
| CPU 后端 | 通过根目录 `tools/run_cpu_tests.py` 在测试进程中关闭 fused MHA fastpath 和 oneDNN；避免本机 CPU BF16 后端限制，训练源码保持原样 |
| 独立 CLI smoke | 小型合成数据训练 4 步，生成 best / last、开发验证 JSON、单例预测 JSON、`.states.npz`，正常退出 |
| 20 次训练计划 | `scripts/run_evolution_experiment.py --plan` 正常生成 20 项，指定输出目录未创建 |
| 配置与训练代码 | 对原核实快照逐文件比较，`src/`、`configs/`、`licenses/` 均无字节差异 |
| 脚本语法 | 全部 `scripts/*.py` AST 解析通过 |

完整测试包括 query 请求不影响 pCR、真实未来不进入分类、T3 无需转移、history 有效性 / 因果遮罩、残差初始 copy、自由多步梯度、冻结前端、encoder64 绕过 pre_quant、优化器更新、精确恢复和概率预测。新的工程路径检查还验证只读计划、资源副本只改路径、原 YAML 字节保留、新输出父 checkpoint 重定位及资源变更时拒绝恢复。

发布副本的工程变更在 `scripts/`：识别新 monorepo 版本位置并兼容旧 sibling 布局；新增 runtime 资源路径副本和 `--codec` / `--image-cache` 参数；调度 / checkpoint 校验 / 进程接管共用实际运行配置。历史结果仅修改失效链接和未发布工件的说明，数值保持原记录。

默认 smoke 使用基础 CNN 小配置；encoder64 自主训练路径由现有 evolution 集成测试中的随机合成 codec 检查。合成结果仅证明工程链路可运行，不能作为临床性能或正式结果复现证据。真实 ROI32 数据、codec 和正式训练 checkpoint 仍为外部依赖。
