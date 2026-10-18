# V2：Breast MRI Response-JEPA 联合训练版

本包保留乳腺 MRI / I-SPY2 的 V2 训练、推理、消融、队列和分析代码。`phase: joint` 从随机初始化联合优化 MRI 表征、潜状态世界模型及最终 pCR 分类；原来的 A→B→C 分阶段路径也保留。

- **数据输入输出、安装及运行命令**：[INPUT_OUTPUT_ZH.md](docs/INPUT_OUTPUT_ZH.md)。
- **V2 最优读出消融**：[最终结果](results_summary/branch_ablation_20261010/README_ZH.md)、[结果解释](results_summary/branch_ablation_20261010/INTERPRETATION_ZH.md)。
- **冻结配置**：[18 份配置](historical_runs/branch_ablation_20261010/input_configs/)，包含固定 T0 / 动态及种子 17、43、71。
- **模型与数据**：[model.py](src/response_jepa/model.py)、[contracts.py](src/response_jepa/contracts.py)、[data.py](src/response_jepa/data.py)。
- **损失与更新制训练**：[objectives.py](src/response_jepa/objectives.py)、[update_engine.py](src/response_jepa/update_engine.py)、[外部训练入口](scripts/train_overfit.py)。
- **发布核验**：[251 项测试及合成输入输出检查](docs/RELEASE_VALIDATION_ZH.md)。
- **历史设计说明**：[联合训练](docs/JOINT_TRAINING_ZH.md)、[网络结构](docs/CURRENT_ARCHITECTURE_ZH.md)、[读出消融](docs/BRANCH_ABLATION_20261010_ZH.md)。历史文件的运行状态、绝对路径及 `runs/`、`reports/` 路径描述原环境；当前发布位置以本 README 为准。

V2 读出消融内部，动态 `F_forecast` 的三个种子平均开发验证患者等权 NLL 最低，为 **0.5713 ± 0.0082**（种子间标准差）。对应 [seed17](historical_runs/branch_ablation_20261010/input_configs/F_forecast_dynamic_seed17.yaml)、[seed43](historical_runs/branch_ablation_20261010/input_configs/F_forecast_dynamic_seed43.yaml)、[seed71](historical_runs/branch_ablation_20261010/input_configs/F_forecast_dynamic_seed71.yaml)。实际协议为最多 **3000 次优化器更新**、预热 200 次、每 100 次验证、500 次无改善早停；配置中 `epochs: 60` 是更新制运行时不使用的遗留字段。最终输出直接取 forecast head，备用 direct head 未接受分类监督。

输入为截止当前阶段合法可用的 T0–T3 MRI（每次检查 3 个 DCE 相位）、4 项临床变量及治疗 Arm。输出为最终 pCR 概率和可选的未来潜状态，**没有未来 MRI 像素生成器**。动态模型在新 MRI 到达后可使用同一组权重更新预测；固定 T0 模型只支持基线。

历史实验为 764 名训练患者、102 名反复使用的开发验证患者，没有独立测试结论。本发布包提供源码、配置及聚合说明；患者影像、清单、逐患者预测及正式权重不随代码发布。源码保留原 [MIT LICENSE](LICENSE) 及版权声明；依赖以 [pyproject.toml](pyproject.toml) 和 [requirements.txt](requirements.txt) 为准。
