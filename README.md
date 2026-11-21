# 电动汽车滚动时域充电优化

本项目基于加州理工学院（Caltech）单站的 ACN-Data 历史充电会话，在聚合站点容量约束下重放车辆到站、充电和离站过程，比较四种调度方法的充电满足率、电费、站点负荷峰值和计算开销。

## 完整项目下载

`main` 分支提供可直接浏览的源代码、配置、Notebook、测试和复现入口。原始与处理后数据、完整实验结果、逐步轨迹、验收记录及历史交付包均在 [完整项目 Release](https://github.com/BabyofStar/EV_Rolling_Horizon_Charging_Optimization/releases/tag/project-snapshot-20261121) 中。

请在 Release 的 **Assets** 区域下载以下三个文件：

- `EV_project.tar.gz.part-000`
- `EV_project.tar.gz.part-001`
- `SHA256SUMS`

将三个文件放在同一目录。在 Linux 或 Windows 的 WSL 终端中运行：

```bash
sha256sum -c SHA256SUMS && \
  cat EV_project.tar.gz.part-000 EV_project.tar.gz.part-001 | tar -xzf -
```

解包后进入 `EV_Rolling_Horizon_Charging_Optimization/`，再运行后面的复现命令。历史虚拟环境随快照保存，复现时请按固定依赖创建新环境。更多说明见 [SNAPSHOT.md](SNAPSHOT.md)。

## 调度方法

| 方法 | 调度规则 |
| --- | --- |
| 容量受限无序充电（Uncontrolled） | 在单桩功率和站点总功率限制下，尽早满足已知充电需求。 |
| 分时电价贪心（TOU Greedy） | 根据车辆可用时段和预计离站时间，优先安排在较低电价时段充电。 |
| 静态线性规划（Static LP） | 利用完整未来信息一次性求解整个测试区间，作为离线优化参考。 |
| 滚动时域线性规划（RH-LP） | 定期根据当前信息重算计划，只执行首个控制时段，再依据实际交付更新。 |

RH-LP 先最小化未满足电量，再在服务约束下优化电费与站点负荷峰值。严格主队列与晚提交补充队列分别运行和评价。

## 实验与结果

| 实验 | 内容 |
| --- | --- |
| E1 | 四种调度方法的正式测试对照。 |
| E2／E3 | 目标权重与站点容量的敏感性分析。 |
| E4 | 需求估计误差与提前离站的鲁棒性分析。 |
| E5 | RH-LP 控制间隔与基础窗口对照，共 10 组。 |
| E6 | RH-LP 与 TOU Greedy 的峰谷价差和设备响应对照，共 24 组，含理想与额定响应差异复核。 |

必做实验共归档 **27 个正式运行、172 组方法／队列／情景结果**。测试覆盖 **21 天**，严格主队列含 39 条会话，晚提交补充队列含 468 条。

- **主实验**中，严格主队列的 RH-LP 相比容量受限无序充电，将最高峰值从 **26.62 kW 降至 17.20 kW**，充电满足率也从 **87.78% 降至 83.04%**。
- **E5** 中，控制间隔固定为 15 分钟时，基础窗口从 24 小时缩至 4 或 8 小时，两队列的满足率变化均小于 **0.1 个百分点**。处理耗时来自单批串行实测。
- **E6** 中，峰谷价差减半后，RH-LP 两队列的峰时充电量均增加。设备执行功率上限降至额定值的 80% 后，RH-LP 两队列的满足率分别下降 **6.49 和 7.53 个百分点**。

结果适用于本次单站历史回放与实验电价、容量设定。费用和削峰表现需结合充电服务水平解读。

## 阅读入口

| 内容 | 位置 |
| --- | --- |
| 课程报告 | [report/report.md](report/report.md) |
| 冻结参数 | [configs/final.yaml](configs/final.yaml)、[configs/normalization.yaml](configs/normalization.yaml) |
| E5、E6 运行说明 | [e5_extension/README.md](e5_extension/README.md)、[e6_extension/README.md](e6_extension/README.md) |
| 数据画像、假设、数学规格与最终发现 | 快照内的 `report/data_profile.md`、`report/assumptions.md`、`report/model.md`、`report/findings.md` |
| 正式结果与运行清单 | 快照内的 `results/stage4_all_comparisons.csv`、`results/run_manifest.csv` |
| E5、E6 结果与差异复核 | 快照内的 `results/logs/e5_formal_01/`、`results/logs/e6_formal_01/` 及根目录对应结果包 |

## 复现关键路径

在解包后的项目根目录使用 **Python 3.11**。脚本会创建隔离环境，并从 `requirements.reproduce.txt` 安装固定依赖。安装时需要可用的软件包来源。

```bash
python experiments/reproduce_stage5.py --output results/logs/reproduction_run_001
```

每次运行使用新的输出目录。成功状态记录在 `status.json`，其中 `status=passed`、`stage=complete`。重算表和图位于该输出目录下的 `workspace/results/reproduction/`。

原验收环境为 Python 3.11.18，关键路径用时约 **166 秒**。静态检查 **16/16**、滚动调度检查 **17/17** 均通过，三份重算汇总表与冻结结果的 SHA-256 一致，并重新生成两张图。该验收覆盖合成样例、ACN-Sim 闭环和冻结结果汇总。

## 正式实验入口

以下命令需要完整快照中的数据、配置和依赖。各命令使用新的输出目录，E2–E4 批处理沿用冻结协议指定的 E1 结果。

```bash
python experiments/run_test_evaluation.py results/logs/main_test_replay_001 \
  --split test --alpha 0.5 --capacity-kw 50 --base-hours 24 \
  --static-time-limit 120 \
  --frozen-config-sha256 7cc860d7eb6e76e19e7eff16a47116589e462d571c78d52121f2d4ba0c0c3b7b

python experiments/run_frozen_batch.py sensitivity \
  --output results/logs/sensitivity_replay_001

python experiments/run_frozen_batch.py robustness \
  --output results/logs/robustness_replay_001
```

原环境中，E1 用时约 70 分钟，权重与容量敏感性实验约 4 小时 43 分钟，误差与提前离站实验约 20 小时 22 分钟。E5 和 E6 的启动命令及输出说明见各自的扩展目录。

## 数据来源与归档

数据来源为 [Caltech ACN-Data](https://ev.caltech.edu/dataset)。快照内的原始数据位于 `data/raw/caltech_12weeks_20261111T143402Z_9b087d/`，查询条件和哈希记录在同目录的 `manifest.json`。处理后数据位于 `data/processed/caltech_12weeks_20261111T143402Z_9b087d/`。

清洗与划分脚本入口为 `python src/data_preprocessing.py`。复核正式结果时，请使用归档的冻结数据版本。

`deliverables/` 中保存了阶段五成果包及其 `.verification.json`。该包记录原定必做范围的验收，E5 和 E6 的追加结果随完整项目快照另行归档。
