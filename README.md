# 电动汽车滚动时域充电优化

Caltech单站历史会话重放，比较Uncontrolled、TOU Greedy、
完整信息Static LP与RH-LP。使用聚合连续功率、实验TOU和容量情景；
服务、费用与峰值联合评价。

GitHub 代码视图提供源码和复现入口。获取数据、完整实验记录及下文引用的归档材料，请参阅 [完整项目快照](SNAPSHOT.md)。

## 阅读入口

- 课程报告：report/report.md
- 数据与假设：report/data_profile.md、report/assumptions.md
- 数学规格：report/model.md（保留阶段二历史状态，正式参数以冻结配置为准）
- 正式参数：configs/final.yaml、configs/normalization.yaml
- 最终发现：report/findings.md
- 全部结果：results/stage4_all_comparisons.csv
- 来源清单：results/run_manifest.csv
- 阶段四索引：report/stage4_artifact_index.csv

正式结果172行，对应27个运行中的方法／队列／情景组合，
不是172个独立运行。两个队列分别评价。

## 一条命令验证关键路径

在项目根目录使用Python 3.11运行，先确认python --version。
脚本会自行创建新环境，无须预先激活原项目环境。
安装依赖需要可用的软件包来源。

    python experiments/reproduce_stage5.py --output results/logs/reproduction_run_001

每次使用新的输出目录名。
该命令依次完成：复制必要文件、新建隔离环境、安装固定依赖、
pip check、静态16项检查、RH17项检查、汇总表重算比对与绘图。

状态文件位于指定输出目录的status.json。
成功时status=passed、stage=complete；
数值检查证据和图片位于其workspace/results/reproduction/。
失败时检查error.txt、install.log、pip_check.log和worker.log。

实测总耗时166.092秒；其他机器的安装与运行时间可能不同。
程序使用requirements.reproduce.txt；原requirements.txt保持不变。
该验证不会重跑全量正式实验或原始数据预处理。

## 已通过的干净环境记录

- Python：3.11.18，不继承系统site-packages。
- 静态检查：16/16。
- RH检查：17/17，包含ACN-Sim闭环。
- 三份重算表：40／12／12行，与报告表文件哈希全部一致。
- 新图：E1_summary.png、demand_error_summary.png。
- 证据：results/logs/stage5_clean_reproduction_20261116T074902798172Z/status.json。
- 原始数据预处理与全部正式实验：本次未重跑。

## 数据版本

原始快照：
data/raw/caltech_12weeks_20261111T143402Z_9b087d/，
查询条件与哈希见该目录manifest.json。

处理数据：
data/processed/caltech_12weeks_20261111T143402Z_9b087d/。

完整实验需要原项目对应的处理数据、电价、冻结配置与来源文件。
新下载的数据不能自动视为同一快照。
数据来源为ACN-Data；引用与适用范围见课程报告。

## 正式E1重跑入口

以下命令按已核对的正式入口参数编写；
阶段五本次未重新执行该全量命令。
需要完整项目数据，且输出目录必须是新目录。

    python experiments/run_test_evaluation.py results/logs/main_test_replay_001 --split test --alpha 0.5 --capacity-kw 50 --base-hours 24 --static-time-limit 120 --frozen-config-sha256 7cc860d7eb6e76e19e7eff16a47116589e462d571c78d52121f2d4ba0c0c3b7b

历史E1约耗时70分钟。
权重／容量新增情景历史约4小时43分钟，
误差与提前离站情景合计历史约20小时22分钟；
时间均为原环境观察值。

## E2／E3与E4批处理入口

在已安装requirements.reproduce.txt依赖的Python 3.11环境中执行。
每次使用新的输出目录；以下命令本次仅核对入口，未重新启动长实验。

    python experiments/run_frozen_batch.py sensitivity --output results/logs/sensitivity_replay_001

    python experiments/run_frozen_batch.py robustness --output results/logs/robustness_replay_001

两个入口均复用冻结协议指向的原E1结果，校验来源与输出哈希。
它们不自动改为新重跑E1，也不修改种子、因子清单或参数。
应保留随包的原E1轨迹和审计目录。
若需设计全新的E1及后续实验，应另建、审计并冻结新协议。

## 清洗与数据使用

随包保留原始、清洗、事件与外部参数快照。
清洗划分脚本已有独立进程一致性验证；其入口为：

    python src/data_preprocessing.py

本次干净环境未重新执行原始数据到所有中间文件的全链路重建。
完整实验使用已归档处理快照；快速验证仅使用合成输入和冻结结果。
原始数据来源、筛选、排除和信息释放规则见报告与data/raw下的manifest。

## 最终交付及验证

逐文件清单：report/stage5_final_artifact_index.csv。
成果包包含代码、固定配置、依赖清单、数据快照、报告、
正式结果、轨迹与历史审计证据。

不包含虚拟环境、缓存或重复项目副本；
干净环境副本中用于报告引用的验证输出仍保留。
旧失败记录保留，不以本阶段归档覆盖原实验来源。

压缩包旁的.verification.json记录压缩包SHA-256、
逐文件校验结果和本阶段验收范围。
完整正式实验及原始数据预处理本次没有重跑。


## 结果解释边界

Static LP拥有完整未来信息，其完整时段可行域与在线实际执行口径不同。
较低费用可能伴随较少交付，不能直接解释为同服务节费。
严格主队列测试仅39条，补充队列468条，测试21天。
实验不包含建筑负荷、原网络支路与相位约束或真实账单核算。
