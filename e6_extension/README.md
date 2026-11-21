# E6 电价差和设备响应

从远程项目根目录解压。本扩展读取此前通过 E5 验收的 `e5_extension`、原始 `src`、冻结 E1 对照和处理后数据。

```bash
cd /home/jovyan/work/EV_Rolling_Horizon_Charging_Optimization
.envs/ev-charging/bin/python -m zipfile -e E6_Experiment.zip .
.envs/ev-charging/bin/python -u e6_extension/run.py smoke --out results/logs/e6_smoke_01
```

预验证包括已知功率与能量的单车例子、两队列短回放，以及额定与理想执行的对照。正式实验先核对原 TOU／额定响应与 E1 的 RH-LP 和 TOU Greedy 结果，再运行其余情景。五个对照加共享基准，共六个情景、两个方法和两个队列。

```bash
nohup .envs/ev-charging/bin/python -u e6_extension/run.py formal \
  --smoke-result results/logs/e6_smoke_01/verification.json \
  --out results/logs/e6_formal_01 > results/logs/e6_formal_01.console.log 2>&1 &
```

输出包括 `verification.json`、`run_config.json`、`e1_baseline_check.csv`、`ideal_nominal_check.csv`、`comparison.csv`、`daily_paired.csv`、`session_paired.csv` 和 `energy_satisfaction.png`。`clipped_execution_segments` 按车辆和执行片段计数。限幅电量只对执行层实际收到的请求积分，离站缺口单列。重试可换新的 `--out` 路径保留原运行证据。
