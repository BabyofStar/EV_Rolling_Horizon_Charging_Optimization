# E5 计算效率与滚动窗口

从远程项目根目录执行。ZIP 内的 `e5_extension/` 是独立入口，使用项目冻结的 LP worker、TOU 模块和 ACN-Sim 适配器。源码和输入指纹写入运行目录。开发集预跑与正式批处理均按五个配置、两个队列串行执行。

```bash
cd /home/jovyan/work/EV_Rolling_Horizon_Charging_Optimization
test ! -e e5_extension && .envs/ev-charging/bin/python -m zipfile -e E5_Experiment.zip .
.envs/ev-charging/bin/python -u e5_extension/run.py smoke --out results/logs/e5_smoke_01
```

开发集预跑包括三种步长的真实 LP 合成例、注入超时的 EDF 回退、晚提交与跨午夜事件检查，以及按首个主队列到达日开始的连续两日闭环回放。成功后查验 `results/logs/e5_smoke_01/verification.json`，随后启动正式批处理。

```bash
.envs/ev-charging/bin/python -u e5_extension/run.py formal \
  --smoke-result results/logs/e5_smoke_01/verification.json \
  --out results/logs/e5_formal_01
```

`--out` 指向新的目录。重试时换用新的目录名，以保留原先的运行证据。正式批处理先跑 15 分钟／24 小时基准，逐队列对照冻结 E1 的服务、费用与峰值，再跑其余四个配置。日志按站点自然日打印进度。

| 文件 | 内容 |
|---|---|
| `verification.json` | 完成状态、结果复核标志、源文件和数据指纹 |
| `comparison.csv` | 各配置和队列的服务、费用、峰值、规模和计算性能 |
| `concurrency_summary.csv` | 按开发集并发分档的指标汇总 |
| `daily_paired.csv` | 同队列、同日相对 15 分钟／24 小时基准的差值 |
| `e1_baseline_check.csv` | 正式 15 分钟／24 小时与 E1 的逐项核对 |
| `timing.png`、`e5_report.md` | 计时对照图及结果摘要 |

控制器耗时包括模型输入构造、启动求解子进程、两层求解及首步计划检查。两层求解时间统计成功取得两层结果的步骤。按完整时段保留既有连接边界规则，电费和 15 分钟均值峰值采用共同的评价口径。每日并发档位按真实连接的日峰值分档，仅供结果统计；并列值归入同一档，可能产生空档。每日缺口和满足率按会话实际离站归集，午夜整点离站归到前一日。
