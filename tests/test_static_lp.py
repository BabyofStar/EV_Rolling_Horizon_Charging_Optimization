"""合成静态 LP 验证；不读取 ACN 会话或测试集。"""
from pathlib import Path
from copy import deepcopy
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch
import argparse
import csv
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import optimization_model as opt

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
OUT = Path(args.output)
OUT.mkdir(parents=True, exist_ok=False)

def case(demand=1.0):
    return {
        "dt": 0.25, "alpha": 0.5,
        "C_ref": 1.0, "P_ref": 2.0,
        "epsilon_U": 1e-5, "time_limit_seconds": 30,
        "slots": [
            {"label": f"t{t}", "day": "day1",
             "price": price, "capacity": 2.0}
            for t, price in enumerate([0.1, 0.1, 0.4, 0.4])
        ],
        "sessions": [
            {"id": "A", "demand": demand, "pmax": 2.0,
             "available": [0, 1, 2, 3]}
        ],
    }

def near(actual, expected, tol=3e-5):
    assert abs(actual-expected) <= tol, (actual, expected, tol)

def audit_exports(c, directory, r):
    """从导出的原始功率独立核算，不采用求解器内部表达式。"""
    with (directory / "plan.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        rows = list(csv.DictReader(f))
    with (directory / "sessions.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        vehicle_rows = list(csv.DictReader(f))

    cars = {v["id"]: v for v in c["sessions"]}
    labels = {s["label"]: t for t, s in enumerate(c["slots"])}
    loads = [0.0] * len(c["slots"])
    delivered = {i: 0.0 for i in cars}
    seen = set()
    max_power = 0.0
    for row in rows:
        i, t = row["sessionID"], labels[row["slot"]]
        assert i in cars and (i, t) not in seen
        seen.add((i, t))
        assert row["day"] == c["slots"][t]["day"]
        p = float(row["power_kW"])
        allowed = t in cars[i]["available"]
        limit = cars[i]["pmax"] if allowed else 0.0
        max_power = max(max_power, -p, p-limit)
        near(float(row["energy_kWh"]), p*c["dt"], 1e-10)
        loads[t] += p
        delivered[i] += p*c["dt"]

    assert len(seen) == len(cars)*len(c["slots"])
    for t, slot in enumerate(c["slots"]):
        max_power = max(max_power, loads[t]-slot["capacity"])
    assert max_power <= 1e-6

    vehicle_ids = [v["sessionID"] for v in vehicle_rows]
    assert len(vehicle_ids) == len(set(vehicle_ids))
    assert set(vehicle_ids) == set(cars)
    max_energy = 0.0
    total_u = 0.0
    for row in vehicle_rows:
        i = row["sessionID"]
        u = float(row["unmet_kWh"])
        near(float(row["target_kWh"]), cars[i]["demand"], 1e-10)
        near(float(row["planned_kWh"]), delivered[i], 1e-10)
        max_energy = max(
            max_energy, abs(delivered[i]+u-cars[i]["demand"])
        )
        assert -1e-5 <= u <= cars[i]["demand"]+1e-5
        assert delivered[i] <= cars[i]["demand"]+1e-5
        total_u += u
    assert max_energy <= 1e-5
    near(total_u, r["U_stage2_kWh"], 1e-8)
    assert total_u <= r["U_star_kWh"]+c["epsilon_U"]+1e-7

    cost = sum(
        loads[t]*s["price"]*c["dt"]
        for t, s in enumerate(c["slots"])
    )
    near(cost, r["cost_USD"], 1e-8)
    for day in dict.fromkeys(s["day"] for s in c["slots"]):
        peak = max(
            [c.get("past_peak", {}).get(day, 0)]
            + [loads[t] for t, s in enumerate(c["slots"])
               if s["day"] == day]
        )
        near(peak, r["peak_by_day_kW"][day], 1e-8)
        assert r["peak_variable_by_day_kW"][day] >= peak-1e-6

    return {
        "independent_power_violation_kW": max_power,
        "independent_energy_residual_kWh": max_energy,
    }

records = []
results = {}

def run(name, c, expected_u, expected_cost=None, expected_peak=None):
    directory = OUT / name
    r = opt.solve_lp(c, directory)
    assert r["status"] == "passed"
    near(r["U_star_kWh"], expected_u, 1e-6)
    independent = audit_exports(c, directory, r)
    if expected_cost is not None:
        near(r["cost_USD"], expected_cost)
    if expected_peak is not None:
        near(r["peak_by_day_kW"]["day1"], expected_peak)
    # Urgent case: exact epsilon-aware hand calculation.
    if name == "urgent_departure":
        near(r["U_stage2_kWh"], 0.5 + c["epsilon_U"], 1e-8)
        near(
            r["peak_by_day_kW"]["day1"],
            2.0 - c["epsilon_U"] / c["dt"],
            1e-7,
        )
        near(r["cost_USD"], 0.1 * (0.5-c["epsilon_U"]), 1e-8)
    results[name] = r
    return {
        "U_star_kWh": r["U_star_kWh"],
        "cost_USD": r["cost_USD"],
        **independent,
    }

jobs = []

c = case(0)
jobs.append(("zero_demand", c, 0, 0, 0))

c = case(1)
c["sessions"][0]["available"] = []
jobs.append(("no_available_slots", c, 1, 0, 0))

c = case(1)
c["sessions"][0]["available"] = [0]
jobs.append((
    "urgent_departure", c, 0.5,
    0.1 * (0.5 - c["epsilon_U"]),
    2.0 - c["epsilon_U"] / c["dt"],
))

c = case(3)
for s in c["slots"]:
    s["capacity"] = 100
jobs.append(("device_power_limit", c, 1, 0.5, 2))

c = case(1)
for s in c["slots"]:
    s["capacity"] = 0
jobs.append(("zero_station_capacity", c, 1, 0, 0))

c = case(0.5)
for s in c["slots"]:
    s["price"] = 0.2
jobs.append(("equal_price_peak_minimization", c, 0, 0.1, 0.5))

c = case(0.5)
c["past_peak"] = {"day1": 2.0}
jobs.append(("already_realized_peak", c, 0, 0.05, 2))

c = case(0.5)
c["alpha"] = 0
c["past_peak"] = {"day1": 2.0, "day2": 0.0}
for t, s in enumerate(c["slots"]):
    s["day"] = "day1" if t < 2 else "day2"
    s["price"] = 0.2
jobs.append(("cross_day_peak_sum", c, 0, 0.1, 2))

c = case()
c["sessions"] = []
c["past_peak"] = {"day1": 3.0}
jobs.append(("empty_fleet", c, 0, 0, 3))

for capacity in [1.0, 2.0, 3.0]:
    c = case()
    c["sessions"] = [
        {"id": i, "demand": 1.5, "pmax": 2.0,
         "available": [0, 1, 2, 3]}
        for i in ["A", "B"]
    ]
    for s in c["slots"]:
        s["capacity"] = capacity
    jobs.append((f"capacity_{capacity:g}", c, 3-capacity, None, None))

# 单车纯费用目标：枚举 {0,1,2} kW 的全部 81 个计划。
# 此例需求为 1 kWh，枚举网格恰好包含连续 LP 的最优解。
c = case(1)
c["alpha"] = 1
for s, price in zip(c["slots"], [0.1, 0.2, 0.3, 0.4]):
    s["price"] = price
enumerated_cost = min(
    sum(p*s["price"]*c["dt"] for p, s in zip(powers, c["slots"]))
    for powers in product([0.0, 1.0, 2.0], repeat=4)
    if abs(sum(powers)*c["dt"]-1.0) < 1e-12
)
near(enumerated_cost, 0.15, 1e-12)
jobs.append(("single_car_enumeration", c, 0, enumerated_cost, None))

for name, c, u, cost, peak in jobs:
    try:
        details = run(name, deepcopy(c), u, cost, peak)
        # 跨日峰值之和应为 2 kW；不向第二天增加负荷。
        if name == "cross_day_peak_sum":
            near(results[name]["peak_by_day_kW"]["day2"], 0)
            near(results[name]["objective"], 1)
        if name == "empty_fleet":
            assert results[name]["mode"] == "analytic_empty_fleet"
            assert results[name]["stages"] == []
        records.append({"case": name, "status": "passed", **details})
    except Exception:
        records.append({
            "case": name, "status": "failed",
            "traceback": traceback.format_exc(),
        })

# 显式比较容量序列，而不只核对各自手算值。
try:
    us = [results[f"capacity_{v}"]["U_star_kWh"] for v in ["1", "2", "3"]]
    assert all(b <= a+1e-6 for a, b in zip(us, us[1:]))
    records.append({"case": "capacity_monotonicity", "status": "passed"})
except Exception:
    records.append({
        "case": "capacity_monotonicity", "status": "failed",
        "traceback": traceback.format_exc(),
    })

def expect_rejection(name, c, error_type, solver=None):
    directory = OUT / name
    try:
        if solver is None:
            opt.solve_lp(c, directory)
        else:
            with patch.object(opt.pyo, "SolverFactory", return_value=solver):
                opt.solve_lp(c, directory)
    except error_type:
        failure = json.loads((directory / "report.json").read_text())
        assert failure["status"] == "failed"
        assert not (directory / "plan.csv").exists()
        assert (directory / "error.txt").is_file()
        return
    raise AssertionError("预期失败未发生")

# 非法输入必须留下失败记录，不产生正常计划。
try:
    c = case(-1)
    expect_rejection("invalid_negative_demand", c, ValueError)
    records.append({"case": "invalid_negative_demand", "status": "passed"})
except Exception:
    records.append({
        "case": "invalid_negative_demand", "status": "failed",
        "traceback": traceback.format_exc(),
    })

# 注入超时状态，只验证失败处理；不是实际求解器超时性能实验。
class NonoptimalSolver:
    def __init__(self):
        self.options = {}
    def available(self, exception_flag=False):
        return True
    def solve(self, model, **kwargs):
        assert kwargs["load_solutions"] is False
        return SimpleNamespace(solver=SimpleNamespace(
            status="warning",
            termination_condition=opt.pyo.TerminationCondition.maxTimeLimit,
        ))

try:
    expect_rejection(
        "nonoptimal_solution_guard", case(), RuntimeError, NonoptimalSolver()
    )
    records.append({"case": "nonoptimal_solution_guard", "status": "passed"})
except Exception:
    records.append({
        "case": "nonoptimal_solution_guard", "status": "failed",
        "traceback": traceback.format_exc(),
    })

passed = sum(r["status"] == "passed" for r in records)
summary = {
    "status": "passed" if passed == len(records) else "failed",
    "passed": passed, "total": len(records), "cases": records,
}
(OUT / "validation_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)

lines = [
    "# 静态 LP 合成验证记录", "",
    f"状态：{summary['status']}；通过 {passed}/{len(records)} 项。",
    f"证据目录：`{OUT.relative_to(ROOT)}`。", "",
    "验证使用合成输入及验证专用归一化参考值，不读取正式测试集。",
    "非最优状态检查使用注入状态，不是实际超时性能测量。",
    "本记录不代表真实数据入口、在线信息隔离或阶段二整体验收。", "",
    "| 检查 | 状态 |",
    "|---|---|",
]
lines += [f"| {r['case']} | {r['status']} |" for r in records]
text = "\n".join(lines) + "\n"
(OUT / "model_validation.md").write_text(text, encoding="utf-8")

# 始终保留本次报告；仅全通过时更新报告目录的当前版本。
if summary["status"] == "passed":
    target = ROOT / "report/model_validation.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        (OUT / "model_validation_before.md").write_bytes(target.read_bytes())
    target.write_text(text, encoding="utf-8")

print(json.dumps(summary, ensure_ascii=False, indent=2))
sys.exit(0 if summary["status"] == "passed" else 1)
