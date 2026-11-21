"""聚合连续功率两层 LP。仅接收显式输入，不读取未来事件文件。"""
from pathlib import Path
from time import perf_counter
from contextlib import redirect_stdout, redirect_stderr
import csv
import json
import math
import traceback
import pyomo.environ as pyo


def solve_lp(case, output_dir):
    """
    slots: [{label, day, price, capacity}]
    sessions: [{id, demand, pmax, available: [slot_index, ...]}]
    参数: dt, alpha, C_ref, P_ref, epsilon_U, time_limit_seconds
    available 由调用方构造；核心不判断真实/预计离站。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "input.json").write_text(
        json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    started = perf_counter()
    report = {"status": "running", "stages": []}

    def finite_number(value, name, lower=None):
        if isinstance(value, bool):
            raise ValueError(f"{name} 不得为布尔值")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} 必须有限")
        if lower is not None and value < lower:
            raise ValueError(f"{name} 必须 >= {lower}")
        return value

    try:
        slots = case["slots"]
        cars = case["sessions"]
        dt = finite_number(case["dt"], "dt", 0)
        cref = finite_number(case["C_ref"], "C_ref", 0)
        pref = finite_number(case["P_ref"], "P_ref", 0)
        alpha = finite_number(case.get("alpha", 0.5), "alpha", 0)
        eps = finite_number(case.get("epsilon_U", 1e-5), "epsilon_U", 0)
        limit = finite_number(
            case.get("time_limit_seconds", 30), "time_limit_seconds", 0
        )
        if min(dt, cref, pref, limit) <= 0 or alpha > 1:
            raise ValueError("dt、参考值及时间上限须为正；alpha 须在 [0,1]")
        if not slots:
            raise ValueError("本接口要求至少一个评价时段")

        labels = [s["label"] for s in slots]
        if len(set(labels)) != len(labels):
            raise ValueError("时段标签重复")
        days = list(dict.fromkeys(s["day"] for s in slots))
        prices = [finite_number(s["price"], "price", 0) for s in slots]
        capacities = [
            finite_number(s["capacity"], "capacity", 0) for s in slots
        ]
        ids = [c["id"] for c in cars]
        if any(not isinstance(i, str) or not i for i in ids):
            raise ValueError("会话 ID 必须为非空字符串")
        if len(set(ids)) != len(ids):
            raise ValueError("会话 ID 重复")

        demand, pmax, available = {}, {}, {}
        for c in cars:
            i = c["id"]
            demand[i] = finite_number(c["demand"], f"{i}.demand", 0)
            pmax[i] = finite_number(c["pmax"], f"{i}.pmax", 0)
            raw = c["available"]
            if any(type(t) is not int or not 0 <= t < len(slots) for t in raw):
                raise ValueError(f"{i}: available 时段索引非法")
            if len(set(raw)) != len(raw):
                raise ValueError(f"{i}: available 索引重复")
            available[i] = sorted(raw)

        past_input = case.get("past_peak", {})
        if set(past_input) - set(days):
            raise ValueError("past_peak 包含窗口以外的日期")
        past = {
            d: finite_number(past_input.get(d, 0), f"past_peak[{d}]", 0)
            for d in days
        }

        # Empty-fleet analytic branch: no solver invocation.
        if not cars:
            objective = (1-alpha) * sum(past.values()) / pref
            report.update({
                "status": "passed",
                "mode": "analytic_empty_fleet",
                "U_star_kWh": 0.0,
                "U_stage2_kWh": 0.0,
                "service_change_kWh": 0.0,
                "epsilon_U_kWh": eps,
                "cost_USD": 0.0,
                "objective": objective,
                "peak_by_day_kW": dict(past),
                "peak_variable_by_day_kW": dict(past),
                "max_power_violation_kW": 0.0,
                "max_energy_residual_kWh": 0.0,
                "max_overcharge_kWh": 0.0,
                "max_u_bound_violation_kWh": 0.0,
                "max_peak_violation_kW": 0.0,
                "service_lock_violation_kWh": 0.0,
                "cost_recompute_error_USD": 0.0,
                "objective_recompute_error": 0.0,
            })
            with (out / "plan.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                csv.writer(f).writerow([
                    "sessionID", "slot", "day", "power_kW", "energy_kWh"
                ])
            with (out / "sessions.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                csv.writer(f).writerow([
                    "sessionID", "target_kWh", "planned_kWh", "unmet_kWh"
                ])
            return report

        pairs = [(i, t) for i in ids for t in available[i]]
        at_time = {
            t: [i for i in ids if t in available[i]]
            for t in range(len(slots))
        }

        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=ids, ordered=True)
        m.T = pyo.RangeSet(0, len(slots) - 1)
        m.D = pyo.Set(initialize=days, ordered=True)
        m.A = pyo.Set(dimen=2, initialize=pairs, ordered=True)
        # 不可用时段不创建功率变量，导出时补零。
        m.p = pyo.Var(m.A, bounds=lambda m, i, t: (0, pmax[i]))
        m.u = pyo.Var(m.I, bounds=lambda m, i: (0, demand[i]))
        m.z = pyo.Var(m.D, domain=pyo.NonNegativeReals)
        m.station_load = pyo.Expression(
            m.T,
            rule=lambda m, t: pyo.quicksum(m.p[i, t] for i in at_time[t])
        )
        m.capacity = pyo.Constraint(
            m.T, rule=lambda m, t: m.station_load[t] <= capacities[t]
        )
        m.energy = pyo.Constraint(
            m.I,
            rule=lambda m, i:
                dt * pyo.quicksum(m.p[i, t] for t in available[i])
                + m.u[i] == demand[i]
        )
        m.peak = pyo.Constraint(
            m.T, rule=lambda m, t: m.z[slots[t]["day"]] >= m.station_load[t]
        )
        m.past = pyo.Constraint(
            m.D, rule=lambda m, d: m.z[d] >= past[d]
        )
        m.U = pyo.Expression(expr=pyo.quicksum(m.u[i] for i in m.I))
        m.C = pyo.Expression(
            expr=dt * pyo.quicksum(prices[t] * m.station_load[t] for t in m.T)
        )
        m.J = pyo.Expression(
            expr=alpha * m.C / cref
            + (1-alpha) * pyo.quicksum(m.z[d] / pref for d in m.D)
        )
        m.first = pyo.Objective(expr=m.U)

        solver = pyo.SolverFactory("appsi_highs")
        if not solver.available(exception_flag=False):
            raise RuntimeError("当前内核的 appsi_highs 不可用")
        solver.options["primal_feasibility_tolerance"] = 1e-8
        solver.options["dual_feasibility_tolerance"] = 1e-8

        def solve_stage(name):
            remaining = limit - (perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("两层累计时间预算已用完")
            solver.options["time_limit"] = remaining
            tick = perf_counter()
            with (out / f"{name}.txt").open("w", encoding="utf-8") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    result = solver.solve(m, tee=True, load_solutions=False)
            record = {
                "name": name,
                "status": str(result.solver.status),
                "termination": str(result.solver.termination_condition),
                "seconds": perf_counter() - tick,
            }
            report["stages"].append(record)
            if result.solver.termination_condition != pyo.TerminationCondition.optimal:
                raise RuntimeError(f"{name} 未确认最优：{record}")
            m.solutions.load_from(result)

        solve_stage("stage1")
        ustar = float(pyo.value(m.U))
        report["U_star_kWh"] = ustar
        report["stages"][-1]["objective_kWh"] = ustar

        m.first.deactivate()
        m.service_lock = pyo.Constraint(expr=m.U <= ustar + eps)
        m.second = pyo.Objective(expr=m.J)
        solve_stage("stage2")
        report["stages"][-1]["objective"] = float(pyo.value(m.J))

        # 保留原始求解值，不通过截断掩盖残差。
        power = {
            (i, t): float(pyo.value(m.p[i, t])) if (i, t) in m.A else 0.0
            for i in ids for t in range(len(slots))
        }
        unmet = {i: float(pyo.value(m.u[i])) for i in ids}
        loads = [
            sum(power[i, t] for i in ids) for t in range(len(slots))
        ]
        delivered = {
            i: dt * sum(power[i, t] for t in range(len(slots))) for i in ids
        }
        z = {d: float(pyo.value(m.z[d])) for d in days}
        realized_plan_peak = {
            d: max([past[d]] + [
                loads[t] for t in range(len(slots)) if slots[t]["day"] == d
            ]) for d in days
        }

        power_violation = max(
            [0.0]
            + [-v for v in power.values()]
            + [power[i, t] - pmax[i] for i, t in pairs]
            + [loads[t] - capacities[t] for t in range(len(slots))]
        )
        energy_residual = max(
            abs(delivered[i] + unmet[i] - demand[i]) for i in ids
        )
        overcharge = max([0.0] + [delivered[i] - demand[i] for i in ids])
        u_bounds = max(
            [0.0] + [-unmet[i] for i in ids]
            + [unmet[i] - demand[i] for i in ids]
        )
        peak_violation = max(
            [0.0] + [realized_plan_peak[d] - z[d] for d in days]
        )
        lock_violation = max(0.0, sum(unmet.values()) - ustar - eps)
        cost = dt * sum(prices[t] * loads[t] for t in range(len(slots)))
        objective = alpha * cost / cref + (1-alpha) * sum(z.values()) / pref

        report.update({
            "U_stage2_kWh": sum(unmet.values()),
            "service_change_kWh": sum(unmet.values()) - ustar,
            "epsilon_U_kWh": eps,
            "cost_USD": cost,
            "objective": objective,
            "peak_by_day_kW": realized_plan_peak,
            "peak_variable_by_day_kW": z,
            "max_power_violation_kW": power_violation,
            "max_energy_residual_kWh": energy_residual,
            "max_overcharge_kWh": overcharge,
            "max_u_bound_violation_kWh": u_bounds,
            "max_peak_violation_kW": peak_violation,
            "service_lock_violation_kWh": lock_violation,
            "cost_recompute_error_USD": abs(cost - pyo.value(m.C)),
            "objective_recompute_error": abs(objective - pyo.value(m.J)),
        })

        with (out / "plan.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sessionID", "slot", "day", "power_kW", "energy_kWh"])
            for i in ids:
                for t, s in enumerate(slots):
                    w.writerow([i, s["label"], s["day"], power[i, t],
                                power[i, t] * dt])
        with (out / "sessions.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sessionID", "target_kWh", "planned_kWh", "unmet_kWh"])
            for i in ids:
                w.writerow([i, demand[i], delivered[i], unmet[i]])

        checks = [
            power_violation <= 1e-6,
            peak_violation <= 1e-6,
            energy_residual <= 1e-5,
            overcharge <= 1e-5,
            u_bounds <= 1e-5,
            lock_violation <= 1e-7,
            abs(cost - pyo.value(m.C)) <= 1e-8,
            abs(objective - pyo.value(m.J)) <= 1e-8,
        ]
        if not all(checks):
            raise AssertionError("残差检查未通过，查看 report.json 和导出计划")
        report["status"] = "passed"
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report["total_seconds"] = perf_counter() - started
        (out / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
