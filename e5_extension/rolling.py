"""Rolling controller: visible inputs only; existing two-stage LP in a worker."""
from pathlib import Path
from time import perf_counter
import json
import traceback
import pandas as pd
from src.baselines import _state, _time, _number
from src.rolling_horizon import _invoke, _first_command

def deadlines_for(now, cars, delta):
    result = {}
    for car in cars:
        estimate = _time(car["expected_departure"])
        result[car["id"]] = (
            now + delta if estimate <= now else estimate.floor(delta))
    return result

def edf_command(view, interval_minutes):
    now, delta, capacity, cars = _state(view, interval_minutes)
    deadlines = deadlines_for(now, cars, delta)
    dt_hours = delta.total_seconds() / 3600
    command = {}
    for car in sorted(cars, key=lambda v: (
        _time(v["expected_departure"]), v["arrival"], v["id"])):
        if deadlines[car["id"]] < now + delta:
            continue
        power = min(car["pmax_kw"], capacity, car["remaining_kwh"] / dt_hours)
        if power > 0:
            command[car["id"]] = power
            capacity = max(capacity - power, 0.0)
    return command

class RollingHorizonController:
    def __init__(self, output_dir, price_at, C_ref, P_ref,
                 alpha=.5, base_hours=24, epsilon_U=1e-5,
                 time_limit_seconds=30, timezone="America/Los_Angeles",
                 interval_minutes=15):
        self.root = Path(output_dir)
        self.price_at = price_at
        self.C_ref, self.P_ref = _number(C_ref), _number(P_ref)
        self.alpha = _number(alpha)
        self.base_hours = _number(base_hours)
        self.epsilon = _number(epsilon_U)
        self.limit = _number(time_limit_seconds)
        if min(self.C_ref, self.P_ref, self.base_hours, self.limit) <= 0:
            raise ValueError("References, horizon and time budget must be positive")
        if self.alpha > 1:
            raise ValueError("alpha must be in [0, 1]")
        if interval_minutes not in (5, 15, 30):
            raise ValueError("E5 control interval must be 5, 15 or 30 minutes")
        self.interval_minutes = interval_minutes
        self.timezone = timezone
        self.counter = 0
        self.root.mkdir(parents=True, exist_ok=False)

    def build_case(self, view, past_peak=None):
        now, delta, capacity, cars = _state(view, self.interval_minutes)
        deadlines = deadlines_for(now, cars, delta)
        end = max(
            (now + pd.Timedelta(hours=self.base_hours)).ceil(delta),
            max(deadlines.values(), default=now))
        grid = tuple(pd.date_range(now, end, freq=delta, inclusive="left"))
        days = [str(t.tz_convert(self.timezone).date()) for t in grid]
        today = str(now.tz_convert(self.timezone).date())
        past = {}
        for day, value in (past_peak or {}).items():
            if not isinstance(day, str) or str(pd.Timestamp(day).date()) != day:
                raise ValueError("Invalid peak date")
            if day > today:
                raise ValueError("Future realized peak supplied")
            past[day] = _number(value)
        return {
            "dt": delta.total_seconds() / 3600, "alpha": self.alpha,
            "C_ref": self.C_ref, "P_ref": self.P_ref,
            "epsilon_U": self.epsilon, "time_limit_seconds": self.limit,
            "past_peak": {day: past.get(day, 0.0) for day in dict.fromkeys(days)},
            "slots": [
                {"label": t.isoformat(), "day": day,
                 "price": _number(self.price_at(t)), "capacity": capacity}
                for t, day in zip(grid, days)
            ],
            "sessions": [
                {"id": car["id"], "demand": car["remaining_kwh"],
                 "pmax": car["pmax_kw"],
                 "available": [
                     j for j, t in enumerate(grid)
                     if t >= car["arrival"].ceil(delta) and
                     t + delta <= deadlines[car["id"]]
                 ]}
                for car in sorted(cars, key=lambda v: v["id"])
            ],
        }

    def decide(self, view, past_peak=None):
        tick = perf_counter()
        now, _, _, cars = _state(view, self.interval_minutes)
        self.counter += 1
        step = self.root / f"step_{self.counter:06d}"
        step.mkdir()
        result = {
            "time": now.isoformat(), "mode": "empty", "fallback": False,
            "reason": None, "command": {}, "solve_report": None,
            "first_slot_adjustment_kW": 0.0,
            "effective_horizon_hours": 0.0,
            "lp_slots": 0, "lp_sessions": 0, "lp_power_variables": 0,
            "lp_variables": 0, "lp_stage2_constraints": 0,
        }
        if cars:
            try:
                case = self.build_case(view, past_peak)
                n_slots = len(case["slots"])
                n_sessions = len(case["sessions"])
                n_days = len(case["past_peak"])
                n_power = sum(len(car["available"]) for car in case["sessions"])
                result.update(
                    effective_horizon_hours=n_slots * case["dt"],
                    lp_slots=n_slots, lp_sessions=n_sessions, lp_power_variables=n_power,
                    lp_variables=n_power + n_sessions + n_days,
                    lp_stage2_constraints=2 * n_slots + n_sessions + n_days + 1,
                )
                remaining = self.limit - (perf_counter() - tick)
                if remaining <= 0:
                    raise TimeoutError("Decision budget consumed before solve")
                case["time_limit_seconds"] = remaining
                report = _invoke(case, step, remaining)
                if perf_counter() - tick >= self.limit:
                    raise TimeoutError("Decision budget consumed by solve")
                command, adjustment = _first_command(
                    case, step / "lp/plan.csv")
                if perf_counter() - tick >= self.limit:
                    raise TimeoutError("Decision budget consumed by plan validation")
                result.update(
                    mode="lp", command=command, solve_report=report,
                    first_slot_adjustment_kW=adjustment)
            except Exception as exc:
                (step / "controller_error.txt").write_text(
                    traceback.format_exc(), encoding="utf-8")
                result.update(
                    mode="fallback", fallback=True, reason=repr(exc),
                    command=edf_command(view, self.interval_minutes))
        result["total_seconds"] = perf_counter() - tick
        result["step_directory"] = str(step)
        (step / "decision.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

