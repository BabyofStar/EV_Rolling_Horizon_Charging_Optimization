"""Rolling controller: visible inputs only; existing two-stage LP in a worker."""
from pathlib import Path
from time import perf_counter
import csv
import json
import math
import subprocess
import sys
import traceback
import pandas as pd
from .baselines import _state, _time, _number

ROOT = Path(__file__).resolve().parents[1]
DELTA = pd.Timedelta(minutes=15)

def deadlines_for(now, cars):
    result = {}
    for car in cars:
        estimate = _time(car["expected_departure"])
        result[car["id"]] = (
            now + DELTA if estimate <= now else estimate.floor(DELTA))
    return result

def edf_command(view):
    now, _, capacity, cars = _state(view, 15)
    deadlines = deadlines_for(now, cars)
    command = {}
    for car in sorted(cars, key=lambda v: (
        _time(v["expected_departure"]), v["arrival"], v["id"])):
        if deadlines[car["id"]] < now + DELTA:
            continue
        power = min(car["pmax_kw"], capacity, car["remaining_kwh"] / .25)
        if power > 0:
            command[car["id"]] = power
            capacity = max(capacity - power, 0.0)
    return command

def record_realized_peaks(peaks, segments, observed_until,
                          timezone="America/Los_Angeles"):
    """Update from already-executed segments, never predicted window peaks."""
    observed_until = _time(observed_until)
    for segment in segments:
        a, b = _time(segment["start"]), _time(segment["end"])
        if not a < b <= observed_until:
            raise ValueError("Invalid or future execution feedback")
        power = _number(segment["total_kw"])
        while a < b:
            local = a.tz_convert(timezone)
            day = str(local.date())
            peaks[day] = max(peaks.get(day, 0.0), power)
            midnight = (
                local.normalize() + pd.DateOffset(days=1)).tz_convert("UTC")
            a = min(b, midnight)

def _invoke(case, step, timeout):
    input_path = step / "case.json"
    input_path.write_text(
        json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
    with (step / "worker.txt").open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            [sys.executable, "-m", "src.rolling_horizon", "--worker",
             str(input_path), str(step / "lp")],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            timeout=timeout, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"LP worker exit code {process.returncode}")
    report = json.loads((step / "lp/report.json").read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError("LP report not passed")
    if len(report.get("stages", [])) != 2 or any(
        stage["termination"] != "optimal" for stage in report["stages"]
    ):
        raise RuntimeError("Both LP stages must be optimal")
    return report

def _first_command(case, plan_path):
    cars = {car["id"]: car for car in case["sessions"]}
    labels = {slot["label"]: j for j, slot in enumerate(case["slots"])}
    allowed = {sid: set(car["available"]) for sid, car in cars.items()}
    loads = [0.0] * len(labels)
    energies = {sid: 0.0 for sid in cars}
    first, seen = {}, set()
    with plan_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            sid, label = row["sessionID"], row["slot"]
            if sid not in cars or label not in labels or (sid, label) in seen:
                raise ValueError("Invalid or duplicate exported plan key")
            seen.add((sid, label))
            j = labels[label]
            power = float(row["power_kW"])
            energy = float(row["energy_kWh"])
            if not math.isfinite(power) or not math.isfinite(energy):
                raise ValueError("Nonfinite exported plan")
            limit = cars[sid]["pmax"] if j in allowed[sid] else 0.0
            if power < -1e-6 or power > limit + 1e-6:
                raise ValueError("Exported device/availability violation")
            if row["day"] != case["slots"][j]["day"]:
                raise ValueError("Exported day mismatch")
            if abs(energy - power * case["dt"]) > 1e-8:
                raise ValueError("Exported energy integral mismatch")
            loads[j] += power
            energies[sid] += energy
            if j == 0:
                first[sid] = power
    if len(seen) != len(cars) * len(labels):
        raise ValueError("Incomplete exported plan")
    if any(load > slot["capacity"] + 1e-6
           for load, slot in zip(loads, case["slots"])):
        raise ValueError("Exported station capacity violation")
    if any(energies[sid] > cars[sid]["demand"] + 1e-5 for sid in cars):
        raise ValueError("Exported overcharge")

    # Retain raw files; remove only tolerance-scale errors for execution.
    command, adjustment = {}, 0.0
    capacity = case["slots"][0]["capacity"]
    for sid in sorted(cars):
        raw = first[sid]
        limit = min(cars[sid]["pmax"], cars[sid]["demand"] / case["dt"],
                    capacity) if 0 in allowed[sid] else 0.0
        power = min(max(raw, 0.0), limit)
        if abs(power - raw) > 1e-6:
            raise ValueError("First-slot correction exceeds power tolerance")
        adjustment += abs(power - raw)
        if power > 0:
            command[sid] = power
            capacity = max(capacity - power, 0.0)
    return command, adjustment

class RollingHorizonController:
    def __init__(self, output_dir, price_at, C_ref, P_ref,
                 alpha=.5, base_hours=24, epsilon_U=1e-5,
                 time_limit_seconds=30, timezone="America/Los_Angeles"):
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
        self.timezone = timezone
        self.counter = 0
        self.root.mkdir(parents=True, exist_ok=False)

    def build_case(self, view, past_peak=None):
        now, _, capacity, cars = _state(view, 15)
        deadlines = deadlines_for(now, cars)
        end = max(
            (now + pd.Timedelta(hours=self.base_hours)).ceil(DELTA),
            max(deadlines.values(), default=now))
        grid = tuple(pd.date_range(now, end, freq=DELTA, inclusive="left"))
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
            "dt": .25, "alpha": self.alpha,
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
                     if t >= car["arrival"].ceil(DELTA) and
                     t + DELTA <= deadlines[car["id"]]
                 ]}
                for car in sorted(cars, key=lambda v: v["id"])
            ],
        }

    def decide(self, view, past_peak=None):
        tick = perf_counter()
        now, _, _, cars = _state(view, 15)
        self.counter += 1
        step = self.root / f"step_{self.counter:06d}"
        step.mkdir()
        result = {
            "time": now.isoformat(), "mode": "empty", "fallback": False,
            "reason": None, "command": {}, "solve_report": None,
            "first_slot_adjustment_kW": 0.0,
            "effective_horizon_hours": 0.0,
        }
        if cars:
            try:
                case = self.build_case(view, past_peak)
                result["effective_horizon_hours"] = len(case["slots"]) * .25
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
                    command=edf_command(view))
        result["total_seconds"] = perf_counter() - tick
        result["step_directory"] = str(step)
        (step / "decision.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("Usage: --worker CASE_JSON OUTPUT_DIR")
    from .optimization_model import solve_lp
    solve_lp(
        json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")),
        Path(sys.argv[3]))
