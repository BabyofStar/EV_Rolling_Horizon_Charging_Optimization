"""Metrics from actual execution. Future truth is evaluator-only input."""
from collections import defaultdict
import math
import numpy as np
import pandas as pd

def _time(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")
    return value.tz_convert("UTC")

def _number(value, nonnegative=False):
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError("Invalid numeric value")
    return value

def evaluate_run(station_trace, vehicle_trace, sessions, start, end,
                 price_at, capacity_kw, timezone="America/Los_Angeles",
                 service_epsilon=.01, power_tolerance=1e-6):
    """
    station_trace: start,end,total_kw[,energy_kwh]; complete continuous coverage.
    vehicle_trace: sessionID,start,end,actual_kw,energy_kwh;
                   intervals must match station execution segments.
    sessions: sessionID,connectionTime,disconnectTime,evaluation_target_kwh,pmax_kw.
    Optional checkpoint fields:
      first_expected_departure, expected_checkpoint_target_kwh, first_input_time.
    Checkpoint targets are supplied explicitly by the evaluator's input policy.
    """
    start, end = _time(start), _time(end)
    delta = pd.Timedelta(minutes=15)
    if not start < end or start != start.floor(delta) or end != end.floor(delta):
        raise ValueError("Evaluation boundaries must follow the 15-minute grid")
    capacity_kw = _number(capacity_kw, True)
    service_epsilon = _number(service_epsilon, True)

    records = {}
    for row in sessions.to_dict("records"):
        sid = str(row["sessionID"])
        if sid in records:
            raise ValueError("Duplicate session")
        row = dict(row)
        row["connectionTime"] = _time(row["connectionTime"])
        row["disconnectTime"] = _time(row["disconnectTime"])
        if not start <= row["connectionTime"] < row["disconnectTime"] <= end:
            raise ValueError("Censored or invalid evaluation session")
        row["evaluation_target_kwh"] = _number(row["evaluation_target_kwh"], True)
        row["pmax_kw"] = _number(row["pmax_kw"], True)
        records[sid] = row

    grid = tuple(pd.date_range(start, end, freq=delta, inclusive="left"))
    grid_energy = {t.value: 0.0 for t in grid}
    daily = {}
    for t in grid:
        day = str(t.tz_convert(timezone).date())
        if day not in daily:
            daily[day] = {
                "local_date": day, "energy_kwh": 0.0, "cost_USD": 0.0,
                "peak_kW": 0.0, "minimum_kW": float("inf"), "grid_slots": 0,
            }
        daily[day]["grid_slots"] += 1

    station_rows = []
    for row in station_trace.to_dict("records"):
        a, b = _time(row["start"]), _time(row["end"])
        station_rows.append((a, b, _number(row["total_kw"]), row))
    station_rows.sort(key=lambda row: (row[0], row[1]))
    intervals, cursor = {}, start
    energy_residual = station_violation = device_violation = 0.0
    station_violations = device_violations = connection_violations = 0

    for a, b, power, original in station_rows:
        if a != cursor or not a < b <= end:
            raise ValueError("Station trace has a gap, overlap or invalid boundary")
        cursor = b
        h = (b - a).value / 3_600_000_000_000
        energy = power * h
        if "energy_kwh" in original:
            energy_residual = max(
                energy_residual, abs(_number(original["energy_kwh"]) - energy))
        intervals[(a.value, b.value)] = energy
        violation = max(0.0, -power, power - capacity_kw)
        station_violation = max(station_violation, violation)
        station_violations += int(violation > power_tolerance)

        t = a
        while t < b:
            cut = min(b, t.floor(delta) + delta)
            part = power * (cut - t).value / 3_600_000_000_000
            day = str(t.tz_convert(timezone).date())
            rate = _number(price_at(t), True)
            daily[day]["energy_kwh"] += part
            daily[day]["cost_USD"] += part * rate
            daily[day]["peak_kW"] = max(daily[day]["peak_kW"], power)
            daily[day]["minimum_kW"] = min(daily[day]["minimum_kW"], power)
            grid_energy[t.floor(delta).value] += part
            t = cut
    if cursor != end:
        raise ValueError("Station trace does not cover the full evaluation period")

    delivered = defaultdict(float)
    interval_vehicle_energy = defaultdict(float)
    vehicle_rows = defaultdict(list)
    seen = set()
    for row in vehicle_trace.to_dict("records"):
        sid = str(row["sessionID"])
        a, b = _time(row["start"]), _time(row["end"])
        key = (a.value, b.value)
        if sid not in records or key not in intervals or (sid, key) in seen:
            raise ValueError("Invalid or duplicate vehicle execution segment")
        seen.add((sid, key))
        power, energy = _number(row["actual_kw"]), _number(row["energy_kwh"])
        h = (b - a).value / 3_600_000_000_000
        energy_residual = max(energy_residual, abs(energy - power * h))
        delivered[sid] += energy
        interval_vehicle_energy[key] += energy
        vehicle_rows[sid].append((a, b, power))
        r = records[sid]
        violation = max(0.0, -power, power - r["pmax_kw"])
        device_violation = max(device_violation, violation)
        device_violations += int(violation > power_tolerance)
        if power > power_tolerance and not (
            a >= r["connectionTime"].ceil(delta) and b <= r["disconnectTime"]
        ):
            connection_violations += 1
    for key, energy in intervals.items():
        energy_residual = max(
            energy_residual, abs(energy - interval_vehicle_energy[key]))

    service_rows = []
    expected_evaluable = expected_uncompleted = 0
    for sid, row in records.items():
        target, actual = row["evaluation_target_kwh"], delivered[sid]
        result = {
            "sessionID": sid, "evaluation_target_kwh": target,
            "delivered_kwh": actual, "unmet_kwh": max(target - actual, 0.0),
            "excess_kwh": max(actual - target, 0.0),
            "satisfaction": min(max(actual, 0.0), target) / target if target > 0 else None,
            "uncompleted_at_actual_departure": actual < target - service_epsilon,
            "expected_checkpoint_evaluable": False,
            "uncompleted_at_expected_departure": None,
        }
        deadline = row.get("first_expected_departure")
        checkpoint_target = row.get("expected_checkpoint_target_kwh")
        if deadline is not None and pd.notna(deadline) and (
            checkpoint_target is not None and pd.notna(checkpoint_target)
        ):
            deadline = _time(deadline)
            first_input = row.get("first_input_time", row["connectionTime"])
            if pd.notna(first_input) and deadline >= _time(first_input):
                checkpoint_target = _number(checkpoint_target, True)
                before = sum(
                    power * max(0, (min(b, deadline) - a).value)
                    / 3_600_000_000_000
                    for a, b, power in vehicle_rows[sid] if a < deadline)
                unfinished = before < checkpoint_target - service_epsilon
                expected_evaluable += 1
                expected_uncompleted += int(unfinished)
                result.update(
                    expected_checkpoint_evaluable=True,
                    uncompleted_at_expected_departure=unfinished)
        service_rows.append(result)

    services = pd.DataFrame(service_rows)
    targets = [r["evaluation_target_kwh"] for r in service_rows]
    total_target = sum(targets)
    fulfilled = sum(
        min(max(r["delivered_kwh"], 0.0), r["evaluation_target_kwh"])
        for r in service_rows)
    fractions = [r["satisfaction"] for r in service_rows if r["satisfaction"] is not None]
    unfinished = sum(r["uncompleted_at_actual_departure"] for r in service_rows)
    daily_rows = list(daily.values())
    grid_power = np.array([grid_energy[t.value] / .25 for t in grid])
    for row in daily_rows:
        row["peak_valley_kW"] = row["peak_kW"] - row["minimum_kW"]
        values = [grid_power[j] for j, t in enumerate(grid)
                  if str(t.tz_convert(timezone).date()) == row["local_date"]]
        row["grid_average_peak_kW"] = float(max(values))

    summary = {
        "cost_USD": sum(r["cost_USD"] for r in daily_rows),
        "delivered_kwh": sum(delivered.values()),
        "unmet_kwh": sum(r["unmet_kwh"] for r in service_rows),
        "excess_kwh": sum(r["excess_kwh"] for r in service_rows),
        "energy_satisfaction": fulfilled / total_target if total_target > 0 else None,
        "sessions": len(records), "uncompleted_sessions": int(unfinished),
        "actual_departure_uncompleted_rate": unfinished / len(records) if records else None,
        "expected_checkpoint_evaluable_sessions": expected_evaluable,
        "expected_checkpoint_excluded_sessions": len(records) - expected_evaluable,
        "expected_checkpoint_uncompleted_sessions": expected_uncompleted,
        "expected_departure_uncompleted_rate": (
            expected_uncompleted / expected_evaluable if expected_evaluable else None),
        "max_individual_unmet_kwh": max([0.0] + [r["unmet_kwh"] for r in service_rows]),
        "session_satisfaction_P10": float(np.quantile(fractions, .1)) if fractions else None,
        "max_peak_kW": max(r["peak_kW"] for r in daily_rows),
        "max_grid_average_peak_kW": float(max(grid_power)),
        "max_grid_load_change_kW": float(max(np.abs(np.diff(grid_power)))) if len(grid_power) > 1 else 0.0,
        "station_violation_segments": station_violations,
        "device_violation_segments": device_violations,
        "connection_violation_segments": connection_violations,
        "max_power_violation_kW": max(station_violation, device_violation),
        "max_energy_residual_kwh": energy_residual,
    }
    return summary, pd.DataFrame(daily_rows), services

def summarize_decisions(frame):
    if frame.empty:
        return {"decision_count": 0, "fallback_count": 0,
                "active_P95_seconds": None, "active_max_seconds": None}
    active = frame[frame["mode"] != "empty"]
    timing = [_number(v, True) for v in active["total_seconds"]]
    fallback = sum(str(v).lower() in {"true", "1", "1.0"} for v in frame["fallback"])
    return {
        "decision_count": len(frame), "active_decision_count": len(active),
        "fallback_count": fallback,
        "fallback_fraction_active": fallback / len(active) if len(active) else None,
        "active_mean_seconds": float(np.mean(timing)) if timing else None,
        "active_P95_seconds": float(np.quantile(timing, .95)) if timing else None,
        "active_max_seconds": max(timing) if timing else None,
        "mode_counts": {str(k): int(v) for k, v in frame["mode"].value_counts().items()},
    }
