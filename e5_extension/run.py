"""Run the E5 RH-LP extension from a separate directory inside the EV project."""

import argparse
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import highspy
import numpy as np
import pandas as pd
import pyomo.version
import yaml

from src.acnsim_adapter import ACNSimExecutor
from src.rolling_horizon import RollingHorizonController as FrozenController
from src.rolling_horizon import record_realized_peaks
from src.tariff import attach_tariff

import rolling
from metrics import evaluate_run
from simulator import AggregateEnvironment


DATA = ROOT / "data/processed/caltech_12weeks_20261111T143402Z_9b087d"
E1 = ROOT / "results/main_comparison.csv"
E1_SHA256 = "01a6e2f15540a5f9d57d55f5a9179b70a8d60be036a7a1c26b4b580f3589e293"
CONFIGS = ((15, 24), (5, 24), (30, 24), (15, 4), (15, 8))
COHORTS = {"main_at_arrival": 39, "supplement_after_submission": 468}
TOLERANCE_KWH = 1e-5
SIZE_FIELDS = ("effective_horizon_hours", "lp_slots", "lp_sessions",
               "lp_power_variables", "lp_variables", "lp_stage2_constraints")
BASELINE_FIELDS = ("sessions", "unmet_kwh", "delivered_kwh", "cost_USD",
                   "max_peak_kW", "energy_satisfaction", "uncompleted_sessions")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_day(stamp, tz):
    return str(stamp.tz_convert(tz).date())


def split_bounds(splits, split):
    bounds = splits["splits"][split]
    return tuple(pd.Timestamp(bounds[key]).tz_convert("UTC")
                 for key in ("start_utc_inclusive", "end_utc_exclusive"))


def source_fingerprints():
    package = Path(__file__).resolve().parent
    expected = json.loads((package / "source_hashes.json").read_text(encoding="utf-8"))
    paths = [*(ROOT / name for name in expected), *package.glob("*.py"),
             package / "source_hashes.json", E1,
             *(ROOT / "configs" / name for name in
               ("station.yaml", "splits.yaml", "tariff.yaml", "normalization.yaml", "final.yaml")),
             *(DATA / name for name in ("sessions_with_split.csv", "environment_events.csv",
                                        "input_release_events.csv", "user_updates.csv"))]
    actual = {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}
    for name, digest in expected.items():
        require(actual[name] == digest, f"Source changed since E5 preflight: {name}")
    require(actual[E1.relative_to(ROOT).as_posix()] == E1_SHA256, "E1 comparison fingerprint mismatch")
    return actual


def validate_frozen_settings(station, norm):
    mismatches = []
    for key, expected in (("control_interval_minutes", 15),
                          ("network_model", "aggregate_continuous"),
                          ("original_branch_constraints_enforced", False)):
        actual = station[key]
        if actual != expected:
            mismatches.append(f"station.{key}={actual!r} (expected {expected!r})")
    # Match the tolerance used by E1 for the saved normalization references.
    numeric = (("station", station, "charging_efficiency", 1.0),
               ("station", station, "station_capacity_kW", 50.0),
               ("station", station, "evse_max_power_kW", 6.656),
               ("normalization", norm, "C_ref", 2.88),
               ("normalization", norm, "P_ref", 6.656))
    for section, settings, key, expected in numeric:
        actual = settings[key]
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
            mismatches.append(f"{section}.{key}={actual!r} (expected {expected!r})")
    if mismatches:
        raise RuntimeError("Frozen E1 settings differ: " + ", ".join(mismatches))


def read_inputs():
    def cfg(name):
        return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))

    station, norm, splits = (cfg(name) for name in ("station", "normalization", "splits"))
    validate_frozen_settings(station, norm)
    dtype = {"sessionID": str, "stationID": str}
    sessions, events, releases, updates = (
        pd.read_csv(DATA / name, dtype=dtype) for name in
        ("sessions_with_split.csv", "environment_events.csv", "input_release_events.csv", "user_updates.csv"))
    for frame, column in ((sessions, "connectionTime"), (events, "event_time"), (updates, "modifiedAt")):
        frame[column] = pd.to_datetime(frame[column], utc=True, format="ISO8601")
    updates = updates[updates["valid"].astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})]
    return station, norm, splits, sessions, events, releases, updates


def connected_peaks(connections, start, end, tz):
    days = pd.date_range(start.tz_convert(tz).normalize(), end.tz_convert(tz), freq="D", inclusive="left")
    peaks = {}
    for midnight in days:
        a = max(start, midnight)
        b = min(end, midnight + pd.DateOffset(days=1))
        changes = defaultdict(int)
        for arrival, departure in connections:
            if max(a, arrival) < min(b, departure):
                changes[max(a, arrival)] += 1
                changes[min(b, departure)] -= 1
        current = peak = 0
        for stamp in sorted(changes):
            current += changes[stamp]
            peak = max(peak, current)
        peaks[str(midnight.date())] = peak
    return peaks


def price_lookup(start, end):
    grid = pd.date_range(start, end + pd.Timedelta(days=2), freq="5min")
    prices = attach_tariff(pd.DataFrame({"start_utc": grid}))["price_USD_per_kWh"]
    cache = dict(zip(grid, prices))

    def price_at(stamp):
        key = pd.Timestamp(stamp).floor("5min")
        if key not in cache:
            rate = attach_tariff(pd.DataFrame({"start_utc": [key]}))["price_USD_per_kWh"].iloc[0]
            cache[key] = float(rate)
        return float(cache[key])

    return price_at


def evaluation_rows(selected, departures, updates, pmax):
    rows, histories = [], {}
    for record in selected.itertuples(index=False):
        sid, arrival = record.sessionID, record.connectionTime
        departure = departures[sid]
        if record.cohort == "main_at_arrival":
            times, targets = [arrival.value], [float(record.initial_kWhRequested)]
            deadline = pd.Timestamp(record.initial_requestedDeparture)
        else:
            valid = updates[(updates["sessionID"] == sid) & (updates["modifiedAt"] >= arrival)
                            & (updates["modifiedAt"] < departure)].sort_values(["modifiedAt", "input_order"])
            require(len(valid) > 0, f"No valid in-connection request for {sid}")
            times, targets = [t.value for t in valid["modifiedAt"]], valid["kWhRequested"].astype(float).tolist()
            deadline = pd.Timestamp(valid.iloc[0]["requestedDeparture"])
        histories[sid] = (times, targets)
        checkpoint = bisect_right(times, deadline.value) - 1
        rows.append({"sessionID": sid, "connectionTime": arrival, "disconnectTime": departure,
                     "evaluation_target_kwh": targets[-1], "pmax_kw": pmax,
                     "first_input_time": pd.Timestamp(times[0], tz="UTC"),
                     "first_expected_departure": deadline,
                     "expected_checkpoint_target_kwh": targets[checkpoint] if checkpoint >= 0 else None})
    return pd.DataFrame(rows), histories


def traces(environment, histories, delta):
    station_rows, vehicle_rows = [], []
    cumulative = defaultdict(float)
    for segment in environment.ledger:
        a, b = segment["start"], segment["end"]
        station_rows.append({"start": a, "end": b, "total_kw": segment["total_kw"],
                             "energy_kwh": sum(segment["energy_kwh"].values())})
        for sid, amount in segment["energy_kwh"].items():
            if amount > 0:
                times, targets = histories[sid]
                index = bisect_right(times, a.value) - 1
                require(index >= 0 and a >= pd.Timestamp(times[0], tz="UTC").ceil(delta),
                        f"Energy delivered before request became actionable: {sid}")
                before = cumulative[sid]
                cumulative[sid] += amount
                require(cumulative[sid] <= max(before, targets[index]) + TOLERANCE_KWH,
                        f"Energy delivered beyond the visible target: {sid}")
            vehicle_rows.append({"sessionID": sid, "start": a, "end": b,
                                 "actual_kw": segment["power_kw"][sid], "energy_kwh": amount})
    for record in environment.completed:
        require(abs(cumulative[record["id"]] - record["delivered_kwh"]) <= TOLERANCE_KWH,
                f"Trace disagrees with environment delivery: {record['id']}")
    return (pd.DataFrame(station_rows), pd.DataFrame(
        vehicle_rows, columns=["sessionID", "start", "end", "actual_kw", "energy_kwh"]))


def timing(frame):
    active = frame[frame["mode"] != "empty"]
    result = {"decision_count": len(frame), "active_decisions": len(active),
              "successful_solves": int((active["mode"] == "lp").sum()),
              "fallback_count": int(active["fallback"].sum()),
              "fallback_rate_active": float(active["fallback"].mean()) if len(active) else None,
              "controller_total_seconds": float(frame["online_seconds"].sum()),
              "two_stage_total_seconds": float(active["solver_seconds"].sum())}
    for name, values in (("online", active["online_seconds"]),
                         ("solver", active["solver_seconds"].dropna()),
                         *((key, active.loc[active["lp_slots"] > 0, key]) for key in SIZE_FIELDS)):
        for stat, value in (("mean", values.mean()), ("p95", values.quantile(.95)), ("max", values.max())):
            result[f"{name}_{stat}"] = float(value) if pd.notna(value) else None
    return result


def replay(out, interval, hours, station, norm, selection, price_at, start, end, bands):
    selected, events, releases, evaluation, histories = selection
    capacity = float(station["station_capacity_kW"])
    pmax = float(station["evse_max_power_kW"])
    tz = station["timezone"]
    delta = pd.Timedelta(minutes=interval)
    executor = ACNSimExecutor(selected["stationID"].tolist(), capacity_kw=capacity,
                              voltage=float(station["evse_voltage_V"]),
                              max_current=float(station["evse_max_current_A"]))
    environment = AggregateEnvironment(selected, events, releases, start, capacity_kw=capacity,
                                       pmax_kw=pmax, executor=executor, interval_minutes=interval)
    controller = rolling.RollingHorizonController(
        out / "decisions", price_at, norm["C_ref"], norm["P_ref"],
        alpha=.5, base_hours=hours, epsilon_U=1e-5, time_limit_seconds=30,
        timezone=tz, interval_minutes=interval)
    peaks, rows = {}, []
    wall = perf_counter()
    while environment.now < end:
        now = environment.now
        view = environment.view()
        decision = controller.decide(view, dict(peaks))
        report = decision["solve_report"]
        stages = [float(item["seconds"]) for item in report["stages"]] if report else None
        day = local_day(now, tz)
        rows.append({"time": now, "local_date": day, "load_band": bands.get(day, "tail"),
                     "connected_at_decision": len(view["vehicles"]),
                     "mode": decision["mode"], "fallback": decision["fallback"],
                     "reason": decision["reason"], "online_seconds": decision["total_seconds"],
                     "stage1_seconds": stages[0] if stages else np.nan,
                     "stage2_seconds": stages[1] if stages else np.nan,
                     "solver_seconds": sum(stages) if stages else np.nan,
                     **{key: decision[key] for key in SIZE_FIELDS}})
        index = len(environment.ledger)
        environment.step(decision["command"])
        record_realized_peaks(peaks, environment.ledger[index:], environment.now, tz)
        if environment.now.tz_convert(tz).date() != now.tz_convert(tz).date():
            print(f"{out.parent.name}/{out.name}: finished {day}; {len(rows)} decisions", flush=True)

    require(environment.now == end and not environment.view()["vehicles"]
            and len(environment.completed) == len(selected) and not executor._connected,
            "Incomplete replay or connected vehicle after common tail")
    station_trace, vehicle_trace = traces(environment, histories, delta)
    summary, daily, service = evaluate_run(
        station_trace, vehicle_trace, evaluation, start, end, price_at, capacity, tz,
        control_interval_minutes=interval)
    require(not any(summary[key] for key in ("station_violation_segments", "device_violation_segments",
                                              "connection_violation_segments"))
            and summary["max_energy_residual_kwh"] <= TOLERANCE_KWH,
            "Hard constraint or energy audit failed")
    require(all(abs(row.peak_kW - peaks.get(row.local_date, 0.0)) <= 1e-8
                for row in daily.itertuples(index=False)), "Realized daily peak audit failed")
    for record in environment.completed:
        delivered = executor.evs[record["id"]].energy_delivered
        require(abs(record["delivered_kwh"] - delivered) <= TOLERANCE_KWH, "ACN-Sim energy mismatch")

    decision_table = pd.DataFrame(rows)
    require(decision_table["online_seconds"].max() < interval * 60, "Control deadline exceeded")
    performance = timing(decision_table)
    summary.update(performance, replay_wall_seconds=perf_counter() - wall,
                   requires_review=bool(performance["fallback_count"] or
                                        (performance["online_p95"] or 0) > 30))
    service = service.merge(evaluation[["sessionID", "disconnectTime"]], on="sessionID", validate="one_to_one")
    # An exact-midnight departure belongs to the day on which its final
    # delivery could occur, including departures on the evaluation end.
    service["local_date"] = service["disconnectTime"].map(
        lambda t: local_day(t - pd.Timedelta(nanoseconds=1), tz))
    service["fulfilled_kwh"] = service[["delivered_kwh", "evaluation_target_kwh"]].min(axis=1)
    service_day = service.groupby("local_date")[["evaluation_target_kwh", "fulfilled_kwh", "unmet_kwh"]].sum()
    daily = daily.join(service_day, on="local_date")
    for column in service_day.columns:
        daily[column] = daily[column].fillna(0)
    daily["load_band"] = daily["local_date"].map(bands).fillna("tail")
    for key, table in (("decision_summary", decision_table), ("daily_metrics", daily),
                       ("session_service", service), ("station_trajectory", station_trace),
                       ("session_trajectory", vehicle_trace)):
        table.to_csv(out / f"{key}.csv", index=False)
    write_json(out / "summary.json", summary)
    return summary, daily, decision_table


def smoke_controller(out):
    now = pd.Timestamp("2019-05-07T07:00:00Z")
    view = {"time": now, "capacity_kw": 50, "vehicles": ({
        "id": "test", "station_id": "EVSE", "arrival": now - pd.Timedelta(minutes=1),
        "target_kwh": .2, "delivered_kwh": 0, "remaining_kwh": .2, "pmax_kw": 6.656,
        "expected_departure": now,
    },)}
    price = lambda _: .12
    previous = FrozenController(out / "frozen", price, 2.88, 6.656)
    for interval in (5, 15, 30):
        controller = rolling.RollingHorizonController(
            out / f"synthetic_{interval}", price, 2.88, 6.656, interval_minutes=interval)
        case = controller.build_case(view)
        require(case["sessions"][0]["available"] == [0], "Overdue request must extend one control interval")
        require(case["dt"] == interval / 60 and len(case["slots"]) == 24 * 60 // interval,
                "Grid or energy coefficient mismatch")
        if interval == 15:
            require(case == previous.build_case(view), "E5 baseline LP differs from the frozen controller")
        decision = controller.decide(view)
        delivered = decision["command"].get("test", 0) * case["dt"]
        require(decision["mode"] == "lp" and abs(delivered - .2) <= TOLERANCE_KWH + 1e-8,
                f"Synthetic {interval}-minute solve failed; inspect {decision['step_directory']}")
        with patch.object(rolling, "_invoke", side_effect=TimeoutError("synthetic timeout")):
            recovered = controller.decide(view)
        require(recovered["fallback"] and math.isclose(recovered["command"]["test"] * case["dt"], .2),
                "Interval-aware EDF fallback failed")
        smoke_events(interval)


def smoke_events(interval):
    """Exercise late input and actual departure across site-local midnight."""
    delta = pd.Timedelta(minutes=interval)
    midnight = pd.Timestamp("2019-05-07T07:00:00Z")
    start = midnight - delta
    arrival = start + pd.Timedelta(minutes=1)
    release = start + pd.Timedelta(minutes=2)
    departure = midnight + pd.Timedelta(minutes=1)
    sessions = pd.DataFrame([{"sessionID": "test", "stationID": "EVSE",
                              "connectionTime": arrival, "cohort": "supplement_after_submission"}])
    events = pd.DataFrame([{"sessionID": "test", "event_time": stamp, "event_type": kind}
                           for stamp, kind in ((arrival, "arrival"), (departure, "departure"))])
    updates = pd.DataFrame([{"sessionID": "test", "modifiedAt": release, "valid": True,
                              "input_order": 0, "kWhRequested": 2, "requestedDeparture": midnight + delta}])
    environment = AggregateEnvironment(sessions, events, updates, start, interval_minutes=interval)
    require(not environment.view()["vehicles"], "Future vehicle leaked into the initial view")
    state = environment.step({})
    require(environment.now == midnight and state["vehicles"][0]["remaining_kwh"] == 2,
            "Late request did not become visible at the next control boundary")
    environment.step({"test": 6})
    require(not environment.view()["vehicles"] and environment.now == midnight + delta
            and math.isclose(environment.completed[0]["delivered_kwh"], .1),
            "Cross-midnight replay did not stop power at the actual departure")


def publish(out, summaries, groups, daily_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(groups).to_csv(out / "concurrency_summary.csv", index=False)
    daily = pd.concat(daily_rows, ignore_index=True)
    baseline = daily[(daily["interval_minutes"] == 15) & (daily["base_hours"] == 24)]
    keys = ["cohort", "local_date"]
    metrics = ["cost_USD", "peak_kW", "unmet_kwh"]
    paired = daily.merge(baseline[keys + metrics], on=keys, suffixes=("", "_m15_h24"), validate="many_to_one")
    for metric in metrics:
        paired[f"delta_{metric}"] = paired[metric] - paired[f"{metric}_m15_h24"]
    paired.to_csv(out / "daily_paired.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col", layout="constrained")
    for column, (cohort, block) in enumerate(comparison.groupby("cohort", sort=False)):
        labels = [f"{row.interval_minutes}m / {row.base_hours}h" for row in block.itertuples()]
        axes[0, column].bar(labels, block["online_p95"].astype(float), color="#178176")
        axes[0, column].set_title("At-arrival requests" if column == 0 else "Late-submitted requests")
        axes[0, column].set_ylabel("Active controller P95 (s)")
        axes[1, column].bar(labels, block["controller_total_seconds"], color="#38669a")
        axes[1, column].set_ylabel("Total controller time (s)")
        axes[1, column].tick_params(axis="x", labelrotation=25)
    figure.savefig(out / "timing.png", dpi=160)
    plt.close(figure)

    lines = ["# E5 RH-LP runtime and horizon", "",
             "Controller timings include LP worker startup and plan checks. Solver timing uses",
             "successful two-stage solves. The controller P95 uses active decisions, including fallbacks.",
             "The 15-minute reporting grid is common to all control intervals.", "",
             "| Cohort | Interval | Base horizon | Active decisions | Controller P95 (s) | "
             "Fallbacks | Satisfaction (%) | Unmet (kWh) | Cost (USD) | Peak (kW) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in comparison.itertuples():
        p95 = f"{row.online_p95:.3f}" if pd.notna(row.online_p95) else "NA"
        satisfaction = f"{100 * row.energy_satisfaction:.3f}" if pd.notna(row.energy_satisfaction) else "NA"
        lines.append(f"| {row.cohort} | {row.interval_minutes} min | {row.base_hours} h | "
                     f"{row.active_decisions} | {p95} | {row.fallback_count} | {satisfaction} | "
                     f"{row.unmet_kwh:.3f} | {row.cost_USD:.3f} | {row.max_peak_kW:.3f} |")
    lines.extend(("", "Interpret costs and peaks together with energy service. A change in control grid",
                  "also changes the complete slots available before the visible deadlines.",
                  "See `concurrency_summary.csv` for effective horizons and model sizes by load band.", ""))
    (out / "e5_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "formal"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--smoke-result", type=Path)
    args = parser.parse_args()
    if args.mode == "formal" and args.smoke_result is None:
        parser.error("formal requires --smoke-result from this package")
    fingerprints = source_fingerprints()
    if args.mode == "formal":
        evidence = json.loads(args.smoke_result.read_text(encoding="utf-8"))
        require(evidence["status"] == "passed" and evidence["mode"] == "smoke"
                and evidence["source_and_input_sha256"] == fingerprints,
                "Smoke evidence is incomplete or differs from the current package and inputs")

    station, norm, splits, sessions, events, releases, updates = read_inputs()
    tz = station["timezone"]
    dev_start, dev_end = split_bounds(splits, "development")
    split = "development" if args.mode == "smoke" else "test"
    start, end = split_bounds(splits, split)
    if args.mode == "smoke":
        main = sessions[(sessions["split"] == split) & (sessions["cohort"] == "main_at_arrival")]
        require(len(main) > 0, "No main-cohort development sessions")
        first_arrival = main["connectionTime"].min().tz_convert(tz)
        start = first_arrival.normalize().tz_convert("UTC")
        end = min(dev_end, (first_arrival.normalize() + pd.DateOffset(days=2)).tz_convert("UTC"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = (args.out or ROOT / "results/logs" / f"e5_{args.mode}_{timestamp}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    print(f"E5 output: {out}", flush=True)
    if args.mode == "smoke":
        smoke_controller(out)

    departures = dict(events.loc[events["event_type"] == "departure", ["sessionID", "event_time"]]
                      .itertuples(index=False, name=None))
    selections, all_bands, thresholds, latest_departure = {}, {}, {}, end
    for cohort, expected in COHORTS.items():
        pool = sessions[sessions["cohort"] == cohort]
        development = pool[pool["split"] == "development"]
        dev_connections = [(row.connectionTime, departures[row.sessionID])
                           for row in development.itertuples(index=False)]
        dev_peaks = connected_peaks(dev_connections, dev_start, dev_end, tz)
        thresholds[cohort] = lo, hi = [int(value) for value in np.quantile(
            list(dev_peaks.values()), [1/3, 2/3], method="inverted_cdf")]
        selected = pool[(pool["split"] == split) & pool["connectionTime"].ge(start)
                        & pool["connectionTime"].lt(end)]
        require(len(selected) > 0, f"No {cohort} sessions in {split} replay")
        if args.mode == "formal":
            require(len(selected) == expected, f"Formal cohort size mismatch: {cohort}")
        ids = set(selected["sessionID"])
        connections = [(row.connectionTime, departures[row.sessionID])
                       for row in selected.itertuples(index=False)]
        latest_departure = max(latest_departure, *(departure for _, departure in connections))
        peaks = connected_peaks(connections, start, end, tz)
        bands = {day: "low" if peak <= lo else "middle" if peak <= hi else "high"
                 for day, peak in peaks.items()}
        all_bands[cohort] = bands
        pd.DataFrame({"local_date": list(dev_peaks), "peak_connected": list(dev_peaks.values())}).to_csv(
            out / f"development_concurrency_{cohort}.csv", index=False)
        pd.DataFrame({"local_date": list(peaks), "peak_connected": list(peaks.values()),
                      "load_band": list(bands.values())}).to_csv(out / f"concurrency_{cohort}.csv", index=False)
        evaluation, history = evaluation_rows(selected, departures, updates, float(station["evse_max_power_kW"]))
        selections[cohort] = (selected, events[events["sessionID"].isin(ids)],
                              releases[releases["sessionID"].isin(ids)], evaluation, history)

    common_end = latest_departure.ceil("30min")
    require(start == start.floor("30min"), "All scenarios require a common 30-minute-aligned start")
    if args.mode == "formal":
        require(common_end == end, "Formal replay end differs from the frozen E1 test boundary")
    price_at = price_lookup(start, common_end)
    write_json(out / "run_config.json", {
        "mode": args.mode, "split": split, "scenarios": CONFIGS, "start_utc": str(start),
        "selection_end_utc": str(end), "replay_end_utc": str(common_end),
        "development_peaks_inverted_cdf_terciles": thresholds,
        "load_band_rule": "low <= first cutoff; middle <= second cutoff; high otherwise; ties stay together",
        "alpha": .5, "capacity_kw": 50, "time_limit_seconds": 30, "evaluation_grid_minutes": 15,
        "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                    "processor": platform.processor(), "python": platform.python_version(),
                    "pandas": pd.__version__, "pyomo": pyomo.version.version, "highs": highspy.Highs().version(),
                    "thread_environment": {k: os.environ.get(k) for k in
                                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
        "source_and_input_sha256": fingerprints,
    })
    baseline = pd.read_csv(E1)
    summaries, groups, comparisons, daily_rows = [], [], [], []
    for interval, hours in CONFIGS:
        for cohort, selection in selections.items():
            path = out / f"m{interval:02d}_h{hours:02d}" / cohort
            labels = {"cohort": cohort, "interval_minutes": interval, "base_hours": hours}
            print(f"Starting {path.parent.name}/{cohort}", flush=True)
            summary, daily, decisions = replay(path, interval, hours, station, norm, selection,
                                               price_at, start, common_end, all_bands[cohort])
            summaries.append({**labels, **summary})
            daily_rows.append(daily.assign(**labels))
            if args.mode == "formal" and (interval, hours) == (15, 24):
                ref = baseline[(baseline["cohort"] == cohort) & (baseline["method"] == "RH_LP")]
                require(len(ref) == 1, f"Missing unique E1 RH-LP row for {cohort}")
                for key in BASELINE_FIELDS:
                    reference, actual = float(ref.iloc[0][key]), float(summary[key])
                    comparisons.append({"cohort": cohort, "metric": key, "e1": reference, "e5": actual,
                                        "absolute_difference": abs(reference - actual)})
                pd.DataFrame(comparisons).to_csv(out / "e1_baseline_check.csv", index=False)
                require(all(row["absolute_difference"] <= 1e-4 for row in comparisons),
                        "E5 15-minute/24-hour baseline differs from E1; see e1_baseline_check.csv")
            if args.mode == "smoke":
                require(not summary["fallback_count"], f"Unexpected smoke fallback; inspect {path}")

            for band, time_rows in decisions.groupby("load_band"):
                day_rows = daily[daily["load_band"] == band]
                target = day_rows["evaluation_target_kwh"].sum()
                groups.append({**labels, "load_band": band, "days": len(day_rows),
                               "cost_USD": day_rows["cost_USD"].sum(), "max_peak_kW": day_rows["peak_kW"].max(),
                               "unmet_kwh": day_rows["unmet_kwh"].sum(),
                               "energy_satisfaction": day_rows["fulfilled_kwh"].sum() / target if target else None,
                               **timing(time_rows)})

    publish(out, summaries, groups, daily_rows)
    require(source_fingerprints() == fingerprints, "An input or source changed during this run")
    write_json(out / "verification.json", {"status": "passed", "mode": args.mode,
               "scenarios": len(CONFIGS), "runs": len(summaries),
               "requires_review": any(row["requires_review"] for row in summaries),
               "source_and_input_sha256": fingerprints})
    print(f"E5 {args.mode} passed: {out / 'verification.json'}", flush=True)


if __name__ == "__main__":
    main()
