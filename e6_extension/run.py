"""Replay tariff and actuator sensitivity with the frozen 15-minute controllers."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "e5_extension")]

import pandas as pd

from e5_extension.run import (
    BASELINE_FIELDS, COHORTS, E1, TOLERANCE_KWH, evaluation_rows, local_day,
    price_lookup, read_inputs, require, sha, source_fingerprints, split_bounds,
    traces, write_json,
)
from src.acnsim_adapter import ACNSimExecutor
from src.baselines import tou_greedy
from src.metrics import evaluate_run
from src.rolling_horizon import RollingHorizonController, record_realized_peaks
from src.simulator import AggregateEnvironment

DELTA = pd.Timedelta(minutes=15)
METHODS = ("RH_LP", "TOU_Greedy")
SCENARIOS = (
    ("nominal", 1.0, 1.0),
    ("ideal", 1.0, None),
    ("spread_050", .5, 1.0),
    ("spread_150", 1.5, 1.0),
    ("response_090", 1.0, .9),
    ("response_080", 1.0, .8),
)


def fingerprints():
    return {**source_fingerprints(), "e6_extension/run.py": sha(Path(__file__))}


def tariff_scenarios(original, tz):
    # A known 24-hour local day supplies the nominal time-weighted average.
    midnight = pd.Timestamp("2019-07-08", tz=tz).tz_convert("UTC")
    samples = [original(midnight + i * DELTA) for i in range(96)]
    mean = sum(samples) / len(samples)
    require(math.isclose(mean, 5.6 / 24, abs_tol=1e-12)
            and math.isclose(max(samples) - min(samples), .28, abs_tol=1e-12),
            "Original TOU did not match the E1 average and spread")
    rates = {}
    for scale in (.5, 1., 1.5):
        def transformed(stamp, multiplier=scale):
            price = original(stamp)
            return price if multiplier == 1.0 else mean + multiplier * (price - mean)

        prices = [transformed(midnight + i * DELTA) for i in range(96)]
        require(min(prices) >= 0 and math.isclose(sum(prices) / 96, mean, abs_tol=1e-12)
                and math.isclose(max(prices) - min(prices), scale * .28, abs_tol=1e-12),
                f"Tariff audit failed for scale {scale}")
        rates[scale] = transformed
    return rates, mean


def delivery_audit(executor, response_limit, environment, tz):
    fields = ["requested_on_device_kwh", "executed_on_device_kwh", "clipped_kwh",
              "clipped_execution_segments"]
    vehicle = defaultdict(lambda: dict.fromkeys(fields, 0))
    day = defaultdict(lambda: dict.fromkeys(fields, 0))
    if executor is not None:
        for record in executor.records:
            date = local_day(record["time"], tz)
            actual = record["actual_kw"]
            for sid, request in record["requested_kw"].items():
                response = actual[sid]
                require(math.isclose(response, min(request, response_limit), abs_tol=1e-8),
                        f"ACN-Sim reported unexpected response for {sid}")
                difference = max(request - response, 0.0) if request - response > 1e-8 else 0.0
                values = (request * record["hours"], response * record["hours"],
                          difference * record["hours"], int(difference > 0))
                for name, value in zip(fields, values):
                    vehicle[sid][name] += value
                    day[date][name] += value
        for record in environment.completed:
            sid = record["id"]
            require(abs(record["delivered_kwh"] - executor.evs[sid].energy_delivered) <= TOLERANCE_KWH
                    and abs(record["delivered_kwh"] - vehicle[sid]["executed_on_device_kwh"]) <= TOLERANCE_KWH,
                    f"Device/ledger energy mismatch for {sid}")
    else:
        for segment in environment.ledger:
            date = local_day(segment["start"], tz)
            for sid, energy in segment["energy_kwh"].items():
                for name in fields[:2]:
                    vehicle[sid][name] += energy
                    day[date][name] += energy
    return vehicle, day, fields


def replay(out, scenario, method, selection, dates, station, norm, rate, original):
    selected, events, releases, evaluation, history = selection
    start, end = dates
    _, _, fraction = scenario
    capacity = float(station["station_capacity_kW"])
    pmax = float(station["evse_max_power_kW"])
    tz = station["timezone"]
    limit = None if fraction is None else pmax * fraction
    executor = None if limit is None else ACNSimExecutor(
        selected["stationID"].tolist(), capacity_kw=capacity,
        voltage=float(station["evse_voltage_V"]), max_current=float(station["evse_max_current_A"]),
        response_max_kw=None if fraction == 1 else limit)
    env = AggregateEnvironment(selected, events, releases, start, capacity_kw=capacity,
                               pmax_kw=pmax, executor=executor)
    out.mkdir(parents=True, exist_ok=False)
    controller = None
    if method == "RH_LP":
        controller = RollingHorizonController(
            out / "decisions", rate, C_ref=norm["C_ref"], P_ref=norm["P_ref"],
            alpha=.5, base_hours=24, epsilon_U=1e-5, time_limit_seconds=30, timezone=tz)
    peaks, decisions = {}, []
    clock = perf_counter()
    while env.now < end:
        now, view = env.now, env.view()
        tick = perf_counter()
        if controller is None:
            command = tou_greedy(view, rate, base_hours=24)
            active = any(car["remaining_kwh"] is not None and car["remaining_kwh"] > 1e-9
                         for car in view["vehicles"])
            fallback = False
        else:
            result = controller.decide(view, dict(peaks))
            command = result["command"]
            active = result["mode"] != "empty"
            fallback = result["fallback"]
        seconds = perf_counter() - tick
        decisions.append({"time": now, "active": active, "fallback": fallback,
                          "online_seconds": seconds})
        index = len(env.ledger)
        env.step(command)
        if controller is not None:
            record_realized_peaks(peaks, env.ledger[index:], env.now, tz)
        if local_day(now, tz) != local_day(env.now, tz):
            print(f"{scenario[0]}/{method}/{out.name}: finished {local_day(now, tz)}", flush=True)
    require(env.now == end and not env.view()["vehicles"] and len(env.completed) == len(selected)
            and (executor is None or not executor._connected), "Incomplete session replay")
    station_trace, vehicle_trace = traces(env, history, DELTA)
    summary, daily, service = evaluate_run(station_trace, vehicle_trace, evaluation,
                                           start, end, rate, capacity, tz)
    require(not any(summary[col] for col in ("station_violation_segments", "device_violation_segments",
                                             "connection_violation_segments"))
            and summary["max_energy_residual_kwh"] <= TOLERANCE_KWH, "Power or energy audit failed")
    _, original_daily, _ = evaluate_run(station_trace, vehicle_trace, evaluation,
                                         start, end, original, capacity, tz)
    daily = daily.merge(original_daily[["local_date", "cost_USD"]].rename(
        columns={"cost_USD": "cost_at_e1_tariff_USD"}), on="local_date", validate="one_to_one")
    service = service.merge(evaluation[["sessionID", "disconnectTime"]], on="sessionID", validate="one_to_one")
    service["local_date"] = service["disconnectTime"].map(lambda t: local_day(t - pd.Timedelta(nanoseconds=1), tz))
    by_departure = service.groupby("local_date")["unmet_kwh"].sum()
    daily["unmet_at_departure_kwh"] = daily["local_date"].map(by_departure).fillna(0)

    device_vehicle, device_day, fields = delivery_audit(executor, limit, env, tz)
    for field in fields:
        service[field] = [device_vehicle[sid][field] for sid in service["sessionID"]]
        daily[field] = [device_day[date][field] for date in daily["local_date"]]
        summary[field] = float(service[field].sum())
    if fraction is None or fraction == 1:
        require(summary["clipped_execution_segments"] == 0, "Unexpected clipping at the nominal setting")
    table = pd.DataFrame(decisions)
    require(table["online_seconds"].max() < 900, "Control deadline was exceeded")
    active_seconds = table.loc[table["active"], "online_seconds"]
    summary.update(cost_at_e1_tariff_USD=float(daily["cost_at_e1_tariff_USD"].sum()),
                   decision_count=len(table), fallback_count=int(table["fallback"].sum()),
                   active_online_p95_sec=float(active_seconds.quantile(.95)) if len(active_seconds) else None,
                   wall_seconds=perf_counter() - clock)
    for name, frame in (("daily_metrics", daily), ("session_service", service), ("decisions", table),
                        ("station_trajectory", station_trace), ("session_trajectory", vehicle_trace)):
        frame.to_csv(out / f"{name}.csv", index=False)
    write_json(out / "summary.json", summary)
    return summary, daily, service


def select_inputs(mode):
    station, norm, splits, sessions, events, releases, updates = read_inputs()
    tz = station["timezone"]
    split = "test" if mode == "formal" else "development"
    start, end = split_bounds(splits, split)
    selected_split = sessions[sessions["split"] == split]
    if mode == "smoke":
        earliest = selected_split.loc[selected_split["cohort"] == "main_at_arrival", "connectionTime"].min()
        start = earliest.tz_convert(tz).normalize().tz_convert("UTC")
        end = min(end, (start.tz_convert(tz) + pd.DateOffset(days=2)).tz_convert("UTC"))
    selected_split = selected_split[selected_split["connectionTime"].ge(start)
                                    & selected_split["connectionTime"].lt(end)]
    departures = dict(events.loc[events["event_type"] == "departure", ["sessionID", "event_time"]]
                      .itertuples(index=False, name=None))
    tail = max(end, max(departures[sid] for sid in selected_split["sessionID"]).ceil(DELTA))
    require(mode != "formal" or tail == end, "Replay end differs from the E1 test boundary")
    selections = {}
    for cohort, size in COHORTS.items():
        selected = selected_split[selected_split["cohort"] == cohort]
        require(len(selected) > 0 and (mode != "formal" or len(selected) == size), f"Invalid cohort size: {cohort}")
        ids = set(selected["sessionID"])
        evaluation, history = evaluation_rows(selected, departures, updates, float(station["evse_max_power_kW"]))
        selections[cohort] = (selected, events[events["sessionID"].isin(ids)],
                              releases[releases["sessionID"].isin(ids)], evaluation, history)
    return station, norm, (start, tail), selections


def compare_metrics(reference, summary, group):
    return [{**group, "metric": field, "reference": float(reference[field]),
             "observed": float(summary[field]),
             "difference": float(summary[field]) - float(reference[field])} for field in BASELINE_FIELDS]


def smoke_actuator(station):
    """Known 15-minute full-rate request, with exact delivered energy at each limit."""
    now = pd.Timestamp("2019-05-06T07:00:00Z")
    end = now + 2 * DELTA
    pmax = float(station["evse_max_power_kW"])
    capacity = float(station["station_capacity_kW"])
    sessions = pd.DataFrame([{"sessionID": "probe", "stationID": "EVSE",
                              "connectionTime": now, "cohort": "main_at_arrival",
                              "initial_kWhRequested": 10.0, "initial_requestedDeparture": end}])
    events = pd.DataFrame([{"sessionID": "probe", "event_type": kind, "event_time": time}
                           for kind, time in (("arrival", now), ("departure", end))])
    results = []
    for fraction in (None, 1., .9, .8):
        executor = None if fraction is None else ACNSimExecutor(
            ["EVSE"], capacity_kw=capacity, voltage=float(station["evse_voltage_V"]),
            max_current=float(station["evse_max_current_A"]),
            response_max_kw=None if fraction == 1 else pmax * fraction)
        env = AggregateEnvironment(sessions, events, pd.DataFrame(), now, capacity_kw=capacity,
                                   pmax_kw=pmax, executor=executor)
        env.step({"probe": pmax})
        observed = float(env.view()["vehicles"][0]["delivered_kwh"])
        expected = (fraction if fraction is not None else 1.0) * pmax / 4
        require(math.isclose(observed, expected, abs_tol=1e-8), f"Actuator smoke check failed: {fraction}")
        env.step({})
        require(env.now == end and len(env.completed) == 1, "Synthetic actuator vehicle failed to depart")
        by_vehicle, _, _ = delivery_audit(executor, None if fraction is None else pmax * fraction,
                                           env, station["timezone"])
        results.append({"response_fraction": fraction, "expected_delivered_kwh": expected,
                        "delivered_kwh": observed, **by_vehicle["probe"]})
    return results


def paired(frames, fields, join_keys):
    data = pd.concat(frames, ignore_index=True)
    baseline = data[data["scenario"] == "nominal"]
    result = data.merge(baseline[join_keys + list(fields)], on=join_keys,
                        suffixes=("", "_nominal"), validate="many_to_one")
    require(len(result) == len(data), "Some rows have no matched nominal reference")
    for field in fields:
        result[f"delta_{field}"] = result[field] - result[field + "_nominal"]
    return result


def publish(out, rows, daily, services):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "comparison.csv", index=False)
    daily_fields = ("cost_USD", "cost_at_e1_tariff_USD", "peak_kW", "unmet_at_departure_kwh")
    paired(daily, daily_fields, ["method", "cohort", "local_date"]).to_csv(out / "daily_paired.csv", index=False)
    paired(services, ("delivered_kwh", "unmet_kwh"), ["method", "cohort", "sessionID"]).to_csv(
        out / "session_paired.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, (cohort, block) in zip(axes, comparison.groupby("cohort", sort=False)):
        for method, points in block.groupby("method", sort=False):
            ax.plot(points["scenario"], 100 * points["energy_satisfaction"], "o-", label=method)
        ax.set_title(cohort.replace("_", " "))
        ax.set_ylabel("Energy satisfaction (%)")
        ax.tick_params(axis="x", labelrotation=25)
        ax.legend()
    figure.savefig(out / "energy_satisfaction.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "formal"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--smoke-result", type=Path)
    args = parser.parse_args()
    evidence_hashes = fingerprints()
    if args.mode == "formal":
        if args.smoke_result is None:
            parser.error("formal requires --smoke-result")
        smoke = json.loads(args.smoke_result.read_text(encoding="utf-8"))
        require(smoke["mode"] == "smoke" and smoke["status"] == "passed"
                and smoke["source_and_input_sha256"] == evidence_hashes, "Smoke evidence does not match this run")
    station, norm, dates, selections = select_inputs(args.mode)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    print(f"E6 output: {out}", flush=True)
    original = price_lookup(*dates)
    rates, mean = tariff_scenarios(original, station["timezone"])
    if args.mode == "smoke":
        write_json(out / "actuator_smoke.json", smoke_actuator(station))
    write_json(out / "run_config.json", {"mode": args.mode, "dates_utc": [str(t) for t in dates],
               "alpha": .5, "capacity_kw": 50, "interval_minutes": 15, "base_hours": 24,
               "C_ref": float(norm["C_ref"]), "P_ref": float(norm["P_ref"]), "tariff_mean": mean,
               "scenarios": [dict(scenario=name, spread_scale=scale, response_fraction=fraction)
                             for name, scale, fraction in SCENARIOS],
               "methods": METHODS, "source_and_input_sha256": evidence_hashes})

    e1 = pd.read_csv(E1)
    rows, daily, services, e1_checks, ideal_checks = [], [], [], [], []
    nominal = {}
    for scenario in SCENARIOS:
        name, scale, fraction = scenario
        for method in METHODS:
            for cohort, data in selections.items():
                label = {"scenario": name, "method": method, "cohort": cohort}
                print(f"Starting {name}/{method}/{cohort}", flush=True)
                outcome, day, service = replay(out / name / method / cohort, scenario, method,
                                                data, dates, station, norm, rates[scale], original)
                rows.append({**label, "spread_scale": scale, "response_fraction": fraction, **outcome})
                daily.append(day.assign(**label))
                services.append(service.assign(**label))
                if args.mode == "smoke":
                    require(outcome["fallback_count"] == 0, f"Unexpected smoke fallback: {label}")
                if name == "nominal":
                    nominal[method, cohort] = outcome
                    if args.mode == "formal":
                        ref = e1[(e1["method"] == method) & (e1["cohort"] == cohort)]
                        require(len(ref) == 1, f"Missing E1 baseline row: {label}")
                        e1_checks.extend(compare_metrics(ref.iloc[0], outcome, label))
                        pd.DataFrame(e1_checks).to_csv(out / "e1_baseline_check.csv", index=False)
                        require(all(abs(r["difference"]) <= 1e-4 for r in e1_checks), "E1 baseline comparison failed")
                elif name == "ideal":
                    ideal_checks.extend(compare_metrics(nominal[method, cohort], outcome, label))
                    pd.DataFrame(ideal_checks).to_csv(out / "ideal_nominal_check.csv", index=False)
    ideal_equivalent = all(abs(row["difference"]) <= 1e-4 for row in ideal_checks)
    publish(out, rows, daily, services)
    require(fingerprints() == evidence_hashes, "Source or input changed during the run")
    require(args.mode != "smoke" or ideal_equivalent,
            "Ideal and nominal smoke aggregates differ; see ideal_nominal_check.csv and the trajectories")
    write_json(out / "verification.json", {"status": "passed", "mode": args.mode,
               "runs": len(rows), "scenarios": len(SCENARIOS), "ideal_nominal_equivalent": ideal_equivalent,
               "requires_review": not ideal_equivalent or any(row["fallback_count"] for row in rows),
               "source_and_input_sha256": evidence_hashes})
    print(f"E6 {args.mode} passed: {out / 'verification.json'}", flush=True)


if __name__ == "__main__":
    main()
