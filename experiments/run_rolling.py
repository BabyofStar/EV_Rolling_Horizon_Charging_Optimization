from pathlib import Path
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rolling_horizon import RollingHorizonController, record_realized_peaks
from src.simulator import AggregateEnvironment
from src.acnsim_adapter import ACNSimExecutor
from src.tariff import attach_tariff

OUT = Path(sys.argv[1])
DATA = ROOT / "data/processed/caltech_12weeks_20261111T143402Z_9b087d"
passed = []

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    passed.append(name)

def config(name):
    return yaml.safe_load(
        (ROOT / "configs" / name).read_text(encoding="utf-8"))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

station, splits = config("station.yaml"), config("splits.yaml")
norm_path = ROOT / "configs/normalization.yaml"
norm_before = digest(norm_path)
norm = config("normalization.yaml")
TZ = station["timezone"]
start = pd.Timestamp(splits["splits"]["development"]["start_utc_inclusive"])
dev_end = pd.Timestamp(splits["splits"]["development"]["end_utc_exclusive"])
arrival_end = start + pd.Timedelta(days=7)
capacity = float(station["station_capacity_kW"])
pmax = float(station["evse_max_power_kW"])

check("frozen execution settings",
      station["control_interval_minutes"] == 15 and
      station["charging_efficiency"] == 1 and
      station["network_model"] == "aggregate_continuous" and
      not station["original_branch_constraints_enforced"])
check("formal normalization preserved",
      abs(norm["C_ref"] - 2.88) < 1e-12 and
      abs(norm["P_ref"] - 6.656) < 1e-12)

# This calendar is public/known; it contains no future demand information.
grid = pd.date_range(
    start, dev_end + pd.Timedelta(days=2), freq="15min", inclusive="left")
priced = attach_tariff(pd.DataFrame({"start_utc": grid}))
price_cache = dict(zip(grid.asi8, priced["price_USD_per_kWh"].astype(float)))
def price_at(t):
    t = pd.Timestamp(t)
    if t.value not in price_cache:
        frame = attach_tariff(pd.DataFrame({"start_utc": [t]}))
        price_cache[t.value] = float(frame["price_USD_per_kWh"].iloc[0])
    return price_cache[t.value]

# Independently compare the existing tariff module with the saved tariff table.
saved_tariff = pd.read_csv(
    ROOT / "data/external/tariff.csv",
    usecols=["start_utc", "price_USD_per_kWh"])
saved_tariff["start_utc"] = pd.to_datetime(saved_tariff["start_utc"], utc=True)
sample_grid = pd.date_range(start, arrival_end, freq="15min", inclusive="left")
saved_prices = saved_tariff.set_index("start_utc")[
    "price_USD_per_kWh"].reindex(sample_grid)
check("known tariff agrees with saved table",
      saved_prices.notna().all() and
      np.max(np.abs(saved_prices.to_numpy() -
                    np.array([price_at(t) for t in sample_grid]))) <= 1e-12)

sessions = pd.read_csv(DATA / "sessions_with_split.csv")
sessions["connectionTime"] = pd.to_datetime(sessions["connectionTime"], utc=True)
events = pd.read_csv(DATA / "environment_events.csv")
events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
releases = pd.read_csv(DATA / "input_release_events.csv")
updates = pd.read_csv(DATA / "user_updates.csv")

def valid(value):
    return str(value).strip().lower() in {"true", "1", "1.0"}

summaries = []
for cohort, expected in [
    ("main_at_arrival", 5),
    ("supplement_after_submission", 198),
]:
    selected = sessions[
        (sessions["split"] == "development") &
        (sessions["cohort"] == cohort) &
        (sessions["connectionTime"] >= start) &
        (sessions["connectionTime"] < arrival_end)
    ].copy()
    check(cohort + ": fixed first-week selection", len(selected) == expected)
    ids = set(selected["sessionID"])
    ev = events[events["sessionID"].isin(ids)].copy()
    rel = releases[releases["sessionID"].isin(ids)].copy()
    arrivals = dict(zip(selected["sessionID"], selected["connectionTime"]))
    departures = dict(ev.loc[
        ev["event_type"] == "departure",
        ["sessionID", "event_time"]].itertuples(index=False, name=None))
    end = max(arrival_end, max(departures.values()).ceil("15min"))
    check(cohort + ": tail inside development split", end <= dev_end)

    # Independent target history, from user_updates rather than release events.
    history = {}
    for row in selected.to_dict("records"):
        sid = row["sessionID"]
        if cohort == "main_at_arrival":
            times, targets = [arrivals[sid].value], [float(row["initial_kWhRequested"])]
        else:
            u = updates[updates["sessionID"] == sid].copy()
            u = u[u["valid"].map(valid)]
            u["modifiedAt"] = pd.to_datetime(u["modifiedAt"], utc=True)
            u = u[(u["modifiedAt"] >= arrivals[sid]) &
                  (u["modifiedAt"] < departures[sid])]
            u = u.sort_values(["modifiedAt", "input_order"])
            times = [t.value for t in u["modifiedAt"]]
            targets = u["kWhRequested"].astype(float).tolist()
        assert times
        history[sid] = (times, targets)

    folder = OUT / cohort
    folder.mkdir()
    executor = ACNSimExecutor(
        selected["stationID"].astype(str).tolist(),
        capacity_kw=capacity, voltage=float(station["evse_voltage_V"]),
        max_current=float(station["evse_max_current_A"]))
    e = AggregateEnvironment(
        selected, ev, rel, start=start, capacity_kw=capacity,
        pmax_kw=pmax, executor=executor)
    controller = RollingHorizonController(
        folder / "decisions", price_at,
        C_ref=norm["C_ref"], P_ref=norm["P_ref"],
        alpha=.5, base_hours=24, time_limit_seconds=30, timezone=TZ)
    peaks, decisions = {}, []
    visible_cases = 0

    while e.now < end:
        state = e.view()
        decision = controller.decide(state, dict(peaks))
        report = decision["solve_report"]
        today = str(e.now.tz_convert(TZ).date())
        if decision["mode"] == "lp":
            case = json.loads(
                (Path(decision["step_directory"]) / "case.json").read_text(
                    encoding="utf-8"))
            visible = {
                v["id"]: v for v in state["vehicles"]
                if v["target_kwh"] is not None and
                v["remaining_kwh"] > 1e-9 and
                v["arrival"].ceil("15min") <= e.now
            }
            assert {c["id"] for c in case["sessions"]} == set(visible)
            for c in case["sessions"]:
                assert abs(c["demand"] -
                           visible[c["id"]]["remaining_kwh"]) <= 1e-8
            for day, value in case["past_peak"].items():
                expected_peak = peaks.get(day, 0.0) if day == today else 0.0
                assert abs(value - expected_peak) <= 1e-8
            assert report["peak_by_day_kW"][today] >= peaks.get(today, 0) - 1e-6
            visible_cases += 1

        solver_seconds = (
            sum(s["seconds"] for s in report["stages"])
            if report is not None else None)
        decisions.append({
            "time": e.now, "mode": decision["mode"],
            "total_seconds": decision["total_seconds"],
            "two_stage_solver_seconds": solver_seconds,
            "effective_horizon_hours": decision["effective_horizon_hours"],
            "fallback": decision["fallback"], "reason": decision["reason"],
            "step_directory": decision["step_directory"],
        })
        index = len(e.ledger)
        e.step(decision["command"])
        record_realized_peaks(peaks, e.ledger[index:], e.now, TZ)

        if len(decisions) % 96 == 0:
            progress = {
                "cohort": cohort, "time": str(e.now),
                "steps": len(decisions),
                "fallbacks": sum(d["fallback"] for d in decisions),
            }
            (OUT / "progress.json").write_text(
                json.dumps(progress, indent=2), encoding="utf-8")
            print(json.dumps(progress), flush=True)

    check(cohort + ": LP exercised on real visible requests", visible_cases > 0)
    check(cohort + ": complete selected-session replay",
          len(e.completed) == expected and not e.view()["vehicles"] and
          not executor._connected)

    cumulative = {sid: 0.0 for sid in ids}
    independent_peaks, station_rows, energy_rows = {}, [], []
    energy_residual = power_residual = 0.0
    for segment in e.ledger:
        a, b = segment["start"], segment["end"]
        h = (b - a).value / 3_600_000_000_000
        assert h > 0
        day = str(a.tz_convert(TZ).date())
        assert day == str((b - pd.Timedelta(nanoseconds=1)).tz_convert(TZ).date())
        independent_peaks[day] = max(
            independent_peaks.get(day, 0), segment["total_kw"])
        total_energy = sum(segment["energy_kwh"].values())
        energy_residual = max(
            energy_residual, abs(total_energy - segment["total_kw"] * h))
        power_residual = max(power_residual, segment["total_kw"] - capacity)
        station_rows.append({
            "start": a, "end": b, "total_kw": segment["total_kw"],
            "energy_kwh": total_energy,
        })
        for sid, amount in segment["energy_kwh"].items():
            power = segment["power_kw"][sid]
            power_residual = max(power_residual, power - pmax, -power)
            energy_residual = max(energy_residual, abs(amount - power * h))
            assert a >= arrivals[sid].ceil("15min") and b <= departures[sid]
            times, targets = history[sid]
            j = bisect_right(times, a.value) - 1
            assert j >= 0
            assert a >= pd.Timestamp(times[0], tz="UTC").ceil("15min")
            before = cumulative[sid]
            cumulative[sid] += amount
            assert cumulative[sid] <= max(before, targets[j]) + 1e-5
            energy_rows.append({
                "sessionID": sid, "start": a, "end": b,
                "actual_kw": power, "energy_kwh": amount,
            })

    check(cohort + ": independent input, connection and demand audit", True)
    check(cohort + ": device, station and energy residuals",
          power_residual <= 1e-6 and energy_residual <= 1e-5)
    check(cohort + ": independent actual daily peak feedback",
          peaks.keys() == independent_peaks.keys() and
          all(abs(peaks[d] - independent_peaks[d]) <= 1e-8 for d in peaks))
    check(cohort + ": cumulative ACN-Sim feedback",
          all(abs(cumulative[sid] - executor.evs[sid].energy_delivered) <= 1e-5
              for sid in ids) and
          all(abs(cumulative[s["id"]] - s["delivered_kwh"]) <= 1e-5
              for s in e.completed))

    frame = pd.DataFrame(decisions)
    active = frame[frame["mode"] != "empty"]
    fallbacks = frame[frame["fallback"]]
    check(cohort + ": every decision below control interval",
          frame["total_seconds"].max() < 900)
    frame.to_csv(folder / "decision_summary.csv", index=False)
    pd.DataFrame(station_rows).to_csv(
        folder / "station_trajectory.csv", index=False)
    pd.DataFrame(energy_rows).to_csv(
        folder / "session_trajectory.csv", index=False)
    pd.DataFrame(e.completed).to_csv(
        folder / "completed_sessions.csv", index=False)
    summaries.append({
        "cohort": cohort, "sessions": expected,
        "execution_end_utc": str(end),
        "decision_modes": dict(Counter(frame["mode"])),
        "active_decision_mean_seconds": float(active["total_seconds"].mean()),
        "active_decision_P95_seconds": float(active["total_seconds"].quantile(.95)),
        "active_decision_max_seconds": float(active["total_seconds"].max()),
        "fallback_count": len(fallbacks),
        "fallback_fraction_active": len(fallbacks) / len(active),
        "fallback_examples": fallbacks[["time", "reason"]].head(10).astype(str).to_dict("records"),
        "max_power_residual_kW": power_residual,
        "max_energy_residual_kwh": energy_residual,
        "diagnostic_delivered_kwh": sum(cumulative.values()),
        "requires_review": bool(
            len(fallbacks) or active["total_seconds"].quantile(.95) > 30),
    })

check("normalization file unchanged", digest(norm_path) == norm_before)
tracked = [
    ROOT / "src/rolling_horizon.py", ROOT / "src/optimization_model.py",
    ROOT / "src/simulator.py", ROOT / "src/acnsim_adapter.py",
    ROOT / "src/baselines.py", ROOT / "src/tariff.py", Path(__file__),
    norm_path, ROOT / "configs/station.yaml", ROOT / "configs/splits.yaml",
    DATA / "sessions_with_split.csv", DATA / "environment_events.csv",
    DATA / "input_release_events.csv", DATA / "user_updates.csv",
]
result = {
    "status": "passed", "check_count": len(passed), "checks": passed,
    "replays": summaries,
    "requires_review": any(s["requires_review"] for s in summaries),
    "scope": "development first-week dynamic validation; not formal comparison",
    "source_and_input_sha256": {
        str(p.relative_to(ROOT)): digest(p) for p in tracked},
}
(OUT / "verification.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(
    {k: v for k, v in result.items() if k != "source_and_input_sha256"},
    ensure_ascii=False, indent=2))
