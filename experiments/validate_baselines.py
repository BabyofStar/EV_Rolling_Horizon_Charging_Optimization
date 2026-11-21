from pathlib import Path
from bisect import bisect_right
from statistics import median
import hashlib
import json
import math
import sys
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines import uncontrolled, tou_greedy
from src.simulator import AggregateEnvironment
from src.acnsim_adapter import ACNSimExecutor

OUT = Path(sys.argv[1])
DATA = ROOT / "data/processed/caltech_12weeks_20261111T143402Z_9b087d"
passed = []

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    passed.append(name)

def load_config(name):
    return yaml.safe_load(
        (ROOT / "configs" / name).read_text(encoding="utf-8"))

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

station = load_config("station.yaml")
tariff = load_config("tariff.yaml")
splits = load_config("splits.yaml")
TZ = station["timezone"]
start = pd.Timestamp(splits["splits"]["development"]["start_utc_inclusive"])
end = pd.Timestamp(splits["splits"]["development"]["end_utc_exclusive"])
capacity = float(station["station_capacity_kW"])
pmax = float(station["evse_max_power_kW"])

check("frozen execution scope",
      station["network_model"] == "aggregate_continuous" and
      station["control_interval_minutes"] == 15 and
      station["charging_efficiency"] == 1 and
      not station["original_branch_constraints_enforced"])
check("tariff scope",
      tariff["timezone"] == TZ and tariff["unit"] == "USD/kWh" and
      tariff["calendar"] == "same_schedule_every_day_including_weekends_and_holidays" and
      tariff["extension_policy"] == "repeat_local_daily_schedule_for_any_date")

def minute(value):
    if isinstance(value, int):
        return value
    hour, mins = map(int, str(value).split(":"))
    return hour * 60 + mins

periods = sorted(
    (minute(p["start"]), minute(p["end"]), float(p["price"]))
    for p in tariff["periods"])
check("tariff continuous and nonnegative",
      periods[0][0] == 0 and periods[-1][1] == 1440 and
      all(a < b and math.isfinite(price) and price >= 0
          for a, b, price in periods) and
      all(periods[j][1] == periods[j + 1][0]
          for j in range(len(periods) - 1)))

price_cache = {}
def price_at(t):
    t = pd.Timestamp(t)
    key = t.value
    if key not in price_cache:
        local = t.tz_convert(TZ)
        m = local.hour * 60 + local.minute
        price_cache[key] = next(price for a, b, price in periods if a <= m < b)
    return price_cache[key]

for hour, expected in [(0, .12), (8, .24), (16, .40), (21, .24)]:
    t = pd.Timestamp(f"2019-05-06 {hour:02}:00", tz=TZ)
    check(f"tariff boundary {hour:02}:00", abs(price_at(t) - expected) < 1e-12)

sessions = pd.read_csv(DATA / "sessions_with_split.csv")
sessions = sessions[sessions["split"] == "development"].copy()
sessions["connectionTime"] = pd.to_datetime(sessions["connectionTime"], utc=True)
events = pd.read_csv(DATA / "environment_events.csv")
events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
releases = pd.read_csv(DATA / "input_release_events.csv")
updates = pd.read_csv(DATA / "user_updates.csv")

def valid(v):
    return str(v).strip().lower() in {"true", "1", "1.0"}

dates = [str(t.date()) for t in pd.date_range(
    start.tz_convert(TZ), end.tz_convert(TZ), freq="D", inclusive="left")]
check("49 complete local development days", len(dates) == 49)

def histories(selected, selected_events):
    arrivals = dict(zip(selected["sessionID"], selected["connectionTime"]))
    departures = dict(selected_events.loc[
        selected_events["event_type"] == "departure",
        ["sessionID", "event_time"]].itertuples(index=False, name=None))
    history = {}
    for row in selected.to_dict("records"):
        sid = row["sessionID"]
        if row["cohort"] == "main_at_arrival":
            times = [arrivals[sid].value]
            targets = [float(row["initial_kWhRequested"])]
        else:
            u = updates[updates["sessionID"] == sid].copy()
            u = u[u["valid"].map(valid)]
            u["modifiedAt"] = pd.to_datetime(u["modifiedAt"], utc=True)
            u = u[(u["modifiedAt"] >= arrivals[sid]) &
                  (u["modifiedAt"] < departures[sid])]
            u = u.sort_values(["modifiedAt", "input_order"])
            times = [t.value for t in u["modifiedAt"]]
            targets = u["kWhRequested"].astype(float).tolist()
        assert times, f"No independently valid target: {sid}"
        history[sid] = (times, targets)
    return arrivals, departures, history

summaries, daily_tables = [], {}
for cohort, count in [
    ("main_at_arrival", 64),
    ("supplement_after_submission", 1227),
]:
    selected = sessions[sessions["cohort"] == cohort].copy()
    check(cohort + ": development sample count", len(selected) == count)
    ids = set(selected["sessionID"])
    selected_events = events[events["sessionID"].isin(ids)].copy()
    selected_releases = releases[releases["sessionID"].isin(ids)].copy()
    arrivals, departures, history = histories(selected, selected_events)
    check(cohort + ": complete split boundaries",
          all(start <= arrivals[sid] < departures[sid] <= end for sid in ids))

    for method in ["Uncontrolled", "TOU_Greedy"]:
        print(f"Running {cohort}/{method}", flush=True)
        folder = OUT / cohort / method
        folder.mkdir(parents=True)
        executor = ACNSimExecutor(
            selected["stationID"].astype(str).tolist(),
            capacity_kw=capacity, voltage=float(station["evse_voltage_V"]),
            max_current=float(station["evse_max_current_A"]))
        e = AggregateEnvironment(
            selected, selected_events, selected_releases, start=start,
            capacity_kw=capacity, pmax_kw=pmax, executor=executor)

        with (folder / "requests_and_plans.jsonl").open("w", encoding="utf-8") as stream:
            while e.now < end:
                state = e.view()
                if method == "Uncontrolled":
                    result = uncontrolled(state, details=True)
                    record = {"time": str(e.now), **result}
                else:
                    result = tou_greedy(state, price_at, details=True)
                    # Sparse encoding preserves the entire future plan.
                    record = {
                        "time": str(e.now), "command": result["command"],
                        "slot_count": len(result["slots"]),
                        "effective_horizon_hours": result["effective_horizon_hours"],
                        "forecast_unmet_kwh": result["forecast_unmet_kwh"],
                        "sparse_plan_kw": {
                            sid: [[j, power] for j, power in enumerate(powers) if power > 0]
                            for sid, powers in result["plan_kw"].items()
                        },
                    }
                stream.write(json.dumps(record, default=str) + "\n")
                e.step(result["command"])

        prefix = f"{cohort}/{method}"
        check(prefix + ": all selected sessions completed",
              len(e.completed) == count and not e.view()["vehicles"] and
              not executor._connected)

        cumulative = {sid: 0.0 for sid in ids}
        vehicle_cost = {sid: 0.0 for sid in ids}
        daily = {
            date: {"local_date": date, "energy_kwh": 0.0,
                   "cost_USD": 0.0, "peak_kW": 0.0}
            for date in dates
        }
        station_rows, energy_rows = [], []
        energy_residual = power_residual = 0.0

        for segment in e.ledger:
            a, b = segment["start"], segment["end"]
            h = (b - a).value / 3_600_000_000_000
            assert h > 0
            date = str(a.tz_convert(TZ).date())
            assert date == str((b - pd.Timedelta(nanoseconds=1)).tz_convert(TZ).date())
            assert price_at(a) == price_at(b - pd.Timedelta(nanoseconds=1))
            rate = price_at(a)
            energy = sum(segment["energy_kwh"].values())
            energy_residual = max(
                energy_residual, abs(energy - segment["total_kw"] * h))
            power_residual = max(
                power_residual, segment["total_kw"] - capacity)
            daily[date]["energy_kwh"] += energy
            daily[date]["cost_USD"] += energy * rate
            daily[date]["peak_kW"] = max(
                daily[date]["peak_kW"], segment["total_kw"])
            station_rows.append({
                "start": a, "end": b, "total_kw": segment["total_kw"],
                "energy_kwh": energy, "price_USD_per_kWh": rate,
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
                vehicle_cost[sid] += amount * rate
                energy_rows.append({
                    "sessionID": sid, "start": a, "end": b,
                    "actual_kw": power, "energy_kwh": amount,
                })

        check(prefix + ": independent connection and information audit", True)
        check(prefix + ": power and energy residuals",
              power_residual <= 1e-6 and energy_residual <= 1e-5)
        check(prefix + ": actual cumulative feedback",
              all(abs(cumulative[sid] - executor.evs[sid].energy_delivered) <= 1e-5
                  for sid in ids) and
              all(abs(cumulative[s["id"]] - s["delivered_kwh"]) <= 1e-5
                  for s in e.completed))

        daily_frame = pd.DataFrame(daily.values())
        cost = float(daily_frame["cost_USD"].sum())
        check(prefix + ": independent per-session cost reconciliation",
              abs(cost - sum(vehicle_cost.values())) <= 1e-5)
        check(prefix + ": all 49 days retained", len(daily_frame) == 49)

        service_rows = []
        for sid in sorted(ids):
            target = history[sid][1][-1]
            delivered = cumulative[sid]
            service_rows.append({
                "sessionID": sid, "evaluation_target_kwh": target,
                "delivered_kwh": delivered,
                "unmet_kwh": max(target - delivered, 0),
                "irreversible_excess_kwh": max(delivered - target, 0),
                "actual_departure": departures[sid], "cost_USD": vehicle_cost[sid],
            })
        services = pd.DataFrame(service_rows)
        total_target = float(services["evaluation_target_kwh"].sum())
        satisfaction = sum(
            min(r["delivered_kwh"], r["evaluation_target_kwh"])
            for r in service_rows) / total_target if total_target > 0 else None

        pd.DataFrame(station_rows).to_csv(folder / "station_trajectory.csv", index=False)
        pd.DataFrame(energy_rows).to_csv(folder / "session_trajectory.csv", index=False)
        services.to_csv(folder / "session_service.csv", index=False)
        daily_frame.to_csv(folder / "daily_metrics.csv", index=False)
        daily_tables[(cohort, method)] = daily_frame
        summaries.append({
            "cohort": cohort, "method": method, "sessions": count,
            "cost_USD": cost,
            "max_peak_kW": float(daily_frame["peak_kW"].max()),
            "delivered_kwh": float(services["delivered_kwh"].sum()),
            "unmet_kwh": float(services["unmet_kwh"].sum()),
            "energy_satisfaction": satisfaction,
            "uncompleted_sessions": int((services["unmet_kwh"] > .01).sum()),
            "max_power_residual_kW": power_residual,
            "max_energy_residual_kwh": energy_residual,
        })

# Prespecified normalization: all days, then positive-day fallback, then unit fallback.
reference = daily_tables[("main_at_arrival", "Uncontrolled")]
def reference_value(values):
    values = [float(v) for v in values]
    value = float(median(values))
    positive = [v for v in values if v > 0]
    if value > 0:
        rule = "all_day_median"
    elif positive:
        value, rule = float(median(positive)), "positive_day_median"
    else:
        value, rule = 1.0, "unit_fallback"
    return value, {
        "rule": rule, "zero_days": sum(v == 0 for v in values),
        "positive_days": len(positive),
    }

C_ref, cost_rule = reference_value(reference["cost_USD"])
P_ref, peak_rule = reference_value(reference["peak_kW"])
check("positive finite normalization references",
      math.isfinite(C_ref) and C_ref > 0 and math.isfinite(P_ref) and P_ref > 0)

tracked = [
    ROOT / "src/baselines.py", ROOT / "src/simulator.py",
    ROOT / "src/acnsim_adapter.py", Path(__file__),
    ROOT / "configs/station.yaml", ROOT / "configs/tariff.yaml",
    ROOT / "configs/splits.yaml", DATA / "sessions_with_split.csv",
    DATA / "environment_events.csv", DATA / "input_release_events.csv",
    DATA / "user_updates.csv",
]
hashes = {str(p.relative_to(ROOT)): sha256(p) for p in tracked}
normalization = {
    "C_ref": C_ref, "P_ref": P_ref,
    "C_ref_unit": "USD", "P_ref_unit": "kW",
    "source_split": "development", "source_cohort": "main_at_arrival",
    "source_method": "capacity_limited_uncontrolled",
    "natural_days_including_zero_days": 49,
    "cost_reference_rule": cost_rule, "peak_reference_rule": peak_rule,
    "fixed_for_all_methods_weights_cohorts_and_test_days": True,
    "evidence_directory": str(OUT.relative_to(ROOT)),
    "source_and_input_sha256": hashes,
}
proposed = yaml.safe_dump(normalization, allow_unicode=True, sort_keys=False)
(OUT / "normalization_proposed.yaml").write_text(proposed, encoding="utf-8")

frozen_path = ROOT / "configs/normalization.yaml"
if frozen_path.exists():
    existing = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    check("existing frozen references agree",
          abs(existing["C_ref"] - C_ref) <= 1e-12 and
          abs(existing["P_ref"] - P_ref) <= 1e-12 and
          existing["source_and_input_sha256"] == hashes)
else:
    # Open exclusively: existing frozen settings cannot be silently overwritten.
    with frozen_path.open("x", encoding="utf-8") as stream:
        stream.write(proposed)

pd.DataFrame(summaries).to_csv(OUT / "development_summary.csv", index=False)
result = {
    "status": "passed", "check_count": len(passed), "checks": passed,
    "development_summary": summaries,
    "normalization": {
        "C_ref": C_ref, "P_ref": P_ref,
        "cost_reference_rule": cost_rule, "peak_reference_rule": peak_rule,
        "file": str(frozen_path),
    },
    "scope": "development baseline validation; not the formal test comparison",
}
(OUT / "verification.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
