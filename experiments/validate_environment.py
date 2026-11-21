from pathlib import Path
from bisect import bisect_right
import hashlib
import json
import sys
import shutil
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.simulator import AggregateEnvironment
from src.acnsim_adapter import ACNSimExecutor

OUT = Path(sys.argv[1])
DATA = ROOT / "data/processed/caltech_12weeks_20261111T143402Z_9b087d"
TZ = "America/Los_Angeles"
DT = pd.Timedelta(minutes=15)
passed = []

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    passed.append(name)

def hours(delta):
    return delta.value / 3_600_000_000_000

def valid(value):
    return str(value).strip().lower() in {"true", "1", "1.0"}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

station = yaml.safe_load(
    (ROOT / "configs/station.yaml").read_text(encoding="utf-8"))
splits = yaml.safe_load(
    (ROOT / "configs/splits.yaml").read_text(encoding="utf-8"))
check("frozen aggregate scope",
      station["network_model"] == "aggregate_continuous" and
      not station["original_branch_constraints_enforced"] and
      station["control_interval_minutes"] == 15 and
      abs(station["charging_efficiency"] - 1) < 1e-12)

capacity = float(station["station_capacity_kW"])
pmax = float(station["evse_max_power_kW"])
voltage = float(station["evse_voltage_V"])
current = float(station["evse_max_current_A"])
check("device power and current units agree",
      abs(pmax - voltage * current / 1000) < 1e-10)

start = pd.Timestamp(splits["splits"]["development"]["start_utc_inclusive"])
dev_end = pd.Timestamp(splits["splits"]["development"]["end_utc_exclusive"])
arrival_end = start + pd.Timedelta(days=7)

sessions = pd.read_csv(DATA / "sessions_with_split.csv")
events = pd.read_csv(DATA / "environment_events.csv")
releases = pd.read_csv(DATA / "input_release_events.csv")
independent_updates = pd.read_csv(DATA / "user_updates.csv")
sessions["connectionTime"] = pd.to_datetime(
    sessions["connectionTime"], utc=True)
events["event_time"] = pd.to_datetime(events["event_time"], utc=True)

# Diagnostic control only. It reads view() and has no environment/file access.
def diagnostic_control(view):
    result = {}
    capacity_left = view["capacity_kw"]
    now = view["time"]
    vehicles = sorted(
        view["vehicles"], key=lambda v: (v["arrival"], v["id"]))
    for v in vehicles:
        remaining = v["remaining_kwh"]
        if remaining is None or remaining <= 1e-9:
            continue
        estimate = v["expected_departure"]
        deadline = (now + DT if estimate <= now else estimate.floor("15min"))
        if deadline < now + DT:
            continue
        power = min(v["pmax_kw"], capacity_left, remaining / 0.25)
        if power > 0:
            result[v["id"]] = power
            capacity_left -= power
    return result

def run(selected, selected_events, selected_releases, begin, end,
        backend, run_dir, control=diagnostic_control):
    executor = None
    if backend:
        executor = ACNSimExecutor(
            selected["stationID"].astype(str).tolist(),
            capacity_kw=capacity, voltage=voltage, max_current=current)
    e = AggregateEnvironment(
        selected, selected_events, selected_releases, start=begin,
        capacity_kw=capacity, pmax_kw=pmax, executor=executor)
    while e.now < end:
        e.step(control(e.view()))
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        station_rows, energy_rows = [], []
        for row in e.ledger:
            station_rows.append({
                "start": row["start"], "end": row["end"],
                "total_kw": row["total_kw"],
                "energy_kwh": sum(row["energy_kwh"].values()),
            })
            for sid, energy in row["energy_kwh"].items():
                energy_rows.append({
                    "sessionID": sid, "start": row["start"], "end": row["end"],
                    "actual_kw": row["power_kw"][sid], "energy_kwh": energy,
                })
        pd.DataFrame(station_rows).to_csv(
            run_dir / "station_trajectory.csv", index=False)
        pd.DataFrame(energy_rows).to_csv(
            run_dir / "session_trajectory.csv", index=False)
        pd.DataFrame(e.completed).to_csv(
            run_dir / "completed_sessions.csv", index=False)
    return e, executor

def audit(e, executor, selected, selected_events, prefix):
    arrivals = dict(zip(selected["sessionID"], selected["connectionTime"]))
    departures = dict(
        selected_events.loc[
            selected_events["event_type"] == "departure",
            ["sessionID", "event_time"]].itertuples(index=False, name=None))
    check(prefix + ": complete selected-session replay",
          len(e.completed) == len(selected) and not e.view()["vehicles"])

    # Build an independent request history from user_updates.csv.
    history = {}
    for row in selected.to_dict("records"):
        sid = row["sessionID"]
        if row["cohort"] == "main_at_arrival":
            history[sid] = (
                [arrivals[sid].value], [float(row["initial_kWhRequested"])])
        else:
            u = independent_updates[
                independent_updates["sessionID"] == sid].copy()
            u = u[u["valid"].map(valid)]
            u["modifiedAt"] = pd.to_datetime(u["modifiedAt"], utc=True)
            u = u[(u["modifiedAt"] >= arrivals[sid]) &
                  (u["modifiedAt"] < departures[sid])]
            u = u.sort_values(["modifiedAt", "input_order"])
            history[sid] = (
                [t.value for t in u["modifiedAt"]],
                u["kWhRequested"].astype(float).tolist())

    cumulative = {sid: 0.0 for sid in arrivals}
    max_residual = 0.0
    capacity_residual = 0.0
    device_residual = 0.0
    for segment in e.ledger:
        h = hours(segment["end"] - segment["start"])
        assert h > 0
        capacity_residual = max(
            capacity_residual, segment["total_kw"] - capacity)
        max_residual = max(
            max_residual,
            abs(sum(segment["energy_kwh"].values()) - segment["total_kw"] * h))
        for sid, energy in segment["energy_kwh"].items():
            power = segment["power_kw"][sid]
            device_residual = max(device_residual, power - pmax, -power)
            max_residual = max(max_residual, abs(energy - power * h))
            assert segment["start"] >= arrivals[sid].ceil("15min")
            assert segment["end"] <= departures[sid]
            times, targets = history[sid]
            index = bisect_right(times, segment["start"].value) - 1
            assert index >= 0, f"Unreleased request: {sid}"
            assert segment["start"] >= pd.Timestamp(
                times[0], tz="UTC").ceil("15min")
            before = cumulative[sid]
            cumulative[sid] += energy
            # Demand reductions may leave irreversible excess, but no new excess.
            assert cumulative[sid] <= max(before, targets[index]) + 1e-5

    check(prefix + ": connection and independent information audit", True)
    check(prefix + ": segment energy conservation", max_residual <= 1e-5)
    check(prefix + ": station and device limits",
          capacity_residual <= 1e-6 and device_residual <= 1e-6)
    check(prefix + ": cumulative session energy audit",
          all(abs(cumulative[s["id"]] - s["delivered_kwh"]) <= 1e-5
              for s in e.completed))
    if executor is not None:
        check(prefix + ": actual ACN-Sim cumulative feedback",
              all(abs(cumulative[sid] - ev.energy_delivered) <= 1e-5
                  for sid, ev in executor.evs.items()) and
              not executor._connected)
    return cumulative, max_residual

summaries = {}
for cohort in ["main_at_arrival", "supplement_after_submission"]:
    selected = sessions[
        (sessions["split"] == "development") &
        (sessions["cohort"] == cohort) &
        (sessions["connectionTime"] >= start) &
        (sessions["connectionTime"] < arrival_end)
    ].copy()
    check(cohort + ": nonempty fixed first-week selection", len(selected) > 0)
    ids = set(selected["sessionID"])
    selected_events = events[events["sessionID"].isin(ids)].copy()
    selected_releases = releases[releases["sessionID"].isin(ids)].copy()
    last_departure = selected_events.loc[
        selected_events["event_type"] == "departure", "event_time"].max()
    end = max(arrival_end, last_departure.ceil("15min"))
    check(cohort + ": tail remains inside development split", end <= dev_end)

    results = {}
    for backend, name in [(False, "ideal"), (True, "acnsim")]:
        e, executor = run(
            selected, selected_events, selected_releases,
            start, end, backend, OUT / cohort / name)
        totals, residual = audit(
            e, executor, selected, selected_events, f"{cohort}/{name}")
        results[name] = totals

    check(cohort + ": ideal and adapted real-data energy agree",
          all(abs(results["ideal"][sid] - results["acnsim"][sid]) <= 1e-5
              for sid in ids))
    total = sum(results["acnsim"].values())
    check(cohort + ": replay includes positive actual delivery", total > 0)
    summaries[cohort] = {
        "sessions": len(selected),
        "arrival_start_utc": str(start),
        "arrival_end_utc_exclusive": str(arrival_end),
        "execution_end_utc": str(end),
        "diagnostic_delivered_kwh": total,
    }

# Synthetic DST tests exercise actual execution over a full local natural day.
for date, expected_hours in [("2019-03-10", 23), ("2019-11-03", 25)]:
    local_start = pd.Timestamp(date, tz=TZ)
    local_end = local_start + pd.DateOffset(days=1)
    begin, end = local_start.tz_convert("UTC"), local_end.tz_convert("UTC")
    sid = "synthetic-dst"
    selected = pd.DataFrame([{
        "sessionID": sid, "stationID": "dst-evse",
        "connectionTime": begin, "cohort": "main_at_arrival",
        "initial_kWhRequested": 500.0,
        "initial_requestedDeparture": end,
    }])
    selected_events = pd.DataFrame([
        {"sessionID": sid, "event_type": "arrival", "event_time": begin},
        {"sessionID": sid, "event_type": "departure", "event_time": end},
    ])
    for backend, name in [(False, "ideal"), (True, "acnsim")]:
        e, executor = run(
            selected, selected_events, pd.DataFrame(),
            begin, end, backend, None,
            control=lambda view: {sid: 6.0} if view["vehicles"] else {})
        check(f"DST {date}/{name}: physical duration and energy",
              len(e.ledger) == expected_hours * 4 and
              abs(e.completed[0]["delivered_kwh"] - 6 * expected_hours) <= 1e-5)
        check(f"DST {date}/{name}: one local calendar day",
              all(str(row["start"].tz_convert(TZ).date()) == date
                  for row in e.ledger))

# Verify the previously supplied independent regression evidence.
previous = ROOT / "results/logs/stage3_adapter_20261112T074129922228Z"
for name, count in [
    ("event_regression", 22), ("execution_regression", 13),
    ("adapter_integration", 20),
]:
    evidence = json.loads(
        (previous / f"{name}_verification.json").read_text(encoding="utf-8"))
    check(f"prior {name}: saved evidence",
          evidence["status"] == "passed" and evidence["check_count"] == count)

tracked = [
    ROOT / "src/simulator.py", ROOT / "src/acnsim_adapter.py",
    Path(__file__), DATA / "sessions_with_split.csv",
    DATA / "environment_events.csv", DATA / "input_release_events.csv",
    DATA / "user_updates.csv", DATA / "information_policy.json",
    ROOT / "configs/station.yaml", ROOT / "configs/splits.yaml",
]
hashes = {str(p.relative_to(ROOT)): sha256(p) for p in tracked}
for filename in ["station.yaml", "splits.yaml"]:
    shutil.copy2(ROOT / "configs" / filename, OUT / filename)
shutil.copy2(DATA / "information_policy.json", OUT / "information_policy.json")

interface_note = """# Stage 3.1 environment and adapter interfaces

- AggregateEnvironment owns full events; the controller receives view() copies.
- view() contains current UTC time, capacity and already-connected vehicles.
- Visible vehicle fields: id, station_id, arrival, cohort, released target_kwh,
  expected_departure, delivered_kwh, pmax_kw, remaining_kwh,
  irreversible_excess_kwh. It contains no future disconnectTime or arrivals.
- step({session_id: kW}) executes one 15-minute control interval.
- Arrival rounds up for control eligibility; visible expected departure rounds
  down. Expired estimates use one-slot extension in the diagnostic controller.
- Same-time event order: departure, arrival, input update; last input_order wins.
- Main requests freeze at arrival. Supplementary inputs release at modifiedAt.
- The executor receives only current visible state and requested segment power.
- ACN-Sim network feasibility is checked before update_pilots.
- Actual current feedback drives grid-side delivered energy; efficiency is 1.
- Ledger and completed-session outputs belong to the evaluator, not controller.
- ACN-Sim components execute devices; the project environment advances events.
- Network validation covers the aggregate experimental capacity scenario only.
- Artificial battery reservoirs provide bookkeeping, not inferred SOC/capacity.
- Diagnostic replay is not a formal method comparison or normalization baseline.
"""
(OUT / "environment_interfaces.md").write_text(interface_note, encoding="utf-8")

result = {
    "status": "passed", "check_count": len(passed), "checks": passed,
    "development_replays": summaries, "source_and_input_sha256": hashes,
    "scope": "stage 3.1 correctness checks; no formal algorithm comparison",
    "original_caltech_network_validated": False,
}
(OUT / "verification.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(
    {k: v for k, v in result.items() if k != "source_and_input_sha256"},
    ensure_ascii=False, indent=2))
print("Hash manifest saved in verification.json")
print("Interface note:", OUT / "environment_interfaces.md")
