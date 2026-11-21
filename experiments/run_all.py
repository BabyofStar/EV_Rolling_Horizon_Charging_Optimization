from pathlib import Path
from bisect import bisect_right
from time import perf_counter
import hashlib
import json
import shutil
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines import uncontrolled, tou_greedy
from src.simulator import AggregateEnvironment
from src.acnsim_adapter import ACNSimExecutor
from src.metrics import evaluate_run, summarize_decisions
from src.rolling_horizon import _invoke, _first_command
from src.tariff import attach_tariff

OUT = Path(sys.argv[1])
DATA = ROOT / "data/processed/caltech_12weeks_20261111T143402Z_9b087d"
RH = ROOT / "results/logs/stage3_rolling_replay_20261112T082426223430Z"
passed, comparisons, artifact_hashes = [], [], {}
started = perf_counter()

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    passed.append(name)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def config(name):
    return yaml.safe_load(
        (ROOT / "configs" / name).read_text(encoding="utf-8"))

station, splits, norm = (
    config("station.yaml"), config("splits.yaml"), config("normalization.yaml"))
norm_hash = digest(ROOT / "configs/normalization.yaml")
TZ = station["timezone"]
capacity = float(station["station_capacity_kW"])
pmax = float(station["evse_max_power_kW"])
start = pd.Timestamp(splits["splits"]["development"]["start_utc_inclusive"])
dev_end = pd.Timestamp(splits["splits"]["development"]["end_utc_exclusive"])
arrival_end = start + pd.Timedelta(days=7)

old = json.loads((RH / "verification.json").read_text(encoding="utf-8"))
check("accepted RH evidence", old["status"] == "passed" and not old["requires_review"])
for relative, expected in old["source_and_input_sha256"].items():
    check("RH source/input unchanged: " + relative,
          digest(ROOT / relative) == expected)
check("frozen scope and references",
      station["network_model"] == "aggregate_continuous" and
      station["control_interval_minutes"] == 15 and
      station["charging_efficiency"] == 1 and
      not station["original_branch_constraints_enforced"] and
      abs(norm["C_ref"] - 2.88) < 1e-12 and
      abs(norm["P_ref"] - 6.656) < 1e-12)

grid = pd.date_range(start, dev_end + pd.Timedelta(days=2),
                     freq="15min", inclusive="left")
priced = attach_tariff(pd.DataFrame({"start_utc": grid}))
price_cache = dict(zip(grid.asi8, priced["price_USD_per_kWh"]))
def price_at(t):
    t = pd.Timestamp(t)
    if t.value not in price_cache:
        price_cache[t.value] = float(attach_tariff(
            pd.DataFrame({"start_utc": [t]}))["price_USD_per_kWh"].iloc[0])
    return float(price_cache[t.value])

sessions = pd.read_csv(DATA / "sessions_with_split.csv")
sessions["connectionTime"] = pd.to_datetime(sessions["connectionTime"], utc=True)
events = pd.read_csv(DATA / "environment_events.csv")
events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
releases = pd.read_csv(DATA / "input_release_events.csv")
updates = pd.read_csv(DATA / "user_updates.csv")

def valid(value):
    return str(value).strip().lower() in {"true", "1", "1.0"}

def change(previous, current):
    if previous is None or current is None:
        return None
    old_slots, old_power = previous
    new_slots, new_power = current
    common = old_slots & new_slots
    if not common:
        return None
    keys = {key for key in old_power.keys() | new_power.keys()
            if key[1] in common}
    return sum(abs(new_power.get(key, 0) - old_power.get(key, 0)) for key in keys)

def tou_bundle(result):
    labels = [t.isoformat() for t in result["slots"]]
    powers = {
        (sid, labels[j]): float(power)
        for sid, values in result["plan_kw"].items()
        for j, power in enumerate(values) if power > 0
    }
    return set(labels), powers

def csv_bundle(path):
    frame = pd.read_csv(path, dtype={"sessionID": str, "slot": str})
    powers = {
        (row.sessionID, row.slot): float(row.power_kW)
        for row in frame.itertuples(index=False) if row.power_kW != 0
    }
    return set(frame["slot"]), powers

def execution_frames(environment):
    station_rows, vehicle_rows = [], []
    for row in environment.ledger:
        station_rows.append({
            "start": row["start"], "end": row["end"],
            "total_kw": row["total_kw"],
            "energy_kwh": sum(row["energy_kwh"].values()),
        })
        for sid, energy in row["energy_kwh"].items():
            vehicle_rows.append({
                "sessionID": sid, "start": row["start"], "end": row["end"],
                "actual_kw": row["power_kw"][sid], "energy_kwh": energy,
            })
    return pd.DataFrame(station_rows), pd.DataFrame(
        vehicle_rows,
        columns=["sessionID", "start", "end", "actual_kw", "energy_kwh"])

for cohort, expected_count in [
    ("main_at_arrival", 5), ("supplement_after_submission", 198),
]:
    selected = sessions[
        (sessions["split"] == "development") &
        (sessions["cohort"] == cohort) &
        (sessions["connectionTime"] >= start) &
        (sessions["connectionTime"] < arrival_end)
    ].copy()
    check(cohort + ": fixed shared selection", len(selected) == expected_count)
    ids = set(selected["sessionID"])
    ev = events[events["sessionID"].isin(ids)].copy()
    rel = releases[releases["sessionID"].isin(ids)].copy()
    departures = dict(ev.loc[
        ev["event_type"] == "departure",
        ["sessionID", "event_time"]].itertuples(index=False, name=None))
    arrivals = dict(zip(selected["sessionID"], selected["connectionTime"]))
    end = max(arrival_end, max(departures.values()).ceil("15min"))
    check(cohort + ": shared tail inside development", end <= dev_end)
    prior_summary = next(r for r in old["replays"] if r["cohort"] == cohort)
    check(cohort + ": RH selection and tail match",
          prior_summary["sessions"] == expected_count and
          pd.Timestamp(prior_summary["execution_end_utc"]) == end)

    # One evaluator-only table is shared by all four methods.
    evaluation_rows, histories = [], {}
    for row in selected.to_dict("records"):
        sid = row["sessionID"]
        if cohort == "main_at_arrival":
            times = [arrivals[sid].value]
            targets = [float(row["initial_kWhRequested"])]
            deadline = pd.Timestamp(row["initial_requestedDeparture"])
        else:
            u = updates[updates["sessionID"] == sid].copy()
            u = u[u["valid"].map(valid)]
            u["modifiedAt"] = pd.to_datetime(u["modifiedAt"], utc=True)
            u = u[(u["modifiedAt"] >= arrivals[sid]) &
                  (u["modifiedAt"] < departures[sid])]
            u = u.sort_values(["modifiedAt", "input_order"])
            assert len(u)
            times = [t.value for t in u["modifiedAt"]]
            targets = u["kWhRequested"].astype(float).tolist()
            deadline = pd.Timestamp(u.iloc[0]["requestedDeparture"])
        histories[sid] = (times, targets)
        checkpoint = bisect_right(times, deadline.value) - 1
        evaluation_rows.append({
            "sessionID": sid, "connectionTime": arrivals[sid],
            "disconnectTime": departures[sid],
            "evaluation_target_kwh": targets[-1], "pmax_kw": pmax,
            "first_input_time": pd.Timestamp(times[0], tz="UTC"),
            "first_expected_departure": deadline,
            "expected_checkpoint_target_kwh": (
                targets[checkpoint] if checkpoint >= 0 else None),
        })
    evaluation = pd.DataFrame(evaluation_rows)
    cohort_dir = OUT / cohort
    cohort_dir.mkdir()
    evaluation.to_csv(cohort_dir / "shared_evaluation_sessions.csv", index=False)
    targets_by_id = evaluation.set_index("sessionID")["evaluation_target_kwh"]

    for method in ["Uncontrolled", "TOU_Greedy", "Static_LP", "RH_LP"]:
        print(f"Integrating {cohort}/{method}", flush=True)
        folder = cohort_dir / method
        folder.mkdir()
        offline_seconds, static_report = None, None

        if method == "RH_LP":
            # Reuse accepted execution; audit it again with the common evaluator.
            for filename in [
                "station_trajectory.csv", "session_trajectory.csv",
                "completed_sessions.csv", "decision_summary.csv",
            ]:
                shutil.copy2(RH / cohort / filename, folder / filename)
            station_frame = pd.read_csv(folder / "station_trajectory.csv")
            vehicle_frame = pd.read_csv(
                folder / "session_trajectory.csv", dtype={"sessionID": str})
            decisions = pd.read_csv(folder / "decision_summary.csv")
            previous, changes = None, []
            for row in decisions.to_dict("records"):
                current = (
                    csv_bundle(Path(row["step_directory"]) / "lp/plan.csv")
                    if row["mode"] == "lp" else None)
                changes.append(change(previous, current))
                previous = current
            decisions["plan_change_kW_sum"] = changes
        else:
            runtime_sessions = selected.copy()
            plan_by_time = {}
            if method == "Static_LP":
                # Explicit full-information oracle initialization.
                runtime_sessions["cohort"] = "main_at_arrival"
                runtime_sessions["initial_kWhRequested"] = (
                    runtime_sessions["sessionID"].map(targets_by_id))
                runtime_sessions["initial_requestedDeparture"] = (
                    runtime_sessions["sessionID"].map(departures))
                horizon = tuple(pd.date_range(
                    start, end, freq="15min", inclusive="left"))
                case = {
                    "dt": .25, "alpha": .5,
                    "C_ref": norm["C_ref"], "P_ref": norm["P_ref"],
                    "epsilon_U": 1e-5, "time_limit_seconds": 120,
                    "past_peak": {},
                    "slots": [
                        {"label": t.isoformat(),
                         "day": str(t.tz_convert(TZ).date()),
                         "price": price_at(t), "capacity": capacity}
                        for t in horizon
                    ],
                    "sessions": [
                        {"id": sid, "demand": float(targets_by_id[sid]),
                         "pmax": pmax, "available": list(range(
                             int(pd.Index(horizon).searchsorted(
                                 arrivals[sid].ceil("15min"))),
                             int(pd.Index(horizon).searchsorted(
                                 departures[sid].floor("15min")))))}
                        for sid in sorted(ids)
                    ],
                }
                static_dir = folder / "static_solve"
                static_dir.mkdir()
                tick = perf_counter()
                static_report = _invoke(case, static_dir, 120)
                _first_command(case, static_dir / "lp/plan.csv")
                plan = pd.read_csv(
                    static_dir / "lp/plan.csv",
                    dtype={"sessionID": str, "slot": str})
                for row in plan.itertuples(index=False):
                    if row.power_kW > 0:
                        plan_by_time.setdefault(row.slot, {})[
                            row.sessionID] = float(row.power_kW)
                offline_seconds = perf_counter() - tick

            executor = ACNSimExecutor(
                selected["stationID"].astype(str).tolist(),
                capacity_kw=capacity, voltage=float(station["evse_voltage_V"]),
                max_current=float(station["evse_max_current_A"]))
            environment = AggregateEnvironment(
                runtime_sessions, ev,
                pd.DataFrame() if method == "Static_LP" else rel,
                start=start, capacity_kw=capacity, pmax_kw=pmax,
                executor=executor)
            rows, previous = [], None
            with (folder / "requests_and_plans.jsonl").open(
                "w", encoding="utf-8"
            ) as journal:
                while environment.now < end:
                    tick = perf_counter()
                    state = environment.view()
                    current, diagnostic = None, {}
                    if method == "Uncontrolled":
                        diagnostic = uncontrolled(state, details=True)
                        command = diagnostic["command"]
                    elif method == "TOU_Greedy":
                        result = tou_greedy(state, price_at, details=True)
                        command = result["command"]
                        current = tou_bundle(result)
                        diagnostic = {
                            "forecast_unmet_kwh": result["forecast_unmet_kwh"],
                            "effective_horizon_hours": result["effective_horizon_hours"],
                            "sparse_plan": [
                                [sid, label, power]
                                for (sid, label), power in current[1].items()],
                        }
                    else:
                        raw = plan_by_time.get(environment.now.isoformat(), {})
                        active_ids = {v["id"] for v in state["vehicles"]}
                        assert set(raw).issubset(active_ids)
                        command, left = {}, capacity
                        for v in sorted(state["vehicles"], key=lambda v: v["id"]):
                            raw_power = raw.get(v["id"], 0.0)
                            power = min(raw_power, v["pmax_kw"],
                                        v["remaining_kwh"] / .25, left)
                            assert abs(power - raw_power) <= 1e-6
                            if power > 0:
                                command[v["id"]] = power
                                left = max(left - power, 0)
                    active = any(
                        v["remaining_kwh"] is not None and v["remaining_kwh"] > 1e-9
                        for v in state["vehicles"])
                    journal.write(json.dumps({
                        "time": str(environment.now), "command": command,
                        "diagnostic": diagnostic,
                    }, default=str) + "\n")
                    rows.append({
                        "time": environment.now,
                        "mode": ("static_replay" if method == "Static_LP"
                                 else "rule") if active else "empty",
                        "total_seconds": perf_counter() - tick,
                        "fallback": False,
                        "plan_change_kW_sum": (
                            0.0 if method == "Static_LP" else change(previous, current)),
                    })
                    previous = current
                    environment.step(command)
            check(f"{cohort}/{method}: complete shared-environment replay",
                  len(environment.completed) == expected_count and
                  not environment.view()["vehicles"] and not executor._connected)
            station_frame, vehicle_frame = execution_frames(environment)
            decisions = pd.DataFrame(rows)
            station_frame.to_csv(folder / "station_trajectory.csv", index=False)
            vehicle_frame.to_csv(folder / "session_trajectory.csv", index=False)
            pd.DataFrame(environment.completed).to_csv(
                folder / "completed_sessions.csv", index=False)

        # All online traces must obey the request history at the time of delivery.
        if method != "Static_LP":
            cumulative = {sid: 0.0 for sid in ids}
            trace = vehicle_frame.copy()
            trace["start"] = pd.to_datetime(trace["start"], utc=True, format="ISO8601", errors="raise")
            trace = trace.sort_values(["start", "sessionID"])
            for row in trace.to_dict("records"):
                sid, amount = row["sessionID"], float(row["energy_kwh"])
                if amount <= 0:
                    continue
                times, targets = histories[sid]
                j = bisect_right(times, row["start"].value) - 1
                assert j >= 0
                assert row["start"] >= pd.Timestamp(times[0], tz="UTC").ceil("15min")
                before = cumulative[sid]
                cumulative[sid] += amount
                assert cumulative[sid] <= max(before, targets[j]) + 1e-5
            check(f"{cohort}/{method}: online information audit", True)

        summary, daily, service = evaluate_run(
            station_frame, vehicle_frame, evaluation, start, end,
            price_at, capacity_kw=capacity, timezone=TZ)
        check(f"{cohort}/{method}: unified physical and energy checks",
              summary["max_power_violation_kW"] <= 1e-6 and
              summary["max_energy_residual_kwh"] <= 1e-5 and
              summary["connection_violation_segments"] == 0)
        check(f"{cohort}/{method}: shared target and session accounting",
              summary["sessions"] == expected_count and
              set(service["sessionID"]) == ids)
        if method == "RH_LP":
            check(cohort + ": reused RH energy reconciles with accepted report",
                  abs(summary["delivered_kwh"] -
                      prior_summary["diagnostic_delivered_kwh"]) <= 1e-5)
        if method == "Static_LP":
            check(cohort + ": Static planned and actual shortfall agree",
                  abs(summary["unmet_kwh"] - static_report["U_stage2_kWh"]) <= 1e-5)

        timing = summarize_decisions(decisions)
        check(f"{cohort}/{method}: no unexpected fallback",
              timing["fallback_count"] == 0)
        changes = pd.to_numeric(decisions["plan_change_kW_sum"], errors="coerce").dropna()
        summary.update(
            cohort=cohort, method=method,
            information=("offline_full_information_oracle"
                         if method == "Static_LP" else "online_visible_inputs"),
            rh_reused=(method == "RH_LP"),
            offline_solve_wall_seconds=offline_seconds,
            plan_change_mean_kW_sum=float(changes.mean()) if len(changes) else None,
            plan_change_pairs=int(len(changes)),
            constraint_violation_duration_hours=0.0,
            **timing)
        comparisons.append(summary)
        daily.to_csv(folder / "daily_metrics.csv", index=False)
        service.to_csv(folder / "session_service.csv", index=False)
        decisions.to_csv(folder / "decision_summary.csv", index=False)
        (folder / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        for filename in [
            "station_trajectory.csv", "session_trajectory.csv",
            "session_service.csv", "daily_metrics.csv", "decision_summary.csv",
        ]:
            path = folder / filename
            artifact_hashes[str(path.relative_to(OUT))] = digest(path)

check("normalization unchanged",
      digest(ROOT / "configs/normalization.yaml") == norm_hash)
check("all eight method/cohort combinations completed", len(comparisons) == 8)
table = pd.DataFrame(comparisons)
table.to_csv(OUT / "comparison.csv", index=False)

policy = """# Stage 3 integrated development validation

Evaluation uses the same selected first-week arrivals and full departure tail
for all four methods, with separate cohorts and initial delivered energy zero.

Main target: frozen arrival request.
Supplement target: last valid request released during connection, before departure.
Same-time input conflicts: ascending input_order, last wins; departure-time
updates do not apply.

Expected checkpoint: first valid estimated departure; target is the request
effective at that checkpoint. A checkpoint before first input is not evaluable
and is excluded with its count reported. The main cohort remains frozen.

Static LP is explicitly an offline full-information oracle. Its replay is
initialized with the post-hoc target at arrival and ignores submission updates.
Online environments retain the original input-release policy. All use the
same event engine, aggregate capacity and ACN-Sim device executor.

Static LP uses ceil(real arrival) to floor(real departure), with complete slots.
Online execution can be interrupted by a real departure inside a slot.
The static planned optimum is therefore not automatically a lower bound on
online actual service shortfall. No such bound or formal savings claim is made.

Peak_kW is the maximum actual piecewise power. Grid-average peak is reported
separately. Zero-load time is retained. Fees exclude the optimization peak term.

RH trajectories are reused after source/input hash checks and independent
trajectory audits. Their original controller times are preserved.
Static offline solve time (budget 120 s) is separate from replay decision time.
Uncontrolled has no future plan; its plan-change diagnostic is NA.
Static's fixed future plan has zero revision. TOU and RH revisions compare
common future slots, with absent sparse entries interpreted as zero.
"""
(OUT / "integration_policy.md").write_text(policy, encoding="utf-8")
tracked = [
    Path(__file__), ROOT / "src/metrics.py",
    ROOT / "src/baselines.py", ROOT / "src/rolling_horizon.py",
    ROOT / "src/optimization_model.py", ROOT / "src/simulator.py",
    ROOT / "src/acnsim_adapter.py", ROOT / "src/tariff.py",
    ROOT / "configs/normalization.yaml", ROOT / "configs/station.yaml",
    ROOT / "configs/splits.yaml", ROOT / "configs/tariff.yaml",
    DATA / "sessions_with_split.csv", DATA / "environment_events.csv",
    DATA / "input_release_events.csv", DATA / "user_updates.csv",
    RH / "verification.json",
]
result = {
    "status": "passed", "check_count": len(passed), "checks": passed,
    "rh_reused": True, "rh_source": str(RH),
    "scope": "stage 3 development integration, not formal test results",
    "comparison": comparisons,
    "source_and_input_sha256": {
        str(p.relative_to(ROOT)): digest(p) for p in tracked},
    "output_artifact_sha256": artifact_hashes,
    "wall_seconds": perf_counter() - started,
}
(OUT / "verification.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "status": result["status"], "check_count": len(passed), "checks": passed,
    "rh_reused": True, "wall_seconds": result["wall_seconds"],
}, ensure_ascii=False, indent=2))
columns = [
    "cohort", "method", "cost_USD", "delivered_kwh", "unmet_kwh",
    "energy_satisfaction", "max_peak_kW", "uncompleted_sessions",
    "expected_checkpoint_evaluable_sessions", "expected_departure_uncompleted_rate",
    "fallback_count", "active_P95_seconds", "offline_solve_wall_seconds",
    "plan_change_mean_kW_sum",
]
print("\n=== Development integration comparison ===")
print(table[columns].to_string(index=False))
print("\nFull results:", OUT / "verification.json")
print("Policy:", OUT / "integration_policy.md")
