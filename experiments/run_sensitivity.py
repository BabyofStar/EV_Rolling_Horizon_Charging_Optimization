from pathlib import Path
from bisect import bisect_right
from time import perf_counter
import hashlib
import json
import shutil
import argparse
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
from src.rolling_horizon import (
    _invoke, _first_command, RollingHorizonController,
    record_realized_peaks,
)
from src.tariff import attach_tariff

parser = argparse.ArgumentParser()
parser.add_argument("output_directory")
parser.add_argument("--split", choices=["test"], default="test")
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--capacity-kw", type=float, default=50.0)
parser.add_argument("--base-hours", type=float, default=24.0)
parser.add_argument("--static-time-limit", type=float, default=120.0)
parser.add_argument("--frozen-config-sha256", required=True)
parser.add_argument("--design", required=True)
parser.add_argument("--design-sha256", required=True)
parser.add_argument("--scenario-id", required=True)
args = parser.parse_args()
if not 0 <= args.alpha <= 1:
    parser.error("alpha must be in [0, 1]")
if not all(np.isfinite(v) and v > 0 for v in [
    args.capacity_kw, args.base_hours, args.static_time_limit
]):
    parser.error("capacity, horizon and static budget must be finite and positive")
OUT = Path(args.output_directory).resolve()
if not OUT.is_dir():
    parser.error("output directory must already exist")

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


frozen_path = ROOT / "configs/final.yaml"
assert digest(frozen_path) == args.frozen_config_sha256, (
    "Frozen configuration hash mismatch"
)
frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
assert frozen["status"] == "frozen"
assert frozen["scope"] == "E1_main_test"
assert args.split == "test"
design_path = Path(args.design)
assert digest(design_path) == args.design_sha256
design = yaml.safe_load(design_path.read_text(encoding="utf-8"))
assert design["status"] == "frozen"
assert design["e1_config_sha256"] == args.frozen_config_sha256
matches = [s for s in design["scenarios"] if s["id"] == args.scenario_id]
assert len(matches) == 1
scenario = matches[0]
for key in ["alpha", "capacity_kw", "base_hours", "static_time_limit"]:
    assert getattr(args, key) == scenario[key], (
        "Argument differs from registered scenario: " + key
    )
for rel, expected in design["source_and_input_sha256"].items():
    assert digest(ROOT / rel) == expected, (
        "Sensitivity source/input changed: " + rel
    )
for rel, expected in frozen["source_and_input_sha256"].items():
    assert digest(ROOT / rel) == expected, (
        "Frozen source/input changed: " + rel
    )

station, splits, norm = (
    config("station.yaml"), config("splits.yaml"), config("normalization.yaml"))
norm_hash = digest(ROOT / "configs/normalization.yaml")
TZ = station["timezone"]
capacity = args.capacity_kw
pmax = float(station["evse_max_power_kW"])
start = pd.Timestamp(splits["splits"][args.split]["start_utc_inclusive"])
dev_end = pd.Timestamp(splits["splits"][args.split]["end_utc_exclusive"])
arrival_end = dev_end

check("frozen scope and references",
      station["network_model"] == "aggregate_continuous" and
      station["control_interval_minutes"] == 15 and
      station["charging_efficiency"] == 1 and
      not station["original_branch_constraints_enforced"] and
      abs(norm["C_ref"] - 2.88) < 1e-12 and
      abs(norm["P_ref"] - 6.656) < 1e-12)


input_paths = [
    Path(__file__), ROOT / "requirements.txt",
    ROOT / "configs/final.yaml", Path(args.design),
    *sorted((ROOT / "src").glob("*.py")),
    *[ROOT / "configs" / name for name in [
        "station.yaml", "splits.yaml", "tariff.yaml", "normalization.yaml"
    ]],
    *[DATA / name for name in [
        "sessions_with_split.csv", "environment_events.csv",
        "input_release_events.csv", "user_updates.csv"
    ]],
    ROOT / "data/external/tariff.csv",
    ROOT / "report/stage3_acceptance.md",
    ROOT / "report/stage3_artifact_index.csv",
]
input_before = {
    str(p.relative_to(ROOT)): digest(p) for p in input_paths
}
(OUT / "run_config.json").write_text(
    json.dumps({
        **vars(args), "run_id": OUT.name,
        "scope": "prespecified E2/E3 sensitivity; not parameter selection",
        "source_and_input_sha256": input_before,
    }, ensure_ascii=False, indent=2), encoding="utf-8"
)

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
sessions["connectionTime"] = pd.to_datetime(sessions["connectionTime"], utc=True, format="ISO8601")
events = pd.read_csv(DATA / "environment_events.csv")
events["event_time"] = pd.to_datetime(events["event_time"], utc=True, format="ISO8601")
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
    ("main_at_arrival", 39), ("supplement_after_submission", 468),
]:
    selected = sessions[
        (sessions["split"] == args.split) &
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
    check(cohort + ": shared tail inside selected split", end <= dev_end)
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
            u["modifiedAt"] = pd.to_datetime(u["modifiedAt"], utc=True, format="ISO8601")
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
                "dt": .25, "alpha": args.alpha,
                "C_ref": norm["C_ref"], "P_ref": norm["P_ref"],
                "epsilon_U": 1e-5, "time_limit_seconds": args.static_time_limit,
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
            static_report = _invoke(case, static_dir, args.static_time_limit)
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
        controller = (
            RollingHorizonController(
                folder / "rh_decisions", price_at,
                C_ref=norm["C_ref"], P_ref=norm["P_ref"],
                alpha=args.alpha, base_hours=args.base_hours,
                epsilon_U=1e-5, time_limit_seconds=30, timezone=TZ,
            ) if method == "RH_LP" else None
        )
        peaks = {}
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
                    result = tou_greedy(
                        state, price_at, base_hours=args.base_hours,
                        details=True)
                    command = result["command"]
                    current = tou_bundle(result)
                    diagnostic = {
                        "forecast_unmet_kwh": result["forecast_unmet_kwh"],
                        "effective_horizon_hours": result["effective_horizon_hours"],
                        "sparse_plan": [
                            [sid, label, power]
                            for (sid, label), power in current[1].items()],
                    }
                elif method == "RH_LP":
                    decision = controller.decide(state, dict(peaks))
                    command = decision["command"]
                    diagnostic = decision
                    if decision["mode"] == "lp":
                        step = Path(decision["step_directory"])
                        case = json.loads(
                            (step / "case.json").read_text(encoding="utf-8")
                        )
                        visible = {
                            v["id"]: v for v in state["vehicles"]
                            if v["target_kwh"] is not None
                            and v["remaining_kwh"] > 1e-9
                            and v["arrival"].ceil("15min") <= environment.now
                        }
                        assert {c["id"] for c in case["sessions"]} == set(visible)
                        for car in case["sessions"]:
                            assert abs(
                                car["demand"] -
                                visible[car["id"]]["remaining_kwh"]
                            ) <= 1e-8
                        today = str(environment.now.tz_convert(TZ).date())
                        for day, value in case["past_peak"].items():
                            expected_peak = (
                                peaks.get(day, 0.0) if day == today else 0.0
                            )
                            assert abs(value - expected_peak) <= 1e-8
                        assert decision["solve_report"]["peak_by_day_kW"][
                            today
                        ] >= peaks.get(today, 0.0) - 1e-6
                        current = csv_bundle(step / "lp/plan.csv")
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
                    "mode": (
                        decision["mode"] if method == "RH_LP" else
                        (("static_replay" if method == "Static_LP"
                          else "rule") if active else "empty")
                    ),
                    "total_seconds": perf_counter() - tick,
                    "fallback": (
                        decision["fallback"] if method == "RH_LP" else False
                    ),
                    "reason": (
                        decision["reason"] if method == "RH_LP" else None
                    ),
                    "step_directory": (
                        decision["step_directory"] if method == "RH_LP" else None
                    ),
                    "effective_horizon_hours": (
                        decision["effective_horizon_hours"]
                        if method == "RH_LP" else
                        diagnostic.get("effective_horizon_hours")
                    ),
                    "two_stage_solver_seconds": (
                        sum(s["seconds"] for s in
                            decision["solve_report"]["stages"])
                        if method == "RH_LP" and decision["solve_report"]
                        else None
                    ),
                    "plan_change_kW_sum": (
                        0.0 if method == "Static_LP" else change(previous, current)),
                })
                previous = current
                ledger_index = len(environment.ledger)
                environment.step(command)
                if method == "RH_LP":
                    record_realized_peaks(
                        peaks, environment.ledger[ledger_index:],
                        environment.now, TZ,
                    )
                if len(rows) % 96 == 0:
                    progress = {
                        "cohort": cohort, "method": method,
                        "time": str(environment.now),
                        "steps": len(rows),
                        "fallbacks": sum(bool(r["fallback"]) for r in rows),
                    }
                    (OUT / "progress.json").write_text(
                        json.dumps(progress, indent=2), encoding="utf-8"
                    )
                    print(json.dumps(progress), flush=True)
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
        if method == "Static_LP":
            check(cohort + ": Static planned and actual shortfall agree",
                  abs(summary["unmet_kwh"] - static_report["U_stage2_kWh"]) <= 1e-5)

        timing = summarize_decisions(decisions)
        check(f"{cohort}/{method}: processing below control interval",
              decisions["total_seconds"].max() < 900)
        changes = pd.to_numeric(decisions["plan_change_kW_sum"], errors="coerce").dropna()
        summary.update(
            cohort=cohort, method=method,
            information=("offline_full_information_oracle"
                         if method == "Static_LP" else "online_visible_inputs"),
            split=args.split, run_id=OUT.name,
            scenario_id=args.scenario_id,
            experiment=scenario["experiment"],
            alpha=args.alpha, capacity_kw=capacity,
            base_horizon_hours=args.base_hours,
            rh_reused=False,
            offline_solve_wall_seconds=offline_seconds,
            plan_change_mean_kW_sum=float(changes.mean()) if len(changes) else None,
            plan_change_pairs=int(len(changes)),
            requires_review=bool(
                timing["fallback_count"] > 0 or
                (method == "RH_LP" and
                 timing["active_P95_seconds"] > 30)
            ),
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

check("all tracked source and input files unchanged",
      all(digest(ROOT / rel) == value
          for rel, value in input_before.items()))
check("normalization unchanged",
      digest(ROOT / "configs/normalization.yaml") == norm_hash)
check("all eight method/cohort combinations completed", len(comparisons) == 8)
table = pd.DataFrame(comparisons)
table.to_csv(OUT / "comparison.csv", index=False)
table[[
    "run_id", "split", "cohort", "method", "alpha",
    "capacity_kw", "base_horizon_hours", "requires_review"
]].to_csv(OUT / "run_manifest.csv", index=False)

policy = """# Full validation evaluation

All four methods use the full test split and fresh execution.
Cohorts are evaluated separately. Execution uses the frozen sensitivity design and E1 reference.
References and source inputs are hashed before and after execution.

Static LP is an offline full-information oracle using true connections
and post-hoc evaluation targets. Online methods receive released inputs.
Static complete-slot service is not automatically a lower bound on
online service with intra-slot departures.

Fees, service and actual piecewise peaks are reported jointly.
Fallbacks are retained and flagged for review.
Original Caltech branch and phase constraints are not enforced.
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
    ROOT / "report/stage3_acceptance.md",
]
result = {
    "status": "passed", "check_count": len(passed), "checks": passed,
    "rh_reused": False, "split": args.split,
    "run_id": OUT.name,
    "requires_review": any(r["requires_review"] for r in comparisons),
    "scope": "prespecified E2/E3 sensitivity; not parameter selection",
    "comparison": comparisons,
    "source_and_input_sha256": input_before,
    "output_artifact_sha256": artifact_hashes,
    "wall_seconds": perf_counter() - started,
}
(OUT / "verification.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "status": result["status"], "check_count": len(passed), "checks": passed,
    "rh_reused": False,
    "requires_review": result["requires_review"],
    "wall_seconds": result["wall_seconds"],
}, ensure_ascii=False, indent=2))
columns = [
    "cohort", "method", "cost_USD", "delivered_kwh", "unmet_kwh",
    "energy_satisfaction", "max_peak_kW", "uncompleted_sessions",
    "expected_checkpoint_evaluable_sessions", "expected_departure_uncompleted_rate",
    "fallback_count", "active_P95_seconds", "offline_solve_wall_seconds",
    "plan_change_mean_kW_sum",
]
print("\n=== Prespecified sensitivity comparison ===")
print(table[columns].to_string(index=False))
print("\nFull results:", OUT / "verification.json")
print("Policy:", OUT / "integration_policy.md")
