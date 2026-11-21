"""Extract existing nominal/ideal E6 evidence around the first divergence."""

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


TZ = "America/Los_Angeles"
DELTA = pd.Timedelta(minutes=15)
COHORT = "RH_LP/supplement_after_submission"
CASES = ("nominal", "ideal")
FILES = ("decision.json", "case.json", "lp/report.json", "lp/plan.csv")


def timestamps(series):
    return pd.to_datetime(series, utc=True, format="ISO8601")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_decisions(folder):
    rows = pd.read_csv(folder / "decisions.csv")
    rows["time"] = timestamps(rows["time"])
    rows["step"] = range(1, len(rows) + 1)
    if rows["time"].duplicated().any() or not rows["time"].is_monotonic_increasing:
        raise ValueError(f"Unexpected decision ordering at {folder}")
    return dict(zip(rows["time"], rows["step"]))


def read_trace(folder, name):
    types = {"sessionID": str} if name == "session_trajectory" else None
    data = pd.read_csv(folder / f"{name}.csv", dtype=types)
    data["start"] = timestamps(data["start"])
    data["end"] = timestamps(data["end"])
    data["time"] = data["start"].dt.floor(DELTA)
    if not ((data["end"] > data["start"]) & (data["end"] <= data["time"] + DELTA)).all():
        raise ValueError(f"Found an invalid or cross-step segment in {folder / name}")
    return data


def search_differences(nominal, ideal, path="", limit=40):
    """Report exact input differences, including tolerance-scale float changes."""
    differences = []

    def walk(left, right, key):
        if len(differences) >= limit:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for name in sorted(left.keys() | right.keys()):
                if not key and name == "time_limit_seconds":
                    continue  # The frozen controller computes this from wall-clock time.
                walk(left.get(name), right.get(name), f"{key}.{name}" if key else name)
        elif isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
            for index, (a, b) in enumerate(zip(left, right)):
                walk(a, b, f"{key}[{index}]")
        elif left != right:
            differences.append({"field": key, "nominal": left, "ideal": right})

    walk(nominal, ideal, path)
    return differences


def run_review(run, output):
    verification = read_json(run / "verification.json")
    if verification.get("mode") != "formal" or verification.get("status") != "passed":
        raise ValueError("Expected the completed formal E6 result directory")
    folders = {name: run / name / COHORT for name in CASES}
    steps = {name: read_decisions(folder) for name, folder in folders.items()}
    if list(steps["nominal"]) != list(steps["ideal"]):
        raise ValueError("Nominal and ideal decision clocks differ")
    station = {name: read_trace(folder, "station_trajectory") for name, folder in folders.items()}
    vehicle = {name: read_trace(folder, "session_trajectory") for name, folder in folders.items()}
    totals = pd.concat({name: frame.groupby("time")["energy_kwh"].sum()
                        for name, frame in station.items()}, axis=1)
    if totals.isna().any().any() or set(totals.index) != set(steps["nominal"]):
        raise ValueError("Station evidence does not cover the decision clock")
    totals["ideal_minus_nominal_kwh"] = totals["ideal"] - totals["nominal"]
    changed = totals[totals["ideal_minus_nominal_kwh"].abs() > 1e-5]
    if changed.empty:
        raise ValueError("No meaningful station energy difference in this result directory")
    anchor = changed.index[0]
    # Include the whole first affected local day for detecting earlier per-car
    # discrepancies hidden by matching station totals, plus one hour after it.
    begin = anchor.tz_convert(TZ).normalize().tz_convert("UTC")
    scan = [stamp for stamp in totals.index if begin <= stamp <= anchor + 4 * DELTA]
    trace = {name: frame.groupby(["time", "sessionID"])["energy_kwh"].sum()
             for name, frame in vehicle.items()}
    evidence, decision_rows, session_rows, affected = {}, [], [], {}
    for stamp in scan:
        inputs, commands, modes, energies = {}, {}, {}, {}
        for name in CASES:
            step = folders[name] / "decisions" / f"step_{steps[name][stamp]:06d}"
            evidence[name, stamp] = step
            decision = read_json(step / "decision.json")
            if pd.Timestamp(decision["time"]) != stamp:
                raise ValueError(f"Decision file time does not match CSV at {step}")
            commands[name] = decision["command"]
            modes[name] = decision["mode"]
            case = read_json(step / "case.json") if (step / "case.json").is_file() else None
            inputs[name] = {vehicle["id"]: vehicle["demand"] for vehicle in case["sessions"]} if case else {}
            energies[name] = trace[name].xs(stamp).to_dict() if stamp in trace[name].index.levels[0] else {}
        ids = sorted(set().union(*(set(d) for group in (commands, inputs, energies) for d in group.values())))
        max_command = max((abs(commands["ideal"].get(sid, 0) - commands["nominal"].get(sid, 0))
                           for sid in ids), default=0)
        max_energy = max((abs(energies["ideal"].get(sid, 0) - energies["nominal"].get(sid, 0))
                          for sid in ids), default=0)
        decision_rows.append({"time": stamp, "nominal_mode": modes["nominal"], "ideal_mode": modes["ideal"],
                              "max_command_difference_kw": max_command, "max_vehicle_energy_difference_kwh": max_energy,
                              "station_delta_kwh": totals.at[stamp, "ideal_minus_nominal_kwh"]})
        if max_command > 1e-8 or max_energy > 1e-8:
            affected[stamp] = True
        for sid in ids:
            row = {"time": stamp, "sessionID": sid}
            for name in CASES:
                row.update({f"{name}_demand_kwh": inputs[name].get(sid),
                            f"{name}_command_kw": commands[name].get(sid, 0),
                            f"{name}_delivered_kwh": energies[name].get(sid, 0)})
            session_rows.append(row)

    first = min(affected, default=anchor)
    keep = {stamp for stamp in scan if abs(stamp - first) <= 4 * DELTA or abs(stamp - anchor) <= 4 * DELTA}
    differences = []
    for stamp in sorted(keep):
        paths = [evidence[name, stamp] / "case.json" for name in CASES]
        data = [read_json(path) if path.is_file() else None for path in paths]
        differences.append({"time": stamp.isoformat(), "differences": search_differences(*data)})

    summary = {"run": str(run), "first_station_difference_local": anchor.tz_convert(TZ).isoformat(),
               "first_vehicle_or_command_difference_local": first.tz_convert(TZ).isoformat(),
               "station_delta_at_anchor_kwh": float(totals.at[anchor, "ideal_minus_nominal_kwh"]),
               "source_files": list(FILES), "scanned_steps": len(scan), "raw_steps_per_scenario": len(keep)}
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("review_summary.json", json.dumps(summary, indent=2))
        archive.writestr("case_differences.json", json.dumps(differences, indent=2))
        archive.writestr("decision_comparison.csv", pd.DataFrame(decision_rows).to_csv(index=False))
        archive.writestr("vehicle_step_comparison.csv", pd.DataFrame(session_rows).to_csv(index=False))
        for name in CASES:
            for kind, frame in (("station", station[name]), ("vehicle", vehicle[name])):
                archive.writestr(f"{name}/{kind}_segments.csv", frame[frame["time"].isin(keep)].to_csv(index=False))
            for stamp in sorted(keep):
                step = evidence[name, stamp]
                for label in FILES:
                    path = step / label
                    if path.is_file():
                        archive.write(path, f"{name}/{step.name}/{label}")
    print(json.dumps(summary, indent=2))
    print(f"Review archive: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/logs/e6_formal_01"))
    parser.add_argument("--out", type=Path, default=Path("E6_difference_review.zip"))
    args = parser.parse_args()
    run_review(args.run, args.out)


if __name__ == "__main__":
    main()
