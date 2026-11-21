"""开发集严格主队列的完整信息静态 LP 建模验证。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import sys
import traceback

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optimization_model import solve_lp

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
OUT = Path(args.output)
OUT.mkdir(parents=True, exist_ok=False)

DATA = ROOT / "data/processed/caltech_12weeks_20261111T143402Z_9b087d"
summary = {
    "status": "running",
    "scope": "development_main_static_validation",
    "information": "offline_full_information",
    "normalization": {
        "C_ref_USD": 10.0,
        "P_ref_kW": 50.0,
        "purpose": "validation_only_not_formal_experiment",
    },
}

def aware(series, name):
    if series.isna().any():
        raise ValueError(f"{name} 存在缺失时间")
    if not series.map(lambda x: pd.Timestamp(x).tzinfo is not None).all():
        raise ValueError(f"{name} 存在无时区时间")
    return pd.to_datetime(series, utc=True, errors="raise")

def maxabs(values):
    return float(np.max(np.abs(np.asarray(values, dtype=float))))

try:
    files = {
        "sessions": DATA / "sessions_with_split.csv",
        "truth": DATA / "environment_truth.csv",
        "policy": DATA / "information_policy.json",
        "station": ROOT / "configs/station.yaml",
        "splits": ROOT / "configs/splits.yaml",
        "tariff_config": ROOT / "configs/tariff.yaml",
        "tariff": ROOT / "data/external/tariff.csv",
        "model_code": ROOT / "src/optimization_model.py",
        "runner_code": Path(__file__).resolve(),
    }
    manifest = {
        key: {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for key, path in files.items()
    }
    (OUT / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in ["pyomo", "highspy", "pandas", "numpy", "PyYAML"]
    }
    summary["versions"] = versions
    summary["python"] = sys.version
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()

    station = yaml.safe_load(files["station"].read_text(encoding="utf-8"))
    splits = yaml.safe_load(files["splits"].read_text(encoding="utf-8"))
    tariff_config = yaml.safe_load(
        files["tariff_config"].read_text(encoding="utf-8")
    )
    policy = json.loads(files["policy"].read_text(encoding="utf-8"))

    assert station["network_model"] == "aggregate_continuous"
    assert station["original_branch_constraints_enforced"] is False
    assert station["charging_efficiency"] == 1
    assert station["energy_basis"] == "grid_side"
    assert station["control_interval_minutes"] == 15
    assert station["station_capacity_kW"] == 50
    assert abs(station["evse_max_power_kW"] - 6.656) < 1e-12
    assert station["timezone"] == splits["timezone"] == tariff_config["timezone"]
    assert policy["main_at_arrival"] == "freeze_arrival_request_and_departure_estimate"

    sessions = pd.read_csv(files["sessions"], dtype=str)
    truth = pd.read_csv(
        files["truth"],
        usecols=["sessionID", "disconnectTime"],
        dtype=str,
    )
    assert sessions["sessionID"].notna().all()
    assert truth["sessionID"].notna().all()
    assert not sessions["sessionID"].duplicated().any()
    assert not truth["sessionID"].duplicated().any()
    assert set(sessions["sessionID"]).issubset(set(truth["sessionID"]))

    cohort_counts = sessions.groupby(["split", "cohort"]).size()
    summary["cohort_counts"] = {
        f"{a}/{b}": int(n) for (a, b), n in cohort_counts.items()
    }
    print("实际队列规模：", summary["cohort_counts"], flush=True)

    # 使用明确的队列标签；不按最终字段是否存在来猜测队列。
    chosen = sessions.loc[
        (sessions["split"] == "development")
        & (sessions["cohort"] == "main_at_arrival")
    ].copy()
    if len(chosen) != 64:
        raise ValueError(
            "开发集严格主队列与已冻结的 64 条不符，请核对上方实际队列标签和规模"
        )

    chosen = chosen.merge(
        truth, on="sessionID", how="left", validate="one_to_one"
    )
    for column in [
        "connectionTime", "disconnectTime", "initial_requestedDeparture"
    ]:
        chosen[column] = aware(chosen[column], column)
    chosen["target_kWh"] = pd.to_numeric(
        chosen["initial_kWhRequested"], errors="raise"
    )
    assert np.isfinite(chosen["target_kWh"]).all()
    assert (chosen["target_kWh"] > 0).all()

    bounds = splits["splits"]["development"]
    start = pd.Timestamp(bounds["start_utc_inclusive"])
    end = pd.Timestamp(bounds["end_utc_exclusive"])
    assert start.tzinfo is not None and end.tzinfo is not None
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    assert (chosen["connectionTime"] >= start).all()
    assert (chosen["connectionTime"] < end).all()
    assert (chosen["disconnectTime"] <= end).all()
    assert (chosen["disconnectTime"] > chosen["connectionTime"]).all()

    grid = pd.date_range(start, end, freq="15min", inclusive="left")
    assert len(grid) == 4704
    dt = 0.25
    capacity = float(station["station_capacity_kW"])
    pmax = float(station["evse_max_power_kW"])
    labels = [t.isoformat() for t in grid]
    days = grid.tz_convert(station["timezone"]).strftime("%Y-%m-%d").tolist()

    tariff = pd.read_csv(files["tariff"], dtype=str)
    tariff["start_utc"] = aware(tariff["start_utc"], "tariff.start_utc")
    assert not tariff["start_utc"].duplicated().any()
    tariff["price_USD_per_kWh"] = pd.to_numeric(
        tariff["price_USD_per_kWh"], errors="raise"
    )
    prices = tariff.set_index("start_utc")["price_USD_per_kWh"].reindex(grid)
    assert prices.notna().all()
    assert np.isfinite(prices).all() and (prices >= 0).all()

    # 完整信息静态参考：真实到达向上取整，真实离站向下取整。
    chosen = chosen.sort_values(
        ["connectionTime", "sessionID"], kind="stable"
    ).reset_index(drop=True)
    first = chosen["connectionTime"].dt.ceil("15min")
    last = chosen["disconnectTime"].dt.floor("15min")
    chosen["start_index"] = [int(grid.searchsorted(t)) for t in first]
    chosen["end_index"] = [int(grid.searchsorted(t)) for t in last]
    chosen["full_slots"] = (
        chosen["end_index"] - chosen["start_index"]
    ).clip(lower=0)
    chosen["device_only_shortfall_lower_bound_kWh"] = (
        chosen["target_kWh"] - chosen["full_slots"]*dt*pmax
    ).clip(lower=0)
    chosen.to_csv(OUT / "input_sessions.csv", index=False)

    case = {
        "dt": dt, "alpha": 0.5,
        "C_ref": 10.0, "P_ref": 50.0,
        "epsilon_U": 1e-5, "time_limit_seconds": 30,
        "slots": [
            {"label": label, "day": day,
             "price": float(price), "capacity": capacity}
            for label, day, price in zip(labels, days, prices)
        ],
        "sessions": [
            {
                "id": row.sessionID,
                "demand": float(row.target_kWh),
                "pmax": pmax,
                "available": list(range(row.start_index, row.end_index)),
            }
            for row in chosen.itertuples(index=False)
        ],
    }
    result = solve_lp(case, OUT / "lp")

    # ---------- 从导出文件独立核验 ----------
    plan = pd.read_csv(
        OUT / "lp/plan.csv",
        dtype={"sessionID": str, "slot": str, "day": str},
    )
    vehicle = pd.read_csv(
        OUT / "lp/sessions.csv", dtype={"sessionID": str}
    ).set_index("sessionID")
    ids = chosen["sessionID"].tolist()
    assert set(plan["sessionID"]) == set(ids)
    assert set(plan["slot"]) == set(labels)
    assert len(plan) == len(ids)*len(grid)
    assert not plan.duplicated(["sessionID", "slot"]).any()
    assert not vehicle.index.duplicated().any()
    assert set(vehicle.index) == set(ids)

    index_by_label = dict(zip(labels, range(len(grid))))
    day_by_label = dict(zip(labels, days))
    tindex = plan["slot"].map(index_by_label)
    assert (plan["day"] == plan["slot"].map(day_by_label)).all()
    by_id = chosen.set_index("sessionID")
    allowed = (
        (tindex >= plan["sessionID"].map(by_id["start_index"]))
        & (tindex < plan["sessionID"].map(by_id["end_index"]))
    )
    limits = np.where(allowed, pmax, 0.0)
    power = plan["power_kW"].to_numpy()
    assert np.isfinite(power).all()
    assert np.isfinite(plan["energy_kWh"]).all()

    loads = plan.groupby("slot")["power_kW"].sum().reindex(labels)
    energy = plan.groupby("sessionID")["energy_kWh"].sum().reindex(ids)
    target = by_id.loc[ids, "target_kWh"]
    unmet = vehicle.loc[ids, "unmet_kWh"]
    assert np.isfinite(unmet).all()

    power_violation = float(max(
        0.0, np.max(-power), np.max(power-limits),
        (loads-capacity).max(),
    ))
    energy_residual = maxabs(energy + unmet - target)
    energy_integral_error = maxabs(plan["energy_kWh"] - power*dt)
    overcharge = float(max(0.0, (energy-target).max()))
    unmet_bound_violation = float(max(
        0.0, (-unmet).max(), (unmet-target).max()
    ))
    exported_session_error = max(
        maxabs(vehicle.loc[ids, "planned_kWh"]-energy),
        maxabs(vehicle.loc[ids, "target_kWh"]-target),
    )
    cost = float(np.dot(loads.to_numpy(), prices.to_numpy()) * dt)
    cost_error = abs(cost-result["cost_USD"])
    lock_violation = max(
        0.0, float(unmet.sum())-result["U_star_kWh"]-case["epsilon_U"]
    )
    lower_bound = float(
        chosen["device_only_shortfall_lower_bound_kWh"].sum()
    )

    daily = pd.DataFrame({
        "local_date": days,
        "load_kW": loads.to_numpy(),
        "cost_USD": loads.to_numpy()*prices.to_numpy()*dt,
    }).groupby("local_date").agg(
        peak_kW=("load_kW", "max"), cost_USD=("cost_USD", "sum")
    )
    peak_error = max(
        abs(row.peak_kW-result["peak_by_day_kW"][day])
        for day, row in daily.iterrows()
    )
    peak_variable_violation = max(
        [0.0] + [
            row.peak_kW-result["peak_variable_by_day_kW"][day]
            for day, row in daily.iterrows()
        ]
    )
    daily.to_csv(OUT / "daily_planned_metrics.csv")

    summary.update({
        "sessions": len(ids),
        "slots": len(grid),
        "zero_full_slot_sessions": int((chosen["full_slots"] == 0).sum()),
        "truth_rows_outside_formal_session_ids": len(
            set(truth["sessionID"])-set(sessions["sessionID"])
        ),
        "target_kWh": float(target.sum()),
        "planned_kWh": float(energy.sum()),
        "U_star_kWh": result["U_star_kWh"],
        "U_stage2_kWh": float(unmet.sum()),
        "device_only_shortfall_lower_bound_kWh": lower_bound,
        "planned_cost_USD": cost,
        "planned_max_peak_kW": float(daily["peak_kW"].max()),
        "independent_power_violation_kW": power_violation,
        "independent_energy_residual_kWh": energy_residual,
        "energy_integral_error_kWh": energy_integral_error,
        "overcharge_kWh": overcharge,
        "unmet_bound_violation_kWh": unmet_bound_violation,
        "exported_session_error_kWh": exported_session_error,
        "cost_recompute_error_USD": cost_error,
        "daily_peak_recompute_error_kW": peak_error,
        "peak_variable_violation_kW": peak_variable_violation,
        "service_lock_violation_kWh": lock_violation,
        "solver_stages": result["stages"],
    })
    assert power_violation <= 1e-6
    assert energy_residual <= 1e-5
    assert energy_integral_error <= 1e-8
    assert overcharge <= 1e-5 and unmet_bound_violation <= 1e-5
    assert exported_session_error <= 1e-5
    assert cost_error <= 1e-7
    assert peak_error <= 1e-6 and peak_variable_violation <= 1e-6
    assert lock_violation <= 1e-7
    assert result["U_star_kWh"] >= lower_bound-1e-5
    summary["status"] = "passed"
except Exception:
    summary["status"] = "failed"
    summary["traceback"] = traceback.format_exc()
finally:
    (OUT / "static_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

sys.exit(0 if summary["status"] == "passed" else 1)
