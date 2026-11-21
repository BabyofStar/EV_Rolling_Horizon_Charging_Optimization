
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SUMMARY, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=False)
BATCH = SUMMARY.parent
E1 = ROOT / "results/logs/stage4_main_test_20261112T102930606571Z"
DATA = ROOT / "data/processed/caltech_12weeks_20261111T143402Z_9b087d"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

summary_record = json.loads(
    (SUMMARY / "verification.json").read_text(encoding="utf-8")
)
assert summary_record["status"] == "passed"
assert summary_record["formal_scenarios"] == 22

expected = {}
for rel, value in summary_record["output_artifact_sha256"].items():
    path = SUMMARY / rel
    assert digest(path) == value, rel
    expected[path.resolve()] = value

# 实际轨迹使用已有记录中的哈希；
# 未在旧记录中保存哈希的内部诊断文件，仅建立当前快照。
snapshots, csv_cache, run_cache = {}, {}, {}

def checked(path):
    path = path.resolve()
    assert path.is_relative_to(ROOT)
    value = digest(path)
    if path in expected:
        assert value == expected[path], str(path)
    snapshots[path] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": value,
        "integrity": (
            "prior_recorded_hash" if path in expected else "current_snapshot_only"
        ),
    }
    return path

def register_run(folder):
    folder = folder.resolve()
    if folder in run_cache:
        return
    path = folder / "verification.json"
    rel = str(path.relative_to(ROOT))
    assert digest(path) == summary_record["run_evidence_sha256"][rel]
    expected[path] = summary_record["run_evidence_sha256"][rel]
    record = json.loads(checked(path).read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert not record["requires_review"]
    for key, value in record["source_and_input_sha256"].items():
        expected[(ROOT / key).resolve()] = value
    for key, value in record["output_artifact_sha256"].items():
        expected[(folder / key).resolve()] = value
    run_cache[folder] = record

register_run(E1)

def read_csv(path):
    path = checked(path)
    if path not in csv_cache:
        csv_cache[path] = pd.read_csv(path, dtype={"sessionID": str})
    return csv_cache[path].copy()

def times(series):
    return pd.to_datetime(
        series, utc=True, format="ISO8601", errors="raise"
    ).astype("datetime64[ns, UTC]")

sessions = read_csv(DATA / "sessions_with_split.csv")
events = read_csv(DATA / "environment_events.csv")
releases = read_csv(DATA / "input_release_events.csv")
station_path = checked(ROOT / "configs/station.yaml")
pmax = float(yaml.safe_load(
    station_path.read_text(encoding="utf-8")
)["evse_max_power_kW"])

def service(folder):
    return read_csv(folder / "session_service.csv").set_index("sessionID")

changes = read_csv(SUMMARY / "session_changes_vs_E1.csv")
cohorts = ["main_at_arrival", "supplement_after_submission"]
choices = []

# 固定排序规则：统计量降序、sessionID 升序。
def choose(frame, column):
    assert len(frame)
    return frame.sort_values(
        [column, "sessionID"], ascending=[False, True], kind="mergesort"
    ).iloc[0]["sessionID"]

for kind in ["E1_RH_vs_UC", "early30", "under20_seed00", "over20_seed00"]:
    for cohort in cohorts:
        if kind == "E1_RH_vs_UC":
            rh = service(E1 / cohort / "RH_LP")
            uc = service(E1 / cohort / "Uncontrolled")
            ranked = (
                rh["unmet_kwh"] - uc["unmet_kwh"]
            ).rename("delta").reset_index()
            sid = choose(ranked, "delta")
            scenario_run, baseline_method = E1, "Uncontrolled"
            factor = 1.0
        else:
            scenario_id = (
                "E4_early30" if kind == "early30" else "E4_req20_seed00"
            )
            scenario_run = BATCH / scenario_id
            register_run(scenario_run)
            ranked = changes[
                changes["scenario_id"].eq(scenario_id) &
                changes["cohort"].eq(cohort) &
                changes["method"].eq("RH_LP")
            ].copy()
            factors = read_csv(
                scenario_run / cohort / "request_factors.csv"
            )
            ranked = ranked.merge(
                factors, on="sessionID", validate="one_to_one"
            )
            if kind == "under20_seed00":
                ranked = ranked[ranked["factor"] < 1]
                sid = choose(ranked, "delta_unmet_kwh")
            elif kind == "over20_seed00":
                ranked = ranked[ranked["factor"] > 1]
                sid = choose(ranked, "delta_excess_kwh")
            else:
                sid = choose(ranked, "delta_unmet_kwh")
            factor = float(
                ranked.set_index("sessionID").loc[sid, "factor"]
            )
            baseline_method = "RH_LP"
        choices.append({
            "kind": kind, "cohort": cohort, "sessionID": sid,
            "scenario_run": scenario_run,
            "baseline_method": baseline_method, "factor": factor,
        })

def connection_inputs(run, cohort, sid):
    if run == E1:
        s, e, u = sessions.copy(), events.copy(), releases.copy()
    else:
        s = read_csv(run / cohort / "runtime_sessions.csv")
        e = read_csv(run / cohort / "scenario_environment_events.csv")
        u = read_csv(run / cohort / "runtime_input_release_events.csv")

    row = s[s["sessionID"].eq(sid)].iloc[0]
    boundary = e[e["sessionID"].eq(sid)].set_index("event_type")
    arrival = pd.Timestamp(boundary.loc["arrival", "event_time"])
    departure = pd.Timestamp(boundary.loc["departure", "event_time"])
    if cohort == "main_at_arrival":
        history = pd.DataFrame([{
            "when": arrival,
            "target_kwh": float(row["initial_kWhRequested"]),
            "expected_departure": pd.Timestamp(row["initial_requestedDeparture"]),
        }])
    else:
        valid = u["valid"].astype(str).str.strip().str.lower().isin(
            ["true", "1", "1.0"]
        )
        selected = u[u["sessionID"].eq(sid) & valid].copy()
        selected["modifiedAt"] = times(selected["modifiedAt"])
        selected = selected.sort_values(["modifiedAt", "input_order"])
        history = pd.DataFrame({
            "when": selected["modifiedAt"],
            "target_kwh": pd.to_numeric(selected["kWhRequested"]),
            "expected_departure": pd.to_datetime(
                selected["requestedDeparture"],
                utc=True, format="ISO8601", errors="raise"
            ),
        }).reset_index(drop=True)
    history["released_before_departure"] = (
        (history["when"] >= arrival) & (history["when"] < departure)
    )
    return arrival, departure, history

def trace(folder, sid, arrival, departure):
    station = read_csv(folder / "station_trajectory.csv")[["start", "end"]]
    vehicle = read_csv(folder / "session_trajectory.csv")
    vehicle = vehicle[vehicle["sessionID"].eq(sid)][
        ["start", "end", "actual_kw"]
    ]
    for frame in [station, vehicle]:
        frame["start"], frame["end"] = times(frame["start"]), times(frame["end"])
    station = station[
        station["start"].ge(arrival) & station["end"].le(departure)
    ].sort_values("start")
    frame = station.merge(
        vehicle, on=["start", "end"], how="left", validate="one_to_one"
    )
    frame["actual_kw"] = frame["actual_kw"].fillna(0.0)
    assert len(frame)
    assert frame.iloc[0]["start"] == arrival
    assert frame.iloc[-1]["end"] == departure
    assert frame["start"].iloc[1:].reset_index(drop=True).equals(
        frame["end"].iloc[:-1].reset_index(drop=True)
    )
    hours = (frame["end"] - frame["start"]).map(
        lambda value: value.value / 3_600_000_000_000
    )
    frame["energy_kwh"] = frame["actual_kw"] * hours
    return frame

def energy_after(frame, boundary):
    return sum(
        row.actual_kw * max(
            0, (row.end - max(row.start, boundary)).value
        ) / 3_600_000_000_000
        for row in frame.itertuples(index=False)
        if row.end > boundary
    )

def first_charge(frame):
    active = frame[frame["actual_kw"] > 1e-6]
    return None if active.empty else active.iloc[0]["start"]

def lp_context(folder, sid, arrival, departure, save_folder, label):
    # 内部文件仅作辅助上下文，历史哈希和当前快照分别记录。
    if not (folder / "decision_summary.csv").is_file():
        return []
    decisions = read_csv(folder / "decision_summary.csv")
    if "step_directory" not in decisions:
        return []
    decisions["time"] = times(decisions["time"])
    decisions = decisions[
        decisions["mode"].eq("lp") &
        decisions["time"].ge(arrival) &
        decisions["time"].lt(departure)
    ].sort_values("time")
    contexts = []
    for position, ordered in [
        ("first", decisions), ("last", decisions.iloc[::-1])
    ]:
        for row in ordered.to_dict("records"):
            step = Path(row["step_directory"]).resolve()
            assert step.is_relative_to(folder.resolve())
            case_path = checked(step / "case.json")
            case = json.loads(case_path.read_text(encoding="utf-8"))
            car = next(
                (c for c in case["sessions"] if str(c["id"]) == sid), None
            )
            if car is None:
                continue
            plan = read_csv(step / "lp/plan.csv")
            plan = plan[plan["sessionID"].eq(sid)].copy()
            slot_times = times(plan["slot"])
            late = sum(
                float(power) * max(
                    0, (slot + pd.Timedelta(minutes=15) -
                        max(slot, departure)).value
                ) / 3_600_000_000_000
                for slot, power in zip(slot_times, plan["power_kW"])
            )
            plan.to_csv(
                save_folder / f"{label}_{position}_vehicle_plan.csv",
                index=False
            )
            contexts.append({
                "side": label, "position": position,
                "time": row["time"],
                "remaining_demand_kwh": car["demand"],
                "available_slots": len(car["available"]),
                "planned_after_actual_departure_kwh": late,
                "case_integrity": snapshots[case_path]["integrity"],
                "plan_integrity": snapshots[
                    (step / "lp/plan.csv").resolve()
                ]["integrity"],
            })
            break
    return contexts

def curve(frame, arrival):
    x = [(frame.iloc[0]["start"] - arrival).value / 3.6e12]
    y = [0.0]
    for row in frame.itertuples(index=False):
        x.append((row.end - arrival).value / 3.6e12)
        y.append(y[-1] + row.energy_kwh)
    return x, y

def power_curve(frame, arrival):
    x, y = [], []
    for row in frame.itertuples(index=False):
        x.extend([
            (row.start - arrival).value / 3.6e12,
            (row.end - arrival).value / 3.6e12,
        ])
        y.extend([row.actual_kw, row.actual_kw])
    return x, y

records, all_contexts = [], []
overview, overview_axes = plt.subplots(4, 2, figsize=(14, 14))
short_names = {
    "E1_RH_vs_UC": "E1 RH vs Uncontrolled",
    "early30": "30-minute earlier departure",
    "under20_seed00": "20% underestimate, seed 00",
    "over20_seed00": "20% overestimate, seed 00",
}

for index, choice in enumerate(choices, start=1):
    sid, cohort, run = (
        choice["sessionID"], choice["cohort"], choice["scenario_run"]
    )
    folder = OUT / f"case{index:02d}"
    folder.mkdir()
    baseline_folder = E1 / cohort / choice["baseline_method"]
    scenario_folder = run / cohort / "RH_LP"
    arrival, departure, history = connection_inputs(run, cohort, sid)
    old_arrival, old_departure, old_history = connection_inputs(E1, cohort, sid)
    assert arrival == old_arrival

    baseline_service = service(baseline_folder).loc[sid]
    scenario_service = service(scenario_folder).loc[sid]
    target = float(scenario_service["evaluation_target_kwh"])
    assert abs(target - baseline_service["evaluation_target_kwh"]) <= 1e-8

    baseline_trace = trace(baseline_folder, sid, arrival, old_departure)
    scenario_trace = trace(scenario_folder, sid, arrival, departure)
    for frame, saved_service in [
        (baseline_trace, baseline_service),
        (scenario_trace, scenario_service)
    ]:
        delivered = float(frame["energy_kwh"].sum())
        assert abs(delivered - saved_service["delivered_kwh"]) <= 1e-5
        assert abs(max(target - delivered, 0) -
                   saved_service["unmet_kwh"]) <= 1e-5
        assert abs(max(delivered - target, 0) -
                   saved_service["excess_kwh"]) <= 1e-5

    baseline_trace.to_csv(folder / "baseline_trace.csv", index=False)
    scenario_trace.to_csv(folder / "scenario_trace.csv", index=False)
    old_history.to_csv(folder / "baseline_inputs.csv", index=False)
    history.to_csv(folder / "scenario_inputs.csv", index=False)

    before = history[history["released_before_departure"]]
    first_input = None if before.empty else before.iloc[0]["when"]
    latest_target = None if before.empty else float(before.iloc[-1]["target_kwh"])
    eligible = None if first_input is None else max(arrival, first_input).ceil("15min")
    available_hours = (
        0.0 if eligible is None else
        max(0, (departure - eligible).value) / 3.6e12
    )
    lower_bound = max(target - pmax * available_hours, 0)

    baseline_tail = energy_after(baseline_trace, departure)
    delivered_loss = (
        float(baseline_service["delivered_kwh"]) -
        float(scenario_service["delivered_kwh"])
    )
    prefix_difference = (
        float(baseline_service["delivered_kwh"]) - baseline_tail -
        float(scenario_service["delivered_kwh"])
    )
    assert abs(
        delivered_loss - baseline_tail - prefix_difference
    ) <= 1e-5

    contexts = []
    if choice["baseline_method"] == "RH_LP":
        contexts += lp_context(
            baseline_folder, sid, arrival, old_departure, folder, "baseline"
        )
    contexts += lp_context(
        scenario_folder, sid, arrival, departure, folder, "scenario"
    )
    for context in contexts:
        context.update(case_id=f"case{index:02d}", sessionID=sid)
    all_contexts.extend(contexts)
    pd.DataFrame(contexts).to_csv(folder / "LP_context.csv", index=False)

    first_b, first_s = first_charge(baseline_trace), first_charge(scenario_trace)
    flags = []
    if before.empty:
        flags.append("no_request_released_before_departure")
    if (first_b is not None and first_s is not None and
            first_s > first_b + pd.Timedelta(minutes=1)):
        flags.append("scenario_first_charge_later")
    if baseline_tail > 0.01:
        flags.append("baseline_charged_in_removed_tail")
    if (choice["factor"] < 1 and latest_target is not None and
            scenario_service["delivered_kwh"] >= latest_target - 0.01 and
            scenario_service["unmet_kwh"] > 0.01):
        flags.append("estimated_target_met_but_evaluation_target_unmet")
    if (choice["factor"] > 1 and
            scenario_service["excess_kwh"] > baseline_service["excess_kwh"] + 0.01):
        flags.append("excess_increased_with_overestimate")
    if any(
        c["side"] == "scenario" and
        c["planned_after_actual_departure_kwh"] > 0.01
        for c in contexts
    ):
        flags.append("LP_context_planned_service_after_actual_departure")

    record = {
        "case_id": f"case{index:02d}", "kind": choice["kind"],
        "cohort": cohort, "sessionID": sid, "factor": choice["factor"],
        "evaluation_target_kwh": target,
        "baseline_delivered_kwh": baseline_service["delivered_kwh"],
        "scenario_delivered_kwh": scenario_service["delivered_kwh"],
        "delta_unmet_kwh":
            scenario_service["unmet_kwh"] - baseline_service["unmet_kwh"],
        "delta_excess_kwh":
            scenario_service["excess_kwh"] - baseline_service["excess_kwh"],
        "baseline_tail_energy_kwh": baseline_tail,
        "prefix_delivery_difference_kwh": prefix_difference,
        "device_time_shortfall_lower_bound_kwh": lower_bound,
        "latest_released_operational_target_kwh": latest_target,
        "inputs_at_or_after_departure": int(
            (~history["released_before_departure"]).sum()
        ),
        "observations": "; ".join(flags) or "no predefined observation flag",
        "scenario_directory": str(scenario_folder.relative_to(ROOT)),
        "baseline_directory": str(baseline_folder.relative_to(ROOT)),
    }
    records.append(record)

    # 每个案例保留功率及累计交付图
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    baseline_label = f'E1 {choice["baseline_method"]}'
    for frame, label, color in [
        (baseline_trace, baseline_label, "#64748b"),
        (scenario_trace, "Scenario RH_LP", "#2563eb"),
    ]:
        x, y = power_curve(frame, arrival)
        axes[0].plot(x, y, color=color, label=label)
        x, y = curve(frame, arrival)
        axes[1].plot(x, y, color=color, label=label)
    for ax in axes:
        ax.axvline((departure - arrival).value / 3.6e12,
                   color="#2563eb", linestyle=":", label="Scenario departure")
        ax.grid(alpha=0.22)
    axes[1].axhline(target, color="#009e73", linestyle="--",
                    label="Fixed evaluation target")
    if not before.empty:
        hx = [(when - arrival).value / 3.6e12 for when in before["when"]]
        hy = before["target_kwh"].tolist()
        hx.append((departure - arrival).value / 3.6e12)
        hy.append(hy[-1])
        axes[1].step(hx, hy, where="post", color="#e69f00",
                     linestyle=":", label="Released operational target")
    axes[0].set_ylabel("Actual power (kW)")
    axes[1].set_ylabel("Cumulative delivery (kWh)")
    axes[1].set_xlabel("Hours since connection")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.suptitle(f'case{index:02d}: {short_names[choice["kind"]]}')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(folder / "trajectory.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    ax = overview_axes.flat[index - 1]
    for frame, label, color in [
        (baseline_trace, baseline_label, "#64748b"),
        (scenario_trace, "Scenario RH_LP", "#2563eb"),
    ]:
        x, y = curve(frame, arrival)
        ax.plot(x, y, color=color, label=label)
    ax.axhline(target, color="#009e73", linestyle="--")
    ax.axvline((departure - arrival).value / 3.6e12,
               color="#2563eb", linestyle=":")
    cohort_label = "Arrival-input" if cohort == "main_at_arrival" else "Late-input"
    ax.set_title(
        f'case{index:02d} | {cohort_label}\n{short_names[choice["kind"]]}',
        fontsize=10
    )
    ax.set_xlabel("Hours since connection")
    ax.set_ylabel("Delivered energy (kWh)")
    ax.grid(alpha=0.22)
    print(f"Checked case{index:02d}: {choice['kind']} / {cohort}", flush=True)

overview.suptitle(
    "Targeted trajectory diagnostics: eight selected comparisons", y=0.995
)
overview.text(
    0.02, 0.005,
    "Gray: E1 baseline (Uncontrolled in row 1; RH otherwise). "
    "Blue: scenario RH. Green dashed: fixed evaluation target. "
    "Blue dotted: scenario departure. Targeted cases are not a representative sample.",
    fontsize=8
)
overview.tight_layout(rect=(0, 0.04, 1, 0.97))
overview.savefig(OUT / "case_overview.png", dpi=170, bbox_inches="tight")
overview.savefig(OUT / "case_overview.svg", bbox_inches="tight")
plt.close(overview)

records = pd.DataFrame(records)
records.to_csv(OUT / "case_audit.csv", index=False)
pd.DataFrame(all_contexts).to_csv(OUT / "LP_context_all.csv", index=False)

# 确认读取期间文件未变化
for path, snapshot in snapshots.items():
    assert digest(path) == snapshot["sha256"], str(path)
pd.DataFrame(snapshots.values()).to_csv(
    OUT / "source_snapshot.csv", index=False
)

notes = [
    "# 定向案例核查",
    "",
    "8 个比较按事先确定的排序选取，不代表随机样本或独立统计重复。",
    "实际交付、缺口、超额均与已审计服务表一致。",
    "提前离站的交付差分解为基准被截去尾段的交付和截断前交付差；",
    "该恒等分解不把重调度、竞争容量等因素单独识别为因果效应。",
    "设备—时间缺口下界是忽略共享容量及后续请求变化的乐观下界。",
    "内部 LP 上下文仅辅助解释；旧哈希验证与当前快照在 source_snapshot.csv 分开记录。",
    "观察标记不是完整因果证明，不据此修改冻结参数。",
]
(OUT / "case_notes.md").write_text(
    "\n".join(notes) + "\n", encoding="utf-8"
)
artifacts = {
    str(path.relative_to(OUT)): digest(path)
    for path in sorted(OUT.rglob("*")) if path.is_file()
}
result = {
    "status": "passed",
    "scope": "eight targeted trajectory diagnostics; not whole stage 4",
    "case_count": 8,
    "checks": [
        "source summary and selected run evidence",
        "continuous case traces and service-table energy reconciliation",
        "fixed evaluation targets",
        "departure-tail delivery decomposition",
        "input visibility and operational-target records",
        "source files unchanged during inspection",
    ],
    "script_sha256": digest(Path(__file__)),
    "output_artifact_sha256": artifacts,
}
(OUT / "verification.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("\n=== CASE AUDIT ===")
print(records[[
    "case_id", "kind", "cohort", "sessionID", "factor",
    "delta_unmet_kwh", "delta_excess_kwh", "baseline_tail_energy_kwh",
    "prefix_delivery_difference_kwh", "device_time_shortfall_lower_bound_kwh",
    "observations",
]].round(6).to_string(index=False))
print("\n=== LP CONTEXT ===")
print(pd.DataFrame(all_contexts).to_string(index=False))
print("\nAudit:", result["status"])
print("Evidence directory:", OUT)
