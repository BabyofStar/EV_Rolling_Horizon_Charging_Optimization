
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
BATCH, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=False)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )

batch = json.loads(
    (BATCH / "batch_verification.json").read_text(encoding="utf-8")
)
assert batch["status"] == "passed"
assert len(batch["scenarios"]) == 22
assert all(
    s["status"] == "passed" and s["returncode"] == 0
    and s["check_count"] == 38
    for s in batch["scenarios"]
)
assert batch["config_sha256"] == (
    "5c52d926f1f94598f9f285babd0cacf0365db4b7fc085029f70d7be2618aa5e8"
)
assert batch["results_sha256"] == (
    "285dc3972c5702146ea508b46cd7f7e89f3f78b8647e175a3a0571d026129c3f"
)
assert digest(BATCH / "robustness_results.csv") == batch["results_sha256"]

config_path = ROOT / "configs/robustness.yaml"
assert digest(config_path) == batch["config_sha256"]
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
assert config["status"] == "frozen"
for rel, expected in config["source_and_input_sha256"].items():
    assert digest(ROOT / rel) == expected, rel

registered = {s["id"]: s for s in config["scenarios"]}
assert len(registered) == 22
assert set(registered) == {s["scenario_id"] for s in batch["scenarios"]}

E1 = ROOT / config["e1_directory"]
table = pd.read_csv(BATCH / "robustness_results.csv")
methods = ["Uncontrolled", "TOU_Greedy", "RH_LP"]
cohorts = ["main_at_arrival", "supplement_after_submission"]
metrics = [
    "cost_USD", "delivered_kwh", "unmet_kwh", "excess_kwh",
    "energy_satisfaction", "max_peak_kW",
    "actual_departure_uncompleted_rate", "active_P95_seconds",
]
assert len(table) == 138
assert not table.duplicated(["scenario_id", "cohort", "method"]).any()
assert table["fallback_count"].eq(0).all()
assert table["requires_review"].astype(str).str.lower().eq("false").all()
assert set(table["scenario_id"]) == set(registered) | {"E1_reference"}

base = table[table["scenario_id"].eq("E1_reference")].copy()
assert len(base) == 6
base_services = {
    (cohort, method): pd.read_csv(
        E1 / cohort / method / "session_service.csv",
        dtype={"sessionID": str}
    ).set_index("sessionID")
    for cohort in cohorts for method in methods
}
factor_manifest = pd.read_csv(
    ROOT / config["factor_manifest"], dtype={"sessionID": str}
)
expected_dates = list(pd.date_range(
    "2019-07-08", "2019-07-29", inclusive="left", freq="D"
).strftime("%Y-%m-%d"))

daily_frames, case_frames, evidence = [], [], {}
runs = [("E1_reference", E1)] + [
    (s["id"], BATCH / s["id"]) for s in config["scenarios"]
]

for index, (scenario_id, folder) in enumerate(runs, start=1):
    verification_path = folder / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    assert verification["status"] == "passed"
    assert verification["requires_review"] is False
    for rel, expected in verification["source_and_input_sha256"].items():
        assert digest(ROOT / rel) == expected, rel
    for rel, expected in verification["output_artifact_sha256"].items():
        assert digest(folder / rel) == expected, rel
    evidence[str(verification_path.relative_to(ROOT))] = digest(verification_path)

    current = pd.read_csv(folder / "comparison.csv")
    current = current[current["method"].isin(methods)]
    exported = table[table["scenario_id"].eq(scenario_id)]
    assert len(current) == len(exported) == 6
    current = current.set_index(["cohort", "method"]).sort_index()
    exported = exported.set_index(["cohort", "method"]).sort_index()
    pd.testing.assert_frame_equal(
        current[metrics], exported[metrics],
        check_dtype=False, check_exact=False, rtol=0, atol=1e-8
    )

    if scenario_id != "E1_reference":
        scenario = registered[scenario_id]
        assert exported["amplitude"].eq(scenario["amplitude"]).all()
        assert exported["seed"].eq(scenario["seed"]).all()
        assert exported["advance_minutes"].eq(scenario["advance_minutes"]).all()

    for cohort in cohorts:
        if scenario_id != "E1_reference":
            wanted = factor_manifest[
                factor_manifest["scenario_id"].eq(scenario_id) &
                factor_manifest["cohort"].eq(cohort)
            ].set_index("sessionID")["factor"].sort_index()
            actual = pd.read_csv(
                folder / cohort / "request_factors.csv",
                dtype={"sessionID": str}
            ).set_index("sessionID")["factor"].sort_index()
            assert wanted.index.equals(actual.index)
            assert np.allclose(wanted, actual, rtol=0, atol=1e-12)

        for method in methods:
            item = folder / cohort / method
            service = pd.read_csv(
                item / "session_service.csv", dtype={"sessionID": str}
            ).set_index("sessionID")
            original = base_services[(cohort, method)]
            assert set(service.index) == set(original.index)
            service = service.reindex(original.index)
            assert np.allclose(
                service["evaluation_target_kwh"],
                original["evaluation_target_kwh"], rtol=0, atol=1e-8
            )

            daily = pd.read_csv(item / "daily_metrics.csv")
            daily = daily.sort_values("local_date").reset_index(drop=True)
            assert daily["local_date"].astype(str).tolist() == expected_dates
            row = exported.loc[(cohort, method)]
            assert abs(daily["cost_USD"].sum() - row["cost_USD"]) <= 1e-5
            assert abs(daily["energy_kwh"].sum() - row["delivered_kwh"]) <= 1e-5
            assert abs(daily["peak_kW"].max() - row["max_peak_kW"]) <= 1e-6
            daily = daily.assign(
                scenario_id=scenario_id, cohort=cohort, method=method
            )
            daily_frames.append(daily)

            if scenario_id != "E1_reference":
                changed = service[[
                    "evaluation_target_kwh", "delivered_kwh",
                    "unmet_kwh", "excess_kwh", "satisfaction"
                ]].copy()
                changed["baseline_unmet_kwh"] = original["unmet_kwh"]
                changed["delta_unmet_kwh"] = (
                    changed["unmet_kwh"] - original["unmet_kwh"]
                )
                changed["delta_excess_kwh"] = (
                    changed["excess_kwh"] - original["excess_kwh"]
                )
                changed = changed.reset_index().assign(
                    scenario_id=scenario_id,
                    scenario_type=registered[scenario_id]["type"],
                    cohort=cohort, method=method,
                    evidence_directory=str(item.relative_to(ROOT)),
                )
                case_frames.append(changed)

    print(f"Audited {index}/23: {scenario_id}", flush=True)

daily = pd.concat(daily_frames, ignore_index=True)
daily.to_csv(OUT / "daily_metrics_all.csv", index=False)

# 1. 原始情景结果与相对各方法自身 E1 基准的变化
reference = base[["cohort", "method"] + metrics]
degradation = table.merge(
    reference, on=["cohort", "method"],
    suffixes=("", "_reference"), validate="many_to_one"
)
for column in ["cost_USD", "unmet_kwh", "excess_kwh", "max_peak_kW"]:
    degradation["delta_" + column] = (
        degradation[column] - degradation[column + "_reference"]
    )
degradation["delta_satisfaction_pp"] = 100 * (
    degradation["energy_satisfaction"] -
    degradation["energy_satisfaction_reference"]
)
degradation.to_csv(OUT / "scenario_changes_vs_E1.csv", index=False)

# 2. 十种子统计；零误差参考不伪装成十次独立重复
errors = table[
    table["scenario_type"].eq("demand_estimation_error")
].copy()
assert len(errors) == 120
errors["seed"] = errors["seed"].astype(int)
for _, group in errors.groupby(["cohort", "method", "amplitude"]):
    assert len(group) == 10
    assert set(group["seed"]) == set(range(10))

seed_stats = errors.groupby(
    ["cohort", "method", "amplitude"]
)[metrics].agg(["count", "mean", "median", "std", "min", "max"])
seed_stats.columns = [
    f"{column}__{statistic}" for column, statistic in seed_stats.columns
]
seed_stats = seed_stats.reset_index()
seed_stats.to_csv(OUT / "seed_statistics.csv", index=False)

# 3. 同种子下 20% 与 10% 的配对差异
low = errors[errors["amplitude"].eq(0.1)]
high = errors[errors["amplitude"].eq(0.2)]
strength = high.merge(
    low, on=["cohort", "method", "seed"],
    suffixes=("_20", "_10"), validate="one_to_one"
)
strength_out = strength[["cohort", "method", "seed"]].copy()
for column in ["cost_USD", "unmet_kwh", "excess_kwh", "max_peak_kW"]:
    strength_out["delta_" + column] = (
        strength[column + "_20"] - strength[column + "_10"]
    )
strength_out["delta_satisfaction_pp"] = 100 * (
    strength["energy_satisfaction_20"] -
    strength["energy_satisfaction_10"]
)
assert len(strength_out) == 60
strength_out.to_csv(OUT / "paired_20_minus_10.csv", index=False)

# 4. 提前离站是确定性单因素情景，不复制成十种子
early = degradation[
    degradation["scenario_type"].eq("early_departure") |
    degradation["scenario_id"].eq("E1_reference")
].sort_values(["cohort", "method", "advance_minutes"])
assert len(early) == 18
early.to_csv(OUT / "early_departure_comparison.csv", index=False)

# 5. RH 与同情景规则基线配对，差值定义为 RH 减基线
indexed = table.set_index(["scenario_id", "cohort", "method"])
method_pairs = []
for row in table[table["method"].eq("RH_LP")].to_dict("records"):
    for baseline_method in ["Uncontrolled", "TOU_Greedy"]:
        baseline = indexed.loc[
            (row["scenario_id"], row["cohort"], baseline_method)
        ]
        method_pairs.append({
            "scenario_id": row["scenario_id"],
            "scenario_type": row["scenario_type"],
            "cohort": row["cohort"],
            "amplitude": row["amplitude"], "seed": row["seed"],
            "baseline_method": baseline_method,
            "delta_cost_USD": row["cost_USD"] - baseline["cost_USD"],
            "delta_unmet_kwh": row["unmet_kwh"] - baseline["unmet_kwh"],
            "delta_excess_kwh": row["excess_kwh"] - baseline["excess_kwh"],
            "delta_peak_kW": row["max_peak_kW"] - baseline["max_peak_kW"],
            "delta_satisfaction_pp": 100 * (
                row["energy_satisfaction"] - baseline["energy_satisfaction"]
            ),
        })
method_pairs = pd.DataFrame(method_pairs)
assert len(method_pairs) == 92
method_pairs.to_csv(OUT / "RH_paired_method_differences.csv", index=False)

random_pairs = method_pairs[
    method_pairs["scenario_type"].eq("demand_estimation_error")
]
pair_columns = [
    "delta_cost_USD", "delta_unmet_kwh", "delta_excess_kwh",
    "delta_peak_kW", "delta_satisfaction_pp"
]
pair_stats = random_pairs.groupby(
    ["cohort", "amplitude", "baseline_method"]
)[pair_columns].agg(["mean", "std", "min", "max"])
pair_stats.columns = [f"{c}__{s}" for c, s in pair_stats.columns]
pair_stats = pair_stats.reset_index()
pair_stats.to_csv(OUT / "RH_paired_seed_statistics.csv", index=False)

# 6. 日级配对保留全部 21 天，包括零负荷日
daily_pairs = []
for baseline_method in ["Uncontrolled", "TOU_Greedy"]:
    rh = daily[daily["method"].eq("RH_LP")]
    rule = daily[daily["method"].eq(baseline_method)]
    joined = rh.merge(
        rule, on=["scenario_id", "cohort", "local_date"],
        suffixes=("_RH", "_baseline"), validate="one_to_one"
    )
    frame = joined[["scenario_id", "cohort", "local_date"]].copy()
    frame["baseline_method"] = baseline_method
    frame["delta_cost_USD"] = (
        joined["cost_USD_RH"] - joined["cost_USD_baseline"]
    )
    frame["delta_daily_peak_kW"] = (
        joined["peak_kW_RH"] - joined["peak_kW_baseline"]
    )
    daily_pairs.append(frame)
daily_pairs = pd.concat(daily_pairs, ignore_index=True)
assert len(daily_pairs) == 1932
daily_pairs.to_csv(OUT / "daily_RH_paired_differences.csv", index=False)

daily_stats = []
for keys, group in daily_pairs.groupby(
    ["scenario_id", "cohort", "baseline_method"]
):
    assert len(group) == 21
    row = dict(zip(["scenario_id", "cohort", "baseline_method"], keys))
    row["days"] = 21
    for column in ["delta_cost_USD", "delta_daily_peak_kW"]:
        values = group[column]
        row.update({
            column + "__mean": float(values.mean()),
            column + "__median": float(values.median()),
            column + "__q25": float(values.quantile(0.25)),
            column + "__q75": float(values.quantile(0.75)),
            column + "__days_lower": int((values < -1e-8).sum()),
            column + "__days_higher": int((values > 1e-8).sum()),
            column + "__days_equal": int((values.abs() <= 1e-8).sum()),
        })
    daily_stats.append(row)
pd.DataFrame(daily_stats).to_csv(
    OUT / "daily_paired_descriptive_statistics.csv", index=False
)

# 7. 保存全部逐车变化及预先固定排序的案例候选
cases = pd.concat(case_frames, ignore_index=True)
cases.to_csv(OUT / "session_changes_vs_E1.csv", index=False)
candidates = cases.sort_values(
    ["delta_unmet_kwh", "scenario_id", "sessionID"],
    ascending=[False, True, True], kind="mergesort"
).groupby(
    ["cohort", "method", "scenario_type"], sort=False
).head(3)
candidates.to_csv(OUT / "largest_shortfall_change_candidates.csv", index=False)

# 8. 图表：误差条是十种子的样本标准差
colors = {
    "Uncontrolled": "#64748b",
    "TOU_Greedy": "#e69f00",
    "RH_LP": "#2563eb",
}
titles = {
    "main_at_arrival": "Arrival-input cohort, n=39",
    "supplement_after_submission": "Late-input cohort, n=468",
}
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False
})
columns = [
    ("cost_USD", "Total electricity cost (USD)", 1),
    ("energy_satisfaction", "Energy satisfaction (%)", 100),
    ("unmet_kwh", "Total unmet energy (kWh)", 1),
    ("excess_kwh", "Total excess energy (kWh)", 1),
]
stats_index = seed_stats.set_index(["cohort", "method", "amplitude"])
base_index = base.set_index(["cohort", "method"])

fig, axes = plt.subplots(2, 4, figsize=(17, 8))
for i, cohort in enumerate(cohorts):
    for j, (metric, ylabel, scale) in enumerate(columns):
        ax = axes[i, j]
        for method in methods:
            means = [float(base_index.loc[(cohort, method), metric])]
            deviations = [0.0]
            for amplitude in [0.1, 0.2]:
                row = stats_index.loc[(cohort, method, amplitude)]
                means.append(float(row[metric + "__mean"]))
                deviations.append(float(row[metric + "__std"]))
            ax.errorbar(
                [0, 10, 20], np.array(means) * scale,
                yerr=np.array(deviations) * scale,
                marker="o", capsize=3, color=colors[method], label=method
            )
        ax.set_title(titles[cohort])
        ax.set_xlabel("Error amplitude (%)")
        ax.set_xticks([0, 10, 20])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
fig.suptitle("E4: controlled demand-estimation errors")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center",
           bbox_to_anchor=(0.5, 0.95), ncol=3)
fig.text(
    0.02, 0.025,
    "Fixed 21-day test period. Nonzero points: mean +/- sample SD over 10 paired "
    "seeds. Zero: reused E1 reference, not 10 repeats. Excess cannot offset unmet energy.",
    fontsize=8
)
fig.tight_layout(rect=(0, 0.09, 1, 0.89))
for extension in ["png", "svg"]:
    fig.savefig(OUT / f"demand_error_summary.{extension}",
                dpi=180, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(2, 4, figsize=(17, 8))
for i, cohort in enumerate(cohorts):
    for j, (metric, ylabel, scale) in enumerate(columns):
        ax = axes[i, j]
        for method in methods:
            frame = early[
                early["cohort"].eq(cohort) & early["method"].eq(method)
            ].sort_values("advance_minutes")
            ax.plot(
                frame["advance_minutes"], frame[metric] * scale,
                marker="o", color=colors[method], label=method
            )
        ax.set_title(titles[cohort])
        ax.set_xlabel("Earlier departure (minutes)")
        ax.set_xticks([0, 15, 30])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
fig.suptitle("E4: earlier departure with unchanged visible estimates")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center",
           bbox_to_anchor=(0.5, 0.95), ncol=3)
fig.text(
    0.02, 0.025,
    "Same evaluation energy targets and 21-day period. All sessions retained. "
    "Short connections clipped to positive duration and audited. Zero reuses E1.",
    fontsize=8
)
fig.tight_layout(rect=(0, 0.09, 1, 0.89))
for extension in ["png", "svg"]:
    fig.savefig(OUT / f"early_departure_summary.{extension}",
                dpi=180, bbox_inches="tight")
plt.close(fig)

notes = [
    "# E4 统计与结果解释准备",
    "",
    "22 个正式情景与 6 组 E1 在线零参考已完成来源、输出与目标一致性核验。",
    "新增 132 组方法／队列结果，共 138 行；零参考未重复运行。",
    "需求误差每幅度 10 个配对种子；提前离站为两个确定性情景。",
    "",
    "## 解释规则",
    "",
    "- 方法配对差值均为 RH 减基线；费用负值表示较低费用，缺口正值表示服务更差。",
    "- 费用与服务、超额同时报告，不能仅由低费用推断节费优势。",
    "- 随机种子统计只描述固定测试期内的受控合成误差，不代表真实预测误差分布。",
    "- 图中误差条为种子样本标准差，不是跨时间、跨站点的置信区间。",
    "- 日级配对保留 21 天，包括零负荷日；仅有三个完整周，不强行宣称显著性。",
    "- 日峰值差与全期最大峰值差口径不同，分别保存。",
    "- 案例按缺口增量确定性排序；候选排序不能直接作为原因解释，仍需逐段轨迹核查。",
    "- 本文件为统计准备，不代表阶段四整体验收。",
]
(OUT / "interpretation_notes.md").write_text(
    "\n".join(notes) + "\n", encoding="utf-8"
)

for rel, expected in config["source_and_input_sha256"].items():
    assert digest(ROOT / rel) == expected, rel
assert digest(config_path) == batch["config_sha256"]
assert digest(BATCH / "robustness_results.csv") == batch["results_sha256"]

artifacts = {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file()}
result = {
    "status": "passed",
    "scope": "E4 result audit and descriptive summary; not whole stage 4",
    "formal_scenarios": 22,
    "new_method_cohort_results": 132,
    "reused_reference_results": 6,
    "seed_groups": len(seed_stats),
    "daily_paired_rows": len(daily_pairs),
    "zero_fallback": True,
    "checks": [
        "frozen configuration, source and run artifact hashes",
        "138 unique scenario/cohort/method rows",
        "registered paired factors match runtime factors",
        "evaluation energy targets fixed to E1",
        "daily metrics reconcile with period cost, energy and maximum peak",
        "10 paired seeds at each nonzero amplitude",
        "all 21 test days retained in daily pairing",
    ],
    "run_evidence_sha256": evidence,
    "script_sha256": digest(Path(__file__)),
    "output_artifact_sha256": artifacts,
}
write_json(OUT / "verification.json", result)

print("\n=== E4 SUMMARY AUDIT ===")
print(json.dumps({
    k: result[k] for k in [
        "status", "scope", "formal_scenarios",
        "new_method_cohort_results", "reused_reference_results",
        "seed_groups", "daily_paired_rows", "zero_fallback"
    ]
}, ensure_ascii=False, indent=2))

view = seed_stats[["cohort", "method", "amplitude"]].copy()
view["cost_mean_USD"] = seed_stats["cost_USD__mean"]
view["satisfaction_mean_pct"] = seed_stats["energy_satisfaction__mean"] * 100
view["satisfaction_SD_pp"] = seed_stats["energy_satisfaction__std"] * 100
view["unmet_mean_kwh"] = seed_stats["unmet_kwh__mean"]
view["excess_mean_kwh"] = seed_stats["excess_kwh__mean"]
print("\n=== TEN-SEED SUMMARY ===")
print(view.round(5).to_string(index=False))

view = early[[
    "cohort", "method", "advance_minutes", "cost_USD",
    "unmet_kwh", "excess_kwh", "energy_satisfaction",
    "max_peak_kW", "delta_satisfaction_pp"
]].copy()
print("\n=== EARLY-DEPARTURE SUMMARY ===")
print(view.round(5).to_string(index=False))

print("\n=== RH MINUS BASELINE: SEED MEANS ===")
print(pair_stats[[
    "cohort", "amplitude", "baseline_method",
    "delta_cost_USD__mean", "delta_unmet_kwh__mean",
    "delta_satisfaction_pp__mean", "delta_excess_kwh__mean",
]].round(5).to_string(index=False))
print("\nSummary directory:", OUT)
