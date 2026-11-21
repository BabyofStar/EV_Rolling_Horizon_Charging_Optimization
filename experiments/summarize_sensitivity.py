
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

batch = json.loads(
    (BATCH / "batch_verification.json").read_text(encoding="utf-8")
)
assert batch["status"] == "passed"
assert batch["design_sha256"] == (
    "7079f7033ae1bfd38d2bbf4a76115cfc1fe002137b779d034b3c4af36d1f23e4"
)
assert digest(BATCH / "sensitivity_results.csv") == batch["results_sha256"]
assert batch["results_sha256"] == (
    "c732274df130a5660d4f94a5a3a996aaf8417a5a6fc0bebb316939a8a3126fee"
)

design_path = ROOT / "configs/sensitivity.yaml"
assert digest(design_path) == batch["design_sha256"]
design = yaml.safe_load(design_path.read_text(encoding="utf-8"))
for rel, expected in design["source_and_input_sha256"].items():
    assert digest(ROOT / rel) == expected, rel

expected_ids = {s["id"] for s in design["scenarios"]}
assert {s["scenario_id"] for s in batch["scenarios"]} == expected_ids

evidence_hashes = {}
folders = [ROOT / batch["reference_directory"]]
for scenario in batch["scenarios"]:
    assert scenario["status"] == "passed"
    assert scenario["returncode"] == 0
    assert scenario["check_count"] == 48
    folders.append(BATCH / scenario["scenario_id"])

for folder in folders:
    path = folder / "verification.json"
    verification = json.loads(path.read_text(encoding="utf-8"))
    assert verification["status"] == "passed"
    assert verification["requires_review"] is False
    for rel, expected in verification["source_and_input_sha256"].items():
        assert digest(ROOT / rel) == expected, rel
    for rel, expected in verification["output_artifact_sha256"].items():
        assert digest(folder / rel) == expected, rel
    evidence_hashes[str(path.relative_to(ROOT))] = digest(path)

table = pd.read_csv(BATCH / "sensitivity_results.csv")
assert len(table) == 40
assert not table.duplicated(["scenario_id", "cohort", "method"]).any()
assert table["fallback_count"].eq(0).all()
assert table["requires_review"].astype(str).str.lower().eq("false").all()

cohorts = ["main_at_arrival", "supplement_after_submission"]
methods = ["Uncontrolled", "TOU_Greedy", "Static_LP", "RH_LP"]
labels = {
    "main_at_arrival": "Arrival-input cohort, n=39",
    "supplement_after_submission": "Late-input cohort, n=468",
}
colors = dict(zip(methods, ["#64748b", "#e69f00", "#009e73", "#2563eb"]))

weights = table[
    table["scenario_id"].isin(["E1_reference", "E2_alpha020", "E2_alpha080"])
].copy()
capacities = table[
    table["scenario_id"].isin(["E1_reference", "E3_cap0375", "E3_cap0625"])
].copy()
assert len(weights) == len(capacities) == 24

# 独立核对：规则方法不受 alpha 影响
for cohort in cohorts:
    for method in ["Uncontrolled", "TOU_Greedy"]:
        frame = weights[
            (weights["cohort"] == cohort) & (weights["method"] == method)
        ]
        for column in [
            "cost_USD", "delivered_kwh", "unmet_kwh",
            "energy_satisfaction", "max_peak_kW"
        ]:
            assert np.ptp(frame[column].to_numpy()) <= 1e-6, (
                cohort, method, column
            )

# 同一静态模型扩大容量不应增加第一层服务最优缺口。
# 此处比较统一执行后、已核验与静态计划一致的总缺口。
for cohort in cohorts:
    static = capacities[
        (capacities["cohort"] == cohort) &
        (capacities["method"] == "Static_LP")
    ].sort_values("capacity_kw")
    assert np.all(np.diff(static["unmet_kwh"].to_numpy()) <= 1e-5)

weights.to_csv(OUT / "weight_sensitivity.csv", index=False)
capacities.to_csv(OUT / "capacity_sensitivity.csv", index=False)

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.hashsalt": "ev-stage4-sensitivity-v1",
})

def save_figure(fig, name):
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(OUT / f"{name}.png", dpi=180)
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)

def panel(ax, frame, cohort, x, y, ylabel):
    for method in methods:
        data = frame[
            (frame["cohort"] == cohort) & (frame["method"] == method)
        ].sort_values(x)
        values = data[y] * (100 if y == "energy_satisfaction" else 1)
        ax.plot(
            data[x], values, marker="o", label=method,
            color=colors[method], linewidth=1.6, markersize=5
        )
    ax.set_xlabel("Alpha" if x == "alpha" else "Scenario capacity (kW)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)

# 1. 费用—峰值点图；标注 alpha，不将有限点解释成完整 Pareto 前沿
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, cohort in zip(axes, cohorts):
    for method in methods:
        data = weights[
            (weights["cohort"] == cohort) & (weights["method"] == method)
        ].sort_values("alpha")
        ax.plot(
            data["cost_USD"], data["max_peak_kW"],
            marker="o", color=colors[method], label=method
        )
        if method in ["Static_LP", "RH_LP"]:
            for row in data.to_dict("records"):
                ax.annotate(
                    f'a={row["alpha"]:g}',
                    (row["cost_USD"], row["max_peak_kW"]),
                    xytext=(4, 5), textcoords="offset points", fontsize=8
                )
    ax.set_title(labels[cohort])
    ax.set_xlabel("Total electricity cost (USD)")
    ax.set_ylabel("Maximum actual EV power (kW)")
    ax.grid(alpha=0.22)
axes[0].legend(fontsize=8)
fig.suptitle("E2: cost and peak at prescribed weights")
fig.text(
    0.02, 0.015,
    "Test: 2019-07-08 to 2019-07-29 (end exclusive), Los Angeles. "
    "Service levels differ; three weights are not a complete Pareto frontier.",
    fontsize=8
)
save_figure(fig, "weight_cost_peak")

# 2. 权重下的服务指标
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, cohort in zip(axes, cohorts):
    panel(ax, weights, cohort, "alpha",
          "energy_satisfaction", "Energy satisfaction (%)")
    ax.set_title(labels[cohort])
axes[0].legend(fontsize=8)
fig.suptitle("E2: service at prescribed weights")
fig.text(
    0.02, 0.015,
    "Same 21-day test interval and cohort targets. "
    "E1 supplies alpha=0.5. Curves connect only the three evaluated points.",
    fontsize=8
)
save_figure(fig, "weight_service")

# 3. 容量下同时报告费用、峰值、服务
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
columns = [
    ("cost_USD", "Total electricity cost (USD)"),
    ("max_peak_kW", "Maximum actual EV power (kW)"),
    ("energy_satisfaction", "Energy satisfaction (%)"),
]
for row_index, cohort in enumerate(cohorts):
    for column_index, (metric, ylabel) in enumerate(columns):
        ax = axes[row_index, column_index]
        panel(ax, capacities, cohort, "capacity_kw", metric, ylabel)
        ax.set_title(labels[cohort])
axes[0, 0].legend(fontsize=8)
fig.suptitle("E3: aggregate capacity scenarios")
fig.text(
    0.02, 0.015,
    "Same 21-day test interval. Capacities are experimental scenarios. "
    "Lower capacity can reduce delivered service as well as cost and peak.",
    fontsize=8
)
save_figure(fig, "capacity_service_cost_peak")

# 4. 计算时间：在线 P95 与离线全期求解分别标记
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for row_index, cohort in enumerate(cohorts):
    for column_index, (frame, x) in enumerate([
        (weights, "alpha"), (capacities, "capacity_kw")
    ]):
        ax = axes[row_index, column_index]
        for method, field, label in [
            ("RH_LP", "active_P95_seconds", "RH active decision P95"),
            ("Static_LP", "offline_solve_wall_seconds", "Static full-period solve"),
        ]:
            data = frame[
                (frame["cohort"] == cohort) & (frame["method"] == method)
            ].sort_values(x)
            ax.plot(data[x], data[field], marker="o",
                    label=label, color=colors[method])
        ax.set_title(labels[cohort])
        ax.set_xlabel("Alpha" if x == "alpha" else "Scenario capacity (kW)")
        ax.set_ylabel("Seconds")
        ax.grid(alpha=0.22)
axes[0, 0].legend(fontsize=8)
fig.suptitle("E2/E3: online decision and offline solve times")
fig.text(
    0.02, 0.015,
    "Timings are observed measurements, not complexity bounds. "
    "Online per-decision P95 and offline full-period solve have different meanings.",
    fontsize=8
)
save_figure(fig, "computation_time")

notes = [
    "# E2/E3 汇总与验收",
    "",
    "预设权重与容量实验运行及汇总检查通过；不代表阶段四全部完成。",
    f"原始证据目录：`{BATCH.relative_to(ROOT)}`",
    "测试期：洛杉矶当地 [2019-07-08, 2019-07-29)，21 个自然日。",
    "两个队列分别为 39／468 条会话。",
    "4 个新增情景各 48/48 项检查通过；共享基准点复用 E1。",
    "含基准共 40 行结果；按实验轴展开的两张表各 24 行，不能视为独立重复实验。",
    "",
    "## 解释边界",
    "",
    "- RH 的费用、缺口及峰值不随权重简单单调变化；需联合解释服务指标。",
    "- 费用—峰值仅有三个预设权重点，不宣称完整 Pareto 前沿。",
    "- 主队列在本次容量网格下结果不变，不推断所有负荷下均对容量不敏感。",
    "- 补充队列容量增加时服务改善；较低容量下低费用不能直接称为节费。",
    "- Static LP 是完整信息离线参考，在线与离线耗时口径分开标示。",
    "- 未根据测试结果修改 E1 主参数，未开展新的参数搜索。",
    "- 控制间隔、窗口与电价扩展本次未运行；E4 信息误差实验仍待实施。",
]
(OUT / "acceptance.md").write_text(
    "\n".join(notes) + "\n", encoding="utf-8"
)
artifacts = {
    p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file()
}
result = {
    "status": "passed",
    "scope": "required E2/E3 grid summary; not whole stage 4",
    "scenario_count_new": 4,
    "comparison_rows_including_reference": 40,
    "checks": [
        "frozen design and source hashes",
        "all five run evidence and trajectory hashes",
        "40 unique method/cohort/scenario combinations",
        "zero fallback and no review flags",
        "rule baselines invariant to alpha",
        "Static LP shortfall nonincreasing with capacity",
    ],
    "source_evidence_sha256": evidence_hashes,
    "summary_script_sha256": digest(Path(__file__)),
    "output_artifact_sha256": artifacts,
}
(OUT / "verification.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
print("Summary directory:", OUT)
