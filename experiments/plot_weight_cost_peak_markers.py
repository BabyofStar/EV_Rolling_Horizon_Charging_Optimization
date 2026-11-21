
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BATCH, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
source = BATCH / "sensitivity_results.csv"
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    "c732274df130a5660d4f94a5a3a996aaf8417a5a6fc0bebb316939a8a3126fee"
)
table = pd.read_csv(source)
table = table[table["scenario_id"].isin(
    ["E1_reference", "E2_alpha020", "E2_alpha080"]
)]
methods = ["Uncontrolled", "TOU_Greedy", "Static_LP", "RH_LP"]
colors = dict(zip(methods, ["#64748b", "#e69f00", "#009e73", "#2563eb"]))
markers = {0.2: "s", 0.5: "o", 0.8: "^"}
sizes = {0.2: 105, 0.5: 62, 0.8: 28}
cohorts = ["main_at_arrival", "supplement_after_submission"]
titles = ["Arrival-input cohort, n=39", "Late-input cohort, n=468"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
for ax, cohort, title in zip(axes, cohorts, titles):
    for method in methods:
        data = table[
            (table["cohort"] == cohort) & (table["method"] == method)
        ].sort_values("alpha")
        ax.plot(data["cost_USD"], data["max_peak_kW"],
                color=colors[method], linewidth=1.5, zorder=1)
        for level in [0.2, 0.5, 0.8]:
            row = data[data["alpha"].eq(level)].iloc[0]
            ax.scatter(
                row["cost_USD"], row["max_peak_kW"],
                marker=markers[level], s=sizes[level],
                facecolors=("white" if level != 0.8 else colors[method]),
                edgecolors=colors[method], linewidths=1.3,
                zorder=2 + [0.2, 0.5, 0.8].index(level),
            )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Total electricity cost (USD)")
    ax.set_ylabel("Maximum actual EV power (kW)")
    ax.grid(alpha=0.22)
    ax.margins(x=0.10, y=0.14)

handles = [
    Line2D([0], [0], color=colors[m], lw=2, label=m)
    for m in methods
] + [
    Line2D([0], [0], color="black", linestyle="None",
           marker=markers[a], markerfacecolor="white",
           label=f"alpha={a:g}", markersize=7)
    for a in [0.2, 0.5, 0.8]
]
fig.suptitle("E2: cost and peak at prescribed weights", y=0.98)
fig.legend(handles=handles, loc="upper center",
           bbox_to_anchor=(0.5, 0.92), ncol=7, fontsize=9)
fig.text(
    0.02, 0.025,
    "Test: [2019-07-08, 2019-07-29), Los Angeles. "
    "Nested markers indicate coincident weights. Service levels differ; "
    "these points are not a complete Pareto frontier.",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.09, 1, 0.82))
for extension in ["png", "svg"]:
    fig.savefig(OUT / f"weight_cost_peak.{extension}",
                dpi=180, bbox_inches="tight")
plt.close(fig)
result = {
    "status": "generated_for_visual_review",
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "artifacts": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.iterdir() if p.is_file()
    },
}
(OUT / "verification.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(json.dumps(result, indent=2))
print("Figure directory:", OUT)
