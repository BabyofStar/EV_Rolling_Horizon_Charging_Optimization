
from pathlib import Path
from datetime import datetime, timezone
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
OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=False)

E1 = ROOT / "results/logs/stage4_main_test_20261112T102930606571Z"
SENS = ROOT / "results/logs/stage4_sensitivity_20261112T124051538161Z"
SENS_SUM = SENS / "summary_20261113T083157781711Z"
ROB = ROOT / "results/logs/stage4_robustness_20261113T090451907487Z"
ROB_SUM = ROB / "summary_20261114T121318982512Z"
CASES = ROB / "case_audit_20261114T122830521605Z"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

index, frozen_sources = {}, {}

def add_index(path, value=None, kind="artifact", basis="current_file_check"):
    path = path.resolve()
    assert path.is_relative_to(ROOT)
    value = digest(path) if value is None else value
    key = str(path.relative_to(ROOT))
    if key in index:
        assert index[key]["sha256"] == value, key
        if basis not in index[key]["verification_basis"]:
            index[key]["verification_basis"] += "; " + basis
        return
    index[key] = {
        "path": key, "sha256": value,
        "kind": kind, "verification_basis": basis,
    }

def accepted(folder, artifact_key="output_artifact_sha256", allowed=("passed",)):
    path = folder / "verification.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] in allowed, str(path)
    add_index(path, kind="acceptance_record")
    for rel, expected in record.get(artifact_key, {}).items():
        artifact = folder / rel
        assert digest(artifact) == expected, str(artifact)
        add_index(artifact, expected, "summary_or_case_product",
                  str(path.relative_to(ROOT)))
    return record

# ---------- 已通过的汇总、图表与案例 ----------
sens_summary = accepted(SENS_SUM)
rob_summary = accepted(ROB_SUM)
case_summary = accepted(CASES)
assert rob_summary["formal_scenarios"] == 22
assert rob_summary["new_method_cohort_results"] == 132
assert rob_summary["reused_reference_results"] == 6
assert rob_summary["seed_groups"] == 12
assert rob_summary["daily_paired_rows"] == 1932
assert rob_summary["zero_fallback"] is True
assert case_summary["case_count"] == 8

display_records = sorted(SENS.glob("summary_*_displayfix/verification.json"))
assert display_records, "未找到显示修正版图表。"
DISPLAY = display_records[-1].parent
display_summary = accepted(DISPLAY)
assert digest(ROOT / "experiments/summarize_sensitivity_v2.py") == (
    display_summary["summary_script_sha256"]
)

marker_records = sorted(SENS.glob("weight_plot_*/verification.json"))
assert marker_records, "未找到最终点形权重图。"
MARKERS = marker_records[-1].parent
marker_summary = accepted(
    MARKERS, artifact_key="artifacts",
    allowed=("generated_for_visual_review",)
)
assert digest(ROOT / "experiments/plot_weight_cost_peak_markers.py") == (
    marker_summary["script_sha256"]
)

# 案例来源快照：保留“历史记录哈希”和“当前快照”的区别
snapshots = pd.read_csv(CASES / "source_snapshot.csv")
for row in snapshots.to_dict("records"):
    assert digest(ROOT / row["path"]) == row["sha256"], row["path"]

for relative, expected in {
    "report/stage3_acceptance.md":
        "9a15bcd90ce19a4732e99c69453aa5fc83ba013cfe6acec093c8b0c0f501e980",
    "report/stage3_artifact_index.csv":
        "ffc0a0d13db243c2d302111b6ab4005a450a8b4f238e892c890423419113aa85",
}.items():
    assert digest(ROOT / relative) == expected
    add_index(ROOT / relative, expected, "E0_prior_acceptance")

# ---------- 绑定 27 个已验收运行 ----------
sens_config = yaml.safe_load(
    (ROOT / "configs/sensitivity.yaml").read_text(encoding="utf-8")
)
rob_config = yaml.safe_load(
    (ROOT / "configs/robustness.yaml").read_text(encoding="utf-8")
)
assert digest(ROOT / "configs/sensitivity.yaml") == (
    "7079f7033ae1bfd38d2bbf4a76115cfc1fe002137b779d034b3c4af36d1f23e4"
)
assert digest(ROOT / "configs/robustness.yaml") == (
    "5c52d926f1f94598f9f285babd0cacf0365db4b7fc085029f70d7be2618aa5e8"
)
run_specs = [("E1_reference", "E1", E1, 8)]
run_specs += [
    (s["id"], s["experiment"], SENS / s["id"], 8)
    for s in sens_config["scenarios"]
]
run_specs += [
    (s["id"], "E4", ROB / s["id"], 6)
    for s in rob_config["scenarios"]
]
assert len(run_specs) == 27

accepted_run_hashes = dict(sens_summary["source_evidence_sha256"])
for rel, value in rob_summary["run_evidence_sha256"].items():
    if rel in accepted_run_hashes:
        assert accepted_run_hashes[rel] == value
    accepted_run_hashes[rel] = value

frames = []
for number, (scenario_id, experiment, folder, rows) in enumerate(run_specs, 1):
    path = folder / "verification.json"
    relative = str(path.relative_to(ROOT))
    assert digest(path) == accepted_run_hashes[relative], relative
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record.get("requires_review", False) is False
    add_index(path, kind="run_verification")

    # 原始运行产物已在运行及完整汇总中做过全量哈希核验。
    # 索引绑定那些已接受的记录，不把这里称作另一次全量重算。
    for rel, value in record["output_artifact_sha256"].items():
        add_index(
            folder / rel, value, "recorded_run_artifact",
            "accepted run and full-summary audit: " + relative
        )
    for rel, value in record["source_and_input_sha256"].items():
        source_path = (ROOT / rel).resolve()
        if source_path in frozen_sources:
            assert frozen_sources[source_path] == value, rel
        frozen_sources[source_path] = value

    comparison_path = folder / "comparison.csv"
    frame = pd.read_csv(comparison_path)
    assert len(frame) == rows
    assert not frame.duplicated(["cohort", "method"]).any()
    assert frame["fallback_count"].eq(0).all()
    assert frame["requires_review"].astype(str).str.lower().eq("false").all()
    for cohort, count in [
        ("main_at_arrival", 39), ("supplement_after_submission", 468)
    ]:
        assert frame.loc[frame["cohort"].eq(cohort), "sessions"].eq(count).all()
    if experiment == "E4":
        assert frame["smoke"].astype(str).str.lower().eq("false").all()
        assert frame["bypass_transform"].astype(str).str.lower().eq("false").all()

    frame["scenario_id"] = scenario_id
    frame["experiment"] = experiment
    frame["evidence_directory"] = str(folder.relative_to(ROOT))
    frame["run_verification_sha256"] = digest(path)
    frames.append(frame)
    add_index(comparison_path, kind="run_comparison")
    if (folder / "stdout_stderr.txt").exists():
        add_index(folder / "stdout_stderr.txt", kind="run_log")
    print(f"Bound accepted run {number}/27: {scenario_id}", flush=True)

for path, value in frozen_sources.items():
    assert digest(path) == value, str(path)
    add_index(path, value, "frozen_source_or_input", "current frozen-source check")

combined = pd.concat(frames, ignore_index=True)
assert len(combined) == 172
assert not combined.duplicated(["scenario_id", "cohort", "method"]).any()
assert combined.groupby("experiment").size().to_dict() == {
    "E1": 8, "E2": 16, "E3": 16, "E4": 132
}
static = combined["method"].eq("Static_LP")
assert combined.loc[static, "information"].eq(
    "offline_full_information_oracle"
).all()

norm = yaml.safe_load(
    (ROOT / "configs/normalization.yaml").read_text(encoding="utf-8")
)
assert abs(norm["C_ref"] - 2.88) < 1e-12
assert abs(norm["P_ref"] - 6.656) < 1e-12
combined["C_ref_USD"] = norm["C_ref"]
combined["P_ref_kW"] = norm["P_ref"]

# ---------- 固定规则的主实验负荷示例：测试期第一天 ----------
e1_record = json.loads(
    (E1 / "verification.json").read_text(encoding="utf-8")
)
splits = yaml.safe_load(
    (ROOT / "configs/splits.yaml").read_text(encoding="utf-8")
)
day_start = pd.Timestamp(
    splits["splits"]["test"]["start_utc_inclusive"]
).tz_convert("America/Los_Angeles")
day_end = day_start + pd.DateOffset(days=1)
colors = {
    "Uncontrolled": "#64748b", "TOU_Greedy": "#e69f00",
    "Static_LP": "#009e73", "RH_LP": "#2563eb",
}
cohorts = ["main_at_arrival", "supplement_after_submission"]
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, cohort in zip(axes, cohorts):
    for method, color in colors.items():
        path = E1 / cohort / method / "station_trajectory.csv"
        key = str(path.relative_to(E1))
        assert digest(path) == e1_record["output_artifact_sha256"][key]
        trace = pd.read_csv(path)
        trace["start"] = pd.to_datetime(trace["start"], utc=True, format="ISO8601")
        trace["end"] = pd.to_datetime(trace["end"], utc=True, format="ISO8601")
        trace = trace[
            trace["end"].gt(day_start) & trace["start"].lt(day_end)
        ]
        x, y = [], []
        for row in trace.itertuples(index=False):
            a, b = max(row.start, day_start), min(row.end, day_end)
            x.extend([
                (a - day_start).value / 3.6e12,
                (b - day_start).value / 3.6e12,
            ])
            y.extend([row.total_kw, row.total_kw])
        ax.plot(
            x, y, color=color, label=method,
            linestyle="--" if method == "Static_LP" else "-", linewidth=1.3
        )
    ax.set_title(cohort)
    ax.set_ylabel("Actual EV power (kW)")
    ax.grid(alpha=0.22)
axes[1].set_xlabel("Hours after local midnight")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center",
           bbox_to_anchor=(0.5, 0.94), ncol=4)
fig.suptitle(f"E1 fixed example: first test day, {day_start.date()}")
fig.text(
    0.02, 0.015,
    "Day chosen by calendar rule, not by method performance. "
    "Separate cohorts; Static LP is an offline full-information reference.",
    fontsize=8
)
fig.tight_layout(rect=(0, 0.06, 1, 0.88))
for extension in ["png", "svg"]:
    fig.savefig(OUT / f"E1_first_test_day.{extension}",
                dpi=180, bbox_inches="tight")
plt.close(fig)

# ---------- 生成可引用的发现与验收报告 ----------
case_table = pd.read_csv(CASES / "case_audit.csv", dtype={"sessionID": str})
assert len(case_table) == 8
unique_sessions = len(case_table[["cohort", "sessionID"]].drop_duplicates())
seed_stats = pd.read_csv(ROB_SUM / "seed_statistics.csv")
early = pd.read_csv(ROB_SUM / "early_departure_comparison.csv")
pair_stats = pd.read_csv(ROB_SUM / "RH_paired_seed_statistics.csv")
sensitivity = pd.read_csv(SENS / "sensitivity_results.csv")
e1 = combined[combined["experiment"].eq("E1")].copy()

def markdown_table(frame):
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(map(str, columns)) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    def format_value(value):
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}"
        return str(value).replace("|", "/").replace("\n", " ")
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(v) for v in values) + " |")
    return "\n".join(lines)

def link(path):
    return "../" + str(path.relative_to(ROOT))

main_view = e1[[
    "cohort", "method", "cost_USD", "delivered_kwh", "unmet_kwh",
    "energy_satisfaction", "max_peak_kW", "active_P95_seconds",
    "offline_solve_wall_seconds"
]].copy()
seed_view = seed_stats[["cohort", "method", "amplitude"]].copy()
for metric in ["cost_USD", "unmet_kwh", "excess_kwh"]:
    seed_view[metric + "_mean"] = seed_stats[metric + "__mean"]
seed_view["satisfaction_mean_pct"] = seed_stats["energy_satisfaction__mean"] * 100
seed_view["satisfaction_SD_pp"] = seed_stats["energy_satisfaction__std"] * 100

lines = [
    "# 阶段四：实验发现",
    "",
    "本阶段完成必做主实验、权重、容量、需求估计误差及提前离站评价。",
    "结果支持特定数据与假设下的权衡和反例，不预设 RH 优势。",
    "",
    "## 实验口径",
    "",
    "- Caltech 单站聚合连续功率情景，两队列分别运行和评价。",
    "- 测试期为洛杉矶当地 [2019-07-08, 2019-07-29)，21 个完整自然日。",
    "- 严格到达输入队列 39 条，晚提交补充队列 468 条，不混合计算为全站结论。",
    "- 主方案 alpha=0.5、容量 50 kW、控制间隔 15 分钟、基础窗口 24 小时并按已知预计离站延伸。",
    "- C_ref=2.88 USD、P_ref=6.656 kW，开发集确定后保持不变。",
    "- Static LP 为完整信息离线参考，真实连接完整时段口径与在线中途离站执行不同；不自动作为在线缺口下界。",
    "- 电价和容量为实验情景；未验证原 Caltech 支路与相位约束，不代表实际账单。",
    "",
    "## E1 主比较",
    "",
    markdown_table(main_view),
    "",
]
for cohort in cohorts:
    group = e1[e1["cohort"].eq(cohort)].set_index("method")
    rh, uc = group.loc["RH_LP"], group.loc["Uncontrolled"]
    lines.append(
        f'- {cohort}：RH 相对 Uncontrolled 的费用差为 '
        f'{100*(rh["cost_USD"]-uc["cost_USD"])/uc["cost_USD"]:.3f}%，'
        f'全期峰值差为 '
        f'{100*(rh["max_peak_kW"]-uc["max_peak_kW"])/uc["max_peak_kW"]:.3f}%，'
        f'满足率差为 '
        f'{100*(rh["energy_satisfaction"]-uc["energy_satisfaction"]):.3f} 个百分点。'
    )
lines += [
    "",
    "费用降低伴随服务差异，不能将这些费用差直接称为同等服务下的节费收益。",
    "",
    "## E2 权重与 E3 容量",
    "",
    "预设 alpha=0.2/0.5/0.8、容量=37.5/50/62.5 kW；共享基准复用 E1。",
    "RH 实际费用与服务未随权重呈简单单调关系；三个权重点不构成完整 Pareto 前沿。",
    "主队列在本容量网格下结果不变；补充队列容量增加时服务改善，低容量下低费用伴随较多缺口。",
    "",
    markdown_table(sensitivity[sensitivity["method"].eq("RH_LP")][[
        "scenario_id", "cohort", "alpha", "capacity_kw",
        "cost_USD", "unmet_kwh", "energy_satisfaction", "max_peak_kW"
    ]]),
    "",
    "## E4 需求估计误差",
    "",
    "每车固定正负号，误差幅度 10%/20%，10 个种子配对，原始评价目标固定。",
    "同一车在两个幅度、后续输入更新及三种在线方法间共享扰动方向。",
    "零误差参考复用 E1，运行入口等价性仅做过小样本对照，不声称全量零组重跑。",
    "",
    markdown_table(seed_view),
    "",
    "误差增大时三方法平均满足率下降，缺口及超额增加。RH 绝对服务较低，不等于相对自身基准的增量退化始终更大。",
    "补充队列中，RH 相对 TOU 的平均费用较高、满足率较低，本情景没有显示费用与服务优势。",
    "超额交付计入电费，但不能抵消其他车辆缺口；请求电量不是电池物理容量，不将超额服务称为真实电池过充。",
    "",
    markdown_table(pair_stats[[
        "cohort", "amplitude", "baseline_method",
        "delta_cost_USD__mean", "delta_unmet_kwh__mean",
        "delta_satisfaction_pp__mean", "delta_excess_kwh__mean"
    ]]),
    "",
    "## E4 提前离站",
    "",
    "实际离站分别提前 15/30 分钟，调度器可见预计离站不变；保留全部会话并审计短连接截断。",
    "",
    markdown_table(early[[
        "cohort", "method", "advance_minutes", "cost_USD",
        "unmet_kwh", "excess_kwh", "energy_satisfaction",
        "delta_satisfaction_pp"
    ]]),
    "",
    "提前 30 分钟时，RH 主/补充满足率分别下降 5.87410/4.93007 个百分点，均大于本次规则基线的降幅。",
    "费用同时降低主要需要结合少交付电量解释，不能单独作为调度收益。",
    "",
    "## 轨迹案例",
    "",
    f"8 个定向比较涉及 {unique_sessions} 个不同会话；同一会话可用于不同对照，不当作独立重复。",
    "",
    markdown_table(case_table[[
        "case_id", "kind", "cohort", "sessionID", "factor",
        "delta_unmet_kwh", "delta_excess_kwh",
        "baseline_tail_energy_kwh", "prefix_delivery_difference_kwh",
        "device_time_shortfall_lower_bound_kwh"
    ]]),
    "",
    "- case01/02：RH 实际较晚开始充电；case01 同时存在约 18.243 kWh 的设备—可用时间缺口下界，不能把全部缺口归因于调度。",
    "- case03：新增缺口约 3.328 kWh，几乎等于被移去尾段中的基准交付。",
    "- case04：新增缺口约 5.937 kWh，由尾段 3.328 kWh 与截断前交付差约 2.609 kWh 组成；恒等分解不是独立因果识别。",
    "- case05/06：低估目标达到后仍存在原始评价缺口，新增缺口分别为 11.2/16.8 kWh。",
    "- case07/08：高估对应新增超额约 4.8/12 kWh，不能用于抵消其他车辆缺口。",
    "- 实际轨迹、输入和服务表为主证据。E1 内部 LP 上下文仅当前快照，相关完整性层级在案例 source_snapshot.csv 中保留。",
    "",
    "## 统计与适用边界",
    "",
    "- 10 种子标准差描述固定测试期内的合成扰动波动，不是跨日期或跨站点置信区间。",
    "- 日级配对保留全部 21 天和零负荷日；只有三个完整周，不强行宣称显著性，也不把 15 分钟点当独立样本。",
    "- 需求误差为受控合成机制，不是真实可部署预测器的验证。",
    "- 严格队列样本仅 39 条且存在选择偏差；两个队列结论分别解释。",
    "- E5 计算网格/窗口扩展、E6 电价/非理想响应扩展未开展，不能声称已覆盖。",
    "- 本阶段没有按测试结果重新选择主参数；报告归档与干净环境复现属于阶段五。",
    "",
    "## 主要证据",
    "",
    f"- [E1 主表]({link(ROOT / 'results/main_comparison.csv')})",
    f"- [E2/E3 汇总]({link(SENS_SUM / 'verification.json')})",
    f"- [最终权重图]({link(MARKERS / 'weight_cost_peak.png')})",
    f"- [E4 汇总]({link(ROB_SUM / 'verification.json')})",
    f"- [日级配对差异]({link(ROB_SUM / 'daily_RH_paired_differences.csv')})",
    f"- [八案例核查]({link(CASES / 'verification.json')})",
    f"- [固定首日负荷示例]({link(OUT / 'E1_first_test_day.png')})",
]

# ---------- 发布派生产物，保护既有文件 ----------
new_paths = [
    ROOT / "results/stage4_all_comparisons.csv",
    ROOT / "report/findings.md",
    ROOT / "report/stage4_acceptance.md",
    ROOT / "report/stage4_artifact_index.csv",
]
for path in new_paths:
    assert not path.exists(), f"归档目标已存在，未覆盖：{path}"

manifest_path = ROOT / "results/run_manifest.csv"
assert manifest_path.resolve() not in frozen_sources
if manifest_path.exists():
    assert manifest_path.read_bytes() == (E1 / "run_manifest.csv").read_bytes(), (
        "现有全局运行清单已变化，未覆盖。"
    )
    (OUT / "previous_run_manifest.csv").write_bytes(manifest_path.read_bytes())

assert digest(ROOT / "results/main_comparison.csv") == digest(E1 / "comparison.csv")

all_path = ROOT / "results/stage4_all_comparisons.csv"
combined.to_csv(all_path, index=False)
manifest_columns = [
    "scenario_id", "experiment", "run_id", "split", "cohort", "method",
    "alpha", "capacity_kw", "base_horizon_hours",
    "evidence_directory", "run_verification_sha256", "requires_review",
]
combined[manifest_columns].to_csv(manifest_path, index=False)
assert len(pd.read_csv(all_path)) == len(pd.read_csv(manifest_path)) == 172

findings_path = ROOT / "report/findings.md"
findings_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

acceptance_text = f"""# 阶段四：实验评价验收

结论：必做范围内通过。归档 UTC 时间：{datetime.now(timezone.utc).isoformat()}。

- E0：引用已通过的阶段三正确性与集成证据；各组检查不简单相加为独立测试总数。
- E1：四方法、两个队列，共 8 组正式主实验结果。
- E2/E3：4 个新增预设情景，共 32 组结果，共享基准复用 E1。
- E4：22 个情景，共 132 组新增结果；6 组在线零参考复用 E1，不重复计入新实验。
- 共 27 个正式运行、172 组方法/队列/情景结果，均无回退或待审查标记。
- E4 固定目标、配对扰动、12 个十种子组、1932 行日级配对及图表已验收。
- 8 个定向案例比较、{unique_sessions} 个不同会话已核对实际交付与服务表。
- 所有方法保留费用、服务、峰值、硬约束及计算时间字段；Static LP 离线身份清楚。
- 结果包括无改进与反例，不以预定节费百分比作为验收条件。
- E5/E6 扩展未开展；阶段五报告与干净环境复现不在本次验收范围。

最终发现见 `findings.md`，证据索引见 `stage4_artifact_index.csv`。
原始运行产物的哈希沿用已接受的运行和完整汇总审计，并绑定记录哈希；
本归档另行核验冻结来源、新发布文件及选定示例轨迹，不声称重新执行全部仿真。

索引不包含索引自身和闭合验收 JSON，二者关系由归档目录的 verification.json 记录。
"""
acceptance_path = ROOT / "report/stage4_acceptance.md"
acceptance_path.write_text(acceptance_text, encoding="utf-8")

for path in [
    all_path, manifest_path, ROOT / "results/main_comparison.csv",
    findings_path, acceptance_path, Path(__file__),
    OUT / "E1_first_test_day.png", OUT / "E1_first_test_day.svg",
]:
    add_index(path, kind="stage4_published_product")
if (OUT / "previous_run_manifest.csv").exists():
    add_index(OUT / "previous_run_manifest.csv", kind="preserved_previous_manifest")

index_path = ROOT / "report/stage4_artifact_index.csv"
pd.DataFrame(index.values()).sort_values("path").to_csv(index_path, index=False)

# 发布派生文件没有改变冻结的模型、输入及配置
for path, value in frozen_sources.items():
    assert digest(path) == value, str(path)

published = [
    all_path, manifest_path, ROOT / "results/main_comparison.csv",
    findings_path, acceptance_path, index_path,
    OUT / "E1_first_test_day.png", OUT / "E1_first_test_day.svg",
]
result = {
    "status": "passed",
    "scope": "stage 4 required experiments and results evaluation",
    "accepted_runs": 27,
    "unique_method_cohort_scenario_rows": 172,
    "rows_by_experiment": {"E1": 8, "E2": 16, "E3": 16, "E4": 132},
    "E4_reused_reference_rows_not_counted_as_new": 6,
    "case_comparisons": 8,
    "unique_case_sessions": unique_sessions,
    "artifact_index_rows": len(index),
    "source_acceptance_record_sha256": {
        str(path.relative_to(ROOT)): digest(path)
        for path in [
            SENS_SUM / "verification.json", DISPLAY / "verification.json",
            MARKERS / "verification.json", ROB_SUM / "verification.json",
            CASES / "verification.json",
        ]
    },
    "published_artifact_sha256": {
        str(path.relative_to(ROOT)): digest(path) for path in published
    },
    "optional_not_run": ["E5", "E6"],
    "stage5_accepted": False,
}
(OUT / "verification.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("\n=== STAGE 4 ACCEPTANCE ===")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("\nArchive directory:", OUT)
print("Findings:", findings_path)
print("Acceptance report:", acceptance_path)
print("Artifact index:", index_path)
