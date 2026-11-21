"""Replay stage-one cleaning and chronological splits."""
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
SNAPSHOT = PROJECT / 'data/raw/caltech_12weeks_20261111T143402Z_9b087d'

from pathlib import Path
import json
import math
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
source = Path(SNAPSHOT)
manifest = json.loads((source / "manifest.json").read_text())
assert manifest.get("complete"), "快照尚未完整下载"

raw = []
for page in manifest["pages"]:
    raw.extend(json.loads((source / page["file"]).read_bytes())["_items"])
assert len(raw) == manifest["records_downloaded"]

out = PROJECT / "data/processed" / source.name
out.mkdir(parents=True, exist_ok=True)

def ts(value):
    if value is None:
        return pd.NaT
    return pd.to_datetime(value, utc=True, errors="coerce")

def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

# 完全重复保留一条；内容冲突的同 ID 记录全部留待审计。
groups = {}
for index, s in enumerate(raw):
    key = s.get("sessionID") or f"__missing_{index}"
    groups.setdefault(key, []).append((index, s))

online, updates, truth, exclusions, inventory = [], [], [], [], []
exact_duplicates = 0

for sid, group in groups.items():
    versions = {json.dumps(s, sort_keys=True) for _, s in group}
    if len(versions) > 1:
        for index, _ in group:
            exclusions.append({
                "raw_index": index, "sessionID": sid,
                "reason": "conflicting_duplicate_id",
            })
        continue

    exact_duplicates += len(group) - 1
    index, s = group[0]
    reasons = []
    arrival, departure = ts(s.get("connectionTime")), ts(s.get("disconnectTime"))

    if not s.get("sessionID"):
        reasons.append("missing_session_id")
    if not s.get("stationID"):
        reasons.append("missing_station_id")
    if str(s.get("siteID")) != "0002" or str(s.get("clusterID")) != "0039":
        reasons.append("unexpected_site_or_cluster")
    if s.get("timezone") != "America/Los_Angeles":
        reasons.append("unexpected_timezone")
    if pd.isna(arrival) or pd.isna(departure) or departure <= arrival:
        reasons.append("invalid_connection_times")
    delivered = number(s.get("kWhDelivered"))
    if delivered is not None and delivered < 0:
        reasons.append("negative_historical_energy")

    if reasons:
        exclusions.append({
            "raw_index": index, "sessionID": sid,
            "reason": ";".join(reasons),
        })
        continue

    # 包括无用户输入车辆，用于核查原始 EVSE 占用。
    inventory.append({
        "sessionID": sid, "stationID": s["stationID"],
        "connectionTime": arrival, "disconnectTime": departure,
    })

    usable = []
    for order, u in enumerate(s.get("userInputs") or []):
        modified, predicted = ts(u.get("modifiedAt")), ts(u.get("requestedDeparture"))
        energy = number(u.get("kWhRequested"))
        reason = None

        if energy is None or energy <= 0:
            reason = "invalid_requested_energy"
        elif pd.isna(modified) or pd.isna(predicted):
            reason = "invalid_input_timestamp"
        elif predicted <= modified:
            reason = "departure_not_after_submission"
        elif modified >= departure:
            reason = "submitted_after_disconnect"

        updates.append({
            "sessionID": sid, "input_order": order,
            "modifiedAt": modified, "kWhRequested": energy,
            "requestedDeparture": predicted,
            "valid": reason is None, "invalid_reason": reason,
        })
        if reason is None:
            usable.append((modified, order, energy, predicted))

    usable.sort(key=lambda u: (u[0], u[1]))
    # 同时刻输入按原始列表顺序处理，后者覆盖前者。
    visible = [u for u in usable if u[0] <= arrival and u[3] > arrival]
    later = [u for u in usable if arrival < u[0] < departure]

    if visible:
        cohort, initial = "main_at_arrival", visible[-1]
    elif later:
        cohort, initial = "supplement_after_submission", None
    else:
        exclusions.append({
            "raw_index": index, "sessionID": sid,
            "reason": "no_usable_request_during_connection",
        })
        continue

    online.append({
        "sessionID": sid, "stationID": s["stationID"],
        "siteID": s["siteID"], "clusterID": s["clusterID"],
        "connectionTime": arrival, "cohort": cohort,
        "initial_kWhRequested": initial[2] if initial else None,
        "initial_requestedDeparture": initial[3] if initial else pd.NaT,
    })
    # 真实离站和历史交付只放入环境/审计文件。
    truth.append({
        "sessionID": sid, "disconnectTime": departure,
        "historical_kWhDelivered": delivered,
        "historical_doneChargingTime": ts(s.get("doneChargingTime")),
    })

sessions_df = pd.DataFrame(online)
inventory_df = pd.DataFrame(inventory)

# 查找同一 EVSE 的占用重叠；暂不自动删除。
overlaps = []
for station, group in inventory_df.groupby("stationID"):
    active = []
    for row in group.sort_values("connectionTime").to_dict("records"):
        active = [p for p in active if p["disconnectTime"] > row["connectionTime"]]
        for previous in active:
            overlaps.append({
                "stationID": station,
                "sessionID_1": previous["sessionID"],
                "sessionID_2": row["sessionID"],
                "overlap_start": row["connectionTime"],
                "overlap_end": min(previous["disconnectTime"], row["disconnectTime"]),
            })
        active.append(row)

tables = {
    "sessions.csv": sessions_df,
    "user_updates.csv": pd.DataFrame(updates),
    "environment_truth.csv": pd.DataFrame(truth),
    "exclusions.csv": pd.DataFrame(
        exclusions, columns=["raw_index", "sessionID", "reason"]
    ),
    "evse_inventory.csv": inventory_df,
    "evse_overlaps.csv": pd.DataFrame(
        overlaps,
        columns=["stationID", "sessionID_1", "sessionID_2",
                 "overlap_start", "overlap_end"],
    ),
}
for name, table in tables.items():
    table.to_csv(out / name, index=False)

summary = {
    "source_snapshot": str(source),
    "raw_records": len(raw),
    "exact_duplicate_records_removed": exact_duplicates,
    "candidate_sessions": len(sessions_df),
    "cohorts": sessions_df["cohort"].value_counts().to_dict(),
    "excluded_records": len(exclusions),
    "evse_overlap_pairs": len(overlaps),
    "status": "pending_grid_and_split_checks",
}
(out / "cleaning_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n排除原因：")
print(tables["exclusions.csv"]["reason"].value_counts().to_string())
print("\n输出目录：", out)

from pathlib import Path
import json
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data/processed/caltech_12weeks_20261111T143402Z_9b087d"
TZ = "America/Los_Angeles"

sessions = pd.read_csv(DATA / "sessions.csv", dtype="string")
truth = pd.read_csv(DATA / "environment_truth.csv", dtype="string")
updates = pd.read_csv(DATA / "user_updates.csv", dtype="string")

for table, columns in [
    (sessions, ["connectionTime", "initial_requestedDeparture"]),
    (truth, ["disconnectTime"]),
    (updates, ["modifiedAt", "requestedDeparture"]),
]:
    for column in columns:
        table[column] = pd.to_datetime(
            table[column], format="mixed", utc=True, errors="coerce"
        )

valid_updates = updates[
    updates["valid"].str.lower().eq("true")
].copy()

bounds = [
    ("development", "2019-05-06", "2019-06-24"),
    ("validation",  "2019-06-24", "2019-07-08"),
    ("test",        "2019-07-08", "2019-07-29"),
]
ranges = {
    name: (
        pd.Timestamp(start, tz=TZ).tz_convert("UTC"),
        pd.Timestamp(end, tz=TZ).tz_convert("UTC"),
    )
    for name, start, end in bounds
}

# 合并真实离站仅用于离线边界审计。
audit = sessions.merge(
    truth[["sessionID", "disconnectTime"]],
    on="sessionID", validate="one_to_one",
)
audit["split"] = pd.Series(pd.NA, index=audit.index, dtype="string")
audit["boundary_reason"] = ""

for name, (start, end) in ranges.items():
    arrived = audit["connectionTime"].ge(start) & audit["connectionTime"].lt(end)
    contained = arrived & audit["disconnectTime"].le(end)
    audit.loc[contained, "split"] = name
    audit.loc[arrived & ~contained, "boundary_reason"] = "crosses_split_end"

audit.loc[
    audit["split"].isna() & audit["boundary_reason"].eq(""),
    "boundary_reason",
] = "outside_selected_range"

# 网格诊断为离线审计，不作为调度器在到达时可见的输入。
updates_by_session = {
    sid: group.sort_values(["modifiedAt", "input_order"])
    for sid, group in valid_updates.assign(
        input_order=pd.to_numeric(valid_updates["input_order"])
    ).groupby("sessionID")
}
checks = []

for row in audit.to_dict("records"):
    arrival = row["connectionTime"]
    if row["cohort"] == "main_at_arrival":
        visible_time = arrival
        predicted = row["initial_requestedDeparture"]
    else:
        group = updates_by_session[row["sessionID"]]
        later = group[
            group["modifiedAt"].gt(arrival)
            & group["modifiedAt"].lt(row["disconnectTime"])
        ]
        first_time = later["modifiedAt"].min()
        # 同时刻多次输入取最后一次。
        initial = later[later["modifiedAt"].eq(first_time)].iloc[-1]
        visible_time = first_time
        predicted = initial["requestedDeparture"]

    first_control = visible_time.ceil("15min")
    expected_end = predicted.floor("15min")
    slots = max(int((expected_end - first_control).total_seconds() // 900), 0)

    checks.append({
        "sessionID": row["sessionID"],
        "split": row["split"],
        "cohort": row["cohort"],
        "first_visible_time": visible_time,
        "first_control_time": first_control,
        "initial_expected_grid_end": expected_end,
        "initial_expected_full_slots": slots,
        "rounding_wait_minutes": (
            first_control - visible_time
        ).total_seconds() / 60,
        "expected_end_rounding_loss_minutes": (
            predicted - expected_end
        ).total_seconds() / 60,
        "departed_before_first_control": row["disconnectTime"] <= first_control,
    })

checks = pd.DataFrame(checks)

# 在线会话文件不加入真实离站，也不提前填入晚提交的请求。
assigned = sessions.merge(
    audit.loc[audit["split"].notna(), ["sessionID", "split"]],
    on="sessionID", validate="one_to_one",
)
assigned.to_csv(DATA / "sessions_with_split.csv", index=False)
audit.to_csv(DATA / "split_boundary_audit.csv", index=False)
checks.to_csv(DATA / "grid_checks.csv", index=False)

config = {
    "timezone": TZ,
    "interval_minutes": 15,
    "boundary_policy": "exclude_sessions_disconnecting_after_split_end",
    "source_snapshot": str(DATA),
    "splits": {},
}
grid_counts = {}

for name, (start, end) in ranges.items():
    utc = pd.date_range(start, end, freq="15min", inclusive="left")
    local = utc.tz_convert(TZ)
    grid = pd.DataFrame({
        "start_utc": utc,
        "end_utc": utc + pd.Timedelta(minutes=15),
        "start_local": local,
        "local_date": local.strftime("%Y-%m-%d"),
        "local_time": local.strftime("%H:%M"),
        "weekday": local.dayofweek,
    })
    grid.to_csv(DATA / f"grid_{name}.csv", index=False)
    grid_counts[name] = len(grid)
    config["splits"][name] = {
        "start_utc_inclusive": start.isoformat(),
        "end_utc_exclusive": end.isoformat(),
        "start_local_inclusive": start.tz_convert(TZ).isoformat(),
        "end_local_exclusive": end.tz_convert(TZ).isoformat(),
    }

(PROJECT / "configs/splits.yaml").write_text(
    yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)

counts = pd.crosstab(assigned["split"], assigned["cohort"]).reindex(
    ["development", "validation", "test"], fill_value=0
)
retained_checks = checks[checks["split"].notna()]

assert assigned["sessionID"].is_unique
assert len(assigned) + int(audit["split"].isna().sum()) == len(sessions)

print("各集合候选会话数：")
print(counts.to_string())
print("\n边界排除：")
print(audit.loc[audit["split"].isna(), "boundary_reason"].value_counts().to_string())
print("\n各集合网格数：", grid_counts)
print("\n保留会话的网格诊断：")
print(retained_checks.groupby("cohort").agg(
    sessions=("sessionID", "size"),
    zero_initial_slots=("initial_expected_full_slots", lambda s: int((s == 0).sum())),
    departed_before_control=("departed_before_first_control", "sum"),
    mean_rounding_wait_minutes=("rounding_wait_minutes", "mean"),
).round(3).to_string())
print("\n已保存划分配置、网格和审计表。")