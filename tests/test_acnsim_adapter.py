from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.simulator import AggregateEnvironment
from src.acnsim_adapter import ACNSimExecutor

T = pd.Timestamp("2019-05-06T07:00:00Z")
passed = []

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    passed.append(name)

def build(rows, updates=(), backend=False, response=None):
    # rows: id, station, arrival_minute, departure_minute, demand, cohort
    sessions, events, releases = [], [], []
    for sid, station, arrival, departure, demand, cohort in rows:
        sessions.append({
            "sessionID": sid, "stationID": station,
            "connectionTime": T + pd.Timedelta(minutes=arrival),
            "cohort": cohort, "initial_kWhRequested": demand,
            "initial_requestedDeparture": T + pd.Timedelta(hours=2),
        })
        for kind, minute in [("arrival", arrival), ("departure", departure)]:
            events.append({
                "sessionID": sid, "event_type": kind,
                "event_time": T + pd.Timedelta(minutes=minute),
            })
    for sid, minute, demand, order in updates:
        releases.append({
            "sessionID": sid, "modifiedAt": T + pd.Timedelta(minutes=minute),
            "input_order": order, "valid": True, "kWhRequested": demand,
            "requestedDeparture": T + pd.Timedelta(hours=2),
        })
    executor = (
        ACNSimExecutor([row[1] for row in rows], capacity_kw=10,
                       response_max_kw=response) if backend else None)
    environment = AggregateEnvironment(
        pd.DataFrame(sessions), pd.DataFrame(events), pd.DataFrame(releases),
        start=T, capacity_kw=10, executor=executor)
    return environment, executor

def command(view):
    # Deterministic diagnostic control only; not a formal baseline.
    remaining_capacity = view["capacity_kw"]
    result = {}
    for v in view["vehicles"]:
        if v["remaining_kwh"] is not None and v["remaining_kwh"] > 1e-9:
            power = min(6.0, remaining_capacity)
            if power > 0:
                result[v["id"]] = power
                remaining_capacity -= power
    return result

def totals(e):
    return {s["id"]: s["delivered_kwh"] for s in e.completed}

main = "main_at_arrival"
supplement = "supplement_after_submission"
scenarios = [
    ("two vehicles",
     [("a", "A", 0, 60, 10, main), ("b", "B", 0, 60, 10, main)], []),
    ("mid-slot departure",
     [("a", "A", 0, 7.5, 10, main)], []),
    ("late input",
     [("a", "A", 0, 60, None, supplement)], [("a", 5, 10, 0)]),
    ("downward update",
     [("a", "A", 0, 60, None, supplement)],
     [("a", 0, 10, 0), ("a", 7.5, 0.5, 1)]),
    ("same-EVSE handoff",
     [("a", "A", 0, 7.5, 10, main), ("b", "A", 7.5, 60, 10, main)], []),
    ("fractional completion",
     [("a", "A", 0, 60, 0.123456789, main)], []),
]

for name, rows, updates in scenarios:
    ideal, _ = build(rows, updates)
    adapted, executor = build(rows, updates, backend=True)
    for _ in range(4):
        ideal.step(command(ideal.view()))
        adapted.step(command(adapted.view()))
    a, b = totals(ideal), totals(adapted)
    check(f"{name}: ideal and ACN-Sim delivery agree",
          a.keys() == b.keys() and
          all(abs(a[sid] - b[sid]) < 1e-7 for sid in a))
    check(f"{name}: environment agrees with ACN-Sim cumulative feedback",
          all(abs(b[sid] - executor.evs[sid].energy_delivered) < 1e-7
              for sid in b))
    check(f"{name}: no final connection or capacity violation",
          not executor._connected and
          all(row["total_kw"] <= 10 + 1e-8 for row in adapted.ledger))

e, executor = build(
    [("a", "A", 0, 60, 10, main)], backend=True, response=3.0)
e.step({"a": 6.0})
actual = e.view()["vehicles"][0]["delivered_kwh"]
check("reduced response updates environment with actual energy",
      abs(actual - 0.75) < 1e-8 and
      abs(executor.evs["a"].energy_delivered - actual) < 1e-8)

# Test the adapter's own network gate independently of the environment gate.
e, executor = build(
    [("a", "A", 0, 60, 10, main), ("b", "B", 0, 60, 10, main)],
    backend=True)
try:
    executor.execute(e.view(), {"a": 6, "b": 6}, 0.25)
except ValueError:
    check("network rejects excessive pilots before energy execution",
          all(ev.energy_delivered == 0 for ev in executor.evs.values()))
else:
    raise AssertionError("Network accepted excessive aggregate current")

result = {
    "status": "passed", "check_count": len(passed), "checks": passed,
    "backend": "ACN-Sim network/EVSE/EV/Battery components",
    "event_engine": "project AggregateEnvironment",
    "network_scope": "aggregate experimental capacity; common artificial phase",
    "physical_battery_parameters_inferred": False,
    "original_caltech_branch_constraints_validated": False,
}
Path(sys.argv[1]).write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
