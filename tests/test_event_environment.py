from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.simulator import AggregateEnvironment

T = pd.Timestamp("2019-05-06T07:00:00Z")
passed = []

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    passed.append(name)

def fixture(cohort="main_at_arrival", target=10.0, departure=60,
            updates=None, arrival=0):
    sessions = pd.DataFrame([{
        "sessionID": "a", "stationID": "evse-a",
        "connectionTime": T + pd.Timedelta(minutes=arrival),
        "cohort": cohort, "initial_kWhRequested": target,
        "initial_requestedDeparture": T + pd.Timedelta(hours=1),
    }])
    events = pd.DataFrame([
        {"sessionID": "a", "event_type": "arrival",
         "event_time": T + pd.Timedelta(minutes=arrival)},
        {"sessionID": "a", "event_type": "departure",
         "event_time": T + pd.Timedelta(minutes=departure)},
    ])
    records = []
    for minute, demand, order in (updates or []):
        records.append({
            "sessionID": "a", "modifiedAt": T + pd.Timedelta(minutes=minute),
            "input_order": order, "valid": True, "kWhRequested": demand,
            "requestedDeparture": T + pd.Timedelta(hours=1),
            "disconnectTime": T + pd.Timedelta(minutes=departure),
        })
    return sessions, events, pd.DataFrame(records)

def env(**kwargs):
    return AggregateEnvironment(*fixture(**kwargs), start=T)

e = env()
e.step({"a": 6.0})
check("15-minute energy integral",
      abs(e.view()["vehicles"][0]["delivered_kwh"] - 1.5) < 1e-8)

e = env(departure=7.5)
e.step({"a": 6.0})
check("mid-slot departure stops power",
      not e.view()["vehicles"] and
      abs(e.completed[0]["delivered_kwh"] - 0.75) < 1e-8)

e = env(target=1.0)
e.step({"a": 6.0})
check("stop at target without overcharging",
      abs(e.view()["vehicles"][0]["delivered_kwh"] - 1.0) < 1e-8)

e = env(cohort="supplement_after_submission", updates=[(5, 10, 0)])
check("late input initially hidden",
      e.view()["vehicles"][0]["target_kwh"] is None)
try:
    e.step({"a": 6.0})
except ValueError:
    check("no charging before released request", e.now == T)
else:
    raise AssertionError("Unreleased request accepted")
e.step({})
check("late input released without retroactive energy",
      e.view()["vehicles"][0]["target_kwh"] == 10 and
      e.view()["vehicles"][0]["delivered_kwh"] == 0)

e = env(cohort="supplement_after_submission",
        updates=[(0, 10, 0), (7.5, 0.5, 1)])
e.step({"a": 6.0})
car = e.view()["vehicles"][0]
check("downward update stops power and preserves irreversible excess",
      abs(car["delivered_kwh"] - 0.75) < 1e-8 and
      abs(car["irreversible_excess_kwh"] - 0.25) < 1e-8)

e = env(cohort="supplement_after_submission", departure=15,
        updates=[(0, 10, 0), (15, 20, 1)])
e.step({"a": 6.0})
check("departure precedes simultaneous update",
      not e.view()["vehicles"] and e.completed[0]["target_kwh"] == 10)

e = env(cohort="supplement_after_submission",
        updates=[(0, 10, 0), (0, 12, 1)])
check("same-time last input_order wins",
      e.view()["vehicles"][0]["target_kwh"] == 12)

e = env(updates=[(0, 99, 0)])
check("main cohort ignores subsequent changes",
      e.view()["vehicles"][0]["target_kwh"] == 10)

a, b = env(departure=30), env(departure=60)
check("future departure hidden from current view", a.view() == b.view())
v = a.view()
v["vehicles"][0]["target_kwh"] = 999
check("view mutation cannot change environment",
      a.view()["vehicles"][0]["target_kwh"] == 10)

e = env(arrival=5)
check("future arrival hidden", not e.view()["vehicles"])
e.step({})
check("mid-slot arrival waits until next decision",
      e.view()["vehicles"][0]["delivered_kwh"] == 0)

e = env()
try:
    e.step({"a": 7.0})
except ValueError:
    check("invalid device command rejected before advancing", e.now == T)
else:
    raise AssertionError("Device violation accepted")

e = AggregateEnvironment(*fixture(), start=T, capacity_kw=5)
try:
    e.step({"a": 6.0})
except ValueError:
    check("invalid station command rejected before advancing", e.now == T)
else:
    raise AssertionError("Station violation accepted")

# Cross-local-midnight execution remains continuous.
s, events, updates = fixture(target=10, departure=60)
offset = pd.Timedelta(hours=23, minutes=45)
s["connectionTime"] = s["connectionTime"] + offset
s["initial_requestedDeparture"] = s["initial_requestedDeparture"] + offset
events["event_time"] = events["event_time"] + offset
e = AggregateEnvironment(s, events, updates, start=T + offset)
e.step({"a": 6})
e.step({"a": 6})
check("local midnight preserves delivered energy",
      abs(e.view()["vehicles"][0]["delivered_kwh"] - 3) < 1e-8)

integrated = sum(
    row["total_kw"] *
    (row["end"] - row["start"]).total_seconds() / 3600
    for row in e.ledger)
check("ledger independently conserves energy", abs(integrated - 3) < 1e-8)

# Actual-data zero-command replay: development only, cohorts separately.
data = ROOT / "data/processed/caltech_12weeks_20261111T143402Z_9b087d"
sessions = pd.read_csv(data / "sessions_with_split.csv")
events = pd.read_csv(data / "environment_events.csv")
updates = pd.read_csv(data / "input_release_events.csv")
counts = {}
for cohort, expected_count in [
    ("main_at_arrival", 64),
    ("supplement_after_submission", 1227),
]:
    selected = sessions[
        (sessions["split"] == "development") &
        (sessions["cohort"] == cohort)
    ]
    check(f"{cohort}: selected count", len(selected) == expected_count)
    e = AggregateEnvironment(selected, events, updates, start=T)
    end = pd.Timestamp("2019-06-24T07:00:00Z")
    while e.now < end:
        e.step({})
    check(f"{cohort}: all sessions depart with zero delivered",
          len(e.completed) == expected_count and
          not e.view()["vehicles"] and
          all(s["delivered_kwh"] == 0 for s in e.completed))
    counts[cohort] = len(e.completed)

result = {
    "status": "passed", "checks": passed, "check_count": len(passed),
    "development_zero_replay_counts": counts,
    "scope": "aggregate environment only; ACN-Sim adapter not yet validated",
}
output = Path(sys.argv[1])
output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                  encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
