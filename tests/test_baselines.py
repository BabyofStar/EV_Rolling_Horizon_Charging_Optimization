from pathlib import Path
from copy import deepcopy
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines import uncontrolled, tou_greedy
from src.simulator import AggregateEnvironment
from src.acnsim_adapter import ACNSimExecutor

T = pd.Timestamp("2019-05-06T07:00:00Z")
passed = []

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    passed.append(name)

def car(sid, demand=1.5, deadline=60, arrival=None):
    return {
        "id": sid, "station_id": sid,
        "arrival": T if arrival is None else arrival,
        "target_kwh": demand, "delivered_kwh": 0.0,
        "remaining_kwh": demand, "pmax_kw": 6.0,
        "expected_departure": T + pd.Timedelta(minutes=deadline),
    }

def view(cars=(), capacity=6.0):
    return {"time": T, "capacity_kw": capacity, "vehicles": tuple(cars)}

check("empty state", uncontrolled(view()) == {} and
      tou_greedy(view(), lambda t: 1) == {})

v = view([car("b"), car("a")])
check("Uncontrolled deterministic ID tie-break",
      uncontrolled(v) == {"a": 6.0})
older = car("b", arrival=T - pd.Timedelta(minutes=15))
check("Uncontrolled arrival order precedes ID",
      uncontrolled(view([car("a"), older])) == {"b": 6.0})

check("remaining energy caps first-slot power",
      uncontrolled(view([car("a", demand=0.25)])) == {"a": 1.0})
unknown = car("unknown")
unknown.update(target_kwh=None, remaining_kwh=None)
check("unreleased request skipped",
      uncontrolled(view([unknown])) == {} and
      tou_greedy(view([unknown]), lambda t: 1) == {})

before = deepcopy(v)
uncontrolled(v)
tou_greedy(v, lambda t: 1)
check("controllers do not mutate visible state", v == before)

cheap_later = lambda t: 1.0 if t == T else 0.1
check("TOU waits for cheaper available slot",
      tou_greedy(view([car("a")]), cheap_later, base_hours=1) == {})
check("TOU equal prices prefer earlier slot",
      tou_greedy(view([car("a")]), lambda t: 1, base_hours=1) == {"a": 6.0})
check("TOU respects imminent expected departure",
      tou_greedy(view([car("a", deadline=15)]),
                 cheap_later, base_hours=1) == {"a": 6.0})
check("TOU prioritizes earlier-departing vehicle",
      tou_greedy(view([car("a", deadline=30), car("b", deadline=15)]),
                 lambda t: 1, base_hours=1) == {"b": 6.0})
check("TOU expired estimate extends by one slot",
      tou_greedy(view([car("a", deadline=-15)]),
                 lambda t: 1, base_hours=1) == {"a": 6.0})
check("TOU future partial slot is unavailable",
      tou_greedy(view([car("a", deadline=10)]),
                 lambda t: 1, base_hours=1) == {})

extended = tou_greedy(
    view([car("a", deadline=30 * 60)]),
    lambda t: 0.1 if t >= T + pd.Timedelta(hours=29) else 1,
    details=True)
check("TOU extends horizon using visible estimate",
      extended["effective_horizon_hours"] == 30 and
      extended["command"] == {} and
      abs(sum(extended["plan_kw"]["a"]) * 0.25 - 1.5) < 1e-8)

r = tou_greedy(
    view([car("a", demand=10, deadline=30),
          car("b", demand=10, deadline=30)]),
    lambda t: 1, base_hours=1, details=True)
check("TOU all future slots respect shared capacity",
      all(sum(powers[j] for powers in r["plan_kw"].values()) <= 6 + 1e-8
          for j in range(len(r["slots"]))))
check("TOU keeps infeasible residual demand",
      abs(sum(r["forecast_unmet_kwh"].values()) - 17.0) < 1e-8)
check("TOU planned energy and residual conserve each demand",
      all(abs(sum(r["plan_kw"][sid]) * 0.25 +
                  r["forecast_unmet_kwh"][sid] - 10) < 1e-8
          for sid in ["a", "b"]))

reversed_v = view(list(reversed(v["vehicles"])))
check("TOU deterministic under input permutation",
      tou_greedy(v, lambda t: 1, details=True) ==
      tou_greedy(reversed_v, lambda t: 1, details=True))

# Run both controllers through the accepted ACN-Sim execution environment.
for method in ["Uncontrolled", "TOU Greedy"]:
    sessions, events = [], []
    for sid in ["a", "b"]:
        sessions.append({
            "sessionID": sid, "stationID": sid,
            "connectionTime": T, "cohort": "main_at_arrival",
            "initial_kWhRequested": 1.5,
            "initial_requestedDeparture": T + pd.Timedelta(minutes=30),
        })
        for kind, minute in [("arrival", 0), ("departure", 60)]:
            events.append({
                "sessionID": sid, "event_type": kind,
                "event_time": T + pd.Timedelta(minutes=minute),
            })
    executor = ACNSimExecutor(["a", "b"], capacity_kw=6)
    e = AggregateEnvironment(
        pd.DataFrame(sessions), pd.DataFrame(events), pd.DataFrame(),
        start=T, capacity_kw=6, executor=executor)
    for _ in range(4):
        state = e.view()
        command = (
            uncontrolled(state) if method == "Uncontrolled" else
            tou_greedy(state, lambda t: 0.12))
        e.step(command)
    check(method + ": ACN-Sim integrated delivery",
          len(e.completed) == 2 and
          all(abs(s["delivered_kwh"] - 1.5) < 1e-8 for s in e.completed) and
          all(row["total_kw"] <= 6 + 1e-8 for row in e.ledger))

result = {
    "status": "passed", "check_count": len(passed), "checks": passed,
    "scope": "baseline synthetic checks; real development replay still pending",
}
Path(sys.argv[1]).write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
