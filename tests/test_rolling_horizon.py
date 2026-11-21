from pathlib import Path
from unittest.mock import patch
import json
import sys
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rolling_horizon import (
    RollingHorizonController, record_realized_peaks, edf_command)
from src.simulator import AggregateEnvironment
from src.acnsim_adapter import ACNSimExecutor

OUT = Path(sys.argv[1]).parent / "rolling_cases"
OUT.mkdir()
NORM = yaml.safe_load(
    (ROOT / "configs/normalization.yaml").read_text(encoding="utf-8"))
T = pd.Timestamp("2019-05-06T07:00:00Z")
passed, controllers = [], []

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    passed.append(name)

def car(sid="a", demand=1.5, deadline=30):
    return {
        "id": sid, "station_id": sid, "arrival": T,
        "target_kwh": demand, "delivered_kwh": 0.0,
        "remaining_kwh": demand, "pmax_kw": 6.0,
        "expected_departure": T + pd.Timedelta(minutes=deadline),
    }

def view(cars=()):
    return {"time": T, "capacity_kw": 6.0, "vehicles": tuple(cars)}

cheap_later = lambda t: .4 if t == T else .1
def controller(**kwargs):
    c = RollingHorizonController(
        OUT / f"controller_{len(controllers):03d}",
        kwargs.pop("price_at", cheap_later),
        C_ref=NORM["C_ref"], P_ref=NORM["P_ref"],
        base_hours=kwargs.pop("base_hours", 1), **kwargs)
    controllers.append(c)
    return c

check("uses frozen references",
      abs(NORM["C_ref"] - 2.88) < 1e-12 and
      abs(NORM["P_ref"] - 6.656) < 1e-12)

r = controller().decide(view())
check("empty fleet bypasses LP", r["mode"] == "empty" and not r["command"])

unknown = car()
unknown.update(target_kwh=None, remaining_kwh=None)
r = controller().decide(view([unknown]))
check("unreleased request bypasses LP", r["mode"] == "empty")

r = controller(alpha=1).decide(view([car()]))
check("real LP waits for cheaper slot and executes only first slot",
      r["mode"] == "lp" and not r["command"] and
      r["solve_report"]["status"] == "passed")

r = controller(alpha=1).decide(view([car(deadline=15)]))
check("real LP serves imminent departure",
      r["mode"] == "lp" and
      abs(r["command"].get("a", 0) * .25 - 1.5) <= 1.1e-5)

r = controller(alpha=1).decide(view([car(demand=3, deadline=15)]))
check("service priority preserves unavoidable shortfall",
      r["mode"] == "lp" and
      abs(r["solve_report"]["U_star_kWh"] - 1.5) < 1e-7 and
      r["solve_report"]["U_stage2_kWh"] <= 1.5 + 1.01e-5)

day = str(T.tz_convert("America/Los_Angeles").date())
r = controller(alpha=0).decide(view([car()]), {day: 5.0})
check("real LP respects already-realized daily peak",
      r["mode"] == "lp" and
      r["solve_report"]["peak_by_day_kW"][day] >= 5 - 1e-7)

c = controller(base_hours=24)
case = c.build_case(view([car(deadline=30 * 60)]), {day: 5})
check("visible departure extends effective horizon",
      len(case["slots"]) == 120)
check("new local day does not inherit yesterday peak",
      len(case["past_peak"]) == 2 and
      case["past_peak"][day] == 5 and
      list(case["past_peak"].values())[1] == 0)

v1, v2 = view([car()]), view([car()])
v1["vehicles"][0]["disconnectTime"] = T + pd.Timedelta(hours=1)
v2["vehicles"][0]["disconnectTime"] = T + pd.Timedelta(hours=20)
check("case ignores extraneous future truth",
      c.build_case(v1) == c.build_case(v2))

partial = c.build_case(view([car(deadline=10)]))
expired = c.build_case(view([car(deadline=-15)]))
check("partial estimated slot remains unavailable",
      partial["sessions"][0]["available"] == [])
check("expired estimate extends exactly one slot",
      expired["sessions"][0]["available"] == [0])

with patch("src.rolling_horizon._invoke",
           side_effect=RuntimeError("injected solver failure")):
    r = controller().decide(
        view([car("b", deadline=15), car("a", deadline=30)]))
check("solver failure triggers deterministic feasible EDF",
      r["fallback"] and r["command"] == {"b": 6.0} and
      "injected solver failure" in r["reason"])

r = controller(time_limit_seconds=1e-6).decide(view([car()]))
check("exhausted outer budget triggers feasible fallback",
      r["fallback"] and r["command"] == {"a": 6.0})

peaks = {}
segment = {"start": T, "end": T + pd.Timedelta(minutes=15), "total_kw": 4}
record_realized_peaks(peaks, [segment], segment["end"])
check("peak feedback uses actual executed power", peaks[day] == 4)
try:
    record_realized_peaks({}, [segment], T)
except ValueError:
    check("future feedback rejected", True)
else:
    raise AssertionError("Future feedback accepted")

# Two-step closed-loop execution through the accepted ACN-Sim backend.
sessions = pd.DataFrame([{
    "sessionID": "a", "stationID": "a", "connectionTime": T,
    "cohort": "main_at_arrival", "initial_kWhRequested": 1.5,
    "initial_requestedDeparture": T + pd.Timedelta(minutes=30),
}])
events = pd.DataFrame([
    {"sessionID": "a", "event_type": "arrival", "event_time": T},
    {"sessionID": "a", "event_type": "departure",
     "event_time": T + pd.Timedelta(minutes=30)},
])
executor = ACNSimExecutor(["a"], capacity_kw=6)
e = AggregateEnvironment(
    sessions, events, pd.DataFrame(), start=T,
    capacity_kw=6, executor=executor)
c, peaks, modes = controller(alpha=1), {}, []
for _ in range(2):
    decision = c.decide(e.view(), peaks)
    modes.append(decision["mode"])
    index = len(e.ledger)
    e.step(decision["command"])
    record_realized_peaks(peaks, e.ledger[index:], e.now)
check("RH-LP closed loop uses actual ACN-Sim delivery",
      modes == ["lp", "lp"] and len(e.completed) == 1 and
      abs(e.completed[0]["delivered_kwh"] - 1.5) <= 1.1e-5 and
      abs(executor.evs["a"].energy_delivered -
          e.completed[0]["delivered_kwh"]) < 1e-8)

result = {
    "status": "passed", "check_count": len(passed), "checks": passed,
    "C_ref": NORM["C_ref"], "P_ref": NORM["P_ref"],
    "scope": "RH-LP synthetic validation; development replay pending",
    "timeout_test": "budget guard; not a computation-performance benchmark",
}
Path(sys.argv[1]).write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
