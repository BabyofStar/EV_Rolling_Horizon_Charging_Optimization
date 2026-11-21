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

def run(target, power, departure=60, adapted=False):
    sessions = pd.DataFrame([{
        "sessionID": "a", "stationID": "A",
        "connectionTime": T, "cohort": "main_at_arrival",
        "initial_kWhRequested": target,
        "initial_requestedDeparture": T + pd.Timedelta(hours=1),
    }])
    events = pd.DataFrame([
        {"sessionID": "a", "event_type": "arrival", "event_time": T},
        {"sessionID": "a", "event_type": "departure",
         "event_time": T + pd.Timedelta(minutes=departure)},
    ])
    executor = ACNSimExecutor(["A"], capacity_kw=50) if adapted else None
    e = AggregateEnvironment(
        sessions, events, pd.DataFrame(), start=T, executor=executor)
    e.step({"a": power})
    states = list(e.view()["vehicles"]) + e.completed
    return e, executor, states[0]["delivered_kwh"]

e, _, energy = run(10, 1e-20)
check("tiny positive command avoids duration overflow",
      e.now == T + pd.Timedelta(minutes=15) and
      abs(energy - 2.5e-21) < 1e-25)

# Artificial stress case only; this does not change or filter real requests.
e, _, energy = run(1e12, 6)
check("distant completion never needs a huge Timedelta",
      abs(energy - 1.5) < 1e-8)

e, _, energy = run(10, 6)
check("ordinary noncompletion energy unchanged", abs(energy - 1.5) < 1e-8)

e, _, energy = run(0.5, 6)
check("completion inside interval still stops exactly",
      abs(energy - 0.5) < 1e-8)

e, _, energy = run(1.5, 6)
check("completion exactly at interval end remains correct",
      abs(energy - 1.5) < 1e-8)

e, _, energy = run(10, 1e-20, departure=7.5)
check("tiny command still stops at actual departure",
      not e.view()["vehicles"] and abs(energy - 1.25e-21) < 1e-25)

e, executor, energy = run(10, 1e-20, adapted=True)
check("tiny command conserves ACN-Sim actual feedback",
      abs(energy - 2.5e-21) < 1e-25 and
      abs(energy - executor.evs["a"].energy_delivered) < 1e-25)

result = {"status": "passed", "check_count": len(passed), "checks": passed}
Path(sys.argv[1]).write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
