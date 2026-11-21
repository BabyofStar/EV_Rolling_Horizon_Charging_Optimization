from pathlib import Path
import inspect
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.simulator import AggregateEnvironment
from acnportal.acnsim.models import Battery, EV, EVSE
from acnportal.acnsim.network import ChargingNetwork

T = pd.Timestamp("2019-05-06T07:00:00Z")
passed = []

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    passed.append(name)

def make(rows, capacity=10):
    # rows: (id, station, arrival_minutes, departure_minutes, target)
    sessions, events = [], []
    for sid, station, arrival, departure, target in rows:
        sessions.append({
            "sessionID": sid, "stationID": station,
            "connectionTime": T + pd.Timedelta(minutes=arrival),
            "cohort": "main_at_arrival",
            "initial_kWhRequested": target,
            "initial_requestedDeparture": T + pd.Timedelta(hours=2),
        })
        for kind, minute in [("arrival", arrival), ("departure", departure)]:
            events.append({
                "sessionID": sid, "event_type": kind,
                "event_time": T + pd.Timedelta(minutes=minute),
            })
    return AggregateEnvironment(
        pd.DataFrame(sessions), pd.DataFrame(events),
        pd.DataFrame(), start=T, capacity_kw=capacity)

def delivered(environment, sid):
    for state in environment.view()["vehicles"]:
        if state["id"] == sid:
            return state["delivered_kwh"]
    return next(s["delivered_kwh"] for s in environment.completed
                if s["id"] == sid)

e = make([("a", "A", 0, 60, 10), ("b", "B", 0, 60, 10)])
try:
    e.step({"a": 6, "b": 6})
except ValueError:
    check("shared-capacity excess rejected without state change",
          e.now == T and delivered(e, "a") == 0 and delivered(e, "b") == 0)
else:
    raise AssertionError("Shared-capacity violation accepted")

e.step({"a": 5, "b": 5})
check("two vehicles execute within shared capacity",
      abs(delivered(e, "a") - 1.25) < 1e-8 and
      abs(delivered(e, "b") - 1.25) < 1e-8)
check("every executed segment respects station capacity",
      all(row["total_kw"] <= 10 + 1e-8 for row in e.ledger))

e = make([("a", "A", 0, 60, 0.5), ("b", "B", 0, 60, 10)])
e.step({"a": 6, "b": 4})
check("one vehicle completing does not stop another",
      abs(delivered(e, "a") - 0.5) < 1e-8 and
      abs(delivered(e, "b") - 1.0) < 1e-8)
check("target completion preserves realized peak",
      abs(max(row["total_kw"] for row in e.ledger) - 10) < 1e-8)

integral = sum(
    row["total_kw"] * (row["end"] - row["start"]).value
    / 3_600_000_000_000
    for row in e.ledger)
check("multi-vehicle ledger independently conserves energy",
      abs(integral - 1.5) < 1e-8)

e = make([("a", "A", 0, 7.5, 10), ("b", "A", 7.5, 60, 10)])
e.step({"a": 6})
check("same-EVSE handoff does not inherit prior vehicle command",
      abs(delivered(e, "a") - 0.75) < 1e-8 and
      delivered(e, "b") == 0)
e.step({"b": 6})
check("new session charges at next control boundary",
      abs(delivered(e, "b") - 1.5) < 1e-8)

e = make([("a", "A", 0, 60, 1.5e-9)])
e.step({"a": 6})
check("sub-microsecond completion integrates without zero-time loop",
      abs(delivered(e, "a") - 1.5e-9) < 1e-12)

e = make([("a", "A", 0, 60, 0.123456789)])
e.step({"a": 6.656})
check("fractional target completion remains accurate",
      abs(delivered(e, "a") - 0.123456789) < 1e-8)

# Synthetic bookkeeping batteries only: these are not inferred real SOC/capacity.
def device(max_power):
    battery = Battery(capacity=100.0, init_charge=0.0,
                      max_power=max_power)
    ev = EV(arrival=0, departure=100, requested_energy=10.0,
            station_id="test", session_id="synthetic",
            battery=battery)
    evse = EVSE("test", max_rate=32.0, min_rate=0.0)
    evse.plugin(ev)
    return ev, evse

ev, evse = device(6.656)
evse.set_pilot(32.0, 208.0, 15.0)
check("ACN-Sim ideal response: A/V/min to kWh",
      abs(ev.current_charging_rate - 32) < 1e-8 and
      abs(ev.energy_delivered - 1.664) < 1e-8)

ev, evse = device(3.0)
evse.set_pilot(32.0, 208.0, 15.0)
check("ACN-Sim actual delivery reflects device power limit",
      abs(ev.energy_delivered - 0.75) < 1e-8 and
      ev.current_charging_rate < 32)

before = ev.energy_delivered
evse.set_pilot(0.0, 208.0, 15.0)
check("ACN-Sim zero pilot adds zero energy",
      abs(ev.energy_delivered - before) < 1e-10)

# Capture installed-version interfaces needed for the network adapter.
interfaces = {}
for obj in [Battery, EV, EVSE, ChargingNetwork,
            ChargingNetwork.register_evse,
            ChargingNetwork.plugin, ChargingNetwork.unplug,
            ChargingNetwork.update_pilots,
            ChargingNetwork.is_feasible]:
    interfaces[obj.__qualname__] = str(inspect.signature(obj))

network_source = inspect.getsource(ChargingNetwork.update_pilots)
ev_charge_source = inspect.getsource(EV.charge)

output = Path(sys.argv[1])
result = {
    "status": "passed",
    "check_count": len(passed),
    "checks": passed,
    "installed_interfaces": interfaces,
    "scope": "aggregate execution and isolated ACN-Sim device checks; "
             "full network adapter not yet validated",
}
output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                  encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("\n=== Installed ChargingNetwork.update_pilots ===")
print(network_source)
print("\n=== Installed EV.charge ===")
print(ev_charge_source)
