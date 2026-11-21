from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import evaluate_run, summarize_decisions

T = pd.Timestamp("2019-05-06T07:00:00Z")
passed = []
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    passed.append(name)

def fixture(powers=(6, 0), target=1.5, start=T):
    end = start + pd.Timedelta(minutes=15 * len(powers))
    station, vehicle = [], []
    for j, power in enumerate(powers):
        a = start + pd.Timedelta(minutes=15 * j)
        b = a + pd.Timedelta(minutes=15)
        station.append({"start": a, "end": b, "total_kw": power,
                        "energy_kwh": power * .25})
        if power:
            vehicle.append({"sessionID": "a", "start": a, "end": b,
                            "actual_kw": power, "energy_kwh": power * .25})
    sessions = [{
        "sessionID": "a", "connectionTime": start, "disconnectTime": end,
        "evaluation_target_kwh": target, "pmax_kw": 10,
    }]
    return pd.DataFrame(station), pd.DataFrame(vehicle), pd.DataFrame(sessions), start, end

def evaluate(args, capacity=10):
    return evaluate_run(*args, price_at=lambda t: .2, capacity_kw=capacity)

s, daily, vehicles = evaluate(fixture())
check("hand-calculated cost and energy",
      abs(s["cost_USD"] - .3) < 1e-10 and abs(s["delivered_kwh"] - 1.5) < 1e-10)
check("zero-load segments retained in peak-valley",
      daily.iloc[0]["peak_kW"] == 6 and daily.iloc[0]["peak_valley_kW"] == 6)
check("fulfilled request service metrics",
      s["energy_satisfaction"] == 1 and s["uncompleted_sessions"] == 0)
check("unspecified expected checkpoint remains NA",
      s["expected_departure_uncompleted_rate"] is None and
      s["expected_checkpoint_excluded_sessions"] == 1)

args = list(fixture(powers=(0, 6)))
args[2]["first_expected_departure"] = T + pd.Timedelta(minutes=15)
args[2]["expected_checkpoint_target_kwh"] = 1.5
s, _, _ = evaluate(args)
check("expected and actual departure indicators remain separate",
      s["actual_departure_uncompleted_rate"] == 0 and
      s["expected_departure_uncompleted_rate"] == 1)

args = list(fixture(powers=(8,), target=1))
extra = args[2].iloc[0].to_dict()
extra["sessionID"] = "b"
args[2] = pd.concat([args[2], pd.DataFrame([extra])], ignore_index=True)
s, _, _ = evaluate(args)
check("excess cannot hide another vehicle shortfall",
      s["energy_satisfaction"] == .5 and s["unmet_kwh"] == 1 and
      s["excess_kwh"] == 1)

args = list(fixture(powers=(0,), target=0))
s, _, _ = evaluate(args)
check("zero total demand satisfaction is NA", s["energy_satisfaction"] is None)

args = list(fixture(powers=(0,)))
args[2] = args[2].iloc[:0]
s, _, _ = evaluate(args)
check("empty fleet rates are NA",
      s["actual_departure_uncompleted_rate"] is None and s["sessions"] == 0)

a, cut, end = T, T + pd.Timedelta(minutes=5), T + pd.Timedelta(minutes=15)
station = pd.DataFrame([
    {"start": a, "end": cut, "total_kw": 6},
    {"start": cut, "end": end, "total_kw": 0},
])
trace = pd.DataFrame([{
    "sessionID": "a", "start": a, "end": cut, "actual_kw": 6, "energy_kwh": .5}])
sessions = fixture(powers=(0,), target=.5)[2]
s, _, _ = evaluate((station, trace, sessions, a, end))
check("instantaneous and grid-average peaks distinguished",
      s["max_peak_kW"] == 6 and abs(s["max_grid_average_peak_kW"] - 2) < 1e-10)

late = pd.Timestamp("2019-05-06T23:45:00", tz="America/Los_Angeles").tz_convert("UTC")
s, daily, _ = evaluate(fixture(start=late))
check("local midnight assigns energy to correct day",
      len(daily) == 2 and abs(daily.iloc[0]["energy_kwh"] - 1.5) < 1e-10 and
      daily.iloc[1]["energy_kwh"] == 0)

for date, duration in [("2019-03-10", 23), ("2019-11-03", 25)]:
    local = pd.Timestamp(date, tz="America/Los_Angeles")
    a = local.tz_convert("UTC")
    b = (local + pd.DateOffset(days=1)).tz_convert("UTC")
    station = pd.DataFrame([{"start": a, "end": b, "total_kw": 6}])
    trace = pd.DataFrame([{
        "sessionID": "a", "start": a, "end": b,
        "actual_kw": 6, "energy_kwh": 6 * duration}])
    sessions = pd.DataFrame([{
        "sessionID": "a", "connectionTime": a, "disconnectTime": b,
        "evaluation_target_kwh": 500, "pmax_kw": 10}])
    s, daily, _ = evaluate((station, trace, sessions, a, b))
    check(f"DST {duration}-hour daily integration",
          abs(s["delivered_kwh"] - 6 * duration) < 1e-8 and
          len(daily) == 1 and daily.iloc[0]["grid_slots"] == duration * 4)

args = list(fixture(powers=(7,)))
args[2]["pmax_kw"] = 6
s, _, _ = evaluate(args, capacity=6)
check("constraint violations are reported",
      s["station_violation_segments"] == 1 and
      s["device_violation_segments"] == 1 and s["max_power_violation_kW"] == 1)

args = list(fixture())
args[0] = args[0].iloc[:1]
try:
    evaluate(args)
except ValueError:
    check("missing zero-load coverage rejected", True)
else:
    raise AssertionError("Incomplete coverage accepted")

timing = summarize_decisions(pd.DataFrame([
    {"mode": "empty", "total_seconds": .01, "fallback": "False"},
    {"mode": "lp", "total_seconds": 2, "fallback": "False"},
    {"mode": "fallback", "total_seconds": 3, "fallback": "True"},
]))
check("timing excludes empty steps and parses CSV booleans",
      timing["fallback_count"] == 1 and timing["active_mean_seconds"] == 2.5)

result = {
    "status": "passed", "check_count": len(passed), "checks": passed,
    "scope": "metrics hand-calculation checks; four-method integration pending",
}
Path(sys.argv[1]).write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
