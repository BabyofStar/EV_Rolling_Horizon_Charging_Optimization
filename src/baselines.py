"""Online rule baselines. No file access or environment truth is used."""
import math
import pandas as pd

EPS = 1e-9

def _number(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Expected finite nonnegative value")
    return value

def _time(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")
    return value.tz_convert("UTC")

def _state(view, interval_minutes):
    if interval_minutes <= 0 or int(interval_minutes) != interval_minutes:
        raise ValueError("Positive integer interval required")
    delta = pd.Timedelta(minutes=int(interval_minutes))
    now = _time(view["time"])
    if now.value % delta.value:
        raise ValueError("Decision time is not on the control grid")
    capacity = _number(view["capacity_kw"])
    cars, ids, stations = [], set(), set()
    for original in view["vehicles"]:
        sid = original["id"]
        station = original["station_id"]
        if sid in ids or station in stations:
            raise ValueError("Duplicate session or EVSE occupancy")
        ids.add(sid)
        stations.add(station)
        arrival = _time(original["arrival"])
        if arrival > now:
            raise ValueError("Future arrival supplied to online controller")
        if original["target_kwh"] is None:
            continue
        target = _number(original["target_kwh"])
        delivered = _number(original["delivered_kwh"])
        remaining = max(target - delivered, 0.0)
        if abs(remaining - _number(original["remaining_kwh"])) > 1e-8:
            raise ValueError("Inconsistent remaining demand")
        if remaining <= EPS or arrival.ceil(delta) > now:
            continue
        cars.append({
            **original, "arrival": arrival, "remaining_kwh": remaining,
            "pmax_kw": _number(original["pmax_kw"]),
        })
    return now, delta, capacity, cars

def uncontrolled(view, interval_minutes=15, details=False):
    """Capacity-limited immediate charging; arrival then session ID order.

    Raw requests are device maxima. Allocation also caps each first-slot
    energy by the current remaining demand.
    """
    now, delta, capacity, cars = _state(view, interval_minutes)
    dt = delta.value / 3_600_000_000_000
    command = {}
    requests = {}
    left = capacity
    remaining_after = {}
    for car in sorted(cars, key=lambda v: (v["arrival"], v["id"])):
        sid = car["id"]
        requests[sid] = car["pmax_kw"]
        power = min(car["pmax_kw"], left, car["remaining_kwh"] / dt)
        if power > 0:
            command[sid] = power
            left = max(left - power, 0.0)
        remaining_after[sid] = max(car["remaining_kwh"] - power * dt, 0.0)
    if details:
        return {
            "command": command, "raw_request_kw": requests,
            "slot": now, "remaining_after_first_slot_kwh": remaining_after,
        }
    return command

def tou_greedy(view, price_at, interval_minutes=15,
               base_hours=24.0, details=False):
    """Rebuild a price-sorted plan for currently visible vehicles only.

    Vehicles: expected departure, arrival, ID.
    Slots: price, then earlier time.
    Expired estimates extend by one slot. Future arrivals are not predicted.
    price_at must represent a preannounced tariff calendar.
    """
    now, delta, capacity, cars = _state(view, interval_minutes)
    if not math.isfinite(float(base_hours)) or base_hours <= 0:
        raise ValueError("Positive finite base horizon required")
    dt = delta.value / 3_600_000_000_000

    deadlines = {}
    for car in cars:
        estimate = _time(car["expected_departure"])
        car["expected_departure"] = estimate
        deadlines[car["id"]] = (
            now + delta if estimate <= now else estimate.floor(delta))

    if not cars:
        result = {
            "command": {}, "slots": (), "plan_kw": {},
            "forecast_unmet_kwh": {}, "effective_horizon_hours": 0.0,
        }
        return result if details else {}

    end = max(
        (now + pd.Timedelta(hours=base_hours)).ceil(delta),
        max(deadlines.values()))
    slots = tuple(pd.date_range(now, end, freq=delta, inclusive="left"))
    prices = [_number(price_at(t)) for t in slots]
    capacity_left = [capacity] * len(slots)
    plan, unmet = {}, {}

    for car in sorted(
        cars, key=lambda v: (v["expected_departure"], v["arrival"], v["id"])
    ):
        sid = car["id"]
        powers = [0.0] * len(slots)
        remaining = car["remaining_kwh"]
        available = [
            j for j, t in enumerate(slots)
            if t >= car["arrival"].ceil(delta) and
            t + delta <= deadlines[sid]
        ]
        for j in sorted(available, key=lambda j: (prices[j], slots[j])):
            power = min(car["pmax_kw"], capacity_left[j], remaining / dt)
            powers[j] = power
            capacity_left[j] = max(capacity_left[j] - power, 0.0)
            remaining = max(remaining - power * dt, 0.0)
            if remaining <= EPS:
                break
        plan[sid] = tuple(powers)
        unmet[sid] = remaining

    command = {sid: powers[0] for sid, powers in plan.items() if powers[0] > 0}
    result = {
        "command": command, "slots": slots, "plan_kw": plan,
        "forecast_unmet_kwh": unmet,
        "effective_horizon_hours": (end - now).value / 3_600_000_000_000,
    }
    return result if details else command
