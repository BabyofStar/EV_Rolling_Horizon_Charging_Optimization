"""Aggregate event replay. Controllers receive view() copies, never this environment."""
import math
import pandas as pd

EPS = 1e-9

def timestamp(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError(f"Expected timezone-aware timestamp: {value}")
    return value.tz_convert("UTC")

def number(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Expected finite nonnegative number: {value}")
    return value

class AggregateEnvironment:
    def __init__(self, sessions, events, updates, start,
                 capacity_kw=50.0, pmax_kw=6.656, executor=None):
        self._executor = executor
        self.now = timestamp(start)
        if self.now != self.now.floor("15min"):
            raise ValueError("Start must be on the 15-minute UTC grid")
        self.capacity = number(capacity_kw)
        self.pmax = number(pmax_kw)
        self._sessions = {}
        self._events = []
        self._index = 0
        self._active = {}
        self.ledger = []       # Environment/evaluator output; not controller input.
        self.completed = []
        self.update_audit = []

        for row in sessions.to_dict("records"):
            sid = str(row["sessionID"])
            if sid in self._sessions:
                raise ValueError(f"Duplicate session: {sid}")
            cohort = row["cohort"]
            if cohort not in {"main_at_arrival", "supplement_after_submission"}:
                raise ValueError(f"Unknown cohort: {cohort}")
            record = {
                "id": sid, "station_id": str(row["stationID"]),
                "arrival": timestamp(row["connectionTime"]),
                "cohort": cohort, "target_kwh": None,
                "expected_departure": None,
            }
            if cohort == "main_at_arrival":
                record["target_kwh"] = number(row["initial_kWhRequested"])
                record["expected_departure"] = timestamp(
                    row["initial_requestedDeparture"])
            if record["arrival"] < self.now:
                raise ValueError("Warm-start sessions require an explicit policy")
            self._sessions[sid] = record

        # Environment events are filtered by the selected official sessions.
        boundaries = {sid: {} for sid in self._sessions}
        for row in events.to_dict("records"):
            sid = str(row["sessionID"])
            if sid not in self._sessions:
                continue
            kind = row["event_type"]
            if kind not in {"arrival", "departure"}:
                raise ValueError(f"Unknown event type: {kind}")
            if kind in boundaries[sid]:
                raise ValueError(f"Duplicate {kind}: {sid}")
            when = timestamp(row["event_time"])
            boundaries[sid][kind] = when
            priority = 0 if kind == "departure" else 1
            self._events.append((when, priority, 0, sid, kind, None))

        for sid, bounds in boundaries.items():
            if set(bounds) != {"arrival", "departure"}:
                raise ValueError(f"Missing boundary event: {sid}")
            if bounds["arrival"] != self._sessions[sid]["arrival"]:
                raise ValueError(f"Arrival mismatch: {sid}")
            if bounds["departure"] <= bounds["arrival"]:
                raise ValueError(f"Nonpositive connection duration: {sid}")

        seen_updates = set()
        for row in updates.to_dict("records"):
            sid = str(row["sessionID"])
            if sid not in self._sessions:
                continue
            if self._sessions[sid]["cohort"] != "supplement_after_submission":
                continue                    # Main-cohort requests stay frozen.
            if str(row["valid"]).strip().lower() not in {"true", "1", "1.0"}:
                continue
            when = timestamp(row["modifiedAt"])
            order = int(row["input_order"])
            key = (sid, when, order)
            if key in seen_updates:
                raise ValueError(f"Duplicate update key: {key}")
            seen_updates.add(key)
            if when < boundaries[sid]["arrival"]:
                raise ValueError(f"Input release before arrival: {sid}")
            # Whitelist: disconnectTime and other future fields are discarded.
            payload = {
                "target_kwh": number(row["kWhRequested"]),
                "expected_departure": timestamp(row["requestedDeparture"]),
            }
            self._events.append((when, 2, order, sid, "input_update", payload))

        self._events.sort(key=lambda event: event[:4])
        self._release()

    def _release(self):
        while self._index < len(self._events):
            event = self._events[self._index]
            when, _, _, sid, kind, payload = event
            if when > self.now:
                break
            self._index += 1
            if kind == "departure":
                state = self._active.pop(sid, None)
                if state is not None:
                    self.completed.append(dict(state, actual_departure=when))
            elif kind == "arrival":
                record = self._sessions[sid]
                if any(s["station_id"] == record["station_id"]
                       for s in self._active.values()):
                    raise ValueError(f"Overlapping EVSE occupancy: {sid}")
                self._active[sid] = dict(record, delivered_kwh=0.0)
            elif sid in self._active:
                self._active[sid].update(payload)
                self.update_audit.append({
                    "time": when, "id": sid, **payload,
                    "delivered_kwh": self._active[sid]["delivered_kwh"],
                })
            # An input at/after actual departure cannot reactivate the session.

        if self._executor is not None:
            self._executor.sync(self.view())

    def view(self):
        """Only already-arrived, still-connected sessions and released inputs."""
        cars = []
        for sid in sorted(self._active):
            state = self._active[sid]
            target = state["target_kwh"]
            delivered = state["delivered_kwh"]
            cars.append({
                **state,
                "pmax_kw": self.pmax,
                "remaining_kwh": (
                    None if target is None else max(target - delivered, 0.0)),
                "irreversible_excess_kwh": (
                    0.0 if target is None else max(delivered - target, 0.0)),
            })
        return {"time": self.now, "capacity_kw": self.capacity,
                "vehicles": tuple(cars)}

    def _integrate(self, boundary, command):
        while self.now < boundary:
            powers = {}
            cut = boundary
            for sid, state in self._active.items():
                target = state["target_kwh"]
                remaining = (0.0 if target is None else
                             max(target - state["delivered_kwh"], 0.0))
                power = command.get(sid, 0.0) if remaining > EPS else 0.0
                if power > 0:
                    powers[sid] = power
                    # Construct completion time only inside this event interval.
                    interval_hours = (
                        (boundary - self.now).value / 3_600_000_000_000)
                    if remaining < power * interval_hours:
                        until_full = max(
                            pd.Timedelta(hours=remaining / power),
                            pd.Timedelta(nanoseconds=1))
                        cut = min(cut, self.now + until_full)

            hours = (cut - self.now).value / 3_600_000_000_000
            if hours <= 0:
                raise RuntimeError("Nonpositive integration interval")
            executed = (powers if self._executor is None else
                        self._executor.execute(self.view(), powers, hours))
            if set(executed) - set(powers):
                raise ValueError('Unexpected executor session')
            executed = {sid: number(p) for sid, p in executed.items()}
            if any(p > powers[sid] + 1e-8
                   for sid, p in executed.items()):
                raise ValueError('Actual power exceeds requested power')
            energy = {}
            actual_power = {}
            for sid, power in executed.items():
                state = self._active[sid]
                remaining = max(
                    state["target_kwh"] - state["delivered_kwh"], 0.0)
                amount = min(power * hours, remaining)
                state["delivered_kwh"] += amount
                energy[sid] = amount
                actual_power[sid] = amount / hours

            self.ledger.append({
                "start": self.now, "end": cut,
                "power_kw": actual_power,
                "energy_kwh": energy,
                "total_kw": sum(actual_power.values()),
            })
            self.now = cut

    def step(self, command):
        """Hold this first-slot command for 15 minutes; events can stop power."""
        command = {str(sid): number(power) for sid, power in command.items()}
        for sid, power in command.items():
            if sid not in self._active:
                raise ValueError(f"Command for a nonactive session: {sid}")
            if power > self.pmax + EPS:
                raise ValueError(f"Device power exceeded: {sid}")
            if power > 0 and self._active[sid]["target_kwh"] is None:
                raise ValueError(f"No released request: {sid}")
        if sum(command.values()) > self.capacity + EPS:
            raise ValueError("Station capacity exceeded")

        end = self.now + pd.Timedelta(minutes=15)
        while self.now < end:
            next_event = (
                self._events[self._index][0]
                if self._index < len(self._events) else end)
            boundary = min(end, next_event)
            if boundary > self.now:
                self._integrate(boundary, command)
            self._release()
        return self.view()
