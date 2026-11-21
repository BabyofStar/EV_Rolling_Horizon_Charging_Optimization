"""ACN-Sim component backend for the aggregate continuous-power scenario.

Event timing remains in AggregateEnvironment.
Synthetic Battery capacity is bookkeeping, not inferred physical capacity/SOC.
All EVSEs use an artificial common phase to express aggregate kW capacity.
Original Caltech branch/phase constraints are not represented here.
"""
import math
import numpy as np
from acnportal.acnsim.models import Battery, EV, EVSE
from acnportal.acnsim.network import ChargingNetwork, Current

class ACNSimExecutor:
    def __init__(self, station_ids, capacity_kw=50.0, voltage=208.0,
                 max_current=32.0, response_max_kw=None):
        self.ids = sorted(set(str(s) for s in station_ids))
        self.capacity_kw = float(capacity_kw)
        self.voltage = float(voltage)
        self.max_current = float(max_current)
        for value in [self.capacity_kw, self.voltage, self.max_current]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Capacity, voltage and current must be positive")
        if not self.ids:
            raise ValueError("At least one EVSE is required")
        nominal = self.voltage * self.max_current / 1000
        self.response_max_kw = (
            nominal if response_max_kw is None else float(response_max_kw))
        if (not math.isfinite(self.response_max_kw) or
                not 0 < self.response_max_kw <= nominal):
            raise ValueError("Invalid synthetic response limit")

        self.network = ChargingNetwork()
        for station in self.ids:
            self.network.register_evse(
                EVSE(station, max_rate=self.max_current, min_rate=0.0),
                voltage=self.voltage, phase_angle=0.0)
        self.network.add_constraint(
            Current({station: 1.0 for station in self.ids}),
            self.capacity_kw * 1000 / self.voltage,
            name="aggregate_experimental_capacity")

        self._connected = {}
        self.evs = {}       # Environment/evaluator access only.
        self.records = []

    def sync(self, view):
        vehicles = view["vehicles"]
        desired = {v["station_id"]: v["id"] for v in vehicles}
        if len(desired) != len(vehicles):
            raise ValueError("Multiple sessions occupy the same EVSE")
        if set(desired) - set(self.ids):
            raise ValueError("Unregistered EVSE")

        for station, sid in list(self._connected.items()):
            if desired.get(station) != sid:
                self.network.unplug(station, session_id=sid)
                del self._connected[station]

        for station, sid in desired.items():
            if station in self._connected:
                continue
            if sid in self.evs:
                raise ValueError("Unexpected repeated session connection")
            # Bookkeeping reservoir; demand limits are enforced by the environment.
            reservoir = 1_000_000.0
            battery = Battery(
                capacity=reservoir, init_charge=0.0,
                max_power=self.response_max_kw)
            ev = EV(
                arrival=0, departure=1_000_000_000,
                requested_energy=reservoir,
                station_id=station, session_id=sid, battery=battery)
            self.network.plugin(ev)
            self.evs[sid] = ev
            self._connected[station] = sid

    def execute(self, view, powers_kw, hours):
        self.sync(view)
        if not math.isfinite(hours) or hours <= 0:
            raise ValueError("Invalid execution duration")
        station_for = {v["id"]: v["station_id"] for v in view["vehicles"]}
        pilots = np.zeros((len(self.ids), 1))
        index = {station: i for i, station in enumerate(self.ids)}

        for sid, value in powers_kw.items():
            power = float(value)
            if sid not in station_for:
                raise ValueError("Command for a nonactive session")
            if not math.isfinite(power) or power < 0:
                raise ValueError("Invalid power")
            current = power * 1000 / self.voltage
            if current > self.max_current + 1e-8:
                raise ValueError("EVSE current exceeded")
            pilots[index[station_for[sid]], 0] = current

        # update_pilots itself does not enforce network constraints.
        if not self.network.is_feasible(pilots):
            raise ValueError("ACN-Sim network rejected the requested pilots")

        self.network.update_pilots(pilots, i=0, period=hours * 60)
        actual = {
            sid: float(self.evs[sid].current_charging_rate)
                 * self.voltage / 1000
            for sid in powers_kw
        }
        self.records.append({
            "time": view["time"], "hours": hours,
            "requested_kw": dict(powers_kw), "actual_kw": dict(actual),
            "network_feasible": True,
        })
        return actual
