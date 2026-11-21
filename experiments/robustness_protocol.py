
"""Controlled synthetic errors; original evaluation targets remain separate."""
import hashlib
import math
import pandas as pd

def factor_for(session_id, amplitude, seed):
    amplitude = float(amplitude)
    if not math.isfinite(amplitude) or not 0 <= amplitude < 1:
        raise ValueError("Amplitude must be finite and in [0, 1)")
    if int(seed) != seed or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    payload = f"ev-e4-v1|{int(seed)}|{session_id}".encode("utf-8")
    sign = 1 if hashlib.sha256(payload).digest()[0] & 1 else -1
    return 1.0 + sign * amplitude

def perturb_requests(sessions, releases, amplitude, seed):
    """Copy operational inputs; never alter their release times."""
    selected = sessions.copy(deep=True)
    updates = releases.copy(deep=True)
    factors = {
        str(sid): factor_for(str(sid), amplitude, seed)
        for sid in sessions["sessionID"]
    }
    if float(amplitude) == 0:
        return selected, updates, factors

    main = selected["cohort"].eq("main_at_arrival")
    selected.loc[main, "initial_kWhRequested"] = (
        pd.to_numeric(selected.loc[main, "initial_kWhRequested"], errors="raise")
        * selected.loc[main, "sessionID"].astype(str).map(factors)
    )
    supplemental_ids = set(
        selected.loc[~main, "sessionID"].astype(str)
    )
    valid = updates["valid"].astype(str).str.strip().str.lower().isin(
        ["true", "1", "1.0"]
    )
    mask = updates["sessionID"].astype(str).isin(supplemental_ids) & valid
    updates.loc[mask, "kWhRequested"] = (
        pd.to_numeric(updates.loc[mask, "kWhRequested"], errors="raise")
        * updates.loc[mask, "sessionID"].astype(str).map(factors)
    )
    return selected, updates, factors

def advance_departures(events, minutes):
    """Change only environment departure events; retain every session."""
    minutes = float(minutes)
    if not math.isfinite(minutes) or minutes < 0:
        raise ValueError("Advance must be finite and nonnegative")
    result = events.copy(deep=True)
    parsed = pd.to_datetime(
        events["event_time"], utc=True, format="ISO8601", errors="raise"
    )
    frame = events.copy(deep=True)
    frame["sessionID"] = frame["sessionID"].astype(str)
    frame["event_time"] = parsed
    if frame.duplicated(["sessionID", "event_type"]).any():
        raise ValueError("Duplicate boundary events")
    bounds = frame.pivot(
        index="sessionID", columns="event_type", values="event_time"
    )
    if not {"arrival", "departure"}.issubset(bounds.columns):
        raise ValueError("Missing boundary event")
    if bounds[["arrival", "departure"]].isna().any().any():
        raise ValueError("Incomplete session boundaries")

    shifted, audit = {}, []
    for sid, row in bounds.iterrows():
        arrival, departure = row["arrival"], row["departure"]
        if not departure > arrival:
            raise ValueError("Invalid original duration")
        proposed = departure - pd.Timedelta(minutes=minutes)
        new_departure = max(proposed, arrival + pd.Timedelta(nanoseconds=1))
        shifted[sid] = new_departure
        audit.append({
            "sessionID": sid,
            "original_departure": departure,
            "scenario_departure": new_departure,
            "requested_advance_minutes": minutes,
            "effective_advance_minutes":
                (departure - new_departure).value / 60_000_000_000,
            "clipped_to_positive_duration": proposed <= arrival,
        })
    if minutes > 0:
        result["event_time"] = parsed
        mask = result["event_type"].eq("departure")
        result.loc[mask, "event_time"] = (
            result.loc[mask, "sessionID"].astype(str).map(shifted)
        )
    return result, pd.DataFrame(audit)
