import numpy as np
import pandas as pd

TARIFF_ID = "experimental_daily_tou_v1"
TIMEZONE = "America/Los_Angeles"

def attach_tariff(frame, time_column="start_utc"):
    result = frame.copy()
    utc = pd.to_datetime(result[time_column], utc=True, errors="raise")
    local = utc.dt.tz_convert(TIMEZONE)
    hour = local.dt.hour + local.dt.minute / 60
    result["price_USD_per_kWh"] = np.select(
        [hour < 8, hour < 16, hour < 21],
        [0.12, 0.24, 0.40],
        default=0.24,
    )
    result["tariff_id"] = TARIFF_ID
    return result
