from pathlib import Path
from io import StringIO
import json
import sys
import numpy as np
import pandas as pd

passed = []
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    passed.append(name)

def parse(values):
    return pd.to_datetime(
        pd.Series(values), utc=True, format="ISO8601", errors="raise")

def same_nanoseconds(raw, parsed):
    expected = np.array([pd.Timestamp(v).value for v in raw], dtype="int64")
    return np.array_equal(expected, parsed.array.asi8)

raw = [
    "2019-05-06 17:44:59+00:00",
    "2019-05-06 17:44:59.999999999+00:00",
    "2019-05-06T17:45:00.000001+00:00",
]
check("mixed precision preserves exact nanoseconds",
      same_nanoseconds(raw, parse(raw)))

repeated_hour = parse([
    "2019-11-03T01:30:00-07:00",
    "2019-11-03T01:30:00-08:00",
])
check("DST repeated hour remains two distinct instants",
      (repeated_hour.iloc[1] - repeated_hour.iloc[0]).value
      == 3_600_000_000_000)

frame = pd.read_csv(sys.argv[2], usecols=["start", "end"])
for column in ["start", "end"]:
    parsed = parse(frame[column])
    check(f"actual CSV {column} preserves nanoseconds",
          same_nanoseconds(frame[column], parsed))

parsed = parse(frame["start"])
buffer = StringIO()
pd.DataFrame({"stamp": parsed}).to_csv(buffer, index=False)
buffer.seek(0)
roundtrip = parse(pd.read_csv(buffer)["stamp"])
check("CSV roundtrip preserves nanoseconds",
      np.array_equal(parsed.array.asi8, roundtrip.array.asi8))

try:
    parse(["not-a-timestamp"])
except ValueError:
    check("invalid timestamp still raises", True)
else:
    raise AssertionError("Invalid timestamp silently accepted")

result = {"status": "passed", "check_count": len(passed), "checks": passed}
Path(sys.argv[1]).write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
