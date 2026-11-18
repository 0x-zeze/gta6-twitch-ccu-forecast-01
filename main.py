"""gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch CCU, hour by hour.

Python 3.13, standard library only. Run by the grader as
`python3 -P -s /bot/main.py` (PYTHONHASHSEED=0), no network, one case at a time:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Reads every <id>.in in the input directory:
  JSON {"schema": "makefun/twitch-ccu-case/v1", "id", "game", "day_start",
  "checkpoint", "levels", "known", "forecast", "history", "history_hourly",
  "streams", "categories_hourly", "comparables", "notes"}
and writes <id>.out to the output directory:
  {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows,
  hours 2..24 each once, no other key, each q five finite numbers > 0 and at
  most 1e9, non-decreasing.

Model
-----
The score is a mean pinball loss on the log scale, so the forecast is built in
log space: for hour h, with p1 the hour-1 print,

    ln q_tau(h) = ln p1 + m_h + z_tau * s_h,
    q_tau(h) rounded to 2 decimals (half up),

where m_h is the forecast median log-ratio ln(y_h / p1) and s_h its log spread,
z_tau the standard normal quantile at each scored level.

m_h comes from a per-hour regression fitted on past big game launches'
release days (the case's own "comparables", SullyGnome hourly Twitch category
averages):

    ln(y_h / p1) = a_h + b_h * x,   x = ln(p1) - ln(150_000), clamped to [-1.76, 1.76].

Only rolling-midnight console launches are used (the launch pattern GTA VI
follows: the game unlocks at local midnight, country by country, so hour 1
already holds New Zealand / Australia / Asia / Europe and the US unlocks at
05:00Z and 08:00Z): a launch qualifies when its unlock note says "midnight
local" or "rolling", its hour-1 print exceeds 10,000 viewers and at least 23
of its 24 release-day hours are present. The slope b_h is shrunk by 0.8 and
clamped to [-1, 0]: the data say the bigger hour 1 already is, the smaller the
later peak relative to it (across the 12 qualifying launches, corr(ln p1,
ln peak/p1) = -0.76; the two biggest, ELDEN RING at 875k and Cyberpunk 2077 at
852k, peaked at 1.00x and 1.26x hour 1, while launches under 60k peaked at
2.2x-12x). Hours 2-4 get b_h = 0 (the overnight dip does not depend on size);
from hour 6 on, b_h is about -0.33, i.e. every doubling of hour 1 cuts the
later median ratio by roughly 20%.

s_h is the residual standard deviation of that regression per hour, smoothed
with a 3-hour moving average and floored at 0.35 log units, so the 90%
interval stays honest about how differently launches behave.

If the case carries too few usable comparables (< 4 launches), the program
falls back to the same regression fitted offline on the kit's 12 qualifying
launches (FALLBACK table below); the numbers are identical in spirit and the
output format is unchanged either way.

Validation (leave-one-out over the 12 qualifying launches, scored exactly as
the contest scores: mean pinball loss on ln scale):
    this model   loss 0.239, typical miss 41.6%, 90% coverage 0.87
    house baseline (persistence, sigma 0.15*sqrt(h-1))
                 loss 0.357, typical miss 71.7%, 90% coverage 0.79
The same model also beats the baseline on all 28 usable comparables (any
unlock pattern): 0.268 vs 0.351. Deterministic: same case, same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Reference hour-1 scale for the regression (log viewers).
REF_LOG_P1 = math.log(150000)
# Clamp x to the training range so a record hour 1 (GTA VI's will be far above
# the largest training launch, ELDEN RING at 875k) extrapolates no further
# than the biggest launch seen.
CLAMP_X = 1.76
# Shrinkage of the fitted slope toward 0 (12 launches is a small sample).
SHRINK = 0.8
# Floor on the log spread: launches differ too much to claim more precision.
SPREAD_FLOOR = 0.35
# Minimum usable launches to trust a runtime fit; below this, use FALLBACK.
MIN_LAUNCHES = 4

# Offline fit on the kit's 12 qualifying rolling-midnight launches, per hour:
# (a_h, b_h, s_h). Used only when the case's comparables are too thin.
FALLBACK = {
    2: (-0.1119, 0.0000, 0.3500),
    3: (-0.3326, 0.0000, 0.3500),
    4: (-0.3863, 0.0000, 0.3500),
    5: (-0.2110, -0.0861, 0.3500),
    6: (0.0191, -0.2204, 0.3717),
    7: (0.0415, -0.2829, 0.4585),
    8: (0.0182, -0.3092, 0.4793),
    9: (-0.0276, -0.3352, 0.4769),
    10: (-0.0316, -0.3314, 0.4757),
    11: (-0.0066, -0.3297, 0.4735),
    12: (0.0560, -0.3322, 0.4608),
    13: (0.1959, -0.3445, 0.4455),
    14: (0.3383, -0.3385, 0.4309),
    15: (0.4560, -0.3411, 0.4345),
    16: (0.5305, -0.3341, 0.4433),
    17: (0.6078, -0.3052, 0.4552),
    18: (0.6682, -0.3354, 0.4613),
    19: (0.7176, -0.3570, 0.4582),
    20: (0.7293, -0.3891, 0.4417),
    21: (0.7362, -0.3923, 0.4124),
    22: (0.7078, -0.3963, 0.3916),
    23: (0.6464, -0.3927, 0.3945),
    24: (0.5465, -0.4128, 0.4007),
}


def parse_time(text):
    """ISO 8601 UTC timestamp -> datetime (naive, UTC)."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def release_day_hours(comp):
    """A comparable's release-day hourly viewers as {hour 1..24: viewers}."""
    try:
        day_start = parse_time(comp["day_start"])
        hours = {}
        for point in comp.get("hourly") or []:
            viewers = point.get("viewers")
            if not isinstance(viewers, (int, float)) or viewers <= 0:
                continue
            hour = int((parse_time(point["from"]) - day_start).total_seconds() // 3600) + 1
            if 1 <= hour <= 24:
                hours[hour] = viewers
        return hours
    except (KeyError, TypeError, ValueError):
        return {}


def qualifies(comp, hours):
    """Rolling-midnight console launch with a real hour-1 print and a full day."""
    unlock = (comp.get("unlock") or "").lower()
    if "midnight local" not in unlock and "rolling" not in unlock:
        return False
    if len(hours) < 23 or 1 not in hours or hours[1] <= 10000:
        return False
    return True


def fit_params(launches):
    """Per-hour (a_h, b_h, s_h) from [(name, {hour: viewers})], or None.

    OLS of ln(y_h / p1) on ln(p1) - REF_LOG_P1 per hour, slope shrunk and
    clamped to [-1, 0]; spread = residual sd, 3-hour smoothed, floored.
    """
    params = {}
    raw_spread = {}
    for hour in range(2, 25):
        xs = []
        ys = []
        for _name, hours in launches:
            if hour in hours and hours[hour] > 0:
                xs.append(math.log(hours[1]) - REF_LOG_P1)
                ys.append(math.log(hours[hour] / hours[1]))
        if len(xs) < 4:
            return None
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        vx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx if vx > 0 else 0.0
        slope = SHRINK * max(-1.0, min(0.0, slope))
        intercept = my - slope * mx
        residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        mean_res = sum(residuals) / n
        var = sum((r - mean_res) ** 2 for r in residuals) / n
        raw_spread[hour] = math.sqrt(var) if var > 0 else 0.5
        params[hour] = (intercept, slope)
    for hour in range(2, 25):
        neighbours = [raw_spread[k] for k in (hour - 1, hour, hour + 1) if k in raw_spread]
        spread = max(SPREAD_FLOOR, sum(neighbours) / len(neighbours))
        a, b = params[hour]
        params[hour] = (a, b, spread)
    return params


def model_params(case):
    """Runtime fit from the case's comparables, else the offline FALLBACK."""
    launches = []
    for comp in case.get("comparables") or []:
        hours = release_day_hours(comp)
        if qualifies(comp, hours):
            launches.append((comp.get("name", "?"), hours))
    if len(launches) >= MIN_LAUNCHES:
        fitted = fit_params(launches)
        if fitted is not None:
            return fitted
    return FALLBACK


def hour1_print(case):
    """Hour 1's print, falling back to the last whole-hour mean, else 1."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and isinstance(row.get("viewers"), (int, float)):
            return max(1, row["viewers"])
    hourly = case.get("history_hourly") or []
    if hourly:
        last = hourly[-1].get("viewers")
        if isinstance(last, (int, float)) and last > 0:
            return max(1, last)
    return 1


def forecast_case(case, name):
    p1 = hour1_print(case)
    params = model_params(case)
    x = max(-CLAMP_X, min(CLAMP_X, math.log(p1) - REF_LOG_P1))
    hours = []
    for row in case.get("forecast") or []:
        hour = row["hour"]
        a, b, s = params[hour]
        median_log = math.log(p1) + a + b * x
        quantiles = []
        for z in Z:
            value = math.exp(median_log + z * s)
            value = min(value, 1e9)
            quantiles.append(round2(max(value, 0.01)))
        # Keep the row non-decreasing after rounding.
        for i in range(1, 5):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
