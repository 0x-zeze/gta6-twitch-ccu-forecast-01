# GTA VI Twitch CCU Forecast — gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast how many people watch Grand Theft Auto VI live on Twitch in each hour
of its release day, 2026-11-19 (UTC), after the first hour's print arrives.
One hour in, the program says how many will be watching in each of hours 2–24,
and how sure it is — five quantiles (5%, 25%, 50%, 75%, 95%) an hour.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no install step, no network. In the
grader's container it reads the case at `/input/<id>.in` and writes
`/output/<id>.out`. Deterministic: two runs write the same bytes. About 1
second per case, far inside the 8-second cap.

## Model

The score is a mean pinball loss on the log scale, so the forecast is built in
log space: for hour `h`, with `p1` the hour-1 print,

```
ln q_tau(h) = ln p1 + m_h + z_tau * s_h
```

`m_h`, the median log-ratio `ln(y_h / p1)`, comes from a per-hour regression
fitted on past big game launches' release days (the case's own
`comparables`, SullyGnome hourly Twitch category averages):

```
ln(y_h / p1) = a_h + b_h * x,   x = ln(p1) - ln(150_000), clamped to [-1.76, 1.76]
```

Only rolling-midnight console launches are used — GTA VI unlocks at local
midnight, country by country, so hour 1 already holds New Zealand / Australia
/ Asia / Europe and the US unlocks at 05:00Z and 08:00Z. The slope `b_h` is
shrunk by 0.8 and clamped to `[-1, 0]`: the data say the bigger hour 1 already
is, the smaller the later peak relative to it (corr(ln p1, ln peak/p1) = -0.76
across the 12 qualifying launches; ELDEN RING at 875k and Cyberpunk 2077 at
852k peaked at 1.00x and 1.26x hour 1, while launches under 60k peaked at
2.2x–12x). `s_h` is the regression's residual spread per hour, smoothed and
floored at 0.35 log units. If the case carries too few usable comparables,
the program falls back to the same regression fitted offline on the kit's 12
qualifying launches (the `FALLBACK` table in `main.py`).

Leave-one-out over the 12 qualifying launches, scored exactly as the contest
scores: **loss 0.239, typical miss 41.6%, 90% coverage 0.87**, against the
house baseline's **0.357 / 71.7% / 0.79**. It also beats the baseline on all
28 usable comparables (any unlock pattern): 0.268 vs 0.351.

## Files

- `main.py` — the whole entry: reads the case, fits the regression from its
  comparables (or the embedded fallback), writes the 23-row forecast.
- `LICENSE` — MIT.
- `.makefun/owner.json` — the payout wallet.
