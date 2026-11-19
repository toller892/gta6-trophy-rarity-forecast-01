[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/toller892/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/toller892/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)

# Week-one trophy earn-rate forecast (Exophase)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its
earn rate on Exophase at 2026-11-26T00:00Z — as five quantiles
(5 / 25 / 50 / 75 / 95%) per trophy. Built for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## How it works

`main.py` (Python 3.13, standard library only, deterministic, no
network) reads one case file per run from `$MAKEFUN_INPUT_DIR` (or
`/input`) and writes one forecast per case to `$MAKEFUN_OUTPUT_DIR`
(or `/output`).

An earn rate is earners divided by players tracked, and Exophase only
refreshes its counts every so often — on launch night the page sat
unchanged at 638 players tracked, then refreshed to 3,868 players with
almost every rate lower. A flat random walk misses that completely, so
this program forecasts in counts:

1. **Player growth.** When the case history already shows growth, log
   players is fitted against log(1 + hours/24) over the recent reads
   and extrapolated (damped and capped). When the history is perfectly
   flat, it falls back on the stated launch-night prior: one refresh
   before the target, multiplying tracked players about 6×.
2. **Cohort earning.** Players who join in an hour earn each trophy by
   the target with probability `u * (1 - exp(-H / tau))`, where `H` is
   their effective play hours and `(u, tau)` come from the trophy's
   class: story chapters climb fast and high, easy activities climb
   moderately, grinds and the platinum barely move in week one. The
   launch-night rate ranks trophies inside their class and nudges `u`
   and `tau`. The already-tracked cohort keeps its observed earners
   and adds only the model's predicted increment.
3. **The 100% trophy** (earned by starting the game) is kept at ~100%.

Quantiles sit on the logit scale around the modelled median, with a
spread that widens with the horizon and with class uncertainty — the
score is mean pinball loss on logits, so an honest wide band beats a
confident narrow one when the median is uncertain.

## Local check

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

On the practice cases the medians land at roughly 0.17–0.52× the last
read rates (dilution-dominated over 90 minutes, as the refresh
implies), with roughly 0.2–0.4 pinball loss against the plausible
refreshed answers — against about 1.51 for the house baseline. Runs in
about 0.05 s per case, byte-identical on repeat runs.

## License

MIT — see [LICENSE](LICENSE).
