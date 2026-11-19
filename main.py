#!/usr/bin/env python3
"""Week-one forecast of GTA VI PS5 trophy earn rates on Exophase.

Model (written for the make.fun bounty gta6-trophy-rarity-forecast-01).

The earn rate of a trophy is earners / players-tracked.  Exophase only
refreshes its counts every so often, so a read shows the rates as of its
last refresh; between refreshes the page is flat.  When a refresh lands,
the tracked-player count jumps (launch night: 638 -> 3,868) and almost
every rate falls, because the new players have barely played yet.

So this program forecasts in counts, not in rates:

* Project the tracked-player count P(t) to the target.  When the case's
  history already shows growth, fit log P against log(1 + hours/24) over
  the recent reads and extrapolate (damped, capped).  When the history is
  perfectly flat, no growth is visible yet, so fall back on the prior the
  competition states: one refresh before the target, multiplying the
  tracked players by about 6 (the launch-night 638 -> 3,868 jump).
* Simulate earners per trophy with a cohort model.  Players who join in
  an hour earn trophy i by the target with probability
  u_i * (1 - exp(-H / tau_i)), where H is their effective play hours
  (elapsed hours since joining times a play fraction) and (u_i, tau_i)
  come from the trophy's class: story chapters climb fast and high,
  easy activities climb moderately, grinds and the platinum barely move
  in week one.  The launch-night rate ranks trophies inside their class
  and nudges u_i / tau_i.  The cohort already tracked at the last read
  keeps its observed earners and adds only the model's predicted
  increment between the last read and the target.
* The one trophy already at 100% (earned by starting the game) is kept
  at ~100%: every cohort earns it immediately.

Quantiles are placed on the logit scale around the modelled median, with
a spread that widens with the horizon and with the trophy's class
uncertainty -- the pinball loss is scored on logits, so an honest wide
band beats a confident narrow one when the median is uncertain.

Deterministic: pure arithmetic on the case file, no randomness, no I/O
beyond the case inputs and the forecast outputs.  Standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# Fraction of elapsed time the average tracked player actually spends
# playing (launch week, broad player base, not just hardcore players).
PLAY_FRACTION = 0.08

# Launch-night refresh prior: tracked players multiplied by this when a
# refresh lands while the history is still flat (638 -> 3,868).
REFRESH_MULTIPLIER = 6.06

# (ceiling u, timescale tau in effective play hours, extra logit spread)
CLASSES = {
    "opener":        (1.00,   0.3, 0.00),
    "platinum":      (0.002, 160.0, 0.25),
    "completion":    (0.004, 160.0, 0.25),
    "chapter1":      (0.72,  10.0, 0.00),
    "chapter2":      (0.62,  16.0, 0.00),
    "chapter3":      (0.52,  24.0, 0.00),
    "chapter4":      (0.42,  34.0, 0.00),
    "story_final":   (0.28,  48.0, 0.05),
    "story_mission": (0.38,  28.0, 0.05),
    "easy":          (0.45,  14.0, 0.05),
    "medium":        (0.22,  30.0, 0.10),
    "grind":         (0.05,  90.0, 0.15),
}


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def logit(p):
    p = min(max(p, 0.0001), 0.9999)
    return math.log(p / (1.0 - p))


def logistic(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def clamp_rate(r):
    return min(max(r, 0.01), 99.99)


def classify(trophy):
    """Assign a trophy to a difficulty class from its tier/description."""
    tier = trophy.get("tier", "")
    text = ((trophy.get("name", "") + " " +
             trophy.get("description", "")).lower())
    if tier == "platinum":
        return "platinum"
    if "100% completion" in text:
        return "completion"
    if "complete chapter 1" in text:
        return "chapter1"
    if "complete chapter 2" in text:
        return "chapter2"
    if "complete chapter 3" in text:
        return "chapter3"
    if "complete chapter 4" in text:
        return "chapter4"
    if "complete the final mission" in text:
        return "story_final"
    desc = trophy.get("description", "").lower()
    if desc.startswith("complete ") and (
            "mission" in text or "heist" in text or "takedown" in text
            or "operation" in text or "run" in text or "play" in text):
        return "story_mission"
    grind_marks = ("250 stunts", "50 challenges", "100 locations",
                   "card sets", "20 different races", "every location",
                   "every spot", "gold medal in all",
                   "all garage properties", "laundering",
                   "8 mission strands", "unique animal",
                   "win 20 different races")
    if any(m in text for m in grind_marks):
        return "grind"
    easy_marks = ("try each type", "arcade game", "discover 100 vehicles",
                  "take down 5 enemies", "rob 3 shops",
                  "lose a 4 star", "top fun drone", "beat cal",
                  "bowl", "cocktail", "follow-through",
                  "smash your way", "register 10 vehicles",
                  "spend $100,000", "jewelry store", "armored truck",
                  "complete an arcade")
    if any(m in text for m in easy_marks):
        return "easy"
    return "medium"


def project_players(history, target_dt):
    """Project tracked players at the target; return (P_target, hourly P).

    hourly P is a list of (hour_start_datetime, players) from the first
    read to the target, used to size the joining cohorts.
    """
    reads = []
    for h in history:
        try:
            reads.append((parse_time(h["at"]), float(h["players_tracked"])))
        except (KeyError, TypeError, ValueError):
            continue
    reads.sort()
    if not reads:
        return 1000.0, []
    t_first, p_first = reads[0]
    t_last, p_last = reads[-1]
    horizon_h = max((target_dt - t_last).total_seconds() / 3600.0, 0.0)

    p_min = min(p for _, p in reads)
    p_max = max(p for _, p in reads)
    if p_max <= p_min * 1.001:
        # Flat history: no growth visible.  Prior: one refresh before the
        # target, multiplying tracked players ~6x (launch-night jump).
        if horizon_h <= 12.0:
            p_target = p_last * REFRESH_MULTIPLIER
        else:
            p_target = (p_last * REFRESH_MULTIPLIER
                        * (1.0 + 0.5 * (horizon_h - 12.0) / 24.0))
    else:
        # Fit log P against log(1 + hours/24) over the recent reads.
        recent = [(t, p) for (t, p) in reads
                  if (t_last - t).total_seconds() <= 48 * 3600.0]
        if len(recent) < 3:
            recent = reads
        xs, ys = [], []
        for t, p in recent:
            hours = max((t - t_first).total_seconds() / 3600.0, 0.0)
            xs.append(math.log1p(hours / 24.0))
            ys.append(math.log(max(p, 1.0)))
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx if sxx > 1e-12 else 0.0
        slope = max(slope, 0.0)
        intercept = my - slope * mx
        x_target = math.log1p(
            max((target_dt - t_first).total_seconds() / 3600.0, 0.0) / 24.0)
        p_target = math.exp(intercept + slope * x_target)
        # Damp and cap: growth after the last read cannot explode.
        horizon_days = horizon_h / 24.0
        cap = p_last * max(1.0, 1.0 + 2.5 * horizon_days)
        p_target = min(max(p_target, p_last), min(cap, 2_000_000.0))

    # Hourly player counts: observed steps up to the last read, then a
    # smooth (log-linear) ramp to the projected target.
    hourly = []
    cur = t_first
    idx = 0
    # players at any time = last read at or before that time
    steps = reads + [(target_dt, p_target)]
    while cur < target_dt:
        while idx + 1 < len(steps) and steps[idx + 1][0] <= cur:
            idx += 1
        if cur < t_last:
            p_cur = steps[idx][1]
        else:
            # log-linear ramp from p_last at t_last to p_target at target
            total = max((target_dt - t_last).total_seconds(), 1.0)
            frac = (cur - t_last).total_seconds() / total
            p_cur = math.exp(math.log(p_last) * (1 - frac)
                             + math.log(p_target) * frac)
        hourly.append((cur, p_cur))
        cur = cur + timedelta(hours=1)
    return p_target, hourly


def forecast_case(case):
    trophies = case["trophies"]
    history = sorted(case.get("history", []),
                     key=lambda h: h.get("at", ""))
    target_dt = parse_time(case["target"])
    horizon_days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)
    horizon_h = horizon_days * 24.0

    last = history[-1] if history else None
    last_rates = {}
    last_players = None
    first_players = None
    if last is not None:
        last_players = float(last.get("players_tracked") or 0) or None
        for i, t in enumerate(trophies):
            r = None
            for h in reversed(history):
                rates = h.get("rates") or []
                if i < len(rates) and rates[i] is not None:
                    r = float(rates[i])
                    break
            last_rates[t["id"]] = r
    if history:
        first_players = float(history[0].get("players_tracked") or 0) or None
    if last_players is None:
        last_players = first_players or 638.0
    if first_players is None:
        first_players = last_players

    p_target, hourly = project_players(history, target_dt)
    t_first = parse_time(history[0]["at"]) if history else target_dt
    t_last = parse_time(history[-1]["at"]) if history else target_dt

    # Class medians of the last rates, for within-class rank modulation.
    classes = {t["id"]: classify(t) for t in trophies}
    class_rates = {}
    for t in trophies:
        r = last_rates.get(t["id"])
        if r is not None:
            class_rates.setdefault(classes[t["id"]], []).append(r)
    class_median = {}
    for c, rs in class_rates.items():
        s = sorted(rs)
        class_median[c] = s[len(s) // 2] if s else 1.0

    # Per-trophy trend over the last 24h (only used if rates moved).
    trend_logit = {}
    if history:
        for i, t in enumerate(trophies):
            pts = []
            for h in history:
                rates = h.get("rates") or []
                if i < len(rates) and rates[i] is not None:
                    try:
                        tt = parse_time(h["at"])
                    except ValueError:
                        continue
                    if (t_last - tt).total_seconds() <= 24 * 3600.0:
                        pts.append((tt, logit(float(rates[i]) / 100.0)))
            if len(pts) >= 3:
                vals = [v for _, v in pts]
                if max(vals) - min(vals) > 0.15:
                    ts = [(tt - pts[0][0]).total_seconds() / 3600.0
                          for tt, _ in pts]
                    n = len(ts)
                    mt = sum(ts) / n
                    mv = sum(vals) / n
                    sxx = sum((x - mt) ** 2 for x in ts)
                    sxy = sum((x - mt) * (v - mv)
                              for x, (_, v) in zip(ts, pts))
                    slope = sxy / sxx if sxx > 1e-12 else 0.0
                    trend_logit[t["id"]] = (
                        vals[-1] + slope * horizon_h * 0.5)

    rows = []
    for t in trophies:
        tid = t["id"]
        cls = classes[tid]
        u0, tau0, extra = CLASSES[cls]
        r_last = last_rates.get(tid)
        if r_last is None:
            r_last = 1.0
        if r_last >= 99.9:
            # Earned by starting the game: stays ~100%.
            rows.append({"id": tid,
                         "q": [99.9, 99.97, 99.99, 99.99, 99.99]})
            continue
        # Rank modulation inside the class.
        med = class_median.get(cls, 1.0) or 1.0
        factor = math.sqrt(max(r_last, 0.01) / max(med, 0.01))
        factor = min(max(factor, 0.6), 1.6)
        u = min(max(u0 * factor, 0.0005), 0.98)
        tau = min(max(tau0 / factor, 2.0), 300.0)

        # Earners at the last read.
        earners_last = r_last / 100.0 * last_players
        # Modelled earners of the first cohort at last read vs target.
        def frac_played(elapsed_h):
            return 1.0 - math.exp(-elapsed_h * PLAY_FRACTION / tau)

        h_last = max((t_last - t_first).total_seconds() / 3600.0, 0.0)
        h_target = max((target_dt - t_first).total_seconds() / 3600.0, 0.0)
        earners = earners_last + first_players * u * (
            frac_played(h_target) - frac_played(h_last))
        # Joining cohorts on the hourly grid after the first read.
        prev_p = None
        last_grid = None
        for (ht, p_cur) in hourly:
            last_grid = ht
            if ht <= t_first:
                prev_p = p_cur
                continue
            inflow = max(p_cur - (prev_p if prev_p is not None else p_cur),
                         0.0)
            prev_p = p_cur
            if inflow <= 0:
                continue
            elapsed = max((target_dt - ht).total_seconds() / 3600.0, 0.0)
            earners += inflow * u * frac_played(elapsed)
        # Tail cohort: players who join after the last grid hour, up to
        # the target count; they have barely played, so count them with
        # half the remaining time as their elapsed play window.
        if prev_p is not None and p_target > prev_p:
            tail_gap_h = 0.0
            if last_grid is not None:
                tail_gap_h = max(
                    (target_dt - last_grid).total_seconds() / 3600.0, 0.0)
            earners += ((p_target - prev_p) * u
                        * frac_played(tail_gap_h / 2.0))
        rate = 100.0 * earners / max(p_target, 1.0)
        rate = clamp_rate(rate)

        # NOTE: logit() takes a fraction (0-1); rate here is a percent.
        med_logit = logit(rate / 100.0)
        if tid in trend_logit:
            med_logit = 0.7 * med_logit + 0.3 * trend_logit[tid]
        spread = 0.5 + 0.3 * math.sqrt(horizon_days) + extra
        qs = []
        for z in Z:
            q = 100.0 * logistic(med_logit + z * spread)
            qs.append(round(min(max(q, 0.01), 99.99), 4))
        for i in range(1, len(qs)):
            qs[i] = max(qs[i], qs[i - 1])
        rows.append({"id": tid, "q": qs})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        path = os.path.join(in_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
