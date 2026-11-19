"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate
at the case target as five quantiles. Python 3.13, standard library only.

Method (two paths, chosen by whether the history moves):
  Data-driven path (the final: history has real movement). Per trophy, the
  earn rate follows a saturating curve in hours since release,
      rate(t) = A * (1 - exp(-lam * t)),
  fitted by a deterministic grid search over lam with A by least squares on
  the recent window (last 36 h, half-life 12 h weights): players finish
  trophies at a rate that slows as the easy ones are done, so a saturating
  form extrapolates honestly where a log-time line runs away. The median
  blends it 75/25 in logit space with a direct logit-rate fit against
  ln(1 + t) (the house curve baseline's shape, same window), which is the
  only component that can predict a fall, so a dilution-dominated week is
  not missed entirely.

  Flat-history path (the practice: every read identical, no trend at all).
  A prior fires, taken from the launch-night refresh the rules describe:
  the first refresh multiplied players tracked ~6x (638 -> 3,868) while new
  players had earned ~1 trophy each, concentrated on the earliest trophies
  (the trophy at 100% is earned by every new player and stays at 100%).
  Rates are diluted accordingly; the prior never fires on a case whose
  history already moves (the final).

Spread: per-trophy predictive SD in logit, sqrt((0.5*sqrt(d))^2 + resid^2
(+ 0.5^2 when the flat-history prior fired)), d the horizon in days; the
upper tail is narrowed (positive z * 0.85) because dilution drops are large
while earning growth is gradual, so downside deviations exceed upside ones.

Deterministic: no randomness, no clock, no network; every input comes from
the case file. Reads every *.in in the input directory, writes one *.out
each. Runtime is far under the 8 s cap.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Spread: logit SD per sqrt-day of horizon (random-walk refresh term).
SIGMA_DAY = 0.5
# Upper-tail narrowing (positive z-scores), v4-style asymmetry.
SKEW_UP = 0.85
# Median blend: saturating fit vs direct logit-rate fit (logit space).
SAT_W = 0.75
# Slope fits use only the recent window (growth decelerates).
WINDOW_H = 36.0
# Recency half-life for fit weights (hours).
WEIGHT_HALF_LIFE_H = 12.0
# Flat-history prior: first-refresh player multiplier (rules: 638 -> 3,868).
REFRESH_FACTOR = 6.06
# New players' trophies each at the first refresh (rules: 3,333 new earns / 3,230 new players).
NEW_EARNS_PER_PLAYER = 1.03
# Hard clamp on how far the median may move from the last rate (factor).
MOVE_CAP = 10.0
# Grid of earning speeds (per hour) for the saturating fit.
LAM_GRID = [10.0 ** (x / 10.0) for x in range(-30, 1)]


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100 - p))


def inv_logit(l: float) -> float:
    if l >= 40:
        return 99.99
    if l <= -40:
        return 0.01
    return 100 / (1 + math.exp(-l))


def round4(x: float):
    """Round to 4 decimals, half up (as the starter does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def hours_since_release(ts: str) -> float:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return (dt - RELEASE).total_seconds() / 3600.0


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sd, n)."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    a = my - b * mx
    n = len(xs)
    if n > 2:
        s2 = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
        resid = math.sqrt(max(0.0, s2))
    else:
        resid = 0.0
    return a, b, resid, n


def sat_fit(ts, rs, t_tgt):
    """rate(t) = A * (1 - exp(-lam * t)); grid-search lam, A by least squares.
    Returns (pred_at_target, resid_logit_sd)."""
    best = None
    for lam in LAM_GRID:
        us = [1 - math.exp(-lam * t) for t in ts]
        suu = sum(u * u for u in us)
        if suu <= 0:
            continue
        A = sum(u * r for u, r in zip(us, rs)) / suu
        sse = sum((r - A * u) ** 2 for u, r in zip(us, rs))
        if best is None or sse < best[0]:
            best = (sse, A, lam)
    _, A, lam = best
    pred = A * (1 - math.exp(-lam * t_tgt))
    # Residual in logit units (what the spread needs).
    res = []
    for t, r in zip(ts, rs):
        fitted = A * (1 - math.exp(-lam * t))
        res.append(logit(max(0.01, r)) - logit(max(0.01, fitted)))
    n = len(res)
    sd = math.sqrt(sum(x * x for x in res) / n) if n else 0.0
    return pred, sd


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    hist = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    t_at = hours_since_release(case["at"])
    t_tgt = hours_since_release(case["target"])
    x_tgt = math.log(1 + t_tgt)

    ts_h = [max(0.0, hours_since_release(h["at"])) for h in hist]
    xs = [math.log(1 + t) for t in ts_h]
    ps = [h["players_tracked"] for h in hist]
    win = [j for j in range(len(xs)) if ts_h[j] >= t_at - WINDOW_H]
    if len(win) < 3:
        win = list(range(len(xs)))
    ws = [0.5 ** ((t_at - ts_h[j]) / WEIGHT_HALF_LIFE_H) for j in win]

    p_flat = len(set(ps)) == 1
    prior_fired = p_flat and t_at < 24.0
    if prior_fired:
        # Flat launch-night history: expect the first refresh inside the horizon.
        f = 1.0 + (REFRESH_FACTOR - 1.0) * min(1.0, case["horizon_minutes"] / 60.0)
        p_tgt = ps[-1] * f
    else:
        p_tgt = float(ps[-1])

    rows = []
    for i, trophy in enumerate(trophies):
        last = hist[-1]["rates"][i]
        last_rate = last if last is not None else 0.0
        last_logit = logit(last_rate)
        lo = max(0.01, last_rate / MOVE_CAP) if last_rate > 0 else 0.01
        hi = min(99.99, last_rate * MOVE_CAP) if last_rate > 0 else 99.99

        if prior_fired:
            # Dilution prior: new players earn ~1 trophy each; the trophy at
            # 100% is earned by every new player and stays at 100%; the small
            # remainder spreads over the others in proportion to old counts.
            p_last = ps[-1]
            e_last = max(0.5, last_rate * p_last / 100.0)
            if last_rate >= 99.995:
                e_tgt = p_tgt
            else:
                n100 = sum(1 for k in range(len(trophies))
                           if hist[-1]["rates"][k] is not None and hist[-1]["rates"][k] >= 99.995)
                new_total = (p_tgt - p_last) * NEW_EARNS_PER_PLAYER - n100 * (p_tgt - p_last)
                new_total = max(0.0, new_total)
                others = [k for k in range(len(trophies))
                          if hist[-1]["rates"][k] is not None and hist[-1]["rates"][k] < 99.995]
                wsum = sum(max(0.5, hist[-1]["rates"][k] * p_last / 100.0) for k in others)
                e_tgt = e_last + new_total * e_last / wsum if wsum > 0 else e_last
            rate_prior = clamp(100.0 * e_tgt / p_tgt)
            rate_prior = min(hi, max(lo, rate_prior))
            med_logit = logit(rate_prior)
            s_res = 0.15
        else:
            # Saturating fit on the recent window.
            ts_w, rs_w = [], []
            for j in win:
                r = hist[j]["rates"][i]
                if r is not None:
                    ts_w.append(ts_h[j])
                    rs_w.append(r)
            if len(set(rs_w)) > 1 and len(rs_w) >= 2:
                pred_sat, s_sat = sat_fit(ts_w, rs_w, t_tgt)
            else:
                pred_sat, s_sat = last_rate, 0.0
            # Direct logit-rate fit (same window) as the falling-rate hedge.
            ex, ey, ew = [], [], []
            for k, j in enumerate(win):
                r = hist[j]["rates"][i]
                if r is None:
                    continue
                ex.append(xs[j])
                ey.append(logit(r))
                ew.append(ws[k])
            if len(set(ey)) > 1 and len(ey) >= 2:
                a, b, _, _ = wls(ex, ey, ew)
                med_direct = a + b * x_tgt
                has_direct = True
            else:
                med_direct = last_logit
                has_direct = False
            med_sat = logit(clamp(pred_sat))
            med_logit = SAT_W * med_sat + (1 - SAT_W) * med_direct if has_direct else med_sat
            med_logit = max(logit(lo), min(logit(hi), med_logit))
            s_res = min(0.8, max(0.15, s_sat))

        s = math.sqrt((SIGMA_DAY * math.sqrt(horizon_days)) ** 2 + s_res ** 2
                      + (0.5 ** 2 if prior_fired else 0.0))
        q = []
        for z in Z:
            zz = z * SKEW_UP if z > 0 else z
            q.append(round4(clamp(inv_logit(med_logit + zz * s))))
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
