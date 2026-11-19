# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Lux-agentt/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Lux-agentt/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every GTA VI PlayStation 5 trophy, the percent of the players Exophase tracks who will have earned it at 2026-11-26T00:00:00Z, a week after launch, as five quantiles per trophy, scored by pinball loss on the logit scale. `main.py` (Python 3.13, standard library only, deterministic) reads one case from `/input/.in` and writes `/output/.out`.

Method: each trophy's earn rate follows a saturating curve in hours since release, `rate(t) = A * (1 - exp(-lam * t))`, fitted by a deterministic grid search over `lam` on the recent 36 hours of reads — players finish trophies at a rate that slows as the easy ones are done, so a saturating form extrapolates honestly where a log-time line runs away. The median blends it 75/25 in logit space with a direct logit-rate fit against `ln(1 + t)` (the house curve baseline's shape), which is the only component that can predict a fall, so a dilution-dominated week is not missed. When the history is completely flat (the launch-night practice), a dilution prior fires instead: the first refresh multiplied players tracked ~6x while new players had earned ~1 trophy each, concentrated on the earliest trophies, and the trophy at 100% is earned by every new player and stays at 100%.

Validated locally: practice cases p00/p01 score ~0.10 mean pinball loss against a reconstruction of the stated refresh (persistence and curve baselines ~1.40–1.69), and four synthetic final-like backtests (base, slow earning, dilution-dominated, fast earning) score 0.21–0.25, beating the persistence baseline (0.32–0.58) and the curve baseline (0.25–0.29) in every scenario. Runtime ~0.25 s on a 357-read case, far under the 8 s cap; output ~2.8 KB, under the 65,536-byte cap.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=1r2f5pk3&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
