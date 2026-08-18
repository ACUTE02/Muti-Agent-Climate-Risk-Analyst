# Heat Stress — model / region comparison

Target `heat_anomaly`: standardised monthly Tmax anomaly, climatology fit on train rows only. Split is the project standard — train 1980-2015, val 2016-2019, test 2020-2024. Ridge on flattened lookback windows, alpha from {0.1, 1, 10, 100, 1000} by validation MSE, lookback from {12, 24, 60} months by validation skill. Labels use the same bar as the Drought Agent: >= +0.1 validated, 0 to 0.1 weak/directional, <= 0 no skill.

## Target distribution check (train partition)

| Region | mean | std | skew | excess kurtosis | p5 (vs -1.645) | p95 (vs +1.645) | approx. normal |
|---|---|---|---|---|---|---|---|
| rajasthan | -0.000 | 0.987 | -0.049 | +0.626 | -1.464 | +1.643 | yes |
| barmer | +0.000 | 0.987 | -0.255 | +0.758 | -1.656 | +1.467 | yes |

## Skill by horizon — vs climatology and vs persistence

| Region | Lookback | Horizon | RMSE | R² | Skill vs clim. | Skill vs persistence | Persistence vs clim. | Label |
|---|---|---|---|---|---|---|---|---|
| rajasthan | 24mo | t+1 | 1.074 | -0.110 | -0.0525 | +0.0764 | -0.1396 | no skill — shown for context only, do not rely on this figure |
| rajasthan | 24mo | t+2 | 1.192 | -0.349 | -0.1610 | +0.2144 | -0.4779 | no skill — shown for context only, do not rely on this figure |
| rajasthan | 24mo | t+3 | 1.201 | -0.354 | -0.1631 | +0.2766 | -0.6077 | no skill — shown for context only, do not rely on this figure |
| barmer | 60mo | t+1 | 1.217 | -0.089 | -0.0209 | +0.1757 | -0.2384 | no skill — shown for context only, do not rely on this figure |
| barmer | 60mo | t+2 | 1.211 | -0.099 | -0.0172 | +0.2993 | -0.4518 | no skill — shown for context only, do not rely on this figure |
| barmer | 60mo | t+3 | 1.204 | -0.075 | -0.0118 | +0.3484 | -0.5528 | no skill — shown for context only, do not rely on this figure |

## Lookback selection (validation skill)

| Region | 12mo | 24mo | 60mo | chosen |
|---|---|---|---|---|
| rajasthan | +0.0598 | +0.1101 | +0.0786 | **24mo** |
| barmer | +0.0266 | +0.0375 | +0.0588 | **60mo** |

**No skill at any horizon, either region.** Every skill score sits at or below zero, meaning nothing beats predicting the seasonal normal. Unlike the Drought Agent's first pass, this is not a target artifact: the target was distribution-checked before modelling and is near-normal. Reported and stopped here, per the phase's stopping rule.

---

## Phase 1.1 — new targets and the SPI-3 cross-agent feature

Same split, Ridge, alpha grid and lookback grid as Phase 1. Each run sees its own target's history (see `heat/phase11.py` for why). Phase 1's reference result was -0.0525 (rajasthan) / -0.0209 (barmer) at t+1 on `heat_anomaly` with heat-only features.

| Region | Target | Features | Lookback | t+1 | t+2 | t+3 | mean | best label |
|---|---|---|---|---|---|---|---|---|
| rajasthan | `heat_anomaly` | heat-only | 24mo | -0.0525 | -0.1610 | -0.1631 | -0.1255 | no skill |
| rajasthan | `heat_anomaly` | heat+SPI3 | 60mo | -0.0038 | -0.0094 | -0.0010 | -0.0047 | no skill |
| rajasthan | `heat_extreme` | heat-only | 60mo | -0.2883 | -0.2987 | -0.2554 | -0.2808 | no skill |
| rajasthan | `heat_extreme` | heat+SPI3 | 60mo | -0.0510 | -0.0527 | -0.0435 | -0.0491 | no skill |
| rajasthan | `heatwave_day_count` | heat-only | 12mo | +0.0220 | +0.0032 | +0.0104 | +0.0119 | weak/directional |
| rajasthan | `heatwave_day_count` | heat+SPI3 | 12mo | +0.0088 | -0.0204 | -0.0052 | -0.0056 | weak/directional |
| barmer | `heat_anomaly` | heat-only | 60mo | -0.0209 | -0.0172 | -0.0118 | -0.0166 | no skill |
| barmer | `heat_anomaly` | heat+SPI3 | 60mo | -0.0115 | -0.0012 | +0.0006 | -0.0040 | weak/directional |
| barmer | `heat_extreme` | heat-only | 60mo | +0.0169 | +0.0231 | +0.0213 | +0.0204 | weak/directional |
| barmer | `heat_extreme` | heat+SPI3 | 60mo | +0.0110 | +0.0246 | +0.0234 | +0.0197 | weak/directional |
| barmer | `heatwave_day_count` | heat-only | 60mo | +0.0269 | +0.0219 | +0.0219 | +0.0236 | weak/directional |
| barmer | `heatwave_day_count` | heat+SPI3 | 60mo | +0.0378 | +0.0224 | +0.0190 | +0.0264 | weak/directional |

### The count target is zero-inflated

- 92.4% of train months have zero heat wave days; 75.2% even within April-June.
- Train mean 0.1881 days overall vs 0.6476 in the pre-monsoon months.

Skill against an all-months climatology is therefore flattered by eight easy months a year. The April-June cut is the meaningful one:

| Region | Features | Horizon | n months | Skill vs clim. | Label |
|---|---|---|---|---|---|
| rajasthan | heat-only | t+1 | 15 | +0.0322 | weak/directional |
| rajasthan | heat-only | t+2 | 15 | +0.0221 | weak/directional |
| rajasthan | heat-only | t+3 | 15 | +0.0192 | weak/directional |
| rajasthan | heat+SPI3 | t+1 | 15 | +0.0241 | weak/directional |
| rajasthan | heat+SPI3 | t+2 | 15 | +0.0116 | weak/directional |
| rajasthan | heat+SPI3 | t+3 | 15 | +0.0180 | weak/directional |
| barmer | heat-only | t+1 | 15 | +0.0106 | weak/directional |
| barmer | heat-only | t+2 | 15 | +0.0046 | weak/directional |
| barmer | heat-only | t+3 | 15 | +0.0049 | weak/directional |
| barmer | heat+SPI3 | t+1 | 15 | +0.0218 | weak/directional |
| barmer | heat+SPI3 | t+2 | 15 | +0.0108 | weak/directional |
| barmer | heat+SPI3 | t+3 | 15 | +0.0138 | weak/directional |

May 2024 (a month with real heat wave days) — predicted vs observed day counts:

| Region | Features | Horizon | Predicted | Observed |
|---|---|---|---|---|
| rajasthan | heat-only | t+1 | 0.71 | 8 |
| rajasthan | heat-only | t+2 | 0.92 | 8 |
| rajasthan | heat-only | t+3 | 0.92 | 8 |
| rajasthan | heat+SPI3 | t+1 | 0.7 | 8 |
| rajasthan | heat+SPI3 | t+2 | 0.89 | 8 |
| rajasthan | heat+SPI3 | t+3 | 0.94 | 8 |
| barmer | heat-only | t+1 | 0.75 | 7 |
| barmer | heat-only | t+2 | 0.75 | 7 |
| barmer | heat-only | t+3 | 0.75 | 7 |
| barmer | heat+SPI3 | t+1 | 0.83 | 7 |
| barmer | heat+SPI3 | t+2 | 0.85 | 7 |
| barmer | heat+SPI3 | t+3 | 0.89 | 7 |

Negative predictions clipped to zero at evaluation:

- rajasthan / heat-only: 29.3% of raw predictions were negative (most negative -0.5732) — a linear model on a non-negative count, as expected.
- rajasthan / heat+SPI3: 21.3% of raw predictions were negative (most negative -0.7281) — a linear model on a non-negative count, as expected.
- barmer / heat-only: 12.1% of raw predictions were negative (most negative -0.1225) — a linear model on a non-negative count, as expected.
- barmer / heat+SPI3: 20.7% of raw predictions were negative (most negative -0.7865) — a linear model on a non-negative count, as expected.

### Winter sanity check (Dec-Feb, t+1)

Tmax never approaches the 40 C plains threshold in winter, so both columns should sit near zero. Confirmed, not assumed:

| Region | Target | Features | n months | Observed mean | Observed max | Predicted mean | Predicted max |
|---|---|---|---|---|---|---|---|
| rajasthan | `heat_anomaly` | heat-only | 14 | -0.054 | +1.764 | -0.319 | +0.444 |
| rajasthan | `heat_anomaly` | heat+SPI3 | 14 | -0.054 | +1.764 | -0.078 | +0.174 |
| rajasthan | `heat_extreme` | heat-only | 14 | +1.445 | +2.491 | +1.276 | +2.152 |
| rajasthan | `heat_extreme` | heat+SPI3 | 14 | +1.445 | +2.491 | +1.462 | +1.602 |
| rajasthan | `heatwave_day_count` | heat-only | 14 | +0.000 | +0.000 | +0.026 | +0.222 |
| rajasthan | `heatwave_day_count` | heat+SPI3 | 14 | +0.000 | +0.000 | +0.034 | +0.141 |
| barmer | `heat_anomaly` | heat-only | 14 | +0.333 | +2.798 | -0.001 | +0.211 |
| barmer | `heat_anomaly` | heat+SPI3 | 14 | +0.333 | +2.798 | +0.062 | +0.197 |
| barmer | `heat_extreme` | heat-only | 14 | +1.652 | +2.583 | +1.512 | +1.704 |
| barmer | `heat_extreme` | heat+SPI3 | 14 | +1.652 | +2.583 | +1.581 | +1.784 |
| barmer | `heatwave_day_count` | heat-only | 14 | +0.000 | +0.000 | +0.110 | +0.277 |
| barmer | `heatwave_day_count` | heat+SPI3 | 14 | +0.000 | +0.000 | +0.255 | +0.966 |

**Nothing clears the bar.** The strongest cell anywhere in this phase is barmer / `heatwave_day_count` / heat+SPI3 at t+1, skill +0.0378 — against the +0.1 bar. Three target definitions (mean, extreme, count) and two feature sets (heat-only, heat+drought) have now been tried. Heat Stress forecasting is genuinely unskillful with this data; the operational IMD heat wave counter is the part worth keeping.
