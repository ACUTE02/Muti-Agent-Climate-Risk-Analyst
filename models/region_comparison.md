# Model / region comparison — averaged over horizons t+1, t+2, t+3

Held-out test set 2020-2024. Identical data, features, target and splits everywhere — only the model and the region change, so these numbers are directly comparable. Skill Score is `1 - RMSE_model/RMSE_climatology`: 0 means exactly as good as always predicting normal conditions, below 0 means worse.

| Model | Region | RMSE | MAE | R² | Skill vs. climatology |
|---|---|---|---|---|---|
| Climatology (baseline) | — | — | — | — | 0 by definition |
| LSTM(128→64), Phase 1.1/1.2 | rajasthan | 1.068 | 0.861 | -0.207 | +0.0011 |
| LSTM(128→64), Phase 1.1/1.2 | barmer | 1.416 | 1.086 | -0.013 | -0.0053 |
| Ridge (linear) | rajasthan | 1.140 | 0.891 | -0.375 | -0.0662 |
| Ridge (linear) | barmer | 1.381 | 1.034 | +0.036 | +0.0194 |
| LSTM(16), lr=1e-4 | rajasthan | 1.055 | 0.846 | -0.178 | +0.0132 |
| LSTM(16), lr=1e-4 | barmer | 1.406 | 1.079 | +0.002 | +0.0023 |

## Skill score by horizon

| Model | Region | t+1 | t+2 | t+3 |
|---|---|---|---|---|
| LSTM(128→64), Phase 1.1/1.2 | rajasthan | +0.0005 | -0.0013 | +0.0041 |
| LSTM(128→64), Phase 1.1/1.2 | barmer | -0.0070 | -0.0065 | -0.0023 |
| Ridge (linear) | rajasthan | +0.0844 | -0.0660 | -0.1961 |
| Ridge (linear) | barmer | +0.1314 | +0.0032 | -0.0671 |
| LSTM(16), lr=1e-4 | rajasthan | +0.0336 | -0.0046 | +0.0111 |
| LSTM(16), lr=1e-4 | barmer | +0.0026 | +0.0033 | +0.0010 |

## Did the small LSTM's val_loss ever beat climatology?

- **rajasthan: YES** — best val_loss 0.9999 at epoch 15 of 30, against a climatology benchmark of 1.0237 on the same validation windows.
- **barmer: YES** — best val_loss 0.8153 at epoch 25 of 40, against a climatology benchmark of 0.8338 on the same validation windows.

**No model variant shows meaningful averaged forecast skill.** The best of them (Ridge (linear), barmer) reaches +0.0194, against the +0.1 threshold set in advance. A linear model, a small slow-learning LSTM and a large LSTM all sit at the climatology baseline, at both a moderately and a highly drought-variable site — evidence that the predictors, not the architecture, are the binding constraint.

One exception worth naming rather than burying: **Ridge (linear) at barmer reaches +0.1314 at t+1 alone**, above the threshold, decaying to nothing at the longer horizons. Read it cautiously — it is one horizon out of 18 numbers in the table above, on 58 test windows, so it could be noise. But it is the first positive signal in this project, and it is where a follow-up should look: one month ahead, linear, not three months ahead, deep.

---

## Phase 1.4 — does the t+1 signal replicate?

Same Ridge pipeline, joint 3-horizon model, t+1 skill only, re-run on four independent historical windows. Each window refits its own `month_stats`, `spi_params` and scaler on its own train partition.

| Region | A | B | C | D | positive | mean |
|---|---|---|---|---|---|---|
| rajasthan | +0.0844 | +0.0163 | +0.1895 | +0.0186 | 4/4 | +0.0772 |
| barmer | +0.1314 | +0.0839 | +0.0675 | -0.0108 | 3/4 | +0.0680 |

- rajasthan: **confirmed** — positive in 4 of 4 windows (needs 3 of 4).
- barmer: **confirmed** — positive in 3 of 4 windows (needs 3 of 4).

## Dedicated 1-month-ahead model

Ridge trained on t+1 alone. Lookback chosen from {12, 24, 60} months by mean *validation* skill across all four windows — test sets were never consulted for selection.

Mean VALIDATION skill by lookback (selection metric):
| Region | 12mo | 24mo | 60mo | chosen |
|---|---|---|---|---|
| rajasthan | +0.2973 | +0.2657 | +0.1453 | **12mo** |
| barmer | +0.2725 | +0.2359 | +0.1642 | **12mo** |

TEST skill of the selected model, per window:
| Region | lookback | A | B | C | D | positive | mean |
|---|---|---|---|---|---|---|---|
| rajasthan | 12mo | +0.1594 | +0.3681 | +0.2403 | +0.2809 | 4/4 | +0.2622 |
| barmer | 12mo | +0.2132 | +0.2265 | +0.1546 | +0.2269 | 4/4 | +0.2053 |

Validity check — how much of that is the SPI-3 overlap?
| Region | Ridge vs climatology | Persistence vs climatology | Ridge vs persistence | beats persistence |
|---|---|---|---|---|
| rajasthan | +0.2622 | +0.1024 | +0.1799 | 4/4 |
| barmer | +0.2053 | +0.0105 | +0.1891 | 4/4 |

SPI-3 is a 3-month accumulation, so SPI-3 at t+1 shares two of its three months with SPI-3 at t. Part of this task is therefore nowcasting rather than pure forecasting, and the persistence row above measures exactly how much: persistence alone earns +0.1024 at rajasthan, +0.0105 at barmer. The model beats that naive exploitation of the overlap in every window, so the remainder is real.

---

## Phase 1.5 — recursive vs. direct for t+2 and t+3

Recursive chains the validated t+1 model forward, feeding each prediction back as if observed. Direct trains a dedicated model per horizon (lookback chosen by validation skill, same process as t+1). Both scored on the same test origins and targets. `←` marks the winner used going forward.

| Horizon | Region | Approach | A | B | C | D | mean skill |
|---|---|---|---|---|---|---|---|
| t+2 | rajasthan | recursive | -0.0520 | +0.0421 | +0.1304 | +0.0298 | +0.0376 |
| t+2 | rajasthan | direct | +0.0529 | +0.0890 | +0.0918 | +0.0728 | +0.0766 **←** |
| t+2 | barmer | recursive | +0.0703 | -0.0136 | +0.0093 | +0.0413 | +0.0268 |
| t+2 | barmer | direct | +0.0855 | +0.0224 | +0.0419 | +0.0252 | +0.0438 **←** |
| t+3 | rajasthan | recursive | -0.3736 | -0.2676 | -0.0390 | -0.1839 | -0.2160 |
| t+3 | rajasthan | direct | -0.0209 | -0.0067 | -0.0314 | +0.0011 | -0.0145 **←** |
| t+3 | barmer | recursive | -0.0264 | -0.1324 | -0.0941 | -0.0931 | -0.0865 |
| t+3 | barmer | direct | -0.0026 | -0.0852 | -0.0087 | -0.0989 | -0.0489 **←** |

Limitation of the recursive path, stated rather than hidden: it holds ONI at its last observed value, since fabricating a future ENSO state would be worse. Calendar features (`month_sin`/`month_cos`) are exactly known; rainfall and everything derived from it is reconstructed by inverting the SPI-3 gamma transform, and those errors compound.


---

## Phase 1.6 — does the Indian Ocean Dipole add anything?

One bounded feature test on an otherwise frozen pipeline. Both columns below were produced by the same code in the same run, so the difference is attributable to the feature and nothing else. IOD correlates with ONI at r=+0.386 over 1980-2024 (+0.340 on train alone) — related, but far from redundant, so a null result here is not simply ENSO in disguise.

| Region | Horizon | Skill without IOD | Skill with IOD | Change | Windows improved | Adopt |
|---|---|---|---|---|---|---|
| rajasthan | t+1 | +0.2622 | +0.2389 | -0.0233 | 1/4 | no |
| rajasthan | t+2 | +0.0763 | +0.0774 | +0.0011 | 2/4 | no |
| rajasthan | t+3 | -0.0145 | -0.0305 | -0.0160 | 1/4 | no |
| barmer | t+1 | +0.2053 | +0.1933 | -0.0120 | 2/4 | no |
| barmer | t+2 | +0.0496 | +0.0392 | -0.0104 | 2/4 | no |
| barmer | t+3 | -0.0489 | -0.0757 | -0.0268 | 0/4 | no |

**Rejected.** Five of the six region/horizon cells got worse, and the one nominal gain (+0.0011) held in only 2 of 4 windows against a 3-of-4 bar. The feature is not in the model; `models/iod_comparison.json` keeps the measurement.

## Final per-horizon verdict

What `forecast_drought_risk()` reports at runtime, straight from measurement on window A's split.

| Region | Horizon | Method | Lookback | Skill | Label |
|---|---|---|---|---|---|
| rajasthan | t+1 | direct | 12mo | +0.2622 | validated |
| rajasthan | t+2 | direct | 12mo | +0.0766 | weak/directional |
| rajasthan | t+3 | direct | 24mo | -0.0145 | no skill — shown for context only, do not rely on this figure |
| barmer | t+1 | direct | 12mo | +0.2053 | validated |
| barmer | t+2 | direct | 12mo | +0.0438 | weak/directional |
| barmer | t+3 | direct | 12mo | -0.0489 | no skill — shown for context only, do not rely on this figure |
