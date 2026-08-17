# CLAUDE.md — Multi-Agent Climate Risk Analyst
## Phase 1.3: Architecture ablation — is the flat-line result a model-capacity problem?

**Why this phase exists:** Phase 1.1's diagnostic showed the 2-layer LSTM (128→64 units, ~123,000 trainable parameters against only 358 training sequences — over 300 parameters per example) never beat the climatology baseline on validation, even at epoch 1, and concluded this was a predictor-limitation problem. That conclusion was an inference from one architecture's behavior, not a tested fact. This phase actually tests it: a smaller model and a linear baseline, both bounded, one-shot — this is the last architecture-side check, not the start of an open-ended tuning loop. Report the result honestly whichever way it comes out, and stop after this.

Do not touch data, features, target, or region config. Only new model variants and their evaluation.

---

## 1. Baseline model — Ridge regression (rules out "does *any* signal exist at all")

If a plain linear model can't beat climatology either, that's much stronger evidence of "no exploitable signal" than a deep model failing — deep models can fail for their own reasons (capacity, optimization) that have nothing to do with whether the data has signal.

```python
# forecasting/baseline_ridge.py
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def flatten_windows(X):
    # X shape (n, 60, 11) -> (n, 660)
    return X.reshape(X.shape[0], -1)

def fit_ridge_baseline(X_train, y_train, X_val, y_val, alphas=(0.1, 1, 10, 100, 1000)):
    """Multi-output Ridge (sklearn supports (n, 3) targets natively).
    Select alpha by validation MSE — this is the only 'tuning' here, and it's
    a single small grid over one hyperparameter, not an open search."""
    best = None
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(flatten_windows(X_train), y_train)
        val_mse = mean_squared_error(y_val, model.predict(flatten_windows(X_val)))
        if best is None or val_mse < best[1]:
            best = (model, val_mse, alpha)
    return best  # (fitted model, val_mse, chosen alpha)
```

Evaluate on the test set exactly like the LSTM (same RMSE/MAE/R²/Skill Score per horizon + averaged, same climatology comparison). Save to `models/metrics_ridge_{region}.json`.

## 2. Smaller LSTM — rules out "the big model overfits before it can learn anything"

Deliberately much smaller, much slower-learning, more patient:

```python
model = Sequential([
    LSTM(16, input_shape=(60, len(FEATURES))),   # single layer, 16 units, no return_sequences
    Dropout(0.3),
    Dense(3),
])
model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])
```

Training: `EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)`, `epochs=150, batch_size=32`. Higher patience than Phase 1.1's — a slower learning rate needs more epochs to show its real trajectory before you can conclude anything from it.

Save as `models/lstm_small_{region}.keras`, evaluate the same way → `models/metrics_lstm_small_{region}.json`. Also save `models/training_history_lstm_small_{region}.json` (same format as Phase 1.1's diagnostic) — capture the full loss curve, don't just report the final metric.

## 3. Comparison

Extend `models/region_comparison.md` with two more rows per region — same table, same honesty standard as before:

| Model | Region | RMSE | R² | Skill vs. climatology |
|---|---|---|---|---|
| Climatology (baseline) | — | — | — | 0 by definition |
| LSTM(128→64), Phase 1.1/1.2 | rajasthan / barmer | (existing) | (existing) | (existing) |
| Ridge (linear) | rajasthan / barmer | new | new | new |
| LSTM(16), lr=1e-4 | rajasthan / barmer | new | new | new |

Also report, for the small LSTM, whether its `val_loss` curve ever dips below the climatology MSE benchmark at *any* epoch (not just the final/restored one) — that's the specific question this phase exists to answer.

## 4. Stopping rule — read this before starting

This is a one-shot ablation with two specific, pre-committed variants. Do not iterate further based on results (no "let's also try LSTM(32)", no third learning rate, no second Ridge alpha grid beyond the one specified). Whatever these two variants show, report it and stop:

- **If Ridge and/or the small LSTM show real skill (Skill Score meaningfully > 0, say > 0.1, on the test set)** — that's a genuine, actionable finding: the original architecture was the problem, not the predictors. Worth a real follow-up phase to properly tune from there.
- **If both still show ≈0 or negative skill, matching the original LSTM** — that closes the loop for real this time, with linear-model and small-model evidence added to the target-definition, exogenous-predictor, and site-choice evidence already gathered. Four independent angles agreeing is about as thorough as a portfolio project needs to be before moving on.

## Definition of Done

- [ ] `models/metrics_ridge_rajasthan.json`, `models/metrics_ridge_barmer.json`
- [ ] `models/metrics_lstm_small_rajasthan.json`, `models/metrics_lstm_small_barmer.json`, plus their training history files
- [ ] `models/region_comparison.md` updated with the 4-row-per-region table above
- [ ] Explicit statement of whether the small LSTM's val_loss ever beat the climatology benchmark at any epoch, for both regions
- [ ] `.gitignore` exceptions extended for the new evidence files (`metrics_ridge_*.json`, `metrics_lstm_small_*.json`, `training_history_lstm_small_*.json`) — same pattern as before, easy to forget
- [ ] `PROJECT_LOG.md` gets a Phase 1.3 entry with the honest outcome

## When done

Report back the extended comparison table and the small-LSTM loss curve. This is the real final review gate before the first commit and Phase 2 — whichever way it lands.
