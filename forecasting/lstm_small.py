"""Phase 1.3 ablation — a deliberately small, slow, patient LSTM.

The question this answers: does the 128→64 network fail because it overfits before
it can learn anything? It carries ~123k trainable parameters against 358 training
sequences — over 300 parameters per example. This variant has a fraction of that
capacity, a 10x lower learning rate, heavier dropout and 15 epochs of patience, so
a slow-but-real learning trajectory would have room to show itself.

Pre-committed and one-shot: the values live in config.py and are not to be tuned
further on the strength of what comes out.

Run standalone:  python -m forecasting.lstm_small [region ...]
"""

from __future__ import annotations

import json

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from forecasting import config
from forecasting.evaluate import climatology_val_mse, score_predictions
from forecasting.split import Dataset, prepare_dataset
from forecasting.train import save_history, set_seeds


def build_small_model(n_features: int = len(config.FEATURES)) -> Sequential:
    model = Sequential([
        Input(shape=(config.SEQ_LEN, n_features)),
        LSTM(config.SMALL_LSTM_UNITS),          # single layer, no return_sequences
        Dropout(config.SMALL_DROPOUT),
        Dense(config.HORIZON),
    ])
    model.compile(optimizer=Adam(learning_rate=config.SMALL_LEARNING_RATE),
                  loss="mse", metrics=["mae"])
    return model


def run(region: str = config.DEFAULT_REGION, ds: Dataset | None = None) -> dict:
    config.check_region(region)
    set_seeds()
    ds = ds or prepare_dataset(region, save=False)
    X_train, y_train = ds.get("train")
    X_val, y_val = ds.get("val")

    model = build_small_model(X_train.shape[-1])
    print(f"[{region}] small LSTM trainable params: "
          f"{model.count_params():,} vs {len(X_train)} training sequences")

    early_stopping = EarlyStopping(monitor="val_loss",
                                   patience=config.SMALL_EARLY_STOPPING_PATIENCE,
                                   restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.SMALL_EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[
            early_stopping,
            ModelCheckpoint(str(config.lstm_small_model_path(region)),
                            monitor="val_loss", save_best_only=True),
        ],
        verbose=2,
    )
    # Its own history file — the Phase-1.1/1.2 model's history is left untouched.
    hist = save_history(history, early_stopping, region,
                        path=config.lstm_small_history_path(region))
    model.save(config.lstm_small_model_path(region))

    # The question this phase exists to answer: did val_loss EVER dip below the
    # climatology benchmark — not just at the restored epoch, at any epoch?
    val_curve = hist["val_loss"]
    clim_mse = climatology_val_mse(ds)
    beat = bool(min(val_curve) < clim_mse)

    y_pred = model.predict(ds.splits["test"]["X"], verbose=0)
    metrics = score_predictions(ds, y_pred, region, "lstm_small", extra={
        "trainable_params": int(model.count_params()),
        "epochs_run": hist["epochs_run"],
        "val_loss_min": float(min(val_curve)),
        "val_loss_best_epoch": hist["best_epoch_1indexed"],
        "climatology_val_mse": round(clim_mse, 4),
        "val_loss_beat_climatology": beat,
    })
    config.lstm_small_metrics_path(region).write_text(json.dumps(metrics, indent=2),
                                                      encoding="utf-8")

    a = metrics["averaged"]
    print(f"[{region}] small LSTM test: RMSE {a['rmse']:.3f}  R2 {a['r2']:+.3f}  "
          f"skill {a['skill_score']:+.4f}")
    print(f"[{region}] val_loss min {min(val_curve):.4f} vs climatology "
          f"{clim_mse:.4f} -> {'BEAT' if beat else 'never beat'} the benchmark")
    return metrics


if __name__ == "__main__":
    import sys

    for name in (sys.argv[1:] or list(config.REGIONS)):
        run(name)
