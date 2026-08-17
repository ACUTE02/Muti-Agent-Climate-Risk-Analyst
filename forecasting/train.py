"""Build, train, and save the LSTM drought forecaster.

Run standalone:  python -m forecasting.train
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from forecasting import config
from forecasting.split import Dataset, prepare_dataset


def set_seeds(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(n_features: int = len(config.FEATURES)) -> Sequential:
    model = Sequential([
        Input(shape=(config.SEQ_LEN, n_features)),
        LSTM(config.LSTM_UNITS_1, return_sequences=True),
        Dropout(config.DROPOUT),
        LSTM(config.LSTM_UNITS_2),
        Dropout(config.DROPOUT),
        Dense(config.DENSE_UNITS, activation="relu"),
        Dense(config.HORIZON),          # SPI at t+1, t+2, t+3
    ])
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE),
                  loss="mse", metrics=["mae"])
    return model


def save_history(history, early_stopping: EarlyStopping,
                 region: str = config.DEFAULT_REGION,
                 path: Path | None = None) -> dict:
    """Persist the per-epoch loss curve to ``models/training_history.json``.

    Diagnostic only — nothing downstream reads it. ``best_epoch`` is the epoch
    whose weights ``restore_best_weights`` put back into the returned model.
    """
    curves = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    val_loss = curves.get("val_loss", [])
    best_index = getattr(early_stopping, "best_epoch", None)
    if best_index is None and val_loss:
        best_index = int(np.argmin(val_loss))

    payload = {
        "region": region,
        "epochs_run": len(val_loss),
        "epochs_configured": config.EPOCHS,
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        "restore_best_weights": True,
        "best_epoch_1indexed": None if best_index is None else int(best_index) + 1,
        "best_val_loss": None if not val_loss else float(min(val_loss)),
        "stopped_epoch_1indexed": int(getattr(early_stopping, "stopped_epoch", 0)) + 1,
        **curves,
    }
    out = path or config.history_path(region)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def train(ds: Dataset | None = None, region: str = config.DEFAULT_REGION):
    """Train one region. Architecture, hyperparameters, callbacks and seeds are
    identical for every region — no per-region tuning, so the results compare."""
    config.check_region(region)
    set_seeds()
    ds = ds or prepare_dataset(region)
    X_train, y_train = ds.get("train")
    X_val, y_val = ds.get("val")

    print(f"[{region}] train {X_train.shape} | val {X_val.shape} | "
          f"test {ds.get('test')[0].shape}")

    model = build_model(X_train.shape[-1])
    model.summary()

    early_stopping = EarlyStopping(monitor="val_loss",
                                   patience=config.EARLY_STOPPING_PATIENCE,
                                   restore_best_weights=True)
    callbacks = [
        early_stopping,
        ModelCheckpoint(str(config.model_path(region)), monitor="val_loss",
                        save_best_only=True),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )
    save_history(history, early_stopping, region)
    # EarlyStopping restored the best weights; persist them in .keras format.
    model.save(config.model_path(region))
    print(f"saved -> {config.model_path(region)}")
    return model, history, ds


if __name__ == "__main__":
    import sys

    train(region=sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_REGION)
