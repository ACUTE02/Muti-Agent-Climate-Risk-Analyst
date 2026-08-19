"""Single source of truth for Phase 1: regions, features, split windows, hyperparams.

Every other module imports from here. Nothing in this file depends on the rest of
the package, so it is safe to import from tests.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"

# Per-region artifacts are region-suffixed so regions never overwrite each other.
ONI_PATH = MODELS_DIR / "oni_series.parquet"   # shared: ENSO is a global index
IOD_PATH = MODELS_DIR / "iod_series.parquet"   # shared: the DMI is global too
COMPARISON_PATH = MODELS_DIR / "region_comparison.md"
ROLLING_CHECK_PATH = MODELS_DIR / "rolling_window_check.json"
T1_METRICS_PATH = MODELS_DIR / "metrics_t1_ridge.json"
HORIZON_COMPARISON_PATH = MODELS_DIR / "horizon_comparison.json"
HORIZON_MANIFEST_PATH = MODELS_DIR / "horizon_manifest.json"


def horizon_model_path(region: str, horizon: int) -> Path:
    return MODELS_DIR / f"ridge_t{horizon}_{check_region(region)}.joblib"

for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Regions
# --------------------------------------------------------------------------- #
REGIONS = {
    "rajasthan": {"lat": 26.9, "lon": 75.8, "label": "Rajasthan (Jaipur centroid)"},
    "barmer": {"lat": 25.75, "lon": 71.38, "label": "Barmer (Thar Desert)"},
}
DEFAULT_REGION = "rajasthan"

# --------------------------------------------------------------------------- #
# Data window
# --------------------------------------------------------------------------- #
FETCH_START = "1980-01-01"
FETCH_END = "2024-12-31"

# --------------------------------------------------------------------------- #
# Features / target
# --------------------------------------------------------------------------- #
FEATURES = [
    "rainfall_mm", "spi3", "anomaly", "month_sin", "month_cos",
    "rainfall_mm_lag1", "rainfall_mm_lag3", "rainfall_mm_lag6", "rainfall_mm_lag12",
    "roll3_mean", "roll12_sum", "oni",
]
TARGET = "spi3"         # real SPI-3 (McKee et al. 1993), see split.fit_spi3_params

# Phase 1.6 tests whether the Indian Ocean Dipole adds anything beyond ENSO.
# Kept as a separate list so the before/after comparison runs the *same*
# pipeline twice rather than editing the standing feature set to find out.
FEATURES_IOD = FEATURES + ["iod", "iod_lag1"]
IOD_COMPARISON_PATH = MODELS_DIR / "iod_comparison.json"
SPI_SOURCE = "rainfall_mm"
SPI_ACCUM = "rainfall_roll3_sum"   # 3-month causal accumulation the gamma is fit to
SPI_ACCUM_MONTHS = 3

# Anomaly reference period (Rule C: a *fixed* historical window inside the train range)
BASELINE = slice("1981", "2010")

# --------------------------------------------------------------------------- #
# Chronological split (no shuffling, ever)
# --------------------------------------------------------------------------- #
TRAIN = slice("1980", "2015")
VAL = slice("2016", "2019")
TEST = slice("2020", "2024")

# Phase 1.4 robustness check: four independent historical windows. Window A is the
# standing split above; B/C/D walk the same shape backwards through the record.
# Every window refits its own month_stats / spi_params / scaler on its own train
# partition (Rule A) — they are not reused across windows.
ROLLING_WINDOWS = {
    "A": (slice("1980", "2015"), slice("2016", "2019"), slice("2020", "2024")),
    "B": (slice("1980", "2010"), slice("2011", "2014"), slice("2015", "2019")),
    "C": (slice("1980", "2005"), slice("2006", "2009"), slice("2010", "2014")),
    "D": (slice("1980", "2000"), slice("2001", "2004"), slice("2005", "2009")),
}

# --------------------------------------------------------------------------- #
# Sequence framing
# --------------------------------------------------------------------------- #
SEQ_LEN = 60      # months of input context (5 years)
HORIZON = 3       # predict SPI at t+1, t+2, t+3

# --------------------------------------------------------------------------- #
# Training hyperparameters
# --------------------------------------------------------------------------- #
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = 32
DROPOUT = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 8
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Phase 1.3 ablation — deliberately smaller, slower, more patient than the model
# above. Pre-committed values: this is a one-shot check, not a tuning loop.
# --------------------------------------------------------------------------- #
SMALL_LSTM_UNITS = 16
SMALL_DROPOUT = 0.3
SMALL_LEARNING_RATE = 1e-4
SMALL_EPOCHS = 150
SMALL_EARLY_STOPPING_PATIENCE = 15

RIDGE_ALPHAS = (0.1, 1, 10, 100, 1000)

# --------------------------------------------------------------------------- #
# Heat Stress agent (second risk type) — same split, same discipline
# --------------------------------------------------------------------------- #
HEAT_TARGET = "heat_anomaly"          # standardised monthly Tmax anomaly
HEAT_SOURCE = "tmax_c"
HEAT_FEATURES = [
    "heat_anomaly",
    "heat_anomaly_lag1", "heat_anomaly_lag2", "heat_anomaly_lag3",
    "heat_anomaly_lag12",
    "heat_roll3_mean", "heat_roll6_mean",
    "month_sin", "month_cos", "oni",
]
HEAT_LAGS = (1, 2, 3, 12)
HEAT_ROLL_WINDOWS = (3, 6)
HEAT_LOOKBACKS = (12, 24, 60)

# IMD heat wave criteria for the plains, adapted to a single grid point.
# Sources cited in heat/target.py — https://ndma.gov.in/Natural-Hazards/Heat-Wave
HEAT_WAVE_MIN_TMAX = 40.0        # departure rules only apply at/above this Tmax
HEAT_WAVE_ABS_C = 45.0           # absolute heat wave threshold
HEAT_WAVE_DEP_C = 4.5            # departure from normal for a heat wave
SEVERE_HEAT_WAVE_ABS_C = 47.0
SEVERE_HEAT_WAVE_DEP_C = 6.4
HEAT_WAVE_SPELL_DAYS = 2         # IMD needs >=2 consecutive qualifying days
NORMAL_WINDOW_DAYS = 7           # +/- days pooled for the day-of-year normal

# Heat Phase 1.1 candidate targets. heat_anomaly (the monthly mean) stays the
# reference point; the other two test whether a mean dilutes short spells.
HEAT_EXTREME_TARGET = "heat_extreme"        # hottest single day of the month
HEAT_COUNT_TARGET = "heatwave_day_count"    # IMD-criteria heat wave days, zero-inflated
HEAT_TARGETS = (HEAT_TARGET, HEAT_EXTREME_TARGET, HEAT_COUNT_TARGET)

# Cross-agent dependency: SPI-3 comes from the Drought Agent. Land-atmosphere
# feedback (dry soil -> less evaporative cooling -> hotter days) is the motivation.
HEAT_SPI3_FEATURES = ["spi3", "spi3_lag1", "spi3_lag3"]
HEAT_FEATURES_SPI3 = HEAT_FEATURES + HEAT_SPI3_FEATURES
PRE_MONSOON_MONTHS = (4, 5, 6)              # where heat wave counts are actually nonzero
WINTER_MONTHS = (12, 1, 2)                  # sanity check: these should be ~zero

HEAT_PHASE11_PATH = MODELS_DIR / "heat_phase11.json"
HEAT_COMPARISON_PATH = MODELS_DIR / "heat_region_comparison.md"
HEAT_MANIFEST_PATH = MODELS_DIR / "heat_manifest.json"


def heat_metrics_path(region: str) -> Path:
    return MODELS_DIR / f"heat_metrics_{check_region(region)}.json"


def heat_climatology_path(region: str) -> Path:
    return MODELS_DIR / f"heat_climatology_{check_region(region)}.joblib"


def heat_scaler_path(region: str) -> Path:
    return MODELS_DIR / f"heat_scaler_{check_region(region)}.joblib"


def heat_model_path(region: str, horizon: int) -> Path:
    return MODELS_DIR / f"ridge_heat_t{horizon}_{check_region(region)}.joblib"

# --------------------------------------------------------------------------- #
# Risk thresholds (Phase 5 will calibrate against real NDMA drought records)
# --------------------------------------------------------------------------- #
SPI_MODERATE = -1.0   # spi <  -1.0 -> at least "Moderate"
SPI_SEVERE = -1.5     # spi <  -1.5 -> "Severe"


def check_region(region: str) -> str:
    """Fail loudly on an unsupported region rather than loading the wrong model."""
    if region not in REGIONS:
        raise ValueError(f"Unsupported region {region!r}. Supported: {list(REGIONS)}")
    return region


def raw_path(region: str) -> Path:
    return DATA_RAW / f"{check_region(region)}_raw.parquet"


def daily_path(region: str) -> Path:
    """Daily frame — needed by the Heat Stress agent, which counts heat wave
    *days* and so cannot work from monthly aggregates."""
    return DATA_RAW / f"{check_region(region)}_daily.parquet"


# --- Rolling recent cache (Phase 8) ---------------------------------------- #
# Deliberately *separate files* from raw_path/daily_path. Those two are the fixed
# 1980-2024 archive that every published skill score in PROJECT_LOG.md and
# EVALUATION.md was measured against; appending live data to them would silently
# move the TEST slice and invalidate all of it. The refresh path only ever writes
# here, and the evaluation path only ever reads there.
def recent_path(region: str) -> Path:
    return DATA_RAW / f"{check_region(region)}_recent.parquet"


def recent_daily_path(region: str) -> Path:
    return DATA_RAW / f"{check_region(region)}_recent_daily.parquet"


def processed_path(region: str) -> Path:
    return DATA_PROCESSED / f"{check_region(region)}_clean.parquet"


def model_path(region: str) -> Path:
    return MODELS_DIR / f"lstm_drought_model_{check_region(region)}.keras"


def scaler_path(region: str) -> Path:
    return MODELS_DIR / f"scaler_{check_region(region)}.joblib"


def month_stats_path(region: str) -> Path:
    return MODELS_DIR / f"month_stats_{check_region(region)}.joblib"


def spi_params_path(region: str) -> Path:
    return MODELS_DIR / f"spi_params_{check_region(region)}.joblib"


def metrics_path(region: str) -> Path:
    return MODELS_DIR / f"metrics_{check_region(region)}.json"


def plot_path(region: str) -> Path:
    return MODELS_DIR / f"test_forecast_plot_{check_region(region)}.png"


def history_path(region: str) -> Path:
    return MODELS_DIR / f"training_history_{check_region(region)}.json"


# --- Phase 1.3 ablation variants ------------------------------------------- #
def ridge_metrics_path(region: str) -> Path:
    return MODELS_DIR / f"metrics_ridge_{check_region(region)}.json"


def lstm_small_model_path(region: str) -> Path:
    return MODELS_DIR / f"lstm_small_{check_region(region)}.keras"


def lstm_small_metrics_path(region: str) -> Path:
    return MODELS_DIR / f"metrics_lstm_small_{check_region(region)}.json"


def lstm_small_history_path(region: str) -> Path:
    return MODELS_DIR / f"training_history_lstm_small_{check_region(region)}.json"
