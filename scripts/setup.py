"""Clone -> working local API, in one command.

    python -m scripts.setup

**Why this exists.** Every artifact the running system loads at request time is
regenerable, and therefore git-ignored: the fetched weather data, the fitted
scaler and SPI-3 gamma parameters, the per-horizon Ridge models, and the ChromaDB
index. A fresh clone legitimately has none of them. That is correct — they are
derived data, not source — but it means "clone and run" was previously a
multi-step ritual documented nowhere. This script is that ritual, automated and
verified.

**The non-obvious part, found by actually auditing it.** `forecast_drought_risk()`
loads `scaler_<region>.joblib` and `spi_params_<region>.joblib` at runtime, but
**neither `t1_model` nor `recursive` writes them** — both call
`prepare_dataset(..., save=False)`. The only modules that persist them are
`forecasting.train` and `forecasting.evaluate`, i.e. the *LSTM* path. So the
obvious-looking sequence "fetch, t1_model, recursive" produces a manifest
pointing at models that load, and a tool that then dies on a missing scaler. On
this machine those files existed only as a leftover side effect of Phase-1 LSTM
training — a model that has no skill and no longer serves any forecast.

Step 3 below fixes that directly by persisting exactly those artifacts, so
setting up does not require training a model nobody uses.

Steps are idempotent: anything already present is skipped unless --force.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from forecasting import config as fconfig
from retrieval import config as rconfig


@dataclass
class Step:
    name: str
    why: str
    outputs: Callable[[], list[Path]]
    run: Callable[[], None]
    needs_api_key: bool = False
    costs_quota: str = ""


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def _fetch_data() -> None:
    from forecasting.fetch_data import load_or_fetch_daily, load_or_fetch_region

    for region in fconfig.REGIONS:
        load_or_fetch_region(region)
        load_or_fetch_daily(region)


def _fetch_oni() -> None:
    from forecasting.enso import fetch_oni

    fetch_oni()


def _persist_runtime_artifacts() -> None:
    """Write the scaler / month_stats / SPI-3 gamma params the tool loads.

    This is the step the obvious sequence misses — see the module docstring.
    It calls the project's own `prepare_dataset` rather than reimplementing the
    fit, so the artifacts are identical to the ones every phase measured with.
    """
    from forecasting.split import prepare_dataset

    for region in fconfig.REGIONS:
        prepare_dataset(region, save=True)


def _train_t1() -> None:
    from forecasting.t1_model import run as t1_run

    t1_run()


def _train_horizons() -> None:
    from forecasting.recursive import run as recursive_run

    recursive_run()


def _build_index() -> None:
    from retrieval.build import main as build_main

    build_main()


STEPS: list[Step] = [
    Step(
        name="weather data",
        why="Open-Meteo monthly + daily pulls for every region, cached to data/raw/.",
        outputs=lambda: [fconfig.raw_path(r) for r in fconfig.REGIONS],
        run=_fetch_data,
    ),
    Step(
        name="ENSO series",
        why="NOAA Oceanic Nino Index, the one exogenous feature that survived testing.",
        outputs=lambda: [fconfig.ONI_PATH],
        run=_fetch_oni,
    ),
    Step(
        name="runtime scaler + SPI-3 parameters",
        why=("The artifacts forecast_drought_risk() loads at request time. "
             "Neither t1_model nor recursive writes these — see the module "
             "docstring. Without this step the API starts and then fails on the "
             "first drought question."),
        outputs=lambda: [p for r in fconfig.REGIONS
                         for p in (fconfig.scaler_path(r),
                                   fconfig.spi_params_path(r))],
        run=_persist_runtime_artifacts,
    ),
    Step(
        name="t+1 Ridge models",
        why="The one validated forecast in the project (skill ~+0.21 to +0.26).",
        outputs=lambda: [fconfig.horizon_model_path(r, 1) for r in fconfig.REGIONS],
        run=_train_t1,
    ),
    Step(
        name="t+2 / t+3 models and horizon manifest",
        why="The remaining horizons, plus the manifest the tool reads to decide "
            "which model answers which horizon and with what confidence label.",
        outputs=lambda: [fconfig.HORIZON_MANIFEST_PATH]
        + [fconfig.horizon_model_path(r, h)
           for r in fconfig.REGIONS for h in (2, 3)],
        run=_train_horizons,
    ),
    Step(
        name="ChromaDB retrieval index",
        why="Embeds and indexes the 9-document corpus so reports can cite sources.",
        outputs=lambda: [rconfig.CHUNKS_PATH,
                         rconfig.CHROMA_DIR / "chroma.sqlite3"],
        needs_api_key=True,
        costs_quota=("embedding quota (a separate budget from the 20/day "
                     "generate_content cap); the build backs off on 429 and "
                     "resumes from disk if interrupted"),
        run=_build_index,
    ),
]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _has_api_key() -> bool:
    import os

    if any(os.environ.get(v) for v in rconfig.API_KEY_ENV_VARS):
        return True
    env_file = fconfig.REPO_ROOT / ".env"
    return env_file.exists() and "API_KEY" in env_file.read_text(
        encoding="utf-8", errors="ignore")


def step_is_done(step: Step) -> bool:
    return all(p.exists() for p in step.outputs())


def verify() -> tuple[bool, list[str]]:
    """Exactly what /health checks, so setup and health cannot disagree."""
    missing: list[str] = []
    for step in STEPS:
        missing += [p.name for p in step.outputs() if not p.exists()]
    return not missing, sorted(set(missing))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate everything the local API needs to run.")
    parser.add_argument("--force", action="store_true",
                        help="re-run steps even if their outputs already exist")
    parser.add_argument("--skip-index", action="store_true",
                        help="skip the ChromaDB build (needs an API key and "
                             "spends embedding quota)")
    parser.add_argument("--check", action="store_true",
                        help="report what is missing and exit, changing nothing")
    args = parser.parse_args(argv)

    if args.check:
        ok, missing = verify()
        print("ready" if ok else f"missing {len(missing)} artifact(s):")
        for name in missing:
            print(f"  - {name}")
        return 0 if ok else 1

    print(f"Setting up {fconfig.REPO_ROOT}\n")
    for number, step in enumerate(STEPS, 1):
        if step.name == "ChromaDB retrieval index" and args.skip_index:
            print(f"[{number}/{len(STEPS)}] {step.name}: SKIPPED (--skip-index)")
            continue
        if step_is_done(step) and not args.force:
            print(f"[{number}/{len(STEPS)}] {step.name}: already present, skipping")
            continue
        if step.needs_api_key and not _has_api_key():
            print(f"[{number}/{len(STEPS)}] {step.name}: SKIPPED — no API key. "
                  f"Set GEMINI_API_KEY (or put it in .env) and re-run.")
            continue

        print(f"[{number}/{len(STEPS)}] {step.name}")
        print(f"      {step.why}")
        if step.costs_quota:
            print(f"      cost: {step.costs_quota}")
        started = time.time()
        try:
            step.run()
        except Exception as exc:
            print(f"      FAILED after {time.time() - started:.1f}s: "
                  f"{type(exc).__name__}: {exc}")
            return 1
        print(f"      done in {time.time() - started:.1f}s")

    ok, missing = verify()
    print()
    if ok:
        print("Setup complete. Start the API with:")
        print("    uvicorn api.app:app --reload --port 8000")
        return 0
    print(f"Setup incomplete — still missing: {', '.join(missing)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
