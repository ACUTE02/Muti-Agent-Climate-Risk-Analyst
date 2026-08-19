"""The one place the skill-score formula lives.

``skill = 1 - RMSE_model / RMSE_reference`` was spelled out by hand ten times
across ``forecasting/evaluate.py``, ``forecasting/t1_model.py``,
``heat/model.py`` and ``heat/phase11.py``. Every one of those numbers is already
published in PROJECT_LOG.md, EVALUATION.md or models/*.json, so this module
exists to remove the duplication *without moving a single published value*: the
arithmetic below is what each call site was already doing, and the consolidation
was landed only after a before/after snapshot diffed clean.

Why here and not next to ``check_grounding()``: ``orchestrator/grounding.py``
sits above both ``forecasting`` and ``heat``, and putting a scoring primitive
there would make the two model packages import the orchestrator to compute a
ratio. ``heat`` already imports from ``forecasting``, so this is the lowest
shared point that does not invert the dependency.

Reading the sign: 0 means "exactly as good as the reference", positive is
better, negative is worse. Which reference is in play — climatology in most call
sites, persistence in the heat ones — stays the caller's decision.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error

__all__ = ["rmse", "round4", "skill_score", "skill_from_predictions"]


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, as a plain float rather than a numpy scalar."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def skill_score(rmse_model: float, rmse_reference: float,
                *, on_zero_reference: float | None = None) -> float | None:
    """``1 - RMSE_model / RMSE_reference``, guarded against a zero reference.

    ``on_zero_reference`` is what to return when the reference RMSE is zero — a
    perfect baseline, which no run in this project has ever produced. The guard
    used to differ per call site (two returned ``None``, one returned NaN, two
    would have raised); it is uniform here. Since no reference RMSE in this
    project is zero, unifying it changes no computed value — verified by the
    before/after snapshot rather than argued.
    """
    if not rmse_reference:
        return on_zero_reference
    return float(1 - rmse_model / rmse_reference)


def skill_from_predictions(y_true, y_pred, y_reference,
                           *, on_zero_reference: float | None = None):
    """Skill straight from arrays, for callers holding predictions not RMSEs."""
    return skill_score(rmse(y_true, y_pred), rmse(y_true, y_reference),
                       on_zero_reference=on_zero_reference)


def round4(value: float | None) -> float | None:
    """Round to the 4dp this project publishes at, passing ``None`` through."""
    return None if value is None else round(float(value), 4)
