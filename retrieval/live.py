"""Run report-time fetches in parallel under one deadline.

A report waits on every live source (IMD, NASA POWER, data.gov.in). Run one
after another, each with its own retries, a single slow host could hold a
report open for minutes. These run concurrently, and whatever has not answered
by the deadline is replaced by the caller's "unavailable" record — reported,
never silently dropped. The abandoned request finishes in the background,
bounded by its own per-request timeout.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, TypeVar

T = TypeVar("T")


def gather_with_deadline(calls: list[Callable[[], T]], deadline_s: float,
                         on_timeout: Callable[[int], T]) -> list[T]:
    """Results in call order; ``on_timeout(i)`` stands in for any call not done."""
    pool = ThreadPoolExecutor(max_workers=max(1, len(calls)))
    futures = [pool.submit(call) for call in calls]
    wait(futures, timeout=deadline_s)
    pool.shutdown(wait=False, cancel_futures=True)
    return [f.result() if f.done() else on_timeout(i)
            for i, f in enumerate(futures)]
