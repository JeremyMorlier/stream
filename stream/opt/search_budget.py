"""Wall-clock budget and convergence tracking shared by the design-space searches.

The searches in this repo are normally bounded by iteration counts (generations, evaluations). To compare them
fairly they need a common stopping rule instead: a maximum wall time and a convergence criterion on the
best-so-far energy-delay product (EDP). `SearchBudget` is that rule. A search calls `record(...)` after every
measured design and polls `should_stop()` before starting the next one; every record is appended to a CSV trace
so convergence curves can be drawn afterwards.

The clock starts at construction, so anything the method does before its first measurement (CACTI precompute,
cost-model warm-up, nested searches) is charged to it.

The object is picklable (it holds no open file handle) because it travels through stage kwargs, which some
stages ship to worker processes. Only the process that owns the search should call `record`.
"""

import bisect
import csv
import json
import math
import os
import time
from collections.abc import Callable
from concurrent.futures import Executor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

TRACE_COLUMNS = ("t", "n_eval", "latency", "energy", "area", "edp", "best_edp", "feasible", "info")


class SearchBudget:
    def __init__(
        self,
        max_time_s: float,
        window_s: float,
        rel_tol: float,
        trace_path: str | None = None,
    ):
        """
        Args:
            max_time_s: hard wall-clock limit, in seconds.
            window_s: convergence window. The search counts as converged once the best EDP has improved by less
                than `rel_tol` (relative) over the last `window_s` seconds.
            rel_tol: relative improvement threshold of the convergence rule (0.005 = 0.5%).
            trace_path: CSV file every record is appended to. Overwritten at construction.
        """
        self.max_time_s = max_time_s
        self.window_s = window_s
        self.rel_tol = rel_tol
        self.trace_path = trace_path
        self.t_start = time.perf_counter()
        self.n_eval = 0
        self.best_edp = math.inf
        self.best_point: dict[str, Any] | None = None
        # (time, best_edp) at every improvement, in time order: enough to read the best EDP at any past time.
        self.improvements: list[tuple[float, float]] = []
        self.stop_reason: str | None = None
        self.t_stop: float | None = None
        if trace_path:
            os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
            with open(trace_path, "w", newline="") as f:
                csv.writer(f).writerow(TRACE_COLUMNS)

    def elapsed(self) -> float:
        return time.perf_counter() - self.t_start

    def best_edp_at(self, t: float) -> float:
        """Best EDP known at time `t` (inf before the first feasible point)."""
        index = bisect.bisect_right([time_ for time_, _ in self.improvements], t)
        return self.improvements[index - 1][1] if index else math.inf

    def record(self, latency: float, energy: float, area: float, info: str = "") -> bool:
        """Log one measured design. Returns True when it improved the best EDP."""
        t = self.elapsed()
        self.n_eval += 1
        latency, energy, area = float(latency), float(energy), float(area)
        feasible = all(math.isfinite(v) and v > 0 for v in (latency, energy))
        edp = latency * energy if feasible else math.inf
        improved = edp < self.best_edp
        if improved:
            self.best_edp = edp
            self.best_point = {"t": t, "latency": latency, "energy": energy, "area": area, "edp": edp, "info": info}
            self.improvements.append((t, edp))
        if self.trace_path:
            with open(self.trace_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    (f"{t:.3f}", self.n_eval, latency, energy, area, edp, self.best_edp, int(feasible), info)
                )
        return improved

    def converged(self) -> bool:
        t = self.elapsed()
        if t < self.window_s or not math.isfinite(self.best_edp):
            return False
        previous = self.best_edp_at(t - self.window_s)
        if not math.isfinite(previous):
            return False
        return previous / self.best_edp - 1 < self.rel_tol

    def should_stop(self) -> bool:
        if self.stop_reason is not None:
            return True
        if self.elapsed() >= self.max_time_s:
            self.stop("time")
        elif self.converged():
            self.stop("converged")
        return self.stop_reason is not None

    def time_left(self) -> float:
        return max(0.0, self.max_time_s - self.elapsed())

    def stop(self, reason: str) -> None:
        """Mark the search as finished. The first reason wins, so a search that ran out of candidates after the
        budget already fired keeps `time` / `converged`."""
        if self.stop_reason is None:
            self.stop_reason = reason
            self.t_stop = self.elapsed()

    def time_to_within(self, fraction: float) -> float | None:
        """First time the best EDP came within `fraction` of the final best."""
        for t, edp in self.improvements:
            if edp <= self.best_edp * (1 + fraction):
                return t
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason or "exhausted",
            "time_to_stop": self.t_stop if self.t_stop is not None else self.elapsed(),
            "n_eval": self.n_eval,
            "best_edp": self.best_edp,
            "best_point": self.best_point,
            "time_to_within_1pct": self.time_to_within(0.01),
            "max_time_s": self.max_time_s,
            "window_s": self.window_s,
            "rel_tol": self.rel_tol,
        }

    def save_summary(self, path: str, **extra: Any) -> None:
        with open(path, "w") as f:
            json.dump({**self.summary(), **extra}, f, indent=2, default=str)

    def sub_budget(self, time_cap_s: float) -> "SubBudget":
        """A per-candidate view for nested searches (see `SubBudget`), started now."""
        return SubBudget(self, time_cap_s)


class SubBudget:
    """The slice of a parent `SearchBudget` one candidate of an outer search may spend.

    It has its own clock and stops after `time_cap_s` (or whatever the parent has left, if less), or as soon as
    the parent stops. It has no convergence rule of its own -- the inner stages stop on their own criteria --
    and its `stop` never stops the parent. Records go straight to the parent, so the trace, the best EDP and the
    convergence window stay global. Same interface as `SearchBudget` for everything the stages call."""

    def __init__(self, parent: SearchBudget, time_cap_s: float):
        self.parent = parent
        self.t_start = time.perf_counter()
        self.max_time_s = min(time_cap_s, parent.time_left())
        self.rel_tol = parent.rel_tol
        self.stop_reason: str | None = None

    def elapsed(self) -> float:
        return time.perf_counter() - self.t_start

    def time_left(self) -> float:
        return max(0.0, self.max_time_s - self.elapsed())

    def record(self, latency: float, energy: float, area: float, info: str = "") -> bool:
        return self.parent.record(latency, energy, area, info=info)

    def stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason

    def should_stop(self) -> bool:
        if self.stop_reason is None:
            if self.parent.should_stop():
                self.stop("parent")
            elif self.elapsed() >= self.max_time_s:
                self.stop("time")
        return self.stop_reason is not None


def _hypervolume_2d(points: list[tuple[float, float]], reference: tuple[float, float]) -> float:
    """Area dominated by `points` (minimization) inside the box bounded by `reference`."""
    area, best_y = 0.0, reference[1]
    for x, y in sorted(points):
        if y < best_y:
            area += (reference[0] - x) * (best_y - y)
            best_y = y
    return area


def hypervolume_3d(points: list[tuple[float, ...]], reference: tuple[float, ...] | list[float]) -> float:
    """Exact hypervolume of a 3-objective minimization front, by slicing along the first objective. Points not
    strictly better than `reference` on every objective contribute nothing."""
    points = [p for p in points if all(v < r for v, r in zip(p, reference, strict=True))]
    if not points:
        return 0.0
    points.sort(key=lambda p: p[0])
    volume = 0.0
    for index, point in enumerate(points):
        next_x = points[index + 1][0] if index + 1 < len(points) else reference[0]
        if next_x > point[0]:
            slice_front = [(p[1], p[2]) for p in points[: index + 1]]
            volume += (next_x - point[0]) * _hypervolume_2d(slice_front, (reference[1], reference[2]))
    return volume


def map_until_deadline(
    executor: "Executor",
    func: Callable[[Any], Any],
    items: list[Any],
    budget: SearchBudget | None,
    fallback: Any,
    on_result: Callable[[int, Any], None] | None = None,
) -> list[Any]:
    """`executor.map(func, items)` that honours the budget's hard time limit.

    Results are handed to `on_result(index, result)` as they complete (so the trace advances in real time instead
    of once per batch). Whatever is still running when `max_time_s` is reached gets `fallback`, and the worker
    processes are terminated: a single slow evaluation would otherwise hold the run far past its budget. The
    executor is unusable afterwards, which is fine since the budget has fired."""
    futures = {executor.submit(func, item): index for index, item in enumerate(items)}
    results: list[Any] = [fallback] * len(items)
    timeout = budget.time_left() if budget is not None else None
    try:
        for future in as_completed(futures, timeout=timeout):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception:
                results[index] = fallback
            if on_result is not None:
                on_result(index, results[index])
    except FuturesTimeoutError:
        budget.stop("time")
        for future in futures:
            future.cancel()
        for process in list(getattr(executor, "_processes", {}).values()):
            process.terminate()
    return results
