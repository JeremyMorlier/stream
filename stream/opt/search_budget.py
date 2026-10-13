"""Wall-clock budget and convergence tracking shared by the design-space searches.

The searches in this repo are normally bounded by iteration counts (generations, evaluations). To compare them
fairly they need a common stopping rule instead: a maximum wall time and a convergence criterion on the
best-so-far energy-delay product (EDP). `SearchBudget` is that rule. A search calls `record(...)` after every
measured design and polls `should_stop()` only before starting a new generation: work already started -- the
current generation, or a search without generations -- always runs to completion, so a run may end past
`max_time_s`. Every record is appended to a CSV trace so convergence curves can be drawn afterwards.

With an `artifact_dir`, a record can also carry an artifact (a JSON-safe dict, or a zero-argument callable
returning one -- see `stream.visualization.design_record`): it is written to `<artifact_dir>/<n_eval>.json` when
the point is not dominated by an earlier saved one, and deleted again once a later point dominates it. The
directory therefore ends up holding exactly the (latency, energy, area) Pareto front of the run, and the trace's
`artifact` column names each point's file. The callable is only invoked for points that may join the front.

The clock starts at construction, so anything the method does before its first measurement (CACTI precompute,
cost-model warm-up, nested searches) is charged to it.

The object is picklable (it holds no open file handle) because it travels through stage kwargs, which some
stages ship to worker processes. Only the process that owns the search should call `record`.
"""

import bisect
import csv
import json
import logging
import math
import os
import time
from collections.abc import Callable
from concurrent.futures import Executor, as_completed
from typing import Any

TRACE_COLUMNS = ("t", "n_eval", "latency", "energy", "area", "edp", "best_edp", "feasible", "info", "artifact")

logger = logging.getLogger(__name__)


class SearchBudget:
    """The common stopping rule of the compared searches: a wall-clock limit plus convergence of the best EDP,
    with a CSV trace of every measured design (see the module docstring)."""

    def __init__(
        self,
        max_time_s: float,
        window_s: float,
        rel_tol: float,
        trace_path: str | None = None,
        artifact_dir: str | None = None,
    ):
        """
        Args:
            max_time_s: hard wall-clock limit, in seconds.
            window_s: convergence window. The search counts as converged once the best EDP has improved by less
                than `rel_tol` (relative) over the last `window_s` seconds.
            rel_tol: relative improvement threshold of the convergence rule (0.005 = 0.5%).
            trace_path: CSV file every record is appended to. Overwritten at construction.
            artifact_dir: where the artifacts of the front's points are kept (see the module docstring).
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
        self.artifact_dir = artifact_dir
        # (latency, energy, area) and n_eval of every point whose artifact is on disk: the running front.
        self.saved_front: list[tuple[tuple[float, float, float], int]] = []
        if artifact_dir:
            os.makedirs(artifact_dir, exist_ok=True)
        if trace_path:
            os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
            with open(trace_path, "w", newline="") as f:
                csv.writer(f).writerow(TRACE_COLUMNS)

    def elapsed(self) -> float:
        """Seconds since the budget was created."""
        return time.perf_counter() - self.t_start

    def best_edp_at(self, t: float) -> float:
        """Best EDP known at time `t` (inf before the first feasible point)."""
        index = bisect.bisect_right([time_ for time_, _ in self.improvements], t)
        return self.improvements[index - 1][1] if index else math.inf

    def record(self, latency: float, energy: float, area: float, info: str = "", artifact: Any = None) -> bool:
        """Log one measured design, with its optional `artifact` (see the module docstring). Returns True when it
        improved the best EDP."""
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
        artifact_id = ""
        if feasible and artifact is not None and self.artifact_dir:
            artifact_id = self._save_artifact((latency, energy, area), artifact)
        if self.trace_path:
            with open(self.trace_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    (
                        f"{t:.3f}",
                        self.n_eval,
                        latency,
                        energy,
                        area,
                        edp,
                        self.best_edp,
                        int(feasible),
                        info,
                        artifact_id,
                    )
                )
        return improved

    def is_dominated(self, latency: float, energy: float, area: float) -> bool:
        """Whether a saved point is at least as good as this one on every objective, i.e. its artifact would not
        be kept. Also usable on a copy of the budget in a worker process, to skip building an artifact."""
        point = (float(latency), float(energy), float(area))
        return any(all(s <= p for s, p in zip(saved, point, strict=True)) for saved, _ in self.saved_front)

    def _save_artifact(self, point: tuple[float, float, float], artifact: Any) -> str:
        """Write the artifact of a point joining the front and delete those it dominates; returns its id ("" when
        nothing was saved). A failing artifact never stops the search."""
        assert self.artifact_dir is not None
        if self.is_dominated(*point):
            return ""
        try:
            data = artifact() if callable(artifact) else artifact
            if data is None:
                return ""
            path = os.path.join(self.artifact_dir, f"{self.n_eval}.json")
            with open(path, "w") as f:
                json.dump(data, f, allow_nan=False, separators=(",", ":"))
        except Exception:
            logger.warning(f"SearchBudget: could not save the artifact of evaluation {self.n_eval}.", exc_info=True)
            return ""
        kept = []
        for saved, n_eval in self.saved_front:
            if all(p <= s for p, s in zip(point, saved, strict=True)):
                try:
                    os.remove(os.path.join(self.artifact_dir, f"{n_eval}.json"))
                except OSError:
                    pass
            else:
                kept.append((saved, n_eval))
        self.saved_front = [*kept, (point, self.n_eval)]
        return str(self.n_eval)

    def converged(self) -> bool:
        """Whether the best EDP improved by less than `rel_tol` over the last `window_s` seconds. Never true
        before a full window has passed or before the first feasible point."""
        t = self.elapsed()
        if t < self.window_s or not math.isfinite(self.best_edp):
            return False
        previous = self.best_edp_at(t - self.window_s)
        if not math.isfinite(previous):
            return False
        return previous / self.best_edp - 1 < self.rel_tol

    def should_stop(self) -> bool:
        """Whether a search should start no new generation: the budget was stopped, ran out of time, or
        converged. Latches -- once true it stays true, with the first reason kept in `stop_reason`."""
        if self.stop_reason is not None:
            return True
        if self.elapsed() >= self.max_time_s:
            self.stop("time")
        elif self.converged():
            self.stop("converged")
        return self.stop_reason is not None

    def time_left(self) -> float:
        """Seconds left before `max_time_s` (0 once it has passed)."""
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
        """The run's outcome as plain data: why and when it stopped, how many designs it measured, the best EDP
        and the point that reached it, and how long it took to get within 1% of that best."""
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
        """Write `summary()`, plus any `extra` fields, as JSON to `path`."""
        with open(path, "w") as f:
            json.dump({**self.summary(), **extra}, f, indent=2, default=str)


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


def map_recording(
    executor: "Executor",
    func: Callable[[Any], Any],
    items: list[Any],
    fallback: Any,
    on_result: Callable[[int, Any], None] | None = None,
) -> list[Any]:
    """`executor.map(func, items)` that hands each result to `on_result(index, result)` as soon as it completes,
    so the trace advances in real time instead of once per batch. Every item runs to completion, whatever the
    budget says; an evaluation that raises gets `fallback`."""
    futures = {executor.submit(func, item): index for index, item in enumerate(items)}
    results: list[Any] = [fallback] * len(items)
    for future in as_completed(futures):
        index = futures[future]
        try:
            results[index] = future.result()
        except Exception:
            results[index] = fallback
        if on_result is not None:
            on_result(index, results[index])
    return results
