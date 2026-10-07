"""Smoke tests for the optimizer-comparison plumbing: the search budget's stop rules, the hypervolume helper and
the generated tpu_like grids.

Run: python test_optimizer_comparison.py
"""

import os
import tempfile

from stream.hardware.architecture.grid_generator import tpu_grid_dict, write_tpu_grid
from stream.opt.search_budget import SearchBudget, hypervolume_3d
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage


class FakeClock:
    def __init__(self, budget: SearchBudget):
        self.now = 0.0
        budget.elapsed = lambda: self.now  # type: ignore[method-assign]


def test_budget_converges() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        budget = SearchBudget(max_time_s=1000, window_s=100, rel_tol=0.01, trace_path=os.path.join(tmp, "trace.csv"))
        clock = FakeClock(budget)
        budget.record(10, 10, 1)  # EDP 100 at t=0
        clock.now = 50
        budget.record(9, 10, 1)  # EDP 90
        clock.now = 120
        assert not budget.should_stop(), "improved by 10% over the last window"
        budget.record(float("inf"), float("inf"), 1)  # infeasible: counted, never best
        clock.now = 151
        assert budget.should_stop() and budget.stop_reason == "converged"
        summary = budget.summary()
        assert summary["best_edp"] == 90 and summary["n_eval"] == 3
        assert summary["time_to_within_1pct"] == 50
        with open(os.path.join(tmp, "trace.csv")) as f:
            assert len(f.readlines()) == 4


def test_budget_times_out() -> None:
    budget = SearchBudget(max_time_s=10, window_s=100, rel_tol=0.01)
    clock = FakeClock(budget)
    budget.record(1, 1, 1)
    clock.now = 10
    assert budget.should_stop() and budget.stop_reason == "time"
    budget.stop("exhausted")
    assert budget.stop_reason == "time", "the first reason wins"


def test_hypervolume() -> None:
    assert hypervolume_3d([(0, 0, 0)], (1, 1, 1)) == 1
    assert hypervolume_3d([(0.5, 0, 0), (0, 0.5, 0.5)], (1, 1, 1)) == 0.625
    assert hypervolume_3d([(2, 0, 0)], (1, 1, 1)) == 0


def test_grid() -> None:
    grid = tpu_grid_dict(4, 4)
    links = [c for c in grid["core_connectivity"] if c["type"] == "link"]
    assert len(links) == 2 * 4 * 3 + 16, "24 mesh links + one link per core to the SIMD core"
    hardware_path, mapping_path = write_tpu_grid(2, 2)
    accelerator = AcceleratorParserStage.parse_accelerator_from_yaml(None, hardware_path)  # type: ignore[arg-type]
    assert len(accelerator.core_list) == 6 and accelerator.offchip_core_id == 5
    assert os.path.exists(mapping_path)


if __name__ == "__main__":
    test_budget_converges()
    test_budget_times_out()
    test_hypervolume()
    test_grid()
    print("All optimizer comparison tests passed.")
