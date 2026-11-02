"""Tests for the convergence stop of the core NSGA2 (`_run_nsga2` in
`stream.stages.generation.core_architecture_exploration`).

On raw objectives the front's hypervolume barely moves once a front end is far below the reference, so the old rule
stopped searches whose best area was still halving every few generations. The rule now takes the hypervolume on log
objectives and also waits for the best value of each objective to settle:

- a front whose hypervolume stalls while one of its ends still falls is not converged,
- halving a small area counts on the log hypervolume as halving a large one does,
- a search on a fitness with a floor stops before its generation cap.

Run with `python tests/test_core_convergence.py`, or collect with pytest.
"""

import math
import random

from stream.opt.search_budget import hypervolume_3d
from stream.stages.generation.core_architecture_exploration import (
    CoreGeneLayout,
    CoreParamRanges,
    _front_stalled,
    _log_objectives,
    _make_core_toolbox,
    _run_nsga2,
)


def test_falling_end_is_not_converged() -> None:
    # The hypervolume gained nothing over the window, but the best area halved.
    hv_history = [1.0] * 6
    best_history = [[6.0, 377.6, 0.233]] * 5 + [[6.0, 377.6, 0.116]]
    assert not _front_stalled(hv_history, best_history, patience=5, rel_tol=0.005)
    best_history[-1] = [6.0, 377.6, 0.233]
    assert _front_stalled(hv_history, best_history, patience=5, rel_tol=0.005)


def test_stalled_extremes_alone_do_not_stop() -> None:
    best_history = [[6.0, 377.6, 0.233]] * 6
    assert not _front_stalled([1.0, 1.0, 1.0, 1.0, 1.0, 1.1], best_history, patience=5, rel_tol=0.005)


def test_no_stop_before_patience() -> None:
    assert not _front_stalled([1.0] * 5, [[1.0, 1.0, 1.0]] * 5, patience=5, rel_tol=0.005)


def test_log_hypervolume_counts_ratios() -> None:
    # The fast core sets latency and energy; the small core is the area end, far below the reference. Halving its
    # area (as tile 10 of the 02_11 attention run did right before it was declared converged) is a 0.2% raw gain.
    fast, mid, small = (6.0, 377.6, 0.8), (12.0, 600.0, 0.3), (40.0, 900.0, 0.0568)
    smaller = (40.0, 900.0, 0.0284)
    reference = (60.0, 5000.0, 5.0)
    raw_gain = hypervolume_3d([fast, mid, smaller], reference) / hypervolume_3d([fast, mid, small], reference) - 1
    log_reference = _log_objectives(reference)
    log_gain = (
        hypervolume_3d([_log_objectives(p) for p in (fast, mid, smaller)], log_reference)
        / hypervolume_3d([_log_objectives(p) for p in (fast, mid, small)], log_reference)
        - 1
    )
    assert raw_gain < 0.005 < log_gain
    assert _log_objectives((1.0, math.inf, 1.0)) is None
    assert _log_objectives((1.0, 0.0, 1.0)) is None


def test_search_with_a_floor_converges() -> None:
    random.seed(0)
    layout = CoreGeneLayout(CoreParamRanges())
    toolbox = _make_core_toolbox(layout)
    # Every objective is floored at 1, which a few genes reach: the front must settle and stop the search.
    toolbox.register(
        "evaluate",
        lambda genes: tuple(1 + max(0.0, genes[i] - layout.low[i] - 1) for i in range(3)),
    )
    toolbox.register("map", map)
    designs, history = _run_nsga2(
        toolbox, 16, label="test", history_index=0, nb_generations=200, nb_points=5, convergence_patience=5
    )
    generations = max(row[1] for row in history)
    assert designs
    assert generations < 200


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
