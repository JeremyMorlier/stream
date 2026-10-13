"""Tests for the tiling GA's hypervolume fitness and convergence (`TilingExplorationStage` with
`tiling_fitness="hypervolume_contribution"` and `hypervolume_patience`).

The fitness steers which tilings survive, and the convergence rule decides when hours of search end, so the
properties checked are the ones whose violation would only show as a quietly worse or longer search:

- a candidate's contribution is the volume only it covers: zero when every point is dominated, exactly its own
  box when it alone owns a corner of the front,
- the reference point is fixed by the first generation, so a later outlier beyond it changes nothing,
- the search stops once the hypervolume stalls for `hypervolume_patience` generations, and not while it grows,
- individuals are re-scored with their current contribution.

All synthetic: no workload; the stage is built with `object.__new__` and fed candidate points directly.

Run directly with `python tests/test_tiling_hypervolume.py`, or collect with pytest.
"""

import math
from types import SimpleNamespace

from stream.opt.search_budget import SearchBudget
from stream.stages.generation.tiling_exploration import (
    TilingExplorationStage,
    _candidate_key,
    hypervolume_contributions,
    hypervolume_reference,
)

# Points are chosen in log10 space: 10**x keeps the arithmetic exact enough to compare volumes by hand.
REFERENCE = (2.0, 2.0, 2.0)


def pt(*logs: float) -> tuple[float, ...]:
    return tuple(10.0**value for value in logs)


def test_contributions() -> None:
    points = {
        "a": [pt(0, 1, 1)],  # box (2-0)*(2-1)*(2-1) = 2, partly shared with b
        "b": [pt(1, 0, 1)],
        "dominated": [pt(1.5, 1.5, 1.5)],  # dominated by a and b
        "failed": [],
    }
    total, contributions = hypervolume_contributions(points, REFERENCE)
    # Union of a's box [0,2]x[1,2]x[1,2] (2) and b's [1,2]x[0,2]x[1,2] (2), overlapping on [1,2]x[1,2]x[1,2] (1).
    assert math.isclose(total, 3.0)
    assert math.isclose(contributions["a"], 1.0) and math.isclose(contributions["b"], 1.0)
    assert contributions["dominated"] == 0.0 and contributions["failed"] == 0.0


def test_reference_is_fixed_by_the_first_points() -> None:
    reference = hypervolume_reference([pt(0, 0, 0), pt(1, 1, 1)])
    assert all(math.isclose(r, 1 + math.log10(1.1)) for r in reference)
    # A later outlier beyond the reference adds nothing.
    base, _ = hypervolume_contributions({"a": [pt(0, 0, 0)]}, reference)
    with_outlier, contributions = hypervolume_contributions({"a": [pt(0, 0, 0)], "far": [pt(5, 5, 5)]}, reference)
    assert math.isclose(base, with_outlier) and contributions["far"] == 0.0


def build_stage(patience: int) -> TilingExplorationStage:
    stage = object.__new__(TilingExplorationStage)
    stage.tiling_fitness = "hypervolume_contribution"
    stage.hypervolume_patience = patience
    stage.hypervolume_rel_tol = 0.005
    stage.hypervolume_history = []
    stage.hypervolume_reference = None
    stage.search_budget = SearchBudget(max_time_s=1e9, window_s=math.inf, rel_tol=0.005)
    return stage


def run_generations(stage: TilingExplorationStage, fronts_per_generation: list[dict]) -> list[float]:
    """Feed the stage one generation at a time; `_candidate_points` reads the candidates seen so far."""
    seen: dict[str, list] = {}
    for generation, fronts in enumerate(fronts_per_generation):
        seen.update(fronts)
        stage._candidate_points = lambda keys, seen=seen: {key: seen[key] for key in keys}
        stage._after_generation(generation, [], dict.fromkeys(seen))
        if stage.search_budget.stop_reason:
            break
    return stage.hypervolume_history


def test_stops_when_hypervolume_stalls() -> None:
    stage = build_stage(patience=2)
    history = run_generations(
        stage,
        [{"g0": [pt(1, 1, 1), pt(0.5, 2, 2)]}, {"g1": [pt(1.5, 1.5, 1.5)]}, {"g2": [pt(1.8, 1.8, 1.8)]}, {"g3": []}],
    )
    # Generations 1 and 2 add only dominated points: after 2 stalled generations the rule fires (at generation 2).
    assert stage.search_budget.stop_reason == "converged" and len(history) == 3


def test_keeps_going_while_hypervolume_grows() -> None:
    stage = build_stage(patience=1)
    run_generations(stage, [{"g0": [pt(1, 1, 1)]}, {"g1": [pt(0.5, 0.5, 0.5)]}, {"g2": [pt(0.2, 0.2, 0.2)]}])
    assert stage.search_budget.stop_reason is None


def test_individuals_are_rescored_with_their_contribution() -> None:
    stage = build_stage(patience=None)
    better, dominated = [1, 2], [3, 4]
    key_better, key_dominated = _candidate_key(better), _candidate_key(dominated)
    points = {key_better: [pt(0, 0, 0)], key_dominated: [pt(0.5, 0.5, 0.5)]}
    stage._candidate_points = lambda keys: {key: points[key] for key in keys}

    class Individual(list):
        pass

    individuals = [Individual(better), Individual(dominated)]
    for ind in individuals:
        ind.fitness = SimpleNamespace(values=(0.0,))
    stage._after_generation(0, individuals, dict.fromkeys(points))
    assert individuals[0].fitness.values[0] < 0 and individuals[1].fitness.values == (-0.0,)


if __name__ == "__main__":
    test_contributions()
    test_reference_is_fixed_by_the_first_points()
    test_stops_when_hypervolume_stalls()
    test_keeps_going_while_hypervolume_grows()
    test_individuals_are_rescored_with_their_contribution()
    print("All tiling hypervolume tests passed.")
