"""Tests for `ScheduleExplorationStage`'s branch-and-bound search.

The property that matters is soundness: pruning must never drop a Pareto-optimal schedule. That is checked here
on a synthetic model -- plan, per-design costs and a stubbed "scheduler" -- so it runs in milliseconds instead
of the minutes a real workload would take, and can cover many random instances instead of one.

The stubbed measurement deliberately returns a *coupled* latency/energy above the analytical bounds (that
coupling, from contention and transfers, is exactly what the bounds cannot see), so the test exercises the real
situation rather than one where bounds happen to be tight.

Run directly with `python test_schedule_exploration.py`, or collect with `pytest test_schedule_exploration.py`.
"""

import itertools
import random
from types import SimpleNamespace

from stream.stages.generation.schedule_exploration import ScheduleExplorationStage, dominates


def build_stage(seed: int, nb_classes: int = 3, nb_ranks: int = 4) -> ScheduleExplorationStage:
    """A stage with a synthetic plan: `nb_classes` decision classes of `nb_ranks` designs each, a chain-shaped
    tile DAG and two cores, with `_measure` stubbed by a deterministic pseudo-scheduler."""
    rng = random.Random(seed)
    stage = object.__new__(ScheduleExplorationStage)

    stage.decision_classes = list(range(nb_classes))
    stage.depth_of_class = {class_index: class_index for class_index in range(nb_classes)}
    stage.designs_per_class = [
        [{"core_dict": {"name": f"core_{c}_{r}", "knob": rng.randrange(10**6)}} for r in range(nb_ranks)]
        for c in range(nb_classes)
    ]

    # One cost pair per class, plus an extra pair on class 0 to exercise "this tile shape, that class's design".
    stage.cost_pairs = [(c, c) for c in range(nb_classes)] + [(nb_classes, 0)]
    stage.pair_depth = [c for c in range(nb_classes)] + [0]
    stage.pair_multiplicity = [rng.randint(1, 4) for _ in stage.cost_pairs]

    stage.pair_runtime = [[rng.uniform(10, 200) for _ in range(nb_ranks)] for _ in stage.cost_pairs]
    stage.pair_energy = [[rng.uniform(1e3, 1e5) for _ in range(nb_ranks)] for _ in stage.cost_pairs]
    stage.pair_min_runtime = [min(runtimes) for runtimes in stage.pair_runtime]
    stage.pair_min_energy = [min(energies) for energies in stage.pair_energy]
    stage.class_area = [[rng.uniform(100, 5000) for _ in range(nb_ranks)] for _ in range(nb_classes)]
    stage.class_min_area = [min(areas) for areas in stage.class_area]

    # Two cores, each hosting some of the pairs, and a chain of tiles for the critical-path bound.
    stage.counts_per_core = [
        [(index, count) for index, count in enumerate(stage.pair_multiplicity) if index % 2 == parity]
        for parity in (0, 1)
    ]
    stage.topo_pair_index = [index for index, count in enumerate(stage.pair_multiplicity) for _ in range(count)]
    stage.topo_predecessors = [
        [] if position == 0 else [position - 1] for position in range(len(stage.topo_pair_index))
    ]

    # Cores carrying each decision class -- only used for the `nb_cores` field on a measured record.
    stage.cores_per_class = dict.fromkeys(range(nb_classes), 1)

    stage.rank_order = [list(range(nb_ranks)) for _ in range(nb_classes)]
    stage.nb_combinations = nb_ranks**nb_classes
    stage.max_schedule_evaluations = 10**6
    stage.max_search_nodes = 10**6

    def fake_measure(ranks, save_yaml=False):
        """Stand-in for the scheduler: never faster or cheaper than the bounds, but coupled across classes so
        the bounds stay loose, the way contention and transfers make them in a real schedule."""
        latency_bound, energy_bound, _area = stage._bounds(ranks)
        # Both overheads are >= 0 (as the real ones are) but anti-correlated, so schedules keep trading latency
        # against energy instead of all landing on one ray.
        coupling = (hash(ranks) % 1000) / 1000
        return SimpleNamespace(
            latency=latency_bound * (1 + 0.4 * coupling),
            energy=energy_bound * (1 + 0.4 * (1 - coupling)),
        )

    stage._measure = fake_measure
    return stage


def brute_force_front(stage: ScheduleExplorationStage) -> set[tuple[float, float, float]]:
    """Measure literally every combination and take the non-dominated set, independent of the search code."""
    points = []
    for ranks in itertools.product(*(range(len(ranks)) for ranks in stage.rank_order)):
        scme = stage._measure(ranks)
        points.append((scme.latency, scme.energy, stage._bounds(ranks)[2]))
    return {p for p in points if not any(dominates(q, p) for q in points)}


def test_pruned_search_returns_the_exhaustive_front():
    total_measured = 0
    total_combinations = 0
    total_pruned = 0
    for seed in range(8):
        stage = build_stage(seed)

        stage.prune_schedules = True
        pruned_archive, pruned_stats = stage._search()
        pruned_front = {point for point, _record in pruned_archive.entries}

        assert pruned_front == brute_force_front(stage), f"seed {seed}: pruned front differs from exhaustive"
        total_measured += pruned_stats["measured"]
        total_combinations += stage.nb_combinations
        total_pruned += pruned_stats["pruned_subtrees"]

    # How much gets pruned depends on the instance -- with three objectives and independent random costs most
    # combinations really are non-dominated, and then there is nothing to skip. What must hold across instances
    # is that pruning happens and never costs a front point (asserted per seed above).
    assert total_pruned > 0, "pruning never triggered, so the bounds were never exercised"
    assert total_measured < total_combinations


def test_measured_schedules_respect_their_bounds():
    stage = build_stage(seed=42)
    stage.prune_schedules = True
    stage._search()
    for record in stage.all_results:
        assert record["latency"] >= record["latency_bound"] - 1e-9
        assert record["energy"] >= record["energy_bound"] - 1e-9


def test_bounds_are_monotone_in_the_assignment():
    """A partial assignment's bound can only grow as more classes are fixed -- the property that lets a pruned
    subtree stand in for all of its completions."""
    stage = build_stage(seed=7)
    for ranks in itertools.product(*(range(len(r)) for r in stage.rank_order)):
        previous = stage._bounds(ranks, depth=0)
        for depth in range(1, len(stage.decision_classes) + 1):
            current = stage._bounds(ranks, depth=depth)
            assert all(c >= p - 1e-9 for c, p in zip(current, previous, strict=True)), (
                f"bound decreased at depth {depth} for {ranks}: {previous} -> {current}"
            )
            previous = current


if __name__ == "__main__":
    test_pruned_search_returns_the_exhaustive_front()
    test_measured_schedules_respect_their_bounds()
    test_bounds_are_monotone_in_the_assignment()
    print("All schedule exploration tests passed.")
