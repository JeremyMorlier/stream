"""Tests for the CP-SAT model of `GroupScheduleOptimizationStage` / `ChunkScheduleOptimizationStage`.

The model replaces a search whose every point is measured, so its errors would not show as crashes but as a
front that is quietly wrong. The properties checked here are the ones that would hide:

- the anchors are the true per-objective optima of the model (computed by hand on a tiny instance),
- two layer groups that run in parallel are never forced onto one core when that costs latency, and siblings of
  one layer never share a core,
- precedence holds: a consumer chunk never starts before its producer chunk ends,
- the epsilon bounds of one solve do not leak into the next,
- solution keys ignore how slots are numbered,
- a solution's tile order follows the model's start times and respects precedence,
- `ProvidedSchedulingOrderStage` emits Stream's generated order when given none, and otherwise a full permutation
  with the provided entries first, in the provided order.

All synthetic: the stage is built with `object.__new__` and given its problem data directly, so no workload, cost
model or scheduler is involved.

Run directly with `python tests/test_schedule_optimization.py`, or collect with pytest.
"""

from types import SimpleNamespace

from ortools.sat.python import cp_model

from stream.stages.generation.schedule_optimization import AREA_SCALE, ModelSolution, ScheduleOptimizationStage
from stream.stages.generation.scheduling_order_generation import (
    ProvidedSchedulingOrderStage,
    merge_scheduling_orders,
)

# Two designs: fast (10 cycles, 100 pJ, area 10) and slow (30 cycles, 50 pJ, area 2) per task.
FAST, SLOW = "fast", "slow"


def build_stage(groups=(0, 1, 2), layers=(0, 1, 2), chunks_per_group=1, dependencies=((0, 1),)):
    """Groups `g` with one chunk each (or several, chained), on layers `layers[g]`; `dependencies` between
    groups become dependencies between their last and first chunks."""
    stage = object.__new__(ScheduleOptimizationStage)
    stage.groups = list(groups)
    stage.layer_of_group = layer_of = dict(zip(groups, layers, strict=True))
    stage.design_keys = [FAST, SLOW]
    stage.area_of = {FAST: 10.0, SLOW: 2.0}
    stage.chunks = [(group, index) for group in groups for index in range(chunks_per_group)]
    stage.duration = [{FAST: 10, SLOW: 30} for _ in stage.chunks]
    # One tile per task: tile (layer of the group, chunk index).
    stage.chunk_tiles = [[SimpleNamespace(id=layer_of[group], sub_id=index)] for group, index in stage.chunks]
    stage.group_energy = {group: {FAST: 100 * chunks_per_group, SLOW: 50 * chunks_per_group} for group in groups}
    first = {group: stage.chunks.index((group, 0)) for group in groups}
    last = {group: stage.chunks.index((group, chunks_per_group - 1)) for group in groups}
    stage.dependencies = [(last[a], first[b]) for a, b in dependencies]
    stage.chains = [
        (index, index + 1)
        for index in range(len(stage.chunks) - 1)
        if stage.chunks[index][0] == stage.chunks[index + 1][0]
    ]
    stage.fold_allow_sibling_merge = False
    stage.solve_time_limit_s = 10.0
    stage.solver_workers = 4
    stage.area_levels = 3
    stage.energy_levels = 2
    stage.solve_log = []
    stage._build_model()
    return stage


def test_anchors_are_the_model_optima() -> None:
    stage = build_stage()
    # Latency: groups 0 -> 1 chain (10 + 10) and group 2 in parallel on fast cores -> 20 cycles.
    fastest = stage._lexicographic(("latency", "energy", "area"), {}, "t")
    assert fastest.latency == 20
    # ...and among the 20-cycle solutions, the cheapest in energy is still all-fast (300 pJ), then the smallest
    # area puts the chain on one core and group 2 on another: two fast cores.
    assert fastest.energy == 300 and fastest.area == 20
    # Area: one slow core runs everything in sequence -> area 2, latency 90.
    smallest = stage._lexicographic(("area", "latency", "energy"), {}, "a")
    assert smallest.area == 2 and smallest.latency == 90 and len(set(smallest.slot_of_group.values())) == 1
    # Energy: every task on a slow design (150 pJ); fastest such is two slow cores (chain 60 || 30) -> 60.
    frugal = stage._lexicographic(("energy", "latency", "area"), {}, "e")
    assert frugal.energy == 150 and frugal.latency == 60


def test_siblings_never_share_a_core() -> None:
    # Groups 0 and 1 are two inter-core slices of layer 0: even the area optimum must keep them apart.
    stage = build_stage(layers=(0, 0, 1), dependencies=())
    smallest = stage._lexicographic(("area", "latency", "energy"), {}, "a")
    assert smallest.slot_of_group[0] != smallest.slot_of_group[1]


def test_precedence_between_chunks() -> None:
    stage = build_stage(chunks_per_group=3)
    stage._lexicographic(("latency",), {}, "t")
    # Re-solve and read the schedule: every dependency and chain edge must hold.
    solver = cp_model.CpSolver()
    stage.model.ClearAssumptions()
    stage.model.Minimize(stage.latency)
    assert solver.Solve(stage.model) == cp_model.OPTIMAL
    for a, b in stage.dependencies + stage.chains:
        assert solver.Value(stage.start[b]) >= solver.Value(stage.end[a])


def test_bounds_do_not_leak_between_solves() -> None:
    stage = build_stage()
    # Bounds are in the model's integer units: area is scaled by AREA_SCALE.
    bounded = stage._solve("latency", {"area": round(2 * AREA_SCALE)}, 10.0, "bounded")
    assert bounded.area <= 2 and bounded.latency == 90
    free = stage._solve("latency", {}, 10.0, "free")
    assert free.latency == 20, "the area bound of the previous solve must not constrain this one"


def test_solution_key_ignores_slot_numbering() -> None:
    a = ModelSolution(slot_of_group={0: 0, 1: 0, 2: 1}, design_of_slot={0: FAST, 1: SLOW}, latency=1, energy=1, area=1)
    b = ModelSolution(slot_of_group={0: 3, 1: 3, 2: 5}, design_of_slot={3: FAST, 5: SLOW}, latency=1, energy=1, area=1)
    c = ModelSolution(slot_of_group={0: 0, 1: 1, 2: 1}, design_of_slot={0: FAST, 1: SLOW}, latency=1, energy=1, area=1)
    assert a.key == b.key
    assert a.key != c.key


def test_scheduling_order_follows_the_model() -> None:
    stage = build_stage(chunks_per_group=3)
    fastest = stage._lexicographic(("latency",), {}, "t")
    order = list(fastest.scheduling_order)
    assert sorted(order) == sorted((tile.id, tile.sub_id) for tiles in stage.chunk_tiles for tile in tiles)
    # Group 0 (layer 0) feeds group 1 (layer 1), so every tile of layer 0 comes before layer 1's first.
    position = {item: index for index, item in enumerate(order)}
    assert max(position[(0, k)] for k in range(3)) < min(position[(1, k)] for k in range(3))
    # A group's own chunks stay in order.
    assert [position[(2, k)] for k in range(3)] == sorted(position[(2, k)] for k in range(3))


def test_merge_scheduling_orders() -> None:
    generated = [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert merge_scheduling_orders(generated, generated) == generated
    # Provided entries first, in the given order; unknown and duplicate entries dropped; the rest appended.
    merged = merge_scheduling_orders([(1, 1), (9, 9), (0, 1), (1, 1)], generated)
    assert merged == [(1, 1), (0, 1), (0, 0), (1, 0)]


def test_provided_order_stage() -> None:
    generated = [(0, 0), (1, 0), (0, 1)]
    captured = {}

    class Leaf:
        def __init__(self, list_of_callables, **kwargs):
            captured.update(kwargs)

        def run(self):
            yield None, None

    for provided, expected in ((None, generated), ([(0, 1)], [(0, 1), (0, 0), (1, 0)])):
        stage = object.__new__(ProvidedSchedulingOrderStage)
        stage.list_of_callables, stage.kwargs = [Leaf], {}
        stage.accelerator, stage.workload, stage.layer_stacks = None, None, [(0, 1)]
        stage.get_scheduling_order_fused = lambda: list(generated)
        stage.provided_scheduling_order = provided
        list(stage.run())
        assert captured["scheduling_order"] == expected


if __name__ == "__main__":
    test_anchors_are_the_model_optima()
    test_siblings_never_share_a_core()
    test_precedence_between_chunks()
    test_bounds_do_not_leak_between_solves()
    test_solution_key_ignores_slot_numbering()
    test_scheduling_order_follows_the_model()
    test_merge_scheduling_orders()
    test_provided_order_stage()
    print("All schedule optimization tests passed.")
