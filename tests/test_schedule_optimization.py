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
- the off-chip bus bound holds latency at the bus time, leaving the slack to energy and area, and a tensor shared by
  groups that may merge is loaded once per core; timed loads make each task wait for its own load,
- `ProvidedSchedulingOrderStage` emits Stream's generated order when given none, and otherwise a full permutation
  with the provided entries first, in the provided order,
- with the interconnect area on, the model's area is the exact `interconnect_area` of the fold it chose (up to the
  rounding of the wire pitch), and a link is only paid for when its groups sit apart.

All synthetic: the stage is built with `object.__new__` and given its problem data directly, so no workload, cost
model or scheduler is involved.

Run directly with `python tests/test_schedule_optimization.py`, or collect with pytest.
"""

import math
from types import SimpleNamespace

from ortools.sat.python import cp_model

from stream.hardware.architecture.noc.interconnect_area import (
    InterconnectAreaModel,
    fold_link_widths,
    interconnect_area,
)
from stream.stages.generation.schedule_optimization import (
    AREA_SCALE,
    LinkTransfer,
    ModelSolution,
    ScheduleOptimizationStage,
)
from stream.stages.generation.scheduling_order_generation import (
    ProvidedSchedulingOrderStage,
    merge_scheduling_orders,
)
from stream.stages.generation.tiled_workload_accelerator_generation import renumber_merged_cores

# Two designs: fast (10 cycles, 100 pJ, area 10) and slow (30 cycles, 50 pJ, area 2) per task.
FAST, SLOW = "fast", "slow"


def build_stage(  # noqa: PLR0913, PLR0917
    groups=(0, 1, 2),
    layers=(0, 1, 2),
    chunks_per_group=1,
    dependencies=((0, 1),),
    offchip_bits=None,
    fold_allow_sibling_merge=False,
    load_bits=None,
    group_link_widths=None,
    transfers=None,
):
    """Groups `g` with one chunk each (or several, chained), on layers `layers[g]`; `dependencies` between
    groups become dependencies between their last and first chunks. `offchip_bits` (users -> bits, over a
    10 bits/cycle bus) turns the aggregate bus bound on; `load_bits` (bits per task, totalling `offchip_bits`)
    the timed loads; `group_link_widths` (group pair -> bits) the interconnect area; `transfers` (`LinkTransfer`s,
    whose group pairs must be in `group_link_widths`) the timed links, each pair's width then a decision."""
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
    stage.fold_allow_sibling_merge = fold_allow_sibling_merge
    stage.offchip_bus = "timed" if load_bits is not None else "aggregate" if offchip_bits is not None else "none"
    stage.load_bits = list(load_bits or [])
    stage.offchip_bits = {frozenset(users): bits for users, bits in (offchip_bits or {}).items()}
    stage.bus_bandwidth = 10
    stage.solve_time_limit_s = 10.0
    stage.solver_workers = 4
    stage.area_levels = 3
    stage.energy_levels = 2
    stage.solve_log = []
    stage.link_transfers = "timed" if transfers is not None else "none"
    stage.transfers = list(transfers or [])
    if group_link_widths is not None:
        stage.interconnect_model = InterconnectAreaModel()
        stage.group_link_widths = dict(group_link_widths)
    stage._build_model()
    return stage


def exact_area(stage, solution, group_link_widths=None) -> float:
    """Cores plus `interconnect_area` of the fold a solution describes, as `ScheduleSpace.area_of` measures it."""
    rolled_of = renumber_merged_cores(solution.slot_of_group)
    slot_of_rolled = {rolled_of[group]: slot for group, slot in solution.slot_of_group.items()}
    core_area = {core: stage.area_of[solution.design_of_slot[slot]] for core, slot in slot_of_rolled.items()}
    widths = fold_link_widths(group_link_widths or stage.group_link_widths, rolled_of)
    return sum(core_area.values()) + interconnect_area(widths, core_area, stage.interconnect_model).total


def test_interconnect_area_matches_the_exact_model() -> None:
    widths = {(0, 1): 128, (1, 2): 64}
    stage = build_stage(group_link_widths=widths)
    for order in (("latency", "energy", "area"), ("area", "latency", "energy"), ("energy", "latency", "area")):
        solution = stage._lexicographic(order, {}, "t")
        assert math.isclose(solution.area, exact_area(stage, solution), rel_tol=1e-3), order
    # The latency optimum keeps the 0 -> 1 chain on one core and group 2 apart: only the 64-bit link 1-2 is paid.
    fastest = stage._lexicographic(("latency", "energy", "area"), {}, "t")
    assert fastest.slot_of_group[0] == fastest.slot_of_group[1] != fastest.slot_of_group[2]
    assert fold_link_widths(widths, renumber_merged_cores(fastest.slot_of_group)) == {(0, 1): 64}
    # Interconnect only adds area: the area optimum is still one slow core, now with a router and bus wires.
    smallest = stage._lexicographic(("area", "latency", "energy"), {}, "a")
    assert len(set(smallest.slot_of_group.values())) == 1 and smallest.area > 2


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


def test_bus_bound_holds_latency_and_frees_energy() -> None:
    # 500 bits over 10 bits/cycle: no solution beats 50 cycles, whatever the cores.
    stage = build_stage(offchip_bits={(0,): 300, (2,): 200})
    fastest = stage._lexicographic(("latency", "energy", "area"), {}, "t")
    assert fastest.latency == 50
    # Within 50 cycles the chain 0 -> 1 fits fast + slow (40) and group 2 slow (30): 100 + 50 + 50 pJ, where the
    # compute-only model's fastest point is all fast (300 pJ, see test_anchors_are_the_model_optima).
    assert fastest.energy == 200


def test_bus_bound_loads_a_shared_tensor_once_per_core() -> None:
    # Groups 0 and 1 are slices of layer 0 reading the same 1000-bit tensor: 100 cycles if they share a core,
    # 200 if not. Without merging they cannot share, so the bound is 200.
    shared = {(0, 1): 1000}
    apart = build_stage(layers=(0, 0, 1), dependencies=(), offchip_bits=shared)
    assert apart._lexicographic(("latency",), {}, "t").latency == 200
    merged = build_stage(layers=(0, 0, 1), dependencies=(), offchip_bits=shared, fold_allow_sibling_merge=True)
    fastest = merged._lexicographic(("latency",), {}, "t")
    assert fastest.latency == 100 and fastest.slot_of_group[0] == fastest.slot_of_group[1]


def test_timed_loads_delay_their_tasks() -> None:
    # Group 0 loads 200 bits (20 cycles) before its task, and group 1 follows group 0. The aggregate bound alone
    # predicts max(compute 20, bus 20) = 20; with the load in front of group 0, the chain takes 20 + 10 + 10.
    aggregate = build_stage(offchip_bits={(0,): 200})
    assert aggregate._lexicographic(("latency",), {}, "t").latency == 20
    timed = build_stage(offchip_bits={(0,): 200}, load_bits=[200, 0, 0])
    assert timed._lexicographic(("latency",), {}, "t").latency == 40
    solver = cp_model.CpSolver()
    timed.model.ClearAssumptions()
    timed.model.Minimize(timed.latency)
    assert solver.Solve(timed.model) == cp_model.OPTIMAL
    assert solver.Value(timed.start[0]) >= solver.Value(timed.load_start[0]) + 20


def sibling_fan_in(bits: int = 5120, splits: int = 1):
    """Groups 0 and 1 are slices of layer 0 (never on one core), both feeding group 2 with `bits` each: group 2 can
    share a core with at most one of them, so one transfer always crosses a link."""
    transfers = [LinkTransfer(producer=g, consumer=2, source=g, target=2, bits=bits, splits=splits) for g in (0, 1)]
    return build_stage(
        layers=(0, 0, 1),
        dependencies=((0, 2), (1, 2)),
        group_link_widths={(0, 2): 32, (1, 2): 32},
        transfers=transfers,
    )


def test_transfer_delays_its_consumer_at_the_chosen_width() -> None:
    stage = sibling_fan_in()
    # Fastest: 10 (slices in parallel) + 5120 / 512 = 10 (the one crossing transfer, widest link) + 10.
    fastest = stage._lexicographic(("latency", "energy", "area"), {}, "t")
    assert fastest.latency == 30
    assert list(fastest.rolled_link_widths.values()) == [512]
    # The smallest accelerator keeps the link at its narrowest: 10 + 5120 / 32 = 160 + 30 (slow consumer).
    smallest = stage._lexicographic(("area", "latency", "energy"), {}, "a")
    assert list(smallest.rolled_link_widths.values()) == [32]
    assert smallest.latency == 30 + 160 + 30
    # Under a 60-cycle budget the smallest accelerator is a fast core for one slice and a slow core shared by the
    # other slice and the consumer (30 + 30): the crossing transfer must be in by 30, after the fast slice's 10, so
    # 5120 bits in 20 cycles -- 256 bits, the narrowest width that does it.
    budget = stage._lexicographic(("area", "latency", "energy"), {"latency": 60}, "b")
    assert budget.latency == 60 and sorted(budget.design_of_slot.values()) == [FAST, SLOW]
    assert list(budget.rolled_link_widths.values()) == [256]
    assert math.isclose(budget.area, exact_area(stage, budget, budget.group_link_widths), rel_tol=1e-3)


def test_transfer_between_groups_on_one_core_is_free() -> None:
    # A chain 0 -> 1 with a transfer too long to ever pay: the fastest point merges the two groups, and the
    # transfer then takes no time (10 + 10).
    stage = build_stage(
        group_link_widths={(0, 1): 32},
        transfers=[LinkTransfer(producer=0, consumer=1, source=0, target=1, bits=10**6, splits=1)],
    )
    fastest = stage._lexicographic(("latency", "energy", "area"), {}, "t")
    assert fastest.latency == 20 and fastest.slot_of_group[0] == fastest.slot_of_group[1]


def test_transfers_overlap_compute_and_share_the_link_by_splits() -> None:
    # Two slices each send 5120 bits to group 2 at 1/2 of the link: 5120 / 256 = 20 cycles at 512 bits.
    stage = sibling_fan_in(splits=2)
    assert stage._lexicographic(("latency",), {}, "t").latency == 10 + 20 + 10


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
    test_bus_bound_holds_latency_and_frees_energy()
    test_bus_bound_loads_a_shared_tensor_once_per_core()
    test_timed_loads_delay_their_tasks()
    test_solution_key_ignores_slot_numbering()
    test_scheduling_order_follows_the_model()
    test_merge_scheduling_orders()
    test_provided_order_stage()
    test_interconnect_area_matches_the_exact_model()
    test_transfer_delays_its_consumer_at_the_chosen_width()
    test_transfer_between_groups_on_one_core_is_free()
    test_transfers_overlap_compute_and_share_the_link_by_splits()
    print("All schedule optimization tests passed.")
