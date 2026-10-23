"""Tests for the placement optimization (`stream.stages.generation.placement_optimization`), solved with CP-SAT and
as a Gurobi MILP.

The two backends are meant to be the same model, so the main property is that they agree: on small random
instances covering every option (off-chip bus none / aggregate / timed, links none / timed, interconnect area on
or off, siblings allowed or not, a core cap), both reach the same optimum of every objective, with and without
epsilon bounds. Besides:

- the anchors are the true per-objective optima, computed by hand on a tiny instance,
- every solution respects the core cap, sibling exclusion, precedences, and never overlaps two tasks on a core,
- with the interconnect area on, the model's area is the exact `interconnect_area` of its fold (up to the rounding
  of the wire pitch),
- groups of one layer may share a core by default, and the bus loads a tensor shared by merged groups once.

All synthetic: problems are built directly, without workload, cost model or scheduler. Gurobi tests are skipped
without a license. Run with `python tests/test_placement_optimization.py`, or collect with pytest.
"""

import math
import random
from types import SimpleNamespace

from stream.hardware.architecture.noc.interconnect_area import (
    InterconnectAreaModel,
    fold_link_widths,
    interconnect_area,
)
from stream.stages.generation.placement_optimization.backend import BackendOptions
from stream.stages.generation.placement_optimization.cpsat_backend import CpSatBackend
from stream.stages.generation.placement_optimization.problem import PlacementProblem
from stream.stages.generation.placement_optimization.stage import PlacementSweep
from stream.stages.generation.schedule_optimization import LinkTransfer
from stream.stages.generation.tiled_workload_accelerator_generation import renumber_merged_cores

try:
    import gurobipy as gp

    gp.Model().dispose()
    from stream.stages.generation.placement_optimization.gurobi_backend import GurobiBackend

    BACKENDS = (CpSatBackend, GurobiBackend)
except Exception:  # no gurobipy or no license
    BACKENDS = (CpSatBackend,)

FAST, SLOW = "fast", "slow"
TIME_LIMIT = 20.0


def build_problem(  # noqa: PLR0913, PLR0917
    groups=(0, 1, 2),
    layers=(0, 1, 2),
    tasks_per_group=1,
    dependencies=((0, 1),),
    max_cores=None,
    allow_sibling_merge=True,
    offchip_bits=None,
    load_bits=None,
    group_link_widths=None,
    transfers=None,
    interconnect=False,
) -> PlacementProblem:
    """Groups `g` on layers `layers[g]`, `tasks_per_group` chained tasks each; fast (10 cycles, 100 pJ, area 10) or
    slow (30 cycles, 50 pJ, area 2) per task. `dependencies` between groups join the last task of one to the first
    of the other."""
    layer_of = dict(zip(groups, layers, strict=True))
    tasks = [(g, index) for g in groups for index in range(tasks_per_group)]
    first = {g: tasks.index((g, 0)) for g in groups}
    last = {g: tasks.index((g, tasks_per_group - 1)) for g in groups}
    return PlacementProblem(
        groups=list(groups),
        layer_of_group=layer_of,
        design_keys=[FAST, SLOW],
        area_of={FAST: 10.0, SLOW: 2.0},
        tasks=tasks,
        task_tiles=[[SimpleNamespace(id=layer_of[g], sub_id=index)] for g, index in tasks],
        duration=[{FAST: 10, SLOW: 30} for _ in tasks],
        group_energy={g: {FAST: 100 * tasks_per_group, SLOW: 50 * tasks_per_group} for g in groups},
        dependencies=[(last[a], first[b]) for a, b in dependencies],
        chains=[(i, i + 1) for i in range(len(tasks) - 1) if tasks[i][0] == tasks[i + 1][0]],
        max_cores=max_cores,
        allow_sibling_merge=allow_sibling_merge,
        group_link_widths=dict(group_link_widths or {}),
        transfers=list(transfers or []),
        offchip_bits={frozenset(users): bits for users, bits in (offchip_bits or {}).items()},
        load_bits=list(load_bits or []),
        bus_bandwidth=10,
        interconnect_model=InterconnectAreaModel() if interconnect else None,
        width_levels=InterconnectAreaModel().width_levels,
    )


def options(offchip_bus="none", link_transfers="none") -> BackendOptions:
    return BackendOptions(offchip_bus=offchip_bus, link_transfers=link_transfers, workers=4)


def sweep_of(backend_cls, problem, **kwargs) -> PlacementSweep:
    return PlacementSweep(
        backend_cls(problem, options(**kwargs)), solve_time_limit_s=TIME_LIMIT, area_levels=3, energy_levels=2
    )


def check_schedule(problem: PlacementProblem, result) -> None:
    """Core cap, sibling exclusion, precedences, and no two tasks overlapping on one slot."""
    solution, starts = result.solution, result.starts
    slot_of = solution.slot_of_group
    assert len(set(slot_of.values())) <= problem.nb_slots
    for a, b in problem.sibling_pairs:
        assert slot_of[a] != slot_of[b]
    end = [
        starts[task] + problem.duration[task][solution.design_of_slot[slot_of[g]]]
        for task, (g, _index) in enumerate(problem.tasks)
    ]
    for a, b in problem.dependencies + problem.chains:
        assert starts[b] >= end[a]
    for k in range(len(problem.tasks)):
        for k2 in range(k + 1, len(problem.tasks)):
            if slot_of[problem.tasks[k][0]] == slot_of[problem.tasks[k2][0]]:
                assert end[k] <= starts[k2] or end[k2] <= starts[k], (k, k2)
    assert solution.latency >= max(end)


def exact_area(problem: PlacementProblem, solution, widths=None) -> float:
    rolled_of = renumber_merged_cores(solution.slot_of_group)
    slot_of_rolled = {rolled_of[g]: slot for g, slot in solution.slot_of_group.items()}
    core_area = {core: problem.area_of[solution.design_of_slot[slot]] for core, slot in slot_of_rolled.items()}
    folded = fold_link_widths(widths or problem.group_link_widths, rolled_of)
    return sum(core_area.values()) + interconnect_area(folded, core_area, problem.interconnect_model).total


# ----------------------------------------------------------------------------------------------- tests ------


def test_anchors_are_the_model_optima() -> None:
    for backend_cls in BACKENDS:
        sweep = sweep_of(backend_cls, build_problem())
        # Latency: 0 -> 1 chained (10 + 10) with 2 in parallel, all fast: 20 cycles, 300 pJ, two fast cores.
        fastest = sweep.lexicographic(("latency", "energy", "area"), {}, "t")
        assert (fastest.latency, fastest.energy, fastest.area) == (20, 300, 20), backend_cls.name
        # Area: one slow core runs everything: area 2, 90 cycles.
        smallest = sweep.lexicographic(("area", "latency", "energy"), {}, "a")
        assert (smallest.area, smallest.latency) == (2, 90), backend_cls.name
        # Energy: all slow (150 pJ); fastest such is two slow cores, chain 60 || 30.
        cheapest = sweep.lexicographic(("energy", "latency", "area"), {}, "e")
        assert (cheapest.energy, cheapest.latency, cheapest.area) == (150, 60, 4), backend_cls.name


def test_core_cap_and_siblings() -> None:
    for backend_cls in BACKENDS:
        # Three independent groups, at most two cores: the third must share.
        capped = sweep_of(backend_cls, build_problem(dependencies=(), max_cores=2))
        result = capped.solve("latency", {}, TIME_LIMIT, "t")
        assert result.solution.latency == 20, backend_cls.name
        check_schedule(capped.backend.problem, result)
        # Siblings may share by default: two groups of layer 0 on one slow core is the area optimum.
        merged = sweep_of(backend_cls, build_problem(groups=(0, 1), layers=(0, 0), dependencies=()))
        assert merged.solve("area", {}, TIME_LIMIT, "a").solution.area == 2, backend_cls.name
        # ...and may not when forbidden.
        apart = sweep_of(
            backend_cls, build_problem(groups=(0, 1), layers=(0, 0), dependencies=(), allow_sibling_merge=False)
        )
        result = apart.solve("area", {}, TIME_LIMIT, "a")
        assert result.solution.area == 4, backend_cls.name
        check_schedule(apart.backend.problem, result)


def test_bus_loads_shared_tensors_once_per_core() -> None:
    # Two sibling groups read the same 300-bit tensor over a 10 bits/cycle bus: 30 cycles on one core, 60 on two.
    for backend_cls in BACKENDS:
        problem = build_problem(groups=(0, 1), layers=(0, 0), dependencies=(), offchip_bits={(0, 1): 300})
        sweep = sweep_of(backend_cls, problem, offchip_bus="aggregate")
        fastest = sweep.lexicographic(("latency", "area", "energy"), {}, "t")
        # Apart (both fast, 10 cycles each) the bus takes 60; together 20 cycles of compute and 30 of bus.
        assert fastest.latency == 30 and len(set(fastest.slot_of_group.values())) == 1, backend_cls.name


def test_reported_latency_is_the_makespan_when_not_minimized() -> None:
    # Minimizing energy leaves the MILP's latency variable free above the makespan; the reported latency must still be
    # the makespan of the returned schedule (all slow: chain 0 -> 1 on 60 cycles, group 2 alongside or after it).
    for backend_cls in BACKENDS:
        problem = build_problem()
        result = sweep_of(backend_cls, problem).solve("energy", {}, TIME_LIMIT, "e")
        slot_of = result.solution.slot_of_group
        end = [
            result.starts[task] + problem.duration[task][result.solution.design_of_slot[slot_of[g]]]
            for task, (g, _index) in enumerate(problem.tasks)
        ]
        assert result.solution.latency == max(end), backend_cls.name


def test_interconnect_area_matches_the_exact_model() -> None:
    widths = {(0, 1): 128, (1, 2): 64}
    for backend_cls in BACKENDS:
        problem = build_problem(dependencies=((0, 1), (1, 2)), group_link_widths=widths, interconnect=True)
        sweep = sweep_of(backend_cls, problem)
        for order in (("latency", "energy", "area"), ("area", "latency", "energy")):
            solution = sweep.lexicographic(order, {}, "t")
            assert math.isclose(solution.area, exact_area(problem, solution), rel_tol=1e-3), (backend_cls.name, order)


def random_problem(seed: int) -> tuple[PlacementProblem, dict]:  # noqa: PLR0914
    """A small random instance; the seed sets the options."""
    rng = random.Random(seed)
    nb_groups = rng.randint(3, 5)
    groups = list(range(nb_groups))
    layers = sorted(rng.randint(0, nb_groups - 2) for _ in groups)
    tasks_per_group = rng.randint(1, 2)
    dependencies = sorted(
        {(a, b) for a in groups for b in groups if layers[a] < layers[b] and rng.random() < 0.5}  # noqa: PLR2004
    )
    # Every combination of the options over 12 consecutive seeds.
    offchip_bus = ("none", "aggregate", "timed")[seed % 3]
    link_transfers = ("none", "timed")[seed // 3 % 2]
    problem = build_problem(
        groups=tuple(groups),
        layers=tuple(layers),
        tasks_per_group=tasks_per_group,
        dependencies=dependencies,
        max_cores=(None, 2, 3)[seed % 3 - 1],
        allow_sibling_merge=seed // 6 % 2 == 0,
        interconnect=seed % 2 == 1,
    )
    designs = {"a": (8, 120, 12.0), "b": (20, 60, 3.0), "c": (14, 90, 6.0)}
    problem.design_keys = sorted(designs)
    problem.area_of = {key: area for key, (_d, _e, area) in designs.items()}
    problem.duration = [
        {key: max(1, round(d * rng.uniform(0.5, 1.5))) for key, (d, _e, _a) in designs.items()} for _ in problem.tasks
    ]
    problem.group_energy = {
        g: {key: round(e * tasks_per_group * rng.uniform(0.5, 1.5)) for key, (_d, e, _a) in designs.items()}
        for g in groups
    }
    first = {g: problem.tasks.index((g, 0)) for g in groups}
    last = {g: problem.tasks.index((g, tasks_per_group - 1)) for g in groups}
    levels = problem.width_levels
    problem.group_link_widths = {(a, b): rng.choice(levels) for a, b in dependencies}
    problem.transfers = [
        LinkTransfer(producer=a, consumer=b, source=last[a], target=first[b], bits=rng.randint(500, 4000), splits=1)
        for a, b in dependencies
    ]
    if offchip_bus != "none":
        problem.offchip_bits = {frozenset({g}): rng.randint(50, 300) for g in groups}
        shared = [g for g in groups if layers[g] == layers[0]]
        if len(shared) > 1:
            problem.offchip_bits[frozenset(shared)] = rng.randint(100, 400)
        problem.load_bits = [0] * len(problem.tasks)
        for users, bits in problem.offchip_bits.items():
            for g in users:
                problem.load_bits[first[g]] += bits
    return problem, {"offchip_bus": offchip_bus, "link_transfers": link_transfers}


def test_backends_agree_on_random_instances() -> None:
    if len(BACKENDS) < 2:  # noqa: PLR2004
        return
    for seed in range(12):
        problem, opts = random_problem(seed)
        sweeps = [sweep_of(backend_cls, problem, **opts) for backend_cls in BACKENDS]
        optima = {}
        for objective in ("latency", "energy", "area"):
            results = [sweep.solve(objective, {}, TIME_LIMIT, objective) for sweep in sweeps]
            for result in results:
                assert result.status == "OPTIMAL", (seed, objective, result.status)
                check_schedule(problem, result)
            values = [result.objective_value for result in results]
            assert values[0] == values[1], (seed, objective, opts, values)
            optima[objective] = values[0]
        # Latency under an area budget halfway between the area optimum and the fastest solution's area.
        fastest_area = sweeps[0].lexicographic(("latency", "area"), {}, "t").area
        budget = (optima["area"] + round(fastest_area * 1e9)) // 2
        values = [sweep.solve("latency", {"area": budget}, TIME_LIMIT, "eps").objective_value for sweep in sweeps]
        assert values[0] == values[1], (seed, "eps", opts, values)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"{name}: ok")
