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
- groups of one layer may share a core by default, and the bus loads a tensor shared by merged groups once,
- the CP-SAT refinements (`shared_loads`, `comm_energy`, `capacity="overflow"`, `link_contention`) each change the
  optimum the way Stream's measurement would, and the model's objectives match `PlacementProblem.evaluate`.

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
from stream.stages.generation.placement_optimization.problem import OffchipLoad, PlacementProblem
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


def options(offchip_bus="none", link_transfers="none", **refinements) -> BackendOptions:
    return BackendOptions(offchip_bus=offchip_bus, link_transfers=link_transfers, workers=4, **refinements)


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


def test_area_grid_is_geometric_from_the_slackened_minimum() -> None:
    for backend_cls in BACKENDS:
        sweep = sweep_of(backend_cls, build_problem())
        sweep.run()
        # Area 2 at the minimum, 20 at the fastest point; 3 levels with 10% slack: 2.2, then 2.2 * (20 / 2.2) ** 0.5.
        bounds = sorted({entry["bound_area"] for entry in sweep.solve_log if entry["label"].startswith("grid")})
        assert bounds == [round(2.2e9), round(2.2e9 * (20 / 2.2) ** 0.5)], (backend_cls.name, bounds)


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


# ------------------------------------------------------------------------------------- CP-SAT refinements ------


def solve_cpsat(problem: PlacementProblem, objective: str, pin=(), **kwargs):
    """CP-SAT's optimum of `objective`, with groups pinned to slots by `pin` (`(group, slot)` pairs)."""
    backend = CpSatBackend(problem, options(**kwargs))
    for g, p in pin:
        backend.model.Add(backend.x[g, p] == 1)
    result = backend.solve(objective, {}, TIME_LIMIT)
    assert result.status == "OPTIMAL", result.status
    check_schedule(problem, result)
    return result


def test_shared_loads_load_a_tensor_once_per_slot() -> None:
    # Two independent groups on one core read the same 300-bit tensor (30 bus cycles). Loaded per group, the second
    # load waits for the first: 30 + 30 + 10 = 70. Shared, both tasks wait for one load: 30 + 10 + 10 = 50.
    def problem():
        p = build_problem(groups=(0, 1), layers=(0, 1), dependencies=(), max_cores=1, load_bits=[300, 300])
        p.offchip_bits = {frozenset({0, 1}): 300}
        p.offchip_loads = [OffchipLoad(users=frozenset({0, 1}), first_task={0: 0, 1: 1}, bits=300)]
        return p

    alone = solve_cpsat(problem(), "latency", offchip_bus="timed")
    shared = solve_cpsat(problem(), "latency", offchip_bus="timed", shared_loads=True)
    assert (alone.solution.latency, shared.solution.latency) == (70, 50)


def test_comm_energy_counts_loads_and_transfers() -> None:
    # Two groups of one layer, 0 -> 1, with a transfer (read 5, write 7) and a 100-bit tensor both use (energy 3 per
    # load). Together on one slow core: 100 + 3, the transfer is free. Apart: 100 + 2 * 3 + 5 + 7.
    def problem(allow_sibling_merge):
        p = build_problem(groups=(0, 1), layers=(0, 0), dependencies=((0, 1),), allow_sibling_merge=allow_sibling_merge)
        p.transfers = [LinkTransfer(producer=0, consumer=1, source=0, target=1, bits=64, splits=1)]
        p.transfer_energy = [({FAST: 5, SLOW: 5}, {FAST: 7, SLOW: 7})]
        p.offchip_loads = [
            OffchipLoad(users=frozenset({0, 1}), first_task={0: 0, 1: 1}, bits=100, energy={FAST: 3, SLOW: 3})
        ]
        return p

    for allow, expected in ((True, 103), (False, 118)):
        p = problem(allow)
        result = solve_cpsat(p, "energy", comm_energy=True)
        assert result.objective_value == expected, (allow, result.objective_value)
        solution = result.solution
        evaluated = p.evaluate(solution.slot_of_group, solution.design_of_slot, result.starts, None, "none", True)
        assert evaluated["energy"] == expected


def test_overflow_charges_a_refetch_of_what_a_core_cannot_keep() -> None:
    # Two groups each load their own 100-bit weights; a core keeps 150 bits. On one core 50 bits overflow, re-fetched
    # at 2 pJ/bit (+100 pJ, and 5 more bus cycles); on two cores nothing overflows.
    def problem(max_cores):
        p = build_problem(groups=(0, 1), layers=(0, 1), dependencies=(), max_cores=max_cores, load_bits=[100, 100])
        p.offchip_bits = {frozenset({0}): 100, frozenset({1}): 100}
        p.offchip_loads = [
            OffchipLoad(users=frozenset({g}), first_task={g: g}, bits=100, energy={FAST: 0, SLOW: 0}) for g in (0, 1)
        ]
        p.resident_capacity = {FAST: 150, SLOW: 150}
        p.refetch_pj_per_bit = 2.0
        return p

    one = solve_cpsat(problem(1), "energy", offchip_bus="aggregate", capacity="overflow")
    two = solve_cpsat(problem(2), "energy", offchip_bus="aggregate", capacity="overflow")
    assert (one.objective_value, two.objective_value) == (200, 100)
    # On one core the bus carries 200 loaded + 50 re-fetched bits at 10 bits/cycle (25 cycles), above the 20 cycles
    # of two fast tasks.
    fastest = solve_cpsat(problem(1), "latency", offchip_bus="aggregate", capacity="overflow")
    assert fastest.solution.latency == 25


def test_capacity_filter_forbids_designs_a_group_overflows() -> None:
    # The fast design is forbidden for group 0: the latency optimum runs it slow (30 + 10 after it on 1). If both
    # designs overflow group 0, the filter is dropped for it rather than making the model infeasible.
    problem = build_problem()
    problem.forbidden = {(0, FAST)}
    result = solve_cpsat(problem, "latency", capacity="filter")
    assert result.solution.design_of_slot[result.solution.slot_of_group[0]] == SLOW
    assert result.solution.latency == 40
    problem.forbidden = {(0, FAST), (0, SLOW)}
    assert solve_cpsat(problem, "latency", capacity="filter").solution.latency == 20


def link_problem(transfers):
    """Groups 0, 1 (layers 0, 1) feed groups 2, 3 (layers 0, 1); siblings apart, so at most two slots hold them."""
    p = build_problem(groups=(0, 1, 2, 3), layers=(0, 1, 0, 1), dependencies=(), max_cores=2, allow_sibling_merge=False)
    p.transfers = [
        LinkTransfer(producer=g, consumer=h, source=g, target=h, bits=25_600, splits=1) for g, h in transfers
    ]
    p.group_link_widths = {(min(g, h), max(g, h)): 512 for g, h in transfers}
    return p


def test_pair_contention_serializes_the_transfers_of_one_group_pair() -> None:
    # Group 0 (two tasks, 0-10 and 10-20) sends 50 cycles of data to each of group 1's two tasks. Without contention
    # both transfers overlap (10-60, 20-70): 80. Sharing their link, the second waits for the first (60-110): 120.
    p = build_problem(
        groups=(0, 1), layers=(0, 0), tasks_per_group=2, dependencies=(), allow_sibling_merge=False, max_cores=2
    )
    p.transfers = [
        LinkTransfer(producer=0, consumer=1, source=0, target=2, bits=25_600, splits=1),
        LinkTransfer(producer=0, consumer=1, source=1, target=3, bits=25_600, splits=1),
    ]
    p.group_link_widths = {(0, 1): 512}
    free = solve_cpsat(p, "latency", link_transfers="timed")
    pair = solve_cpsat(p, "latency", link_transfers="timed", link_contention="pair")
    assert (free.solution.latency, pair.solution.latency) == (80, 120)


def test_link_contention_serializes_transfers_of_different_pairs_on_one_link() -> None:
    # 0 and 1 pinned on slot 0, 2 and 3 on slot 1: transfers 0 -> 2 and 1 -> 3 (50 cycles each) share the link
    # from slot 0 to slot 1. Per pair they overlap (10-60, 20-70; tasks 2, 3 at 60-80): 80. Per link: 120.
    p = link_problem(((0, 2), (1, 3)))
    pin = ((0, 0), (1, 0))
    pair = solve_cpsat(p, "latency", pin, link_transfers="timed", link_contention="pair")
    link = solve_cpsat(p, "latency", pin, link_transfers="timed", link_contention="link")
    assert (pair.solution.latency, link.solution.latency) == (80, 120)


def test_refinements_need_cpsat_and_their_data() -> None:
    try:
        CpSatBackend(build_problem(), options(link_contention="pair"))
    except ValueError:
        pass
    else:
        raise AssertionError("link_contention without timed links must be rejected")
    if len(BACKENDS) > 1:
        try:
            BACKENDS[1](build_problem(), options(shared_loads=True))
        except NotImplementedError:
            pass
        else:
            raise AssertionError("GurobiBackend must reject the CP-SAT refinements")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"{name}: ok")
