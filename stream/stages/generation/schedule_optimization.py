"""Joint hardware/schedule optimization over the cost matrix of a schedule search, with OR-Tools CP-SAT.

The rolled search decides the accelerator in two separate steps: a branch and bound over which design each layer's
core gets (`ScheduleExplorationStage`), then a heuristic folding of the resulting schedules onto fewer cores
(`ScheduleRollingStage`). The stages here replace both with one constraint model that decides, at once:

- how many physical cores there are (slots `p`, used or not),
- which design each core gets (`y[p, d]`),
- which layer groups -- the cores of the unrolled topology, one per `(layer, inter-core group)` -- share a core
  (`x[g, p]`),
- and when each piece of work runs,

and returns a (latency, energy, area) Pareto front by an epsilon-constraint sweep. Every distinct solution is then
re-measured with the real scheduler and inserted into the same `ParetoArchive` as the other schedule search
stages, so `ScheduleSelectionStage` and the visualization stages work unchanged. The model's own estimate of each
point is kept next to the measurement (`predicted_latency`, `predicted_energy`), as the evidence for how much the
model can be trusted.

Three levels of detail, as three stages (see `docs/source/rolled_schedule_optimization.md`):

- `GroupScheduleOptimizationStage` (level 1): one task per layer group. Small and fast, but a consumer group only
  starts once its producer groups have finished, so layer fusion -- which Stream exploits by interleaving tiles of
  consecutive layers -- is invisible and latency is over-estimated.
- `ChunkScheduleOptimizationStage` (level 2): each group's tiles are split, in tile order, into
  `chunks_per_group` chunks. A chunk depends on the chunks that hold its tiles' predecessors, so consecutive layers
  pipeline at chunk granularity and fusion is approximated. The hardware decisions are the same; only the time
  model is finer.
- `TileScheduleOptimizationStage` (level 3): one task per tile, with the tile dependency graph (transitively
  reduced) as precedences -- the time model Stream itself schedules, minus what is listed below.

What the model takes from the schedule space (all computed by the full-matrix warm-up the stages request):
per-tile latency (`latency_attr`) and energy for every (tile shape, design) pair, design areas, and the tile
dependency graph. What it leaves out, and leaves to the re-measurement:

- **transfers between cores.** The derived topologies give every link a bandwidth equal to the total traffic of
  its core pair, so any on-chip transfer takes about one cycle in Stream's model; it is neglected here. Transfer
  energy is zero in the accelerator templates (`unit_energy_cost: 0`).
- **off-chip traffic and memory capacity** (weight loads over the shared bus, eviction when a core hosting several
  layers runs out of memory). These are the extensions of section 5 of the note above; they are what most often
  makes a measured point slower than predicted.

Area is exact (the sum of the chosen designs' areas, offchip excluded as everywhere else in the search), and so is
compute energy up to the cost model.
"""

import csv
import logging
import math
import os
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import networkx as nx
from ortools.sat.python import cp_model

from stream.stages.generation.schedule_evaluation import ScheduleSearchStage, tile_key
from stream.stages.generation.schedule_space import ScheduleSearchState, ScheduleSpace
from stream.stages.generation.tiled_workload_accelerator_generation import renumber_merged_cores
from stream.stages.stage import StageCallable
from stream.workload.computation.computation_node import ComputationNode

logger = logging.getLogger(__name__)

# Integer scaling of the objectives for CP-SAT: latency is in whole cycles already; energy (pJ) and area (a.u.)
# are rounded at these resolutions, fine against values of 1e8+ pJ and 1e2+ a.u.
ENERGY_SCALE = 1.0
AREA_SCALE = 100.0
OBJECTIVES = ("latency", "energy", "area")
# An epsilon grid needs at least its two end points.
MIN_GRID_LEVELS = 2


@dataclass
class ModelSolution:
    """One solution of the model: the merge of the unrolled cores and each slot's design, with the model's own
    estimate of its objectives."""

    slot_of_group: dict[int, int]  # unrolled core id -> slot
    design_of_slot: dict[int, str]  # used slot -> design key
    latency: int
    energy: float
    area: float

    @property
    def key(self) -> tuple:
        """The accelerator it describes, independent of how the slots are numbered."""
        rolled_of = renumber_merged_cores(self.slot_of_group)
        slot_of_rolled = {rolled_of[group]: slot for group, slot in self.slot_of_group.items()}
        return (
            tuple(sorted(rolled_of.items())),
            tuple(self.design_of_slot[slot_of_rolled[core]] for core in range(len(slot_of_rolled))),
        )


class ScheduleOptimizationStage(ScheduleSearchStage):
    """Base of the two levels: builds the CP-SAT model at the granularity `chunks_per_group` gives, sweeps the
    epsilon-constraint grid, measures every distinct solution and adds it to the archive. See the module docstring.

    Takes `schedule_search` from `ScheduleExplorationStage` (which only needs to measure enough to build the space,
    e.g. `max_schedule_evaluations=1`) and hands the extended state down to `ScheduleSelectionStage`."""

    requires_full_cost_matrix = True
    level = 0
    default_chunks_per_group = 1

    def __init__(  # noqa: PLR0913
        self,
        list_of_callables: list[StageCallable],
        *,
        schedule_search: ScheduleSearchState,
        optimization_search_dir: str | None = None,
        chunks_per_group: int | None = None,
        area_levels: int = 5,
        energy_levels: int = 3,
        solve_time_limit_s: float = 20.0,
        solver_workers: int = 8,
        max_optimized_evaluations: int = 200,
        fold_allow_sibling_merge: bool = False,
        **kwargs: Any,
    ):
        """
        Args:
            schedule_search: the unrolled search's state; its archive and records are extended in place.
            optimization_search_dir: where this stage's report goes. Defaults to `schedule_optimization_l<level>/`
                next to the tiled workload.
            chunks_per_group: tasks per layer group (level 2); level 1 always uses 1.
            area_levels, energy_levels: the epsilon-constraint grid, between each objective's minimum and its
                value at the latency-optimal point.
            solve_time_limit_s: CP-SAT time limit per solve (half of it for the tie-break solves). A solve that
                times out still contributes its best solution found.
            solver_workers: CP-SAT search workers.
            max_optimized_evaluations: cap on solutions measured with the real scheduler.
            fold_allow_sibling_merge: let two groups of the same layer share a core. Off by default, as for the
                rolled search: those are the layer's inter-core slices, and the cost model bakes in that they run
                in parallel.
        """
        super().__init__(list_of_callables, **kwargs)
        self.schedule_search = schedule_search
        self.space: ScheduleSpace = schedule_search.space
        self.out_dir = optimization_search_dir or os.path.join(
            os.path.dirname(self.space.tiled_workload_path), f"schedule_optimization_l{self.level}"
        )
        self.chunks_per_group = chunks_per_group or self.default_chunks_per_group
        self.area_levels = area_levels
        self.energy_levels = energy_levels
        self.solve_time_limit_s = solve_time_limit_s
        self.solver_workers = solver_workers
        self.max_optimized_evaluations = max_optimized_evaluations
        self.fold_allow_sibling_merge = fold_allow_sibling_merge
        self.search_budget = kwargs.get("search_budget")
        self.solve_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------------------------- problem data ------

    def _build_problem(self) -> None:
        """Everything the model needs, from the schedule space: groups, chunks, per-design durations and energies,
        design areas and the chunk dependency graph."""
        space = self.space
        self.groups: list[int] = list(space.unrolled_core_ids)
        self.layer_of_group: dict[int, int] = {
            core: node_id for node_id, cores in space.core_ids_of_node.items() for core in cores
        }
        needed_shapes = sorted(set(space.shape_class_of_tile.values()))
        self.designs: dict[str, dict[str, Any]] = {
            key: core_dict
            for key, core_dict in space.distinct_designs().items()
            if all((shape, key) in space.costs.runtime for shape in needed_shapes)
        }
        if not self.designs:
            raise ValueError(f"{type(self).__name__}: no design is cost-modelled on every tile shape.")
        self.design_keys: list[str] = sorted(self.designs)
        self.area_of: dict[str, float] = {key: space.costs.area[key] for key in self.design_keys}

        # Each group's tiles in tile order, split into contiguous chunks.
        tiles = [tile for tile in space.reference_workload.node_list if isinstance(tile, ComputationNode)]
        tiles_of_group: dict[int, list[Any]] = {group: [] for group in self.groups}
        for tile in tiles:
            tiles_of_group[space.unrolled_core_of_tile[tile_key(tile)]].append(tile)
        self.chunks: list[tuple[int, int]] = []  # (group, index)
        self.chunk_of_tile: dict[tuple[int, int], int] = {}
        chunk_tiles: list[list[Any]] = []
        for group in self.groups:
            ordered = sorted(tiles_of_group[group], key=lambda tile: tile.sub_id)
            nb_chunks = max(1, min(self.chunks_per_group, len(ordered)))
            size = math.ceil(len(ordered) / nb_chunks)
            for index in range(nb_chunks):
                members = ordered[index * size : (index + 1) * size]
                if not members:
                    continue
                for tile in members:
                    self.chunk_of_tile[tile_key(tile)] = len(self.chunks)
                self.chunks.append((group, index))
                chunk_tiles.append(members)

        runtime, energy = space.costs.runtime, space.costs.energy
        shape_of = space.shape_class_of_tile
        self.duration: list[dict[str, int]] = [
            {
                key: max(1, math.ceil(sum(runtime[(shape_of[tile_key(tile)], key)] for tile in members)))
                for key in self.design_keys
            }
            for members in chunk_tiles
        ]
        self.group_energy: dict[int, dict[str, int]] = {
            group: {
                key: round(
                    sum(energy[(shape_of[tile_key(tile)], key)] for tile in tiles_of_group[group]) / ENERGY_SCALE
                )
                for key in self.design_keys
            }
            for group in self.groups
        }

        # Chunk dependencies from the tile edges. Coarse chunks of two groups can depend on each other both ways
        # when their layers interleave (fusion); such mutually dependent chunks may run in any order relative to
        # each other, so the edges inside a strongly connected component are dropped, except each group's own
        # chunk chain.
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(self.chunks)))
        for producer, consumer in space.reference_workload.edges():
            if not (isinstance(producer, ComputationNode) and isinstance(consumer, ComputationNode)):
                continue
            a, b = self.chunk_of_tile[tile_key(producer)], self.chunk_of_tile[tile_key(consumer)]
            if a != b:
                graph.add_edge(a, b)
        component_of = {
            chunk: index
            for index, component in enumerate(nx.strongly_connected_components(graph))
            for chunk in component
        }
        dependencies = [(a, b) for a, b in graph.edges() if component_of[a] != component_of[b]]
        self.dropped_dependencies = graph.number_of_edges() - len(dependencies)
        self.chains = [
            (index, index + 1)
            for index in range(len(self.chunks) - 1)
            if self.chunks[index][0] == self.chunks[index + 1][0]
        ]
        # Transitive reduction: transfers take no time in the model and every task lasts at least a cycle, so an
        # edge implied by a path of others constrains nothing. Exact, and what keeps the tile level tractable
        # (attention: ~100k tile edges, most of them implied).
        dag = nx.DiGraph()
        dag.add_nodes_from(range(len(self.chunks)))
        dag.add_edges_from(dependencies + self.chains)
        chain_set = set(self.chains)
        self.dependencies = [edge for edge in nx.transitive_reduction(dag).edges() if edge not in chain_set]
        logger.info(
            f"{type(self).__name__}: {len(self.groups)} group(s), {len(self.chunks)} task(s), "
            f"{len(self.design_keys)} design(s), {len(self.dependencies)} dependency edge(s) after transitive "
            f"reduction of {len(dependencies)} ({self.dropped_dependencies} dropped inside mutually dependent "
            f"chunk sets)."
        )

    # ----------------------------------------------------------------------------------------- model ---------

    def _build_model(self) -> None:  # noqa: PLR0912, PLR0915
        """The CP-SAT model of the module docstring. Slots `0..G-1`; group `i` (in unrolled order) may only use
        slots `<= i` and used slots come first, which removes the relabelling symmetry."""
        model = cp_model.CpModel()
        groups, keys = self.groups, self.design_keys
        nb_slots = len(groups)
        slots = range(nb_slots)
        horizon = sum(max(self.duration[c].values()) for c in range(len(self.chunks)))

        self.y = {(p, key): model.NewBoolVar(f"y_{p}_{key}") for p in slots for key in keys}
        self.used = [model.NewBoolVar(f"u_{p}") for p in slots]
        self.x: dict[tuple[int, int], Any] = {}
        for index, group in enumerate(groups):
            for p in range(index + 1):
                self.x[group, p] = model.NewBoolVar(f"x_{group}_{p}")
            model.AddExactlyOne(self.x[group, p] for p in range(index + 1))
        for p in slots:
            model.Add(sum(self.y[p, key] for key in keys) == self.used[p])
            hosted = [self.x[group, p] for group in groups if (group, p) in self.x]
            model.Add(sum(hosted) >= self.used[p])
            for literal in hosted:
                model.AddImplication(literal, self.used[p])
            if p + 1 < nb_slots:
                model.AddImplication(self.used[p + 1], self.used[p])
        if not self.fold_allow_sibling_merge:
            by_layer: dict[int, list[int]] = {}
            for group in groups:
                by_layer.setdefault(self.layer_of_group[group], []).append(group)
            for siblings in by_layer.values():
                for a, b in combinations(siblings, 2):
                    for p in slots:
                        if (a, p) in self.x and (b, p) in self.x:
                            model.AddBoolOr([self.x[a, p].Not(), self.x[b, p].Not()])

        # Group energy and chunk durations follow the design of the slot the group sits on.
        max_energy = {group: max(self.group_energy[group].values()) for group in groups}
        self.group_energy_var = {group: model.NewIntVar(0, max_energy[group], f"e_{group}") for group in groups}
        for (group, p), literal in self.x.items():
            model.Add(
                self.group_energy_var[group] == sum(self.group_energy[group][key] * self.y[p, key] for key in keys)
            ).OnlyEnforceIf(literal)

        # A task's duration follows the design of its group's slot. Tasks of one group with the same per-design
        # durations (same tile-shape composition) share one duration variable, so the "duration = design's
        # duration" constraints are written once per (group, composition, slot) rather than per task -- at the tile
        # level that is the difference between a few hundred constraints and hundreds of thousands.
        duration_var: dict[tuple[int, tuple], Any] = {}
        for c, (group, _index) in enumerate(self.chunks):
            durations = self.duration[c]
            signature = (group, tuple(durations[key] for key in keys))
            if signature in duration_var:
                continue
            duration = model.NewIntVar(min(durations.values()), max(durations.values()), f"d_{len(duration_var)}")
            duration_var[signature] = duration
            for p in slots:
                if (group, p) in self.x:
                    model.Add(duration == sum(durations[key] * self.y[p, key] for key in keys)).OnlyEnforceIf(
                        self.x[group, p]
                    )

        self.start, self.end = [], []
        intervals_of_slot: dict[int, list[Any]] = {p: [] for p in slots}
        for c, (group, _index) in enumerate(self.chunks):
            duration = duration_var[(group, tuple(self.duration[c][key] for key in keys))]
            start = model.NewIntVar(0, horizon, f"s_{c}")
            end = model.NewIntVar(0, horizon, f"c_{c}")
            self.start.append(start)
            self.end.append(end)
            model.Add(end == start + duration)
            for p in slots:
                if (group, p) in self.x:
                    intervals_of_slot[p].append(
                        model.NewOptionalIntervalVar(start, duration, end, self.x[group, p], f"i_{c}_{p}")
                    )
        for intervals in intervals_of_slot.values():
            model.AddNoOverlap(intervals)
        for a, b in self.dependencies + self.chains:
            model.Add(self.start[b] >= self.end[a])

        self.latency = model.NewIntVar(0, horizon, "latency")
        model.AddMaxEquality(self.latency, self.end)
        self.energy = sum(self.group_energy_var.values())
        self.scaled_area = {key: round(self.area_of[key] * AREA_SCALE) for key in keys}
        self.area = sum(self.scaled_area[key] * self.y[p, key] for p in slots for key in keys)
        self.model = model
        self.objective_expr = {"latency": self.latency, "energy": self.energy, "area": self.area}

    # ----------------------------------------------------------------------------------------- solving -------

    def _solve(self, objective: str, bounds: dict[str, int], time_limit: float, label: str) -> ModelSolution | None:
        """Minimize `objective` subject to `objective <= bound` for each entry of `bounds`. The bounds are added as
        temporary constraints (removed after the solve), and the previous solution is kept as a hint."""
        model = self.model
        # Epsilon bounds the incremental way: each bound is a constraint guarded by a fresh literal, and only this
        # solve's literals are assumed true. Bounds of earlier solves stay in the model but their literals are free,
        # so they constrain nothing.
        literals = []
        for name, bound in bounds.items():
            literal = model.NewBoolVar(f"eps_{name}_{len(self.solve_log)}")
            model.Add(self.objective_expr[name] <= bound).OnlyEnforceIf(literal)
            literals.append(literal)
        model.ClearAssumptions()
        model.AddAssumptions(literals)
        model.Minimize(self.objective_expr[objective])
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = self.solver_workers
        t0 = time.perf_counter()
        status = solver.Solve(model)
        elapsed = time.perf_counter() - t0
        entry = {
            "label": label,
            "objective": objective,
            **{f"bound_{name}": bounds.get(name, "") for name in OBJECTIVES},
            "status": solver.StatusName(status),
            "time_s": round(elapsed, 2),
        }
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            self.solve_log.append(entry)
            return None
        solution = self._extract(solver)
        entry.update(latency=solution.latency, energy=solution.energy, area=solution.area)
        self.solve_log.append(entry)
        model.ClearHints()
        for var in self.x.values():
            model.AddHint(var, solver.Value(var))
        for var in self.y.values():
            model.AddHint(var, solver.Value(var))
        return solution

    def _extract(self, solver: cp_model.CpSolver) -> ModelSolution:
        slot_of_group = {group: p for (group, p), var in self.x.items() if solver.Value(var)}
        design_of_slot = {p: key for (p, key), var in self.y.items() if solver.Value(var)}
        return ModelSolution(
            slot_of_group=slot_of_group,
            design_of_slot=design_of_slot,
            latency=solver.Value(self.latency),
            energy=solver.Value(self.energy) * ENERGY_SCALE,
            area=solver.Value(self.area) / AREA_SCALE,
        )

    def _scaled(self, solution: ModelSolution, name: str) -> int:
        if name == "latency":
            return solution.latency
        if name == "energy":
            return round(solution.energy / ENERGY_SCALE)
        return round(solution.area * AREA_SCALE)

    def _lexicographic(self, order: tuple[str, ...], bounds: dict[str, int], label: str) -> ModelSolution | None:
        """Minimize the objectives in `order` one after the other, each fixed at its optimum for the next (the
        tie-breaks get half the time limit), so the result is non-dominated within the model."""
        bounds = dict(bounds)
        best = None
        for step, objective in enumerate(order):
            limit = self.solve_time_limit_s if step == 0 else self.solve_time_limit_s / 2
            solution = self._solve(objective, bounds, limit, f"{label}:{objective}")
            if solution is None:
                return best
            best = solution
            bounds[objective] = self._scaled(solution, objective)
        return best

    def _sweep(self) -> list[ModelSolution]:
        """Anchors (each objective minimized first), then latency minimized over an (area, energy) budget grid
        between each objective's minimum and its value at the latency-optimal point."""
        solutions: dict[tuple, ModelSolution] = {}

        def keep(solution: ModelSolution | None) -> None:
            if solution is not None:
                solutions.setdefault(solution.key, solution)

        anchors = {}
        for first in OBJECTIVES:
            rest = tuple(o for o in OBJECTIVES if o != first)
            anchors[first] = self._lexicographic((first, *rest), {}, f"anchor-{first}")
            keep(anchors[first])
        fastest = anchors["latency"]
        if fastest is None:
            return list(solutions.values())

        def grid(name: str, levels: int) -> list[int]:
            low = anchors[name] and self._scaled(anchors[name], name)
            high = self._scaled(fastest, name)
            if low is None or levels < MIN_GRID_LEVELS or high <= low:
                return []
            return sorted({round(low + (high - low) * i / (levels - 1)) for i in range(levels - 1)})

        for area_bound in grid("area", self.area_levels):
            for energy_bound in [None, *grid("energy", self.energy_levels)]:
                bounds = {"area": area_bound}
                if energy_bound is not None:
                    bounds["energy"] = energy_bound
                keep(self._lexicographic(("latency", "energy", "area"), bounds, "grid"))
        return list(solutions.values())

    # ------------------------------------------------------------------------------------- measuring -------

    def _measure_solution(self, solution: ModelSolution, archive: Any, records: list[dict[str, Any]]) -> str:
        space = self.space
        merge_of_core = dict(solution.slot_of_group)
        rolled_of = renumber_merged_cores(merge_of_core)
        slot_of_rolled = {rolled_of[group]: slot for group, slot in merge_of_core.items()}
        nb_cores = len(slot_of_rolled)
        design_key_of_core = {core: solution.design_of_slot[slot_of_rolled[core]] for core in range(nb_cores)}
        core_dict_of_core = {core: self.designs[key] for core, key in design_key_of_core.items()}
        area = sum(space.costs.area[key] for key in design_key_of_core.values())
        record_id = f"opt{self.level}-c{nb_cores}-{len(records)}"
        try:
            workload, accelerator, cost_lut, topology = space.assemble_rolled(
                merge_of_core, core_dict_of_core, design_key_of_core
            )
            scme = space.measure(workload, accelerator, cost_lut)
        except Exception:
            logger.warning(f"{type(self).__name__}: solution {record_id} could not be scheduled.", exc_info=True)
            if self.search_budget is not None:
                self.search_budget.record(math.inf, math.inf, area, info=f"{record_id} infeasible")
            return "infeasible"
        point = (float(scme.latency), float(scme.energy), area)
        if self.search_budget is not None:
            self.search_budget.record(*point, info=record_id)
        record = {
            "id": record_id,
            "kind": "rolled",
            "ranks": (),
            "source_id": f"model-l{self.level}",
            "nb_cores": nb_cores,
            "design_keys": tuple(design_key_of_core[core] for core in range(nb_cores)),
            "latency": point[0],
            "energy": point[1],
            "area": point[2],
            # The schedule search's CSV calls the analytic estimate a bound; here it is the model's prediction.
            "latency_bound": solution.latency,
            "energy_bound": solution.energy,
            "predicted_latency": solution.latency,
            "predicted_energy": solution.energy,
            "fold_tolerance": "",
            "fold_objective": f"cpsat-level{self.level}",
            "fold_overlap": f"chunks={self.chunks_per_group}",
            "merge_of_core": merge_of_core,
            "design_key_of_core": design_key_of_core,
            "core_dict_of_core": core_dict_of_core,
            "members_of_core": dict(topology.members_of_core),
            "hosted_node_groups": dict(topology.hosted_node_groups),
            "node_core_ids": dict(topology.node_core_ids),
            "link_connections": list(topology.link_connections),
            "bus_connection": dict(topology.bus_connection),
        }
        records.append(record)
        archive.add(point, record)
        logger.info(
            f"{type(self).__name__}: measured {record_id}: {nb_cores} core(s), latency={point[0]:.6g} "
            f"(predicted {solution.latency:.6g}), energy={point[1]:.6g} (predicted {solution.energy:.6g}), "
            f"area={area:.6g} (front: {len(archive.entries)})."
        )
        return "measured"

    def _save_solve_log(self) -> None:
        path = os.path.join(self.out_dir, "solves.csv")
        columns = ["label", "objective", "bound_latency", "bound_energy", "bound_area", "status", "time_s",
                   "latency", "energy", "area"]  # fmt: skip
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for entry in self.solve_log:
                writer.writerow({column: entry.get(column, "") for column in columns})

    # ------------------------------------------------------------------------------------------- run ---------

    def run(self):
        state = self.schedule_search
        if not self.space.full_matrix:
            raise ValueError(
                f"{type(self).__name__}: needs the full shape x design cost matrix; raise `max_cme_matrix_cells`."
            )
        os.makedirs(self.out_dir, exist_ok=True)
        t0 = time.perf_counter()
        self._build_problem()
        self._build_model()
        t_model = time.perf_counter() - t0
        solutions = self._sweep()
        t_solve = time.perf_counter() - t0 - t_model
        solutions.sort(key=lambda s: (s.latency, s.energy, s.area))
        statuses: dict[str, int] = {}
        for solution in solutions[: self.max_optimized_evaluations]:
            status = self._measure_solution(solution, state.archive, state.records)
            statuses[status] = statuses.get(status, 0) + 1
        t_total = time.perf_counter() - t0
        self._save_solve_log()
        state.stats[f"optimization_l{self.level}"] = {
            "groups": len(self.groups),
            "tasks": len(self.chunks),
            "designs": len(self.design_keys),
            "dependencies": len(self.dependencies),
            "dropped_dependencies": self.dropped_dependencies,
            "solves": len(self.solve_log),
            "solutions": len(solutions),
            **statuses,
            "model_time_s": round(t_model, 2),
            "solve_time_s": round(t_solve, 2),
            "measure_time_s": round(t_total - t_model - t_solve, 2),
            "total_time_s": round(t_total, 2),
        }
        logger.info(f"{type(self).__name__}: {state.stats[f'optimization_l{self.level}']}.")
        state.report["out_dir"] = self.out_dir
        kwargs = self.kwargs.copy()
        kwargs["schedule_search"] = state
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        yield from sub_stage.run()


class GroupScheduleOptimizationStage(ScheduleOptimizationStage):
    """Level 1: one task per layer group (see the module docstring)."""

    level = 1
    default_chunks_per_group = 1

    def __init__(self, list_of_callables: list[StageCallable], **kwargs: Any):
        kwargs.pop("chunks_per_group", None)
        super().__init__(list_of_callables, chunks_per_group=1, **kwargs)


class ChunkScheduleOptimizationStage(ScheduleOptimizationStage):
    """Level 2: `chunks_per_group` tasks per layer group, with chunk-to-chunk dependencies (see the module
    docstring)."""

    level = 2
    default_chunks_per_group = 8


class TileScheduleOptimizationStage(ScheduleOptimizationStage):
    """Level 3: one task per tile, with the tile dependency graph itself (transitively reduced) as precedences.
    The hardware decisions are those of levels 1 and 2; only the time model is exact up to the per-tile costs.
    Much larger than level 2 (every tile is an optional interval on every slot its group may use), so give it a
    longer `solve_time_limit_s`."""

    level = 3

    def __init__(self, list_of_callables: list[StageCallable], **kwargs: Any):
        kwargs.pop("chunks_per_group", None)
        # More chunks than any group has tiles: every tile becomes its own task.
        super().__init__(list_of_callables, chunks_per_group=10**9, **kwargs)
