"""Enumerate whole-accelerator schedules over *every* per-tile Pareto core design, instead of collapsing each
tile's front to a single design.

`CoreDesignSelectionStage` picks, per unique tile, the one front design with the best `sort_key` and evaluates
that single accelerator. That throws the multi-objective work away: the front's other designs might combine into
an accelerator that is slower per tile but far cheaper in area, or one whose slow tiles sit off the critical path
and cost nothing at all. This stage keeps every design and searches the *product* of the fronts -- one choice
per decision class -- measuring combinations with the real scheduler and building the Pareto front over
`(latency, energy, area)` of whole schedules.

Done naively that is `prod(len(front) for front in fronts)` full pipeline runs. Three properties of how a
schedule is constructed make it tractable. The first two -- a design-independent topology and per-(tile,
design) costs that can be learned once and reused -- are what `ScheduleSpace` captures. The third is what this
stage adds:

3. **Two objectives are exact without scheduling, the third is boundable.**
   - `area` is a plain sum over cores of each design's area, plus the interconnect's -- exact. The interconnect
     is a constant (its routers) plus a term per core linear in the side length of its design
     (`InterconnectTerms`), so it splits over classes like the core area does.
   - schedule `energy` is the sum of per-tile CME energies *plus* non-negative transfer/eviction energy, so that
     sum is a lower bound.
   - schedule `latency` is bounded below by both the busiest core's serial load (a core runs its tiles one at a
     time) and the tiled DAG's critical path weighted by per-tile runtimes (transfers only ever delay a node).
   Each bound is monotone non-decreasing in the per-class costs, so filling the unassigned classes with their
   per-objective minima bounds *every* completion of a partial assignment.

Together these give a sound branch and bound: if a (partial or complete) assignment's lower-bound triple is
dominated by an already-*measured* schedule, every completion of it has a true triple at least as large and is
dominated too, so the subtree is skipped without ever running the scheduler. Only measured points are ever used
as dominators, so the front found is the one exhaustive enumeration would find -- `prune_schedules=False` runs
the exhaustive search for exactly that comparison.
"""

import logging
import math
import os
import time
from typing import Any

from stream.stages.generation.core_architecture_exploration import ParetoCoreSearchResult
from stream.stages.generation.schedule_evaluation import (
    OBJECTIVES,
    MeasurementPipeline,
    ParetoArchive,
    ScheduleSearchStage,
    core_occupancy,
    downstream_requires,
    format_ranks,
    measurement_tail,
)
from stream.stages.generation.schedule_space import ScheduleSearchState, ScheduleSpace
from stream.stages.stage import StageCallable
from stream.visualization.design_record import design_record

logger = logging.getLogger(__name__)


class ScheduleExplorationStage(ScheduleSearchStage):
    """Branch and bound over the per-tile core Pareto fronts (`core_pareto_fronts`, from
    `CoreArchitectureExplorationStage`), measuring whole schedules with the stages after the chain's last
    schedule search stage. See the module docstring for the properties the search exploits.

    Builds the `ScheduleSpace` -- plan, infeasible-design filter, cost-model warm-up -- and hands it down with the
    archive and every measured record as `schedule_search` (a `ScheduleSearchState`). A `ScheduleRollingStage`
    after it extends the same archive with folded schedules; a `ScheduleSelectionStage` ends the chain by
    re-measuring the best one. When a later stage asks for it (`requires_full_cost_matrix`,
    `requires_core_occupancy`), the warm-up covers the full shape x design matrix and every measured record
    keeps its per-core busy windows.

    The warm-up's cost LUTs are cached as `cost_lut_warmup_<k>.pickle` (diagonal) or
    `cost_lut_design_<design key>.pickle` (full matrix), under `schedule_search/` or, when the full matrix is
    requested, under `rolled_search/`. The search's own artifacts (CSV, figures) are written by the
    visualization stages from `extra_info["schedule_report"]`.
    """

    # Class-level defaults so stages built without `__init__` (the synthetic tests) skip the rolled extras.
    full_matrix = False
    track_occupancy = False

    def __init__(  # noqa: PLR0913
        self,
        list_of_callables: list[StageCallable],
        *,
        core_pareto_fronts: ParetoCoreSearchResult,
        workload: Any,
        original_workload: Any,
        accelerator: Any,
        tiled_workload_path: str,
        schedule_search_dir: str | None = None,
        max_schedule_evaluations: int = 1000,
        max_search_nodes: int = 1_000_000,
        prune_schedules: bool = True,
        max_cme_matrix_cells: int = 2000,
        core_max_workers: int | None = 1,
        **kwargs: Any,
    ):
        """
        Args:
            core_pareto_fronts: the per-tile Pareto fronts to combine.
            schedule_search_dir: where this search's report and (diagonal) warm-up cache go. Defaults to
                `schedule_search/` next to the tiled workload.
            max_schedule_evaluations: stop after this many *measured* schedules. Everything measured is still
                exact, but the reported front is then only the front of what was measured before the cut-off.
            max_search_nodes: budget on branch-and-bound tree nodes, a guard for fronts where pruning cannot
                keep up with the combinatorics.
            prune_schedules: False measures every combination exhaustively -- exponentially slower, and used to
                verify that the pruned search returns the same front.
            max_cme_matrix_cells: guard on the full `shape x design` warm-up; above it the warm-up falls back to
                the diagonal (see `ScheduleSpace.warm_up`).
            core_max_workers: processes for the full-matrix warm-up.
        """
        super().__init__(list_of_callables, **kwargs)
        self.core_pareto_fronts = core_pareto_fronts
        self.workload = workload
        self.original_workload = original_workload
        self.accelerator = accelerator
        self.tiled_workload_path = tiled_workload_path
        self.schedule_search_dir = schedule_search_dir or os.path.join(
            os.path.dirname(tiled_workload_path), "schedule_search"
        )
        self.max_schedule_evaluations = max_schedule_evaluations
        self.max_search_nodes = max_search_nodes
        self.prune_schedules = prune_schedules
        self.max_cme_matrix_cells = max_cme_matrix_cells
        self.max_workers = core_max_workers
        self.search_budget = kwargs.get("search_budget")
        self.full_matrix = downstream_requires(list_of_callables, "requires_full_cost_matrix")
        self.track_occupancy = downstream_requires(list_of_callables, "requires_core_occupancy")
        # The full matrix is much larger and only the rolled search reads it, so it is cached apart, under the
        # rolled search's directory.
        self.cme_cache_dir = (
            kwargs.get("rolled_search_dir") or os.path.join(os.path.dirname(tiled_workload_path), "rolled_search")
            if self.full_matrix
            else self.schedule_search_dir
        )

    # ------------------------------------------------------------------------------ bounds (property 3) ------

    def _index_costs(self) -> None:
        """Tabulate, per cost pair and per rank, the runtime/energy the warm-up measured, plus each pair's and
        class's per-objective minimum -- the minima are what make a partial assignment boundable."""
        space = self.space
        self.pair_runtime: list[list[float]] = []
        self.pair_energy: list[list[float]] = []
        for shape_class, design_class in space.cost_pairs:
            runtimes, energies = [], []
            for rank in range(len(space.designs_per_class[design_class])):
                key = (shape_class, space.design_key(design_class, rank))
                runtimes.append(space.costs.runtime[key])
                energies.append(space.costs.energy[key])
            self.pair_runtime.append(runtimes)
            self.pair_energy.append(energies)
        self.pair_min_runtime = [min(runtimes) for runtimes in self.pair_runtime]
        self.pair_min_energy = [min(energies) for energies in self.pair_energy]

        # Each class's share of the area: its cores, and the interconnect wires along them. The routers do not
        # depend on the designs, so they are one constant on top (`interconnect_routers`).
        terms = space.interconnect_terms()
        self.interconnect_routers = terms.routers
        wire_coefficient_of_class = dict.fromkeys(space.decision_classes, 0.0)
        for core_id in space.unrolled_core_ids:
            wire_coefficient_of_class[space.design_class_of_core[core_id]] += (
                terms.link_coefficient.get(core_id, 0.0) + terms.bus_coefficient
            )
        self.class_area: list[list[float]] = []
        self.class_wire_area: list[list[float]] = []
        for class_index in space.decision_classes:
            areas, wire_areas = [], []
            for rank in range(len(space.designs_per_class[class_index])):
                area = space.costs.area[space.design_key(class_index, rank)]
                wire_areas.append(wire_coefficient_of_class[class_index] * math.sqrt(max(0.0, area)))
                areas.append(space.cores_per_class[class_index] * area + wire_areas[-1])
            self.class_area.append(areas)
            self.class_wire_area.append(wire_areas)
        self.class_min_area = [min(areas) for areas in self.class_area]

    def _bounds(self, ranks: tuple[int, ...], depth: int | None = None) -> tuple[float, float, float]:
        """Lower bound on `(latency, energy, area)` for every completion of `ranks[:depth]`. With `depth` at its
        default (a complete assignment) latency/energy stay bounds and area is exact."""
        space = self.space
        depth = len(space.decision_classes) if depth is None else depth
        runtimes = [
            self.pair_runtime[index][ranks[pair_depth]] if pair_depth < depth else self.pair_min_runtime[index]
            for index, pair_depth in enumerate(space.pair_depth)
        ]
        energies = [
            self.pair_energy[index][ranks[pair_depth]] if pair_depth < depth else self.pair_min_energy[index]
            for index, pair_depth in enumerate(space.pair_depth)
        ]
        area = self.interconnect_routers + sum(
            self.class_area[depth_][ranks[depth_]] if depth_ < depth else self.class_min_area[depth_]
            for depth_ in range(len(space.decision_classes))
        )
        return self._latency_bound(runtimes), self._energy_bound(energies), area

    def _latency_bound(self, runtimes: list[float]) -> float:
        """max(busiest core's serial load, DAG critical path): a core runs its tiles one at a time, and a tile
        never starts before its predecessors finish -- transfers only add to both."""
        space = self.space
        resource_bound = max(
            (sum(count * runtimes[index] for index, count in counts) for counts in space.counts_per_core),
            default=0.0,
        )
        finish = [0.0] * len(space.topo_pair_index)
        critical_path = 0.0
        for position, pair_index in enumerate(space.topo_pair_index):
            predecessors = space.topo_predecessors[position]
            start = max((finish[p] for p in predecessors), default=0.0)
            finish[position] = start + runtimes[pair_index]
            critical_path = max(critical_path, finish[position])
        return max(resource_bound, critical_path)

    def _energy_bound(self, energies: list[float]) -> float:
        """Sum of per-tile compute/memory energy; the schedule only adds non-negative transfer and eviction
        energy on top."""
        return sum(
            multiplicity * energy for multiplicity, energy in zip(self.space.pair_multiplicity, energies, strict=True)
        )

    # ------------------------------------------------------------------------------ branch and bound ---------

    def _measure(self, ranks: tuple[int, ...]):
        """Measure one combination with the real pipeline: build its accelerator, hand the measurement stages a
        cost LUT assembled from the warm-up cache, and let them produce the schedule."""
        return self.space.measure(*self.space.assemble_unrolled(ranks))

    def _measured_extras(self, scme: Any) -> dict[str, Any]:
        """Extra fields to attach to a measured record: each core's busy windows, when a later stage asks for
        them (they are what the rolled search folds), without holding on to the `StreamCostModelEvaluation`."""
        if not self.track_occupancy:
            return {}
        return {"occupancy": core_occupancy(scme, self.space.unrolled_core_ids)}

    def _class_aggregate(self, depth: int, rank: int, objective_index: int) -> float:
        """One class's own contribution on one objective, used only to order and seed the search."""
        if objective_index == OBJECTIVES.index("area"):
            return self.class_area[depth][rank]
        table = self.pair_runtime if objective_index == OBJECTIVES.index("latency") else self.pair_energy
        return sum(
            self.space.pair_multiplicity[index] * table[index][rank]
            for index, pair_depth in enumerate(self.space.pair_depth)
            if pair_depth == depth
        )

    def _search(self) -> tuple[ParetoArchive, dict[str, Any]]:
        space = self.space
        archive = ParetoArchive()
        stats = {
            "combinations": self.nb_combinations,
            "measured": 0,
            "pruned_subtrees": 0,
            "visited": 0,
            "hit_evaluation_budget": False,
            "hit_node_budget": False,
        }
        self.all_results: list[dict[str, Any]] = []
        measured_ranks: set[tuple[int, ...]] = set()

        def measure(ranks: tuple[int, ...]) -> None:
            if ranks in measured_ranks:
                return
            if stats["measured"] >= self.max_schedule_evaluations:
                stats["hit_evaluation_budget"] = True
                return
            measured_ranks.add(ranks)
            bounds = self._bounds(ranks)
            scme = self._measure(ranks)
            point = (float(scme.latency), float(scme.energy), bounds[2])
            stats["measured"] += 1
            if self.search_budget is not None:
                self.search_budget.record(
                    *point, info=f"unrolled {format_ranks(ranks)}", artifact=lambda: design_record(scme)
                )
            record = {
                "id": f"u{format_ranks(ranks)}",
                "kind": "unrolled",
                "ranks": ranks,
                "nb_cores": sum(space.cores_per_class.values()),
                "design_keys": tuple(
                    space.design_key(class_index, ranks[depth])
                    for depth, class_index in enumerate(space.decision_classes)
                ),
                "latency": point[0],
                "energy": point[1],
                "area": point[2],
                "interconnect_area": self.interconnect_routers
                + sum(self.class_wire_area[depth][rank] for depth, rank in enumerate(ranks)),
                "latency_bound": bounds[0],
                "energy_bound": bounds[1],
            }
            record.update(self._measured_extras(scme))
            self.all_results.append(record)
            archive.add(point, record)
            logger.info(
                f"ScheduleExplorationStage: measured schedule {stats['measured']}/{self.max_schedule_evaluations} "
                f"{ranks}: latency={point[0]:.6g}, energy={point[1]:.6g}, area={point[2]:.6g} "
                f"(front: {len(archive.entries)})."
            )

        # Seed with the per-objective extremes: cheap, and a strong dominator set makes the first descents prune
        # hard instead of wandering.
        for objective_index in range(3):
            measure(
                tuple(
                    min(
                        self.rank_order[depth],
                        key=lambda rank, d=depth, o=objective_index: self._class_aggregate(d, rank, o),
                    )
                    for depth in range(len(space.decision_classes))
                )
            )

        def descend(depth: int, ranks: list[int]) -> None:
            stats["visited"] += 1
            if stats["visited"] > self.max_search_nodes:
                stats["hit_node_budget"] = True
                return
            if stats["measured"] >= self.max_schedule_evaluations:
                stats["hit_evaluation_budget"] = True
                return
            if depth == len(space.decision_classes):
                measure(tuple(ranks))
                return
            for rank in self.rank_order[depth]:
                ranks[depth] = rank
                if self.prune_schedules and archive.is_dominated(self._bounds(tuple(ranks), depth + 1)):
                    stats["pruned_subtrees"] += 1
                    continue
                descend(depth + 1, ranks)
            ranks[depth] = self.rank_order[depth][0]

        descend(0, [self.rank_order[depth][0] for depth in range(len(space.decision_classes))])
        return archive, stats

    # -------------------------------------------------------------------------------------------- run --------

    def _prepare_search(self) -> None:
        """Everything the search needs before the first measurement: the schedule space (plan, infeasible-design
        filter, cost-model warm-up), the cost tables and the per-class rank order."""
        os.makedirs(self.schedule_search_dir, exist_ok=True)
        os.makedirs(self.cme_cache_dir, exist_ok=True)
        self.space = ScheduleSpace(
            self.core_pareto_fronts,
            workload=self.workload,
            original_workload=self.original_workload,
            accelerator=self.accelerator,
            tiled_workload_path=self.tiled_workload_path,
            pipeline=MeasurementPipeline(measurement_tail(self.list_of_callables), self.kwargs),
            estimation_kwargs={
                "loma_lpf_limit": self.kwargs["loma_lpf_limit"],
                "layer_stacks": self.kwargs["layer_stacks"],
                "temporal_mapping_type": self.kwargs["temporal_mapping_type"],
            },
            latency_attr=self.kwargs.get("latency_attr", "latency_total2"),
            max_workers=self.max_workers,
            interconnect_model=self.kwargs.get("interconnect_area_model"),
        )
        self.space.drop_infeasible_designs(all_shapes=self.full_matrix)
        self.space.warm_up(self.cme_cache_dir, full_matrix=self.full_matrix, max_cells=self.max_cme_matrix_cells)
        self._index_costs()
        self._build_rank_order()

    def _build_rank_order(self) -> None:
        """Designs that decode to the same core are the same accelerator: deduplicate each class's ranks before
        taking the product, and order them cheapest-runtime-first so good schedules are measured early."""
        space = self.space
        self.rank_order: list[list[int]] = []
        for depth, class_index in enumerate(space.decision_classes):
            unique_ranks: dict[str, int] = {}
            for rank in range(len(space.designs_per_class[class_index])):
                unique_ranks.setdefault(space.design_key(class_index, rank), rank)
            self.rank_order.append(
                sorted(unique_ranks.values(), key=lambda rank, d=depth: self._class_aggregate(d, rank, 0))
            )
        self.nb_combinations = math.prod(len(ranks) for ranks in self.rank_order)
        logger.info(
            f"ScheduleExplorationStage: {self.nb_combinations} schedule combination(s) over "
            f"{len(space.decision_classes)} decision class(es), front sizes {[len(r) for r in self.rank_order]}."
        )

    def _report(self) -> dict[str, Any]:
        """Plain data the visualization stages need to draw this search's results without the space itself."""
        space = self.space
        return {
            "out_dir": self.schedule_search_dir,
            "accelerator_name": self.accelerator.name,
            "decision_classes": list(space.decision_classes),
            "front_sizes": [len(ranks) for ranks in self.rank_order],
            "layer_names": dict(space.layer_names),
            "layer_order": list(space.layer_order),
            "node_core_ids": dict(space.core_ids_of_node),
            "offchip_core_id": space.offchip_core_id,
            "bus_connection": space.bus_connection,
            "link_connections": list(space.link_connections),
            "fold_log": None,
        }

    def run(self):
        t_start = time.perf_counter()
        self._prepare_search()
        t_prepared = time.perf_counter()
        archive, stats = self._search()
        # Wall times, so the searches that extend this one can be compared without the shared preparation.
        stats["prepare_time_s"] = round(t_prepared - t_start, 2)
        stats["search_time_s"] = round(time.perf_counter() - t_prepared, 2)
        if not archive.entries:
            raise ValueError("ScheduleExplorationStage: no schedule could be measured.")
        logger.info(
            f"ScheduleExplorationStage: measured {stats['measured']}/{stats['combinations']} combination(s), "
            f"pruned {stats['pruned_subtrees']} subtree(s), {len(archive.entries)} schedule(s) on the front."
        )
        kwargs = self.kwargs.copy()
        kwargs["schedule_search"] = ScheduleSearchState(
            space=self.space, archive=archive, records=self.all_results, stats=stats, report=self._report()
        )
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        yield from sub_stage.run()
