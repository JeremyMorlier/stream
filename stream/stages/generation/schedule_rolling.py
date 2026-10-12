"""Fold the *unrolled* accelerator -- one dedicated core per `(layer, inter-core group)` -- onto fewer physical
cores by reusing cores that are idle later in the schedule, and put the results on the same Pareto front.

`ScheduleExplorationStage` searches which core *design* each layer gets, but never how many cores there are:
`derive_group_dedicated_topology` gives every `(original node, group)` pair its own core, so the core count
grows with the workload and each core sits idle outside its layer's window. This stage asks the complementary
question -- *is a core that has finished its layer free to run a later one?* -- and answers it by measurement:

1. Every schedule the unrolled search measured reports, per core, the windows in which it was actually busy
   (`core_occupancy`, read off the scheduled nodes' `start`/`end`; the search keeps them because this stage
   sets `requires_core_occupancy`).
2. Cores whose busy windows don't collide can share one physical core. Which cores may share is a graph
   colouring; the colour classes are the folded cores (`_fold_plans`).
3. Each folded core needs a design that runs *all* the layers now on it -- which is why this stage sets
   `requires_full_cost_matrix`, so the warm-up builds the full `shape x design` cost-model matrix instead of the
   diagonal the unrolled search alone needs.
4. The fold changes the hardware graph: traffic between two merged cores becomes free intra-core traffic and
   its link disappears, while a folded core inherits the union of its members' neighbours, so *new* links
   appear and previously separate traffic coalesces onto them (`derive_rolled_topology`).
5. The folded accelerator is then re-measured with the real scheduler and inserted into the *same*
   `ParetoArchive` as the unrolled schedules, so the front spans both families: unrolled designs win latency,
   folded ones win area, and the interesting points are in between.

Two things are deliberately not assumed:

- **A fold is not free.** Busy windows are compute-only; a core is also occupied by blocking offchip transfers,
  and a folded core streams more weights over the same fixed-bandwidth bus. So even a zero-overlap fold can
  lose latency. Nothing here estimates that -- the re-measurement is the ground truth, and the analytic bounds
  are only used to skip folds that are *already* dominated (`_fold_bounds`), which is sound for the same reason
  the unrolled branch and bound is.
- **Layer fusion leaves little idle time.** In `fused` mode a layer's tiles interleave with the next layer's,
  so the strict "never overlap" graph can admit no merge at all. That is why folding is swept over a
  *tolerance* ladder that also admits overlapping cores, paying serialization for area -- the trade-off the
  combined front is there to expose.
"""

import logging
import math
import os
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

from stream.stages.generation.schedule_evaluation import (
    INTERVAL_T,
    CoreOccupancy,
    ParetoArchive,
    ScheduleSearchStage,
    format_ranks,
    interval_overlap,
    union_overlap,
)
from stream.stages.generation.schedule_space import ScheduleSearchState, ScheduleSpace
from stream.stages.stage import StageCallable

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------- colouring -----------


def greedy_interval_colouring(
    core_ids: list[int], conflicts: set[tuple[int, int]], hull: dict[int, INTERVAL_T]
) -> dict[int, int]:
    """Left-endpoint greedy: take cores in order of when they first become busy and give each the lowest colour
    no conflicting core already holds.

    On a pure interval graph this is optimal -- it uses exactly the maximum number of simultaneously live
    intervals. The graph here is that, plus any forbidden pairs, so the result stays a valid colouring but the
    optimality argument only holds when nothing is forbidden."""
    colour_of: dict[int, int] = {}
    for core_id in sorted(core_ids, key=lambda c: (hull[c][0], hull[c][1], c)):
        taken = {colour_of[other] for other in colour_of if canonical(core_id, other) in conflicts}
        colour = next(c for c in range(len(core_ids)) if c not in taken)
        colour_of[core_id] = colour
    return colour_of


def dsatur_colouring(
    core_ids: list[int], conflicts: set[tuple[int, int]], hull: dict[int, INTERVAL_T]
) -> dict[int, int]:
    """DSATUR: repeatedly colour the vertex with the most distinctly-coloured neighbours.

    Used for the exact (union-of-intervals) conflict graph, which is a subgraph of the hull graph and so can
    need strictly fewer cores -- e.g. when one layer's tiles interleave into another layer's gaps. DSATUR is
    exact on interval and chordal graphs, so it never does worse than the greedy above, and O(V^2) is nothing
    at these core counts. Ties break on conflict degree then on hull start, so the result is deterministic."""
    neighbours = {core_id: set() for core_id in core_ids}
    for a, b in conflicts:
        if a in neighbours and b in neighbours:
            neighbours[a].add(b)
            neighbours[b].add(a)

    colour_of: dict[int, int] = {}
    while len(colour_of) < len(core_ids):
        uncoloured = [core_id for core_id in core_ids if core_id not in colour_of]
        core_id = max(
            uncoloured,
            key=lambda c: (
                len({colour_of[n] for n in neighbours[c] if n in colour_of}),
                len(neighbours[c]),
                -hull[c][0],
                -c,
            ),
        )
        taken = {colour_of[n] for n in neighbours[core_id] if n in colour_of}
        colour_of[core_id] = next(c for c in range(len(core_ids)) if c not in taken)
    return colour_of


def canonical(a: int, b: int) -> tuple[int, int]:
    """An unordered core pair in a fixed order, so it can be looked up in a set."""
    return (a, b) if a <= b else (b, a)


def partition_key(colour_of: dict[int, int]) -> frozenset:
    """Identity of a fold that ignores how the colours were numbered."""
    classes: dict[int, set[int]] = {}
    for core_id, colour in colour_of.items():
        classes.setdefault(colour, set()).add(core_id)
    return frozenset(frozenset(members) for members in classes.values())


@dataclass
class FoldPlan:
    """One folded accelerator: which unrolled cores merge, and what design each merged core gets."""

    source_id: str
    source_ranks: tuple[int, ...]
    merge_of_core: dict[int, int]  # unrolled core id -> merge label
    rolled_of: dict[int, int]  # unrolled core id -> rolled core id
    design_key_of_core: dict[int, str]  # rolled core id -> design key
    core_dict_of_core: dict[int, dict[str, Any]]
    nb_cores: int
    tolerance: float
    objective: str
    overlap_mode: str

    @property
    def key(self) -> tuple:
        """Dedup identity: the same partition with the same designs is the same accelerator, whichever source
        schedule or tolerance produced it."""
        return (
            partition_key(self.merge_of_core),
            tuple(self.design_key_of_core[core_id] for core_id in range(self.nb_cores)),
        )

    @property
    def id(self) -> str:
        """Readable identity: source ranks, core count, design objective and tolerance."""
        return f"r{format_ranks(self.source_ranks)}-c{self.nb_cores}-{self.objective}-t{self.tolerance:g}"


class ScheduleRollingStage(ScheduleSearchStage):
    """Folds the schedules the unrolled search found (`schedule_search`, from `ScheduleExplorationStage`) onto
    fewer cores, measures each fold, and inserts it into the same archive, so the next stage receives one Pareto
    front over both families. Every measured fold is appended to the shared records too.

    Its report goes to `rolled_search/` instead of `schedule_search/`: the visualization stages write there
    `schedules.csv` (with the fold parameters that produced each schedule), `fold_log.csv` (every fold
    *considered*, including those skipped as dominated, duplicate or infeasible -- the diagnostic for whether the
    sweep produces variety or collapses) and the topology figures. Rolled entries carry the full fold
    (`merge_of_core`, `members_of_core`, `hosted_node_groups`, `design_key_of_core`, `core_dict_of_core`,
    `node_core_ids`, `link_connections`, `bus_connection`), i.e. enough to rebuild the accelerator or draw its
    topology without re-running.
    """

    requires_full_cost_matrix = True
    requires_core_occupancy = True

    def __init__(  # noqa: PLR0913
        self,
        list_of_callables: list[StageCallable],
        *,
        schedule_search: ScheduleSearchState,
        rolled_search_dir: str | None = None,
        fold_overlap: Literal["hull", "exact"] = "hull",
        fold_tolerances: tuple[float, ...] = (0.0, 0.05, 0.15, 0.35, 0.6, 1.0),
        rolled_core_count_targets: tuple[int, ...] | None = None,
        fold_allow_sibling_merge: bool = False,
        merged_design_pool: Literal["members", "all"] = "members",
        merged_design_objectives: tuple[str, ...] = ("latency", "energy", "area"),
        keep_singleton_designs: bool = True,
        max_rolled_variants_per_schedule: int = 8,
        max_rolled_evaluations: int = 200,
        fold_source: Literal["pareto", "all"] = "pareto",
        prune_rolled: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            schedule_search: the unrolled search's state, extended in place.
            rolled_search_dir: where this stage's report goes. Defaults to `rolled_search/` next to the tiled
                workload (where `ScheduleExplorationStage` also caches the full cost matrix).
            fold_overlap: `"hull"` compares cores by their first-busy-to-last-busy span -- an interval graph, so
                the colouring is optimal, and hull-disjoint implies genuinely disjoint. `"exact"` compares the
                true busy sets, which can fold further when layers interleave, at the cost of a heuristic
                (DSATUR) colouring.
            fold_tolerances: the aggressiveness ladder. A pair conflicts when their overlap exceeds
                `tolerance * min(load)`, so `0.0` merges only genuinely idle cores and `1.0` merges almost
                anything, serializing to save area. Rising tolerance only removes conflicts, so core counts are
                non-increasing along the ladder.
            rolled_core_count_targets: if given, keep for each target the ladder partition closest to it,
                instead of keeping every distinct partition.
            fold_allow_sibling_merge: allow merging cores of the *same* layer. Off by default: those are the
                layer's inter-core slices, merging them is really a tiling decision, and it invalidates the
                cached CMEs, which bake in how many slices run in parallel.
            merged_design_pool: `"members"` picks a merged core's design from the front designs of the cores
                being merged; `"all"` from every design found, turning the fold into a light redesign.
            merged_design_objectives: one variant per objective, each giving merged cores the design that is
                cheapest for the tiles they host on that objective.
            keep_singleton_designs: leave the design of an unmerged core exactly as the source schedule had it.
            max_rolled_variants_per_schedule: cap on folds generated per source schedule, spread over core
                counts rather than clustered.
            max_rolled_evaluations: cap on rolled schedules actually measured.
            fold_source: fold only the schedules that survived on the front (`"pareto"`), or every measured one.
            prune_rolled: skip folds whose lower bound is already dominated by a measured schedule.
        """
        super().__init__(list_of_callables, **kwargs)
        self.schedule_search = schedule_search
        self.space: ScheduleSpace = schedule_search.space
        self.records: list[dict[str, Any]] = schedule_search.records
        self.rolled_search_dir = rolled_search_dir or os.path.join(
            os.path.dirname(self.space.tiled_workload_path), "rolled_search"
        )
        self.fold_overlap = fold_overlap
        self.fold_tolerances = tuple(sorted(set(fold_tolerances)))
        self.rolled_core_count_targets = rolled_core_count_targets
        self.fold_allow_sibling_merge = fold_allow_sibling_merge
        self.merged_design_pool = merged_design_pool
        self.merged_design_objectives = tuple(merged_design_objectives)
        self.keep_singleton_designs = keep_singleton_designs
        self.max_rolled_variants_per_schedule = max_rolled_variants_per_schedule
        self.max_rolled_evaluations = max_rolled_evaluations
        self.fold_source = fold_source
        self.prune_rolled = prune_rolled
        self.search_budget = kwargs.get("search_budget")
        self.fold_log: list[dict[str, Any]] = []

        # Cores of one layer are its inter-core slices: never merged unless explicitly allowed.
        self.forbidden_pairs: set[tuple[int, int]] = set()
        if not self.fold_allow_sibling_merge:
            self.forbidden_pairs = {
                canonical(a, b)
                for core_ids in self.space.core_ids_of_node.values()
                for a, b in combinations(sorted(core_ids), 2)
            }

    # ------------------------------------------------------------------------------------ folding ------------

    def _conflicts(self, occupancy: CoreOccupancy, tolerance: float) -> set[tuple[int, int]]:
        """Which core pairs may not share a physical core at this tolerance.

        A pair conflicts when the time they are busy at once exceeds `tolerance * min(load)` -- normalizing by
        the smaller load so the ladder means the same thing for a big and a small core. Forbidden pairs (a
        layer's own slices) always conflict."""
        conflicts = set(self.forbidden_pairs)
        core_ids = sorted(occupancy.intervals)
        for a, b in combinations(core_ids, 2):
            if (a, b) in conflicts:
                continue
            if self.fold_overlap == "hull":
                mass = interval_overlap(occupancy.hull[a], occupancy.hull[b])
            else:
                mass = union_overlap(occupancy.intervals[a], occupancy.intervals[b])
            if mass > tolerance * min(occupancy.load[a], occupancy.load[b]):
                conflicts.add(canonical(a, b))
        return conflicts

    def _colour(self, occupancy: CoreOccupancy, conflicts: set[tuple[int, int]]) -> dict[int, int]:
        """Colour the unrolled cores so no two conflicting ones share a colour: hull mode's conflicts form an
        interval graph (greedy is optimal), exact mode's do not (DSATUR)."""
        colouring = greedy_interval_colouring if self.fold_overlap == "hull" else dsatur_colouring
        return colouring(self.space.unrolled_core_ids, conflicts, occupancy.hull)

    def _merged_design_key(self, hosted_shape_counts: dict[int, int], candidate_keys: list[str], objective: str) -> str:
        """The cheapest design in `candidate_keys` for the tiles this merged core actually hosts."""
        costs = self.space.costs

        def cost(design_key: str) -> float:
            if objective == "area":
                return costs.area[design_key]
            table = costs.runtime if objective == "latency" else costs.energy
            return sum(count * table[(shape, design_key)] for shape, count in hosted_shape_counts.items())

        feasible = [
            key for key in sorted(candidate_keys) if all((shape, key) in costs.runtime for shape in hosted_shape_counts)
        ]
        if not feasible:
            raise KeyError("no candidate design is cost-modelled for every shape this core hosts")
        return min(feasible, key=cost)

    def _choose_designs(
        self, record: dict[str, Any], colour_of: dict[int, int], objective: str
    ) -> tuple[dict[int, int], dict[int, str], dict[int, dict[str, Any]]]:
        """Assign one design per merged core, and renumber the colours the way `derive_rolled_topology` will."""
        space = self.space
        source_design_of_core = space.design_key_of_core(space.node_core_dicts(record["ranks"]))
        designs = space.distinct_designs()

        # Same first-appearance-over-sorted-ids renumbering `derive_rolled_topology` uses, so the rolled core
        # ids here and the ones in the assembled accelerator agree.
        label_to_rolled: dict[int, int] = {}
        for core_id in sorted(colour_of):
            label_to_rolled.setdefault(colour_of[core_id], len(label_to_rolled))
        rolled_of = {core_id: label_to_rolled[colour_of[core_id]] for core_id in sorted(colour_of)}

        members_of: dict[int, list[int]] = {}
        for core_id, rolled_id in rolled_of.items():
            members_of.setdefault(rolled_id, []).append(core_id)

        design_key_of_core: dict[int, str] = {}
        for rolled_id, members in members_of.items():
            if len(members) == 1 and self.keep_singleton_designs:
                design_key_of_core[rolled_id] = source_design_of_core[members[0]]
                continue
            hosted: dict[int, int] = {}
            for member in members:
                for shape, count in space.shape_counts_of_core[member].items():
                    hosted[shape] = hosted.get(shape, 0) + count
            if self.merged_design_pool == "all":
                candidates = list(designs)
            else:
                candidates = [
                    space.design_key(space.design_class_of_core[member], rank)
                    for member in members
                    for rank in range(len(space.designs_per_class[space.design_class_of_core[member]]))
                ]
            design_key_of_core[rolled_id] = self._merged_design_key(hosted, candidates, objective)

        core_dict_of_core = {rolled_id: designs[design_key] for rolled_id, design_key in design_key_of_core.items()}
        return rolled_of, design_key_of_core, core_dict_of_core

    def _fold_plans(self, record: dict[str, Any]) -> list[FoldPlan]:
        """Every fold worth measuring for one source schedule: walk the tolerance ladder, colour, and give each
        distinct partition one variant per design objective."""
        occupancy: CoreOccupancy | None = record.get("occupancy")
        if occupancy is None:
            return []

        partitions: list[tuple[float, dict[int, int], int]] = []
        seen_partitions: set[frozenset] = set()
        for tolerance in self.fold_tolerances:
            colour_of = self._colour(occupancy, self._conflicts(occupancy, tolerance))
            nb_cores = len(set(colour_of.values()))
            if nb_cores >= len(self.space.unrolled_core_ids):  # nothing merged
                continue
            key = partition_key(colour_of)
            if key in seen_partitions:
                continue
            seen_partitions.add(key)
            partitions.append((tolerance, colour_of, nb_cores))

        if not partitions:
            return []
        partitions = self._select_partitions(partitions)

        plans: list[FoldPlan] = []
        for tolerance, colour_of, nb_cores in partitions:
            for objective in self.merged_design_objectives:
                try:
                    rolled_of, design_key_of_core, core_dict_of_core = self._choose_designs(
                        record, colour_of, objective
                    )
                except KeyError as exc:
                    logger.warning(
                        f"ScheduleRollingStage: skipping a {nb_cores}-core fold of {record['id']} on {objective}: "
                        f"{exc}."
                    )
                    continue
                plans.append(
                    FoldPlan(
                        source_id=record["id"],
                        source_ranks=record["ranks"],
                        merge_of_core=dict(colour_of),
                        rolled_of=rolled_of,
                        design_key_of_core=design_key_of_core,
                        core_dict_of_core=core_dict_of_core,
                        nb_cores=nb_cores,
                        tolerance=tolerance,
                        objective=objective,
                        overlap_mode=self.fold_overlap,
                    )
                )
        return plans

    def _select_partitions(
        self, partitions: list[tuple[float, dict[int, int], int]]
    ) -> list[tuple[float, dict[int, int], int]]:
        """Trim the ladder to the configured budget, spread over core counts rather than clustered at one end,
        so the folds measured actually span the area/latency trade-off."""
        if self.rolled_core_count_targets is not None:
            chosen: dict[int, tuple[float, dict[int, int], int]] = {}
            for target in self.rolled_core_count_targets:
                best = min(partitions, key=lambda p, t=target: (abs(p[2] - t), p[2]))
                chosen.setdefault(best[2], best)
            return sorted(chosen.values(), key=lambda p: p[2])

        budget = max(1, self.max_rolled_variants_per_schedule)
        by_cores = sorted(partitions, key=lambda partition: partition[2])
        if len(by_cores) <= budget:
            return by_cores
        # Keep the extremes -- the most and least aggressive folds bracket the trade-off -- then stride through
        # the rest. Indices, not the partitions themselves, because a partition holds dicts.
        chosen = {0} if budget == 1 else {0, len(by_cores) - 1}
        remaining = [index for index in range(len(by_cores)) if index not in chosen]
        slots = budget - len(chosen)
        if slots > 0 and remaining:
            stride = max(1, len(remaining) // slots)
            chosen.update(remaining[::stride][:slots])
        return [by_cores[index] for index in sorted(chosen)]

    # ------------------------------------------------------------------------------ bounds and measure -------

    def _fold_bounds(self, plan: FoldPlan) -> tuple[float, float, float]:
        """Lower bound on `(latency, energy, area)` for a fold, with area exact.

        The same two arguments the unrolled latency bound rests on still hold after folding: a core runs its
        tiles one at a time (so the busiest folded core's serial load is a floor, and *this* is the term that
        sees the serialization a fold pays for), and a tile never starts before its predecessors finish. Energy
        is the per-tile compute/memory sum, to which the schedule only adds non-negative transfer energy."""
        space = self.space
        runtime_of_tile: dict[tuple[int, int], float] = {}
        energy_of_tile: dict[tuple[int, int], float] = {}
        load_of_core: dict[int, float] = dict.fromkeys(range(plan.nb_cores), 0.0)
        energy = 0.0
        for tile_id, shape_class in space.shape_class_of_tile.items():
            rolled_id = plan.rolled_of[space.unrolled_core_of_tile[tile_id]]
            key = (shape_class, plan.design_key_of_core[rolled_id])
            runtime_of_tile[tile_id] = space.costs.runtime[key]
            energy_of_tile[tile_id] = space.costs.energy[key]
            load_of_core[rolled_id] += runtime_of_tile[tile_id]
            energy += energy_of_tile[tile_id]

        finish = [0.0] * len(space.topo_tile_keys)
        critical_path = 0.0
        for position, tile_id in enumerate(space.topo_tile_keys):
            start = max((finish[p] for p in space.topo_predecessors[position]), default=0.0)
            finish[position] = start + runtime_of_tile[tile_id]
            critical_path = max(critical_path, finish[position])

        latency = max(max(load_of_core.values(), default=0.0), critical_path)
        area = self._fold_area(plan)
        return latency, energy, area

    def _fold_area(self, plan: FoldPlan) -> float:
        """Exact area of a folded accelerator: the sum over its *compute* cores.

        Offchip is excluded, exactly as the unrolled bounds exclude it -- the two numbers share one Pareto
        archive, so they have to be measured the same way."""
        return sum(self.space.costs.area[plan.design_key_of_core[core_id]] for core_id in range(plan.nb_cores))

    def _measure_fold(self, plan: FoldPlan):
        """Build a fold's accelerator, hand the measurement stages a cost LUT assembled from the full matrix, and
        let them schedule it. Returns the schedule and the folded topology."""
        workload, accelerator, cost_lut, topology = self.space.assemble_rolled(
            plan.merge_of_core, plan.core_dict_of_core, plan.design_key_of_core
        )
        return self.space.measure(workload, accelerator, cost_lut), topology

    # ------------------------------------------------------------------------------------- roll --------------

    def _roll(self, archive: ParetoArchive) -> dict[str, Any]:
        """Fold every source schedule and measure what survives the pre-screen, inserting into `archive`."""
        fold_stats = {
            "plans": 0,
            "measured": 0,
            "pruned": 0,
            "infeasible": 0,
            "duplicate": 0,
            "hit_evaluation_budget": False,
        }
        # Snapshot first: `ParetoArchive.add` rewrites `entries` as rolled points evict unrolled ones.
        if self.fold_source == "pareto":
            sources = [record for _point, record in list(archive.entries) if record["kind"] == "unrolled"]
        else:
            sources = [record for record in list(self.records) if record["kind"] == "unrolled"]
        logger.info(
            f"ScheduleRollingStage: folding {len(sources)} source schedule(s) over tolerances "
            f"{list(self.fold_tolerances)} in '{self.fold_overlap}' mode."
        )

        seen: set[tuple] = set()
        for record in sources:
            for plan in self._fold_plans(record):
                fold_stats["plans"] += 1
                status = self._try_plan(plan, archive, fold_stats, seen)
                self.fold_log.append(
                    {
                        "source_id": plan.source_id,
                        "tolerance": plan.tolerance,
                        "objective": plan.objective,
                        "overlap_mode": plan.overlap_mode,
                        "nb_cores": plan.nb_cores,
                        "nb_merges": len(self.space.unrolled_core_ids) - plan.nb_cores,
                        "status": status,
                    }
                )
        logger.info(
            f"ScheduleRollingStage: {fold_stats['plans']} fold(s) considered -- "
            f"{fold_stats['measured']} measured, {fold_stats['pruned']} pruned, "
            f"{fold_stats['duplicate']} duplicate, {fold_stats['infeasible']} infeasible; "
            f"{len(archive.entries)} schedule(s) on the combined front."
        )
        return fold_stats

    def _try_plan(self, plan: FoldPlan, archive: ParetoArchive, fold_stats: dict[str, Any], seen: set[tuple]) -> str:
        """Measure one fold unless it is a duplicate, already dominated by its bound, or over budget; return what
        happened to it (the `status` column of `fold_log.csv`)."""
        if plan.key in seen:
            fold_stats["duplicate"] += 1
            return "duplicate"
        seen.add(plan.key)

        bounds = self._fold_bounds(plan)
        if self.prune_rolled and archive.is_dominated(bounds):
            fold_stats["pruned"] += 1
            return "pruned"
        if fold_stats["measured"] >= self.max_rolled_evaluations:
            fold_stats["hit_evaluation_budget"] = True
            return "budget"

        try:
            scme, topology = self._measure_fold(plan)
        except Exception:
            # A folded core holds several layers' working sets, so it can overflow its memory and leave the
            # scheduler with nothing runnable. That is a property of the fold, not a bug: record and move on.
            logger.warning(
                f"ScheduleRollingStage: fold {plan.id} ({plan.nb_cores} core(s)) is infeasible.", exc_info=True
            )
            fold_stats["infeasible"] += 1
            if self.search_budget is not None:
                self.search_budget.record(math.inf, math.inf, bounds[2], info=f"rolled {plan.id} infeasible")
            return "infeasible"

        fold_stats["measured"] += 1
        point = (float(scme.latency), float(scme.energy), bounds[2])
        if self.search_budget is not None:
            self.search_budget.record(*point, info=f"rolled {plan.id}")
        record = {
            "id": plan.id,
            "kind": "rolled",
            "ranks": plan.source_ranks,
            "source_id": plan.source_id,
            "nb_cores": plan.nb_cores,
            "design_keys": tuple(plan.design_key_of_core[core_id] for core_id in range(plan.nb_cores)),
            "latency": point[0],
            "energy": point[1],
            "area": point[2],
            "latency_bound": bounds[0],
            "energy_bound": bounds[1],
            "fold_tolerance": plan.tolerance,
            "fold_objective": plan.objective,
            "fold_overlap": plan.overlap_mode,
            "merge_of_core": dict(plan.merge_of_core),
            "design_key_of_core": dict(plan.design_key_of_core),
            "core_dict_of_core": dict(plan.core_dict_of_core),
            "members_of_core": dict(topology.members_of_core),
            "hosted_node_groups": dict(topology.hosted_node_groups),
            "node_core_ids": dict(topology.node_core_ids),
            "link_connections": list(topology.link_connections),
            "bus_connection": dict(topology.bus_connection),
        }
        self.records.append(record)
        archive.add(point, record)
        logger.info(
            f"ScheduleRollingStage: measured fold {plan.id}: {len(self.space.unrolled_core_ids)} -> "
            f"{plan.nb_cores} core(s), latency={point[0]:.6g}, energy={point[1]:.6g}, area={point[2]:.6g} "
            f"(front: {len(archive.entries)})."
        )
        return "measured"

    # ------------------------------------------------------------------------------------- run ---------------

    def run(self):
        state = self.schedule_search
        if not self.space.full_matrix:
            logger.warning(
                "ScheduleRollingStage: the cost model only covers the diagonal, so folds are restricted to designs "
                "every hosted shape was cost-modelled on."
            )
        os.makedirs(self.rolled_search_dir, exist_ok=True)
        t_start = time.perf_counter()
        state.stats["rolled"] = self._roll(state.archive)
        state.stats["rolled"]["time_s"] = round(time.perf_counter() - t_start, 2)
        state.report["out_dir"] = self.rolled_search_dir
        state.report["fold_log"] = self.fold_log
        kwargs = self.kwargs.copy()
        kwargs["schedule_search"] = state
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        yield from sub_stage.run()
