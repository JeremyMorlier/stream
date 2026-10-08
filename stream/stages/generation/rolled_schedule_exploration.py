"""Fold the *unrolled* accelerator -- one dedicated core per `(layer, inter-core group)` -- onto fewer physical
cores by reusing cores that are idle later in the schedule, and put the results on the same Pareto front.

`ScheduleExplorationStage` searches which core *design* each layer gets, but never how many cores there are:
`derive_group_dedicated_topology` gives every `(original node, group)` pair its own core, so the core count
grows with the workload and each core sits idle outside its layer's window. This stage asks the complementary
question -- *is a core that has finished its layer free to run a later one?* -- and answers it by measurement:

1. Every schedule the unrolled search measured reports, per core, the windows in which it was actually busy
   (`core_occupancy`, read off the scheduled nodes' `start`/`end`).
2. Cores whose busy windows don't collide can share one physical core. Which cores may share is a graph
   colouring; the colour classes are the folded cores (`_fold_plans`).
3. Each folded core needs a design that runs *all* the layers now on it -- which is why the warm-up here builds
   the full `shape x design` cost-model matrix instead of the diagonal its parent needs (`_warm_up_cme_cache`).
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
  the parent's branch and bound is.
- **Layer fusion leaves little idle time.** In `fused` mode a layer's tiles interleave with the next layer's,
  so the strict "never overlap" graph can admit no merge at all. That is why folding is swept over a
  *tolerance* ladder that also admits overlapping cores, paying serialization for area -- the trade-off the
  combined front is there to expose.
"""

import csv
import logging
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Literal

from zigzag.utils import pickle_deepcopy

from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.core_architecture_exploration import assemble_rolled_accelerator
from stream.stages.generation.schedule_exploration import (
    ParetoArchive,
    ScheduleExplorationStage,
    _UnusedLeafStage,
    core_dict_key,
    format_ranks,
    strip_unpicklable,
)
from stream.stages.stage import StageCallable
from stream.visualization.hardware_graph import topology_from_connectivity
from stream.workload.computation.computation_node import ComputationNode

logger = logging.getLogger(__name__)

# `(stage, reference workload)` a forked warm-up worker reads; set right before the pool forks.
_WARM_UP_STAGE: tuple[Any, Any] | None = None


def _warm_up_column_worker(item: tuple[str, dict[str, Any]]) -> tuple[dict, dict, dict, dict]:
    assert _WARM_UP_STAGE is not None
    stage, workload = _WARM_UP_STAGE
    stage.cme_cache, stage.runtime_cache, stage.energy_cache, stage.area_cache = {}, {}, {}, {}
    stage._warm_up_uniform_design(*item, workload)
    return stage.cme_cache, stage.runtime_cache, stage.energy_cache, stage.area_cache


INTERVAL_T = tuple[int, int]


# ----------------------------------------------------------------------------- interval arithmetic -----------


def coalesce(intervals: list[INTERVAL_T]) -> list[INTERVAL_T]:
    """Sorted, non-overlapping cover of `intervals`. Touching intervals are merged: a core that runs one tile
    straight into the next is busy throughout, and splitting that into two windows would only make the overlap
    arithmetic below more expensive without changing any answer."""
    merged: list[INTERVAL_T] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def interval_overlap(a: INTERVAL_T, b: INTERVAL_T) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def union_overlap(a: list[INTERVAL_T], b: list[INTERVAL_T]) -> int:
    """Total time two coalesced busy sets are busy at once, by a two-pointer merge -- linear, not quadratic."""
    total = 0
    i = j = 0
    while i < len(a) and j < len(b):
        total += interval_overlap(a[i], b[j])
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


@dataclass
class CoreOccupancy:
    """When each compute core was actually busy in one measured schedule."""

    intervals: dict[int, list[INTERVAL_T]] = field(default_factory=dict)
    load: dict[int, int] = field(default_factory=dict)
    hull: dict[int, INTERVAL_T] = field(default_factory=dict)
    node_groups: dict[int, set[tuple[int, int]]] = field(default_factory=dict)
    makespan: int = 0


def core_occupancy(scme: Any, core_ids: list[int]) -> CoreOccupancy:
    """Per-core busy windows of a measured schedule, read off the scheduled tiles.

    `CoalaScheduler` stamps `start`/`end` on the very node objects the `StreamCostModelEvaluation` holds, so
    this needs nothing the cost model doesn't already produce. Cores in `core_ids` that never ran a tile get an
    empty busy set -- they are free to merge with anything.

    Only *compute* occupancy is visible this way; see the module docstring for why that is a floor, not a
    guarantee."""
    offchip_core_id = getattr(scme.accelerator, "offchip_core_id", None)
    raw: dict[int, list[INTERVAL_T]] = {core_id: [] for core_id in core_ids}
    node_groups: dict[int, set[tuple[int, int]]] = {core_id: set() for core_id in core_ids}
    makespan = 0
    for node in scme.workload.node_list:
        if not isinstance(node, ComputationNode):
            continue
        core_id = node.chosen_core_allocation
        if core_id is None or core_id == offchip_core_id or core_id not in raw:
            continue
        start, end = int(node.start), int(node.end)
        if start < 0 or end < start:
            continue
        raw[core_id].append((start, end))
        node_groups[core_id].add((node.id, node.group))
        makespan = max(makespan, end)

    intervals = {core_id: coalesce(windows) for core_id, windows in raw.items()}
    return CoreOccupancy(
        intervals=intervals,
        load={core_id: sum(end - start for start, end in windows) for core_id, windows in intervals.items()},
        hull={
            core_id: (windows[0][0], windows[-1][1]) if windows else (0, 0) for core_id, windows in intervals.items()
        },
        node_groups=node_groups,
        makespan=makespan,
    )


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
        return f"r{format_ranks(self.source_ranks)}-c{self.nb_cores}-{self.objective}-t{self.tolerance:g}"


class RolledScheduleExplorationStage(ScheduleExplorationStage):
    """Runs the inherited unrolled schedule search, then folds each schedule it found onto fewer cores and
    measures the result, returning one Pareto front over both families.

    Artifacts, all under `rolled_search/` (see `_save_results`, `fold_log.csv` and the parent's list):
        - `schedules.csv`: every measured schedule, unrolled and rolled, with the fold parameters that produced
          it and whether it survived on the front.
        - `fold_log.csv`: every fold *considered*, including those skipped as dominated, duplicate or
          infeasible -- the diagnostic for whether the sweep is producing variety or collapsing.
        - `pareto_schedules.pickle`: rolled entries carry the full fold (`merge_of_core`, `members_of_core`,
          `hosted_node_groups`, `design_key_of_core`, `core_dict_of_core`, `node_core_ids`, `link_connections`,
          `bus_connection`), i.e. enough to rebuild the accelerator or draw its topology without re-running.
        - `cost_lut_design_<design_key>.pickle`: the full-matrix warm-up, content-addressed so it is reusable.
    """

    # The full-matrix warm-up (`_warm_up_cme_cache`) runs every design on every tile shape with the complete
    # ZigZag stage, so the core search's lighter cross-tile pass would only recompute the same pairs.
    cross_tile_evaluation = False

    def __init__(  # noqa: PLR0913
        self,
        list_of_callables: list[StageCallable],
        *,
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
        max_cme_matrix_cells: int = 2000,
        **kwargs: Any,
    ):
        """
        Args:
            rolled_search_dir: where this stage's artifacts and its cost-model cache go. Defaults to
                `rolled_search/` next to the tiled workload.
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
            max_cme_matrix_cells: guard on the full `shape x design` warm-up; above it the stage falls back to
                the parent's diagonal warm-up and only folds within what that covers.
        """
        super().__init__(list_of_callables, **kwargs)
        self.rolled_search_dir = rolled_search_dir or os.path.join(
            os.path.dirname(self.tiled_workload_path), "rolled_search"
        )
        self.cme_cache_dir = self.rolled_search_dir
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
        self.max_cme_matrix_cells = max_cme_matrix_cells

        self._full_matrix = False
        self._plan_of_record: dict[str, FoldPlan] = {}
        self.fold_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------------------------- plan --------------

    def _build_plan(self, search: Any, workload: Any) -> None:
        super()._build_plan(search, workload)
        self.unrolled_core_ids: list[int] = sorted(
            {core_id for core_ids in self.core_ids_of_node.values() for core_id in core_ids}
        )
        # Which decision class supplies each unrolled core's design, and how many tiles of each shape class it
        # runs -- the two things choosing a merged core's design needs.
        self.design_class_of_core: dict[int, int] = {
            core_id: self.design_class_of_node[node_id]
            for node_id, core_ids in self.core_ids_of_node.items()
            for core_id in core_ids
        }
        self.shape_counts_of_core: dict[int, dict[int, int]] = {core_id: {} for core_id in self.unrolled_core_ids}
        for tile_id, core_id in self.unrolled_core_of_tile.items():
            counts = self.shape_counts_of_core.setdefault(core_id, {})
            shape_class = self.shape_class_of_tile[tile_id]
            counts[shape_class] = counts.get(shape_class, 0) + 1

        # Cores of one layer are its inter-core slices: never merged unless explicitly allowed.
        self.forbidden_pairs: set[tuple[int, int]] = set()
        if not self.fold_allow_sibling_merge:
            self.forbidden_pairs = {
                canonical(a, b)
                for core_ids in self.core_ids_of_node.values()
                for a, b in combinations(sorted(core_ids), 2)
            }

    # -------------------------------------------------------------------- full shape x design warm-up --------

    def _shapes_needing_design(self, design_class: int) -> list[int]:
        """Every shape in the workload, not just the ones this class's own cores run: a folded core hosts the
        layers of all the cores merged into it, so any design can end up running any shape -- and the
        full-matrix warm-up puts each design on a *uniform* accelerator, cost-modelling it against every shape
        at once. A design that cannot map one of them would abort that accelerator's `update_cost_lut`."""
        return sorted(set(self.shape_class_of_tile.values()))

    def _distinct_designs(self) -> dict[str, dict[str, Any]]:
        """Every core design the fronts hold, deduplicated by what it decodes to."""
        designs: dict[str, dict[str, Any]] = {}
        for class_index in self.decision_classes:
            for rank in range(len(self.designs_per_class[class_index])):
                core_dict = self._design(class_index, rank)["core_dict"]
                designs.setdefault(core_dict_key(core_dict), core_dict)
        return designs

    def _warm_up_cme_cache(self, workload: Any) -> None:
        """Cost-model every `(tile shape, design)` pair, not just the diagonal the parent needs.

        A folded core runs layers that were never meant for its design, so the search needs the whole matrix.
        It is filled one *uniform* accelerator per distinct design -- every node given the same design -- which
        covers that design's entire column at once: all its compute cores are hardware-identical, so ZigZag runs
        once per shape class and `update_cost_lut` copies the rest. That is `S` real ZigZag runs per design
        rather than one per cell."""
        designs = self._distinct_designs()
        nb_shapes = len({shape for shape in self.shape_class_of_tile.values()})
        cells = nb_shapes * len(designs)
        if cells > self.max_cme_matrix_cells:
            logger.warning(
                f"RolledScheduleExplorationStage: the full cost matrix would be {nb_shapes} shape(s) x "
                f"{len(designs)} design(s) = {cells} cell(s), over max_cme_matrix_cells="
                f"{self.max_cme_matrix_cells}. Falling back to the diagonal warm-up; folds will be restricted "
                f"to designs it already covers."
            )
            super()._warm_up_cme_cache(workload)
            return

        self.cme_cache: dict[tuple[int, str], Any] = {}
        self.runtime_cache: dict[tuple[int, str], float] = {}
        self.energy_cache: dict[tuple[int, str], float] = {}
        self.area_cache: dict[str, float] = {}
        os.makedirs(self.cme_cache_dir, exist_ok=True)
        logger.info(
            f"RolledScheduleExplorationStage: warming up the full cost matrix, {nb_shapes} shape(s) x "
            f"{len(designs)} design(s) = {cells} cell(s), with {len(designs)} uniform accelerator(s)."
        )
        if (self.max_workers or 1) > 1 and len(designs) > 1:
            self._warm_up_designs_in_parallel(designs, workload)
        else:
            for design_key, core_dict in designs.items():
                self._warm_up_uniform_design(design_key, core_dict, workload)
        self._full_matrix = True
        self._check_cme_cache_complete()

    def _warm_up_uniform_design(self, design_key: str, core_dict: dict[str, Any], workload: Any) -> None:
        """One column of the cost matrix: every node on `core_dict`, harvested into the caches."""
        node_core_dicts = dict.fromkeys(self.design_class_of_node, core_dict)
        accelerator = self._build_accelerator(node_core_dicts, workload)
        estimation = ZigZagCoreMappingEstimationStage(
            [_UnusedLeafStage],
            workload=workload,
            accelerator=accelerator,
            loma_lpf_limit=self.kwargs["loma_lpf_limit"],
            cost_lut_path=os.path.join(self.cme_cache_dir, f"cost_lut_design_{design_key}.pickle"),
            layer_stacks=self.kwargs["layer_stacks"],
            temporal_mapping_type=self.kwargs["temporal_mapping_type"],
        )
        estimation.update_cost_lut()
        self._harvest_cmes(estimation.cost_lut, workload, accelerator, self._design_key_of_core(node_core_dicts))

    def _warm_up_designs_in_parallel(self, designs: dict[str, dict[str, Any]], workload: Any) -> None:
        """The columns are independent, so fill them on `core_max_workers` forked processes. Each worker starts
        from empty caches and sends back only what it harvested; the parent merges."""
        global _WARM_UP_STAGE  # noqa: PLW0603
        _WARM_UP_STAGE = (self, workload)
        mp_context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=mp_context) as executor:
            for caches in executor.map(_warm_up_column_worker, list(designs.items())):
                for cache, harvested in zip(
                    (self.cme_cache, self.runtime_cache, self.energy_cache, self.area_cache), caches, strict=True
                ):
                    cache.update(harvested)

    def _required_cme_keys(self) -> list[tuple[int, str]]:
        if not self._full_matrix:
            return super()._required_cme_keys()
        shapes = sorted({shape for shape in self.shape_class_of_tile.values()})
        return [(shape, design_key) for shape in shapes for design_key in self._distinct_designs()]

    def _measured_extras(self, scme: Any) -> dict[str, Any]:
        return {"occupancy": core_occupancy(scme, self.unrolled_core_ids)}

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
        colouring = greedy_interval_colouring if self.fold_overlap == "hull" else dsatur_colouring
        return colouring(self.unrolled_core_ids, conflicts, occupancy.hull)

    def _merged_design_key(self, hosted_shape_counts: dict[int, int], candidate_keys: list[str], objective: str) -> str:
        """The cheapest design in `candidate_keys` for the tiles this merged core actually hosts."""

        def cost(design_key: str) -> float:
            if objective == "area":
                return self.area_cache[design_key]
            table = self.runtime_cache if objective == "latency" else self.energy_cache
            return sum(count * table[(shape, design_key)] for shape, count in hosted_shape_counts.items())

        feasible = [
            key
            for key in sorted(candidate_keys)
            if all((shape, key) in self.runtime_cache for shape in hosted_shape_counts)
        ]
        if not feasible:
            raise KeyError("no candidate design is cost-modelled for every shape this core hosts")
        return min(feasible, key=cost)

    def _choose_designs(
        self, record: dict[str, Any], colour_of: dict[int, int], objective: str
    ) -> tuple[dict[int, int], dict[int, str], dict[int, dict[str, Any]]]:
        """Assign one design per merged core, and renumber the colours the way `derive_rolled_topology` will."""
        source_design_of_core = self._design_key_of_core(self._node_core_dicts(record["ranks"]))
        designs = self._distinct_designs()

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
                for shape, count in self.shape_counts_of_core[member].items():
                    hosted[shape] = hosted.get(shape, 0) + count
            if self.merged_design_pool == "all":
                candidates = list(designs)
            else:
                candidates = [
                    self._design_key(self.design_class_of_core[member], rank)
                    for member in members
                    for rank in range(len(self.designs_per_class[self.design_class_of_core[member]]))
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
            if nb_cores >= len(self.unrolled_core_ids):  # nothing merged
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
                        f"RolledScheduleExplorationStage: skipping a {nb_cores}-core fold of {record['id']} "
                        f"on {objective}: {exc}."
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

        The same two arguments the parent's latency bound rests on still hold after folding: a core runs its
        tiles one at a time (so the busiest folded core's serial load is a floor, and *this* is the term that
        sees the serialization a fold pays for), and a tile never starts before its predecessors finish. Energy
        is the per-tile compute/memory sum, to which the schedule only adds non-negative transfer energy."""
        runtime_of_tile: dict[tuple[int, int], float] = {}
        energy_of_tile: dict[tuple[int, int], float] = {}
        load_of_core: dict[int, float] = dict.fromkeys(range(plan.nb_cores), 0.0)
        energy = 0.0
        for tile_id, shape_class in self.shape_class_of_tile.items():
            rolled_id = plan.rolled_of[self.unrolled_core_of_tile[tile_id]]
            key = (shape_class, plan.design_key_of_core[rolled_id])
            runtime_of_tile[tile_id] = self.runtime_cache[key]
            energy_of_tile[tile_id] = self.energy_cache[key]
            load_of_core[rolled_id] += runtime_of_tile[tile_id]
            energy += energy_of_tile[tile_id]

        finish = [0.0] * len(self.topo_tile_keys)
        critical_path = 0.0
        for position, tile_id in enumerate(self.topo_tile_keys):
            start = max((finish[p] for p in self.topo_predecessors[position]), default=0.0)
            finish[position] = start + runtime_of_tile[tile_id]
            critical_path = max(critical_path, finish[position])

        latency = max(max(load_of_core.values(), default=0.0), critical_path)
        area = self._fold_area(plan)
        return latency, energy, area

    def _fold_area(self, plan: FoldPlan) -> float:
        """Exact area of a folded accelerator: the sum over its *compute* cores.

        Offchip is excluded, exactly as the parent's `_bounds` excludes it -- the two numbers share one Pareto
        archive, so they have to be measured the same way."""
        return sum(self.area_cache[plan.design_key_of_core[core_id]] for core_id in range(plan.nb_cores))

    def _measure_fold(self, plan: FoldPlan, save_yaml: bool = False):
        """Build a fold's accelerator, hand the remaining stages a cost LUT assembled from the full matrix, and
        let them schedule it."""
        workload = pickle_deepcopy(self.workload)
        accelerator, topology = assemble_rolled_accelerator(
            workload,
            self.original_workload,
            self.accelerator,
            plan.merge_of_core,
            plan.core_dict_of_core,
            yaml_path=os.path.join(os.path.dirname(self.tiled_workload_path), "accelerator.yaml")
            if save_yaml
            else None,
        )
        cost_lut = self._assemble_cost_lut(workload, accelerator, plan.design_key_of_core)
        return self._measure_accelerator(workload, accelerator, cost_lut), topology

    # ------------------------------------------------------------------------------------- roll --------------

    def _roll(self, archive: ParetoArchive, stats: dict[str, Any]) -> dict[str, Any]:
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
            sources = [record for record in list(self.all_results) if record["kind"] == "unrolled"]
        logger.info(
            f"RolledScheduleExplorationStage: folding {len(sources)} source schedule(s) over tolerances "
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
                        "nb_merges": len(self.unrolled_core_ids) - plan.nb_cores,
                        "status": status,
                    }
                )
        logger.info(
            f"RolledScheduleExplorationStage: {fold_stats['plans']} fold(s) considered -- "
            f"{fold_stats['measured']} measured, {fold_stats['pruned']} pruned, "
            f"{fold_stats['duplicate']} duplicate, {fold_stats['infeasible']} infeasible; "
            f"{len(archive.entries)} schedule(s) on the combined front."
        )
        return fold_stats

    def _try_plan(self, plan: FoldPlan, archive: ParetoArchive, fold_stats: dict[str, Any], seen: set[tuple]) -> str:
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
                f"RolledScheduleExplorationStage: fold {plan.id} ({plan.nb_cores} core(s)) is infeasible.",
                exc_info=True,
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
        self._plan_of_record[plan.id] = plan
        self.all_results.append(record)
        archive.add(point, record)
        logger.info(
            f"RolledScheduleExplorationStage: measured fold {plan.id}: {len(self.unrolled_core_ids)} -> "
            f"{plan.nb_cores} core(s), latency={point[0]:.6g}, energy={point[1]:.6g}, area={point[2]:.6g} "
            f"(front: {len(archive.entries)})."
        )
        return "measured"

    # ------------------------------------------------------------------------------------- run ---------------

    def run(self):
        archive, stats = self._explore()
        stats["rolled"] = self._roll(archive, stats)
        self._report(archive, stats, self.rolled_search_dir)
        best_record = self._best_record(archive)
        best_scme = self._measure_record(best_record, save_yaml=True)
        yield best_scme, self._extra_info(archive, stats)

    def _measure_record(self, record: dict[str, Any], save_yaml: bool = False):
        if record.get("kind") != "rolled":
            return super()._measure_record(record, save_yaml=save_yaml)
        scme, _topology = self._measure_fold(self._plan_of_record[record["id"]], save_yaml=save_yaml)
        return scme

    # ------------------------------------------------------------------------------- artifacts ---------------

    CSV_COLUMNS = (
        *ScheduleExplorationStage.CSV_COLUMNS,
        "fold_tolerance",
        "fold_objective",
        "fold_overlap",
        "source_id",
    )

    def _csv_row(self, index: int, record: dict[str, Any], pareto_ids: set[str]) -> list[Any]:
        return [
            *super()._csv_row(index, record, pareto_ids),
            record.get("fold_tolerance", ""),
            record.get("fold_objective", ""),
            record.get("fold_overlap", ""),
            record.get("source_id", ""),
        ]

    def _topology_of_record(self, record: dict[str, Any]):
        """A folded design has its own hardware graph -- fewer cores, merged traffic gone and new links in its
        place -- so it is drawn from the connectivity the fold produced, not the unrolled one."""
        if record.get("kind") != "rolled":
            return super()._topology_of_record(record)
        design_key_of_core = record["design_key_of_core"]
        return topology_from_connectivity(
            name=f"{self.accelerator.name} [{record['id']}]",
            node_core_ids=record["node_core_ids"],
            offchip_core_id=record["nb_cores"],
            bus_connection=record["bus_connection"],
            link_connections=record["link_connections"],
            design_keys=design_key_of_core,
            areas={core: self.area_cache[key] for core, key in design_key_of_core.items()},
            layer_names=self.layer_names,
            layer_order=self.layer_order,
        )

    def _pareto_payload(self, record: dict[str, Any]) -> dict[str, Any]:
        if record.get("kind") != "rolled":
            return super()._pareto_payload(record)
        # A rolled entry already carries its whole topology; the parent's `node_core_dicts` would describe the
        # unrolled accelerator this was folded from, which is not what it runs on.
        return strip_unpicklable(record)

    def _report(self, archive: ParetoArchive, stats: dict[str, Any], out_dir: str) -> None:
        super()._report(archive, stats, out_dir)
        self._save_fold_log(out_dir)

    def _save_fold_log(self, out_dir: str) -> None:
        columns = ("source_id", "tolerance", "objective", "overlap_mode", "nb_cores", "nb_merges", "status")
        path = os.path.join(out_dir, "fold_log.csv")
        with open(path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(columns)
            for entry in self.fold_log:
                writer.writerow([entry[column] for column in columns])
        logger.info(f"RolledScheduleExplorationStage: saved {len(self.fold_log)} fold attempt(s) to {path}.")
