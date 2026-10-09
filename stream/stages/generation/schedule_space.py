"""The space of whole-accelerator schedules spanned by the per-tile core Pareto fronts, and what the stages that
search it share.

`CoreArchitectureExplorationStage` leaves one Pareto front of core designs per unique tile shape. A *schedule*
picks one design per decision class (the class whose front drives a layer's cores), builds the
group-dedicated accelerator for it and runs the scheduler on it. `ScheduleSpace` holds everything those
stages need that does not depend on which designs are picked:

1. **The topology is design-independent.** `derive_group_dedicated_topology` assigns cores from the workload and
   its tiling alone, so the core count per node, every tile's core pinning and the tile-to-tile links are
   identical in all combinations. All of it is derived once, in `_build_plan`.
2. **Per-(tile, design) cost is separable and reusable.** A tile's `CostModelEvaluation` depends only on its own
   core's design (plus the shared offchip core), never on the rest of the accelerator. So every distinct
   `(tile class, design)` pair is evaluated by ZigZag exactly once, by a warm-up (`warm_up`), and each
   combination's cost LUT is then assembled by copy from `costs`.

On top of that it knows how to assemble and measure both families the searches produce: *unrolled* schedules
(one design rank per decision class) and *rolled* ones (the unrolled cores folded onto fewer physical cores).
"""

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
from zigzag.utils import pickle_deepcopy

from stream.hardware.architecture.core_generator import build_core_from_dict
from stream.stages.estimation.zigzag_core_mapping_estimation import (
    ZigZagCoreMappingEstimationStage,
    lowest_memory_level_violations,
)
from stream.stages.generation.core_architecture_exploration import (
    ParetoCoreSearchResult,
    assemble_custom_accelerator,
    assemble_rolled_accelerator,
)
from stream.stages.generation.schedule_evaluation import (
    TILE_KEY_T,
    CostCaches,
    MeasurementPipeline,
    ParetoArchive,
    _UnusedLeafStage,
    core_dict_key,
    strip_unpicklable,
    tile_key,
)
from stream.stages.generation.tiled_workload_accelerator_generation import derive_group_dedicated_topology
from stream.workload.computation.computation_node import ComputationNode

logger = logging.getLogger(__name__)

# `(space, reference workload, cache dir)` a forked warm-up worker reads; set right before the pool forks.
_WARM_UP_CONTEXT: tuple["ScheduleSpace", Any, str] | None = None


def _warm_up_column_worker(item: tuple[str, dict[str, Any]]) -> CostCaches:
    assert _WARM_UP_CONTEXT is not None
    space, workload, cache_dir = _WARM_UP_CONTEXT
    space.costs = CostCaches(latency_attr=space.costs.latency_attr)
    space.warm_up_uniform_design(*item, workload, cache_dir)
    return space.costs


class ScheduleSpace:
    """The design-independent plan of one tiled workload's group-dedicated accelerator, the cost-model cache over
    its `(tile shape, design)` pairs, and the measurement of any schedule in it. Built once by
    `ScheduleExplorationStage` and handed down the chain inside a `ScheduleSearchState`."""

    def __init__(  # noqa: PLR0913
        self,
        search: ParetoCoreSearchResult,
        *,
        workload: Any,
        original_workload: Any,
        accelerator: Any,
        tiled_workload_path: str,
        pipeline: MeasurementPipeline,
        estimation_kwargs: dict[str, Any],
        latency_attr: str = "latency_total2",
        max_workers: int | None = 1,
    ):
        """
        Args:
            search: the per-tile Pareto fronts. Their design lists are filtered in place by
                `drop_infeasible_designs`.
            workload: the tiled workload every schedule is measured on (copied per measurement).
            original_workload: the untiled workload, which the topology derivation groups tiles by.
            accelerator: the input accelerator; only its name is reused.
            tiled_workload_path: `accelerator.yaml` of a re-measured schedule is written next to it.
            pipeline: the stages that schedule one accelerator.
            estimation_kwargs: `loma_lpf_limit`, `layer_stacks` and `temporal_mapping_type` for the warm-up's
                `ZigZagCoreMappingEstimationStage`.
            latency_attr: the `CostModelEvaluation` attribute read as a tile's runtime.
            max_workers: processes for the full-matrix warm-up (`1` keeps it serial).
        """
        self.workload = workload
        self.original_workload = original_workload
        self.accelerator = accelerator
        self.tiled_workload_path = tiled_workload_path
        self.pipeline = pipeline
        self.estimation_kwargs = estimation_kwargs
        self.max_workers = max_workers
        self.costs = CostCaches(latency_attr=latency_attr)
        self.full_matrix = False
        # One pinned workload copy drives both the plan and the warm-up: the pinning is design-independent, so it
        # only has to be derived once.
        self.reference_workload = pickle_deepcopy(workload)
        self._build_plan(search, self.reference_workload)

    # ------------------------------------------------------------------- design-independent plan (prop. 1) ---

    def _build_plan(self, search: ParetoCoreSearchResult, workload: Any) -> None:
        """Derive everything that does not depend on which designs get picked: which front drives each node's
        cores, how many cores carry each front, every tile's core pinning and cost key, and the topological
        order the critical-path bounds walk."""
        self.designs_per_class = search.designs_per_tile
        self.unique_tiles = search.unique_tiles

        topology = derive_group_dedicated_topology(workload, self.original_workload)
        self.core_ids_of_node: dict[int, list[int]] = topology.node_core_ids
        self.offchip_core_id: int = topology.offchip_core_id
        self.bus_connection: dict[str, Any] = topology.bus_connection
        self.link_connections: list[dict[str, Any]] = topology.link_connections
        # `original_nodes` is already topologically sorted, so it doubles as the left-to-right column order.
        self.layer_names: dict[int, str] = {node.id: node.short_name for node in search.original_nodes}
        self.layer_order: list[int] = [node.id for node in search.original_nodes]
        # Same representative-tile match `build_node_core_dicts` uses, so plan and assembled accelerator agree.
        self.design_class_of_node: dict[int, int] = {
            node.id: self.shape_class_of(search.tiles_by_original[node][0]) for node in search.original_nodes
        }
        self.decision_classes: list[int] = sorted(set(self.design_class_of_node.values()))
        self.depth_of_class = {class_index: depth for depth, class_index in enumerate(self.decision_classes)}
        unused = [i for i in range(len(self.designs_per_class)) if i not in self.depth_of_class]
        if unused:
            logger.info(
                f"ScheduleSpace: {len(unused)} unique tile class(es) never drive a core design (their node is "
                f"represented by another class), so their fronts are not search dimensions."
            )

        # Cores carrying each decision class -> exact area per combination.
        self.cores_per_class = dict.fromkeys(self.decision_classes, 0)
        for node_id, class_index in self.design_class_of_node.items():
            self.cores_per_class[class_index] += len(topology.node_core_ids[node_id])

        # Each tile's cost key: (its own shape class, the class whose design its core carries). The two differ
        # whenever a node's tiles aren't all the same shape -- the core still comes from the node's
        # representative tile, so the CME needed is "this tile, on that other class's design".
        tiles = [tile for tile in workload.node_list if isinstance(tile, ComputationNode)]
        # Each tile's shape class, resolved once: `shape_class_of` is a scan over the unique tiles, and both the
        # cost pairs below and every cost-LUT assembly need it per tile.
        self.shape_class_of_tile: dict[TILE_KEY_T, int] = {tile_key(tile): self.shape_class_of(tile) for tile in tiles}
        self.pair_of_tile: dict[TILE_KEY_T, tuple[int, int]] = {
            tile_key(tile): (self.shape_class_of_tile[tile_key(tile)], self.design_class_of_node[tile.id])
            for tile in tiles
        }
        self.cost_pairs: list[tuple[int, int]] = sorted(set(self.pair_of_tile.values()))
        pair_index = {pair: index for index, pair in enumerate(self.cost_pairs)}
        self.pair_index_of_tile = {key: pair_index[pair] for key, pair in self.pair_of_tile.items()}
        # Which search depth decides each pair's design.
        self.pair_depth = [self.depth_of_class[design_class] for _shape, design_class in self.cost_pairs]

        # Per-core tile counts per cost pair -> busiest-core bound in O(#cores x #pairs), sparsely.
        counts_per_core: dict[int, list[int]] = {}
        self.pair_multiplicity = [0] * len(self.cost_pairs)
        # Which core each tile is pinned to: the rolled search maps a fold's core partition back onto tiles.
        self.unrolled_core_of_tile: dict[TILE_KEY_T, int] = {}
        for tile in tiles:
            index = self.pair_index_of_tile[tile_key(tile)]
            self.pair_multiplicity[index] += 1
            self.unrolled_core_of_tile[tile_key(tile)] = tile.chosen_core_allocation
            core_counts = counts_per_core.setdefault(tile.chosen_core_allocation, [0] * len(self.cost_pairs))
            core_counts[index] += 1
        self.counts_per_core = [
            [(index, count) for index, count in enumerate(counts) if count] for counts in counts_per_core.values()
        ]

        # Topological order + predecessor positions -> critical-path bound as a flat numeric pass.
        order = list(nx.topological_sort(workload))
        position = {tile_key(tile): index for index, tile in enumerate(order)}
        # The same order as tile keys, so a bound can be taken over per-tile costs rather than per-cost-pair
        # ones -- which is what the rolled search needs, since a fold changes which design each tile runs on.
        self.topo_tile_keys: list[TILE_KEY_T] = [tile_key(tile) for tile in order]
        self.topo_pair_index = [self.pair_index_of_tile[tile_key(tile)] for tile in order]
        self.topo_predecessors = [
            [position[tile_key(predecessor)] for predecessor in workload.predecessors(tile)] for tile in order
        ]

        # The unrolled cores, which decision class supplies each one's design, and how many tiles of each shape
        # class it runs -- what choosing a folded core's design needs.
        self.unrolled_core_ids: list[int] = sorted(
            {core_id for core_ids in self.core_ids_of_node.values() for core_id in core_ids}
        )
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
        logger.info(
            f"ScheduleSpace: plan built -- {len(tiles)} tile(s), {len(self.cost_pairs)} cost pair(s), "
            f"{len(self.counts_per_core)} core(s), {len(self.decision_classes)} decision class(es)."
        )

    def shape_class_of(self, tile: ComputationNode) -> int:
        """Index of the unique tile shape `tile` has."""
        return next(i for i, unique in enumerate(self.unique_tiles) if tile.has_same_performance(unique))

    # -------------------------------------------------------------------------- design/cost lookups ----------

    def design(self, class_index: int, rank: int) -> dict[str, Any]:
        """The `rank`-th design on decision class `class_index`'s front."""
        return self.designs_per_class[class_index][rank]

    def design_key(self, class_index: int, rank: int) -> str:
        """Content key of the `rank`-th design on decision class `class_index`'s front."""
        return core_dict_key(self.design(class_index, rank)["core_dict"])

    def node_core_dicts(self, ranks: tuple[int, ...]) -> dict[int, dict[str, Any]]:
        """The core design every original node's cores get under one rank per decision class."""
        return {
            node_id: self.design(class_index, ranks[self.depth_of_class[class_index]])["core_dict"]
            for node_id, class_index in self.design_class_of_node.items()
        }

    def design_key_of_core(self, node_core_dicts: dict[int, dict[str, Any]]) -> dict[int, str]:
        """Which design each core carries, keyed by *core id*. Both the CME harvest and the cost-LUT assembly
        need this rather than a node -> design map: once a core can host tiles of several layers (the rolled
        search) the node no longer identifies the design, but the core always does."""
        return {
            core_id: core_dict_key(node_core_dicts[node_id])
            for node_id, core_ids in self.core_ids_of_node.items()
            for core_id in core_ids
        }

    def distinct_designs(self) -> dict[str, dict[str, Any]]:
        """Every core design the fronts hold, deduplicated by what it decodes to."""
        designs: dict[str, dict[str, Any]] = {}
        for class_index in self.decision_classes:
            for rank in range(len(self.designs_per_class[class_index])):
                core_dict = self.design(class_index, rank)["core_dict"]
                designs.setdefault(core_dict_key(core_dict), core_dict)
        return designs

    # -------------------------------------------------------------------------- infeasible-design filter -----

    def shapes_needing_design(self, design_class: int, all_shapes: bool) -> list[int]:
        """Which tile shape classes will be cost-modelled on `design_class`'s designs. The unrolled search only
        puts a tile on a core its own class supplied, so only those shapes; with `all_shapes` (the full-matrix
        warm-up, where a folded core can host any layer) every shape in the workload."""
        if all_shapes:
            return sorted(set(self.shape_class_of_tile.values()))
        return sorted({shape for shape, design in self.cost_pairs if design == design_class})

    def drop_infeasible_designs(self, all_shapes: bool = False) -> None:
        """Drop front designs that cannot run every tile shape they will be asked to.

        The per-tile core GA only ever guarantees a design maps *its own* tile (see
        `lowest_memory_level_violations`), but the warm-up cost-models each design against every shape
        `shapes_needing_design` lists. A pair ZigZag cannot map raises `NoValidLoopOrderingFoundException`
        inside `update_cost_lut`, which aborts the whole warm-up -- and with it the run -- rather than leaving
        one cell empty, so such designs are removed from the fronts here, before `warm_up`. Ranks are only read
        after this point, so renumbering them is safe."""
        dropped = 0
        for class_index in self.decision_classes:
            shapes = self.shapes_needing_design(class_index, all_shapes)
            kept: list[dict[str, Any]] = []
            for design in self.designs_per_class[class_index]:
                core = build_core_from_dict(design["core_dict"], core_id=0)
                violations = [
                    (shape, violation)
                    for shape in shapes
                    for violation in lowest_memory_level_violations(self.unique_tiles[shape], core)
                ]
                if violations:
                    dropped += 1
                    shape, violation = violations[0]
                    logger.debug(
                        f"ScheduleSpace: dropping design {core_dict_key(design['core_dict'])} of class "
                        f"{class_index} -- it cannot run shape {shape} ({self.unique_tiles[shape]}): {violation}"
                    )
                else:
                    kept.append(design)
            if not kept:
                raise ValueError(
                    f"ScheduleSpace: every design on class {class_index}'s front fails to map at least one of the "
                    f"tile shapes {shapes} it has to run. Re-run the core search (its Pareto fronts may predate "
                    f"this check) or widen `core_param_ranges` so larger innermost memories are reachable."
                )
            self.designs_per_class[class_index] = kept
        if dropped:
            logger.warning(
                f"ScheduleSpace: dropped {dropped} core design(s) that cannot map every tile shape they would "
                f"have to run; front sizes are now {[len(self.designs_per_class[c]) for c in self.decision_classes]}."
            )

    # ---------------------------------------------------------------------- ZigZag warm-up (property 2) ------

    def warm_up(self, cache_dir: str, full_matrix: bool, max_cells: int) -> None:
        """Fill `costs` with every `(tile shape, design)` pair a search will look up.

        Only the "diagonal" (every tile on the designs its own class carries) unless `full_matrix`, which the
        rolled search needs: a folded core runs layers that were never meant for its design. The full matrix
        falls back to the diagonal when it would exceed `max_cells` cells; the rolled search then only folds
        within what the diagonal covers. The warm-up's cost LUTs are written to `cache_dir`, which a later run
        reuses."""
        os.makedirs(cache_dir, exist_ok=True)
        if full_matrix:
            designs = self.distinct_designs()
            nb_shapes = len(set(self.shape_class_of_tile.values()))
            cells = nb_shapes * len(designs)
            if cells <= max_cells:
                self._warm_up_full(designs, nb_shapes, cache_dir)
                return
            logger.warning(
                f"ScheduleSpace: the full cost matrix would be {nb_shapes} shape(s) x {len(designs)} design(s) = "
                f"{cells} cell(s), over max_cme_matrix_cells={max_cells}. Falling back to the diagonal warm-up; "
                f"folds will be restricted to designs it already covers."
            )
        self._warm_up_diagonal(cache_dir)

    def _warm_up_diagonal(self, cache_dir: str) -> None:
        """Evaluate every distinct `(tile class, design)` pair the unrolled search needs with ZigZag exactly once.

        Walking the fronts diagonally -- accelerator k gives every class its rank-k design (clamped to the last
        rank for shorter fronts) -- covers all pairs in `max(len(front))` accelerators, because a tile's CME
        depends only on its own core."""
        workload = self.reference_workload
        nb_ranks = max(len(self.designs_per_class[class_index]) for class_index in self.decision_classes)
        logger.info(
            f"ScheduleSpace: warming up the cost model for {len(self.cost_pairs)} (tile, design) pair(s) with "
            f"{nb_ranks} accelerator(s)."
        )
        for rank in range(nb_ranks):
            ranks = tuple(
                min(rank, len(self.designs_per_class[class_index]) - 1) for class_index in self.decision_classes
            )
            node_core_dicts = self.node_core_dicts(ranks)
            accelerator = self.build_accelerator(node_core_dicts, workload)
            # A fresh LUT per warm-up: sharing one would evict entries through `remove_cores_with_same_id`,
            # since every accelerator reuses the same core ids with different designs.
            cost_lut = self._estimate(workload, accelerator, os.path.join(cache_dir, f"cost_lut_warmup_{rank}.pickle"))
            self.costs.harvest(
                cost_lut, workload, accelerator, self.design_key_of_core(node_core_dicts), self.shape_class_of_tile
            )
        self._check_complete()

    def _warm_up_full(self, designs: dict[str, dict[str, Any]], nb_shapes: int, cache_dir: str) -> None:
        """Cost-model every `(tile shape, design)` pair, one *uniform* accelerator per distinct design -- every
        node given the same design -- which covers that design's entire column at once: all its compute cores
        are hardware-identical, so ZigZag runs once per shape class and `update_cost_lut` copies the rest. That
        is `S` real ZigZag runs per design rather than one per cell."""
        logger.info(
            f"ScheduleSpace: warming up the full cost matrix, {nb_shapes} shape(s) x {len(designs)} design(s) = "
            f"{nb_shapes * len(designs)} cell(s), with {len(designs)} uniform accelerator(s)."
        )
        workload = self.reference_workload
        if (self.max_workers or 1) > 1 and len(designs) > 1:
            self._warm_up_designs_in_parallel(designs, workload, cache_dir)
        else:
            for design_key, core_dict in designs.items():
                self.warm_up_uniform_design(design_key, core_dict, workload, cache_dir)
        self.full_matrix = True
        self._check_complete()

    def warm_up_uniform_design(self, design_key: str, core_dict: dict[str, Any], workload: Any, cache_dir: str) -> None:
        """One column of the cost matrix: every node on `core_dict`, harvested into `costs`."""
        node_core_dicts = dict.fromkeys(self.design_class_of_node, core_dict)
        accelerator = self.build_accelerator(node_core_dicts, workload)
        cost_lut = self._estimate(
            workload, accelerator, os.path.join(cache_dir, f"cost_lut_design_{design_key}.pickle")
        )
        self.costs.harvest(
            cost_lut, workload, accelerator, self.design_key_of_core(node_core_dicts), self.shape_class_of_tile
        )

    def _warm_up_designs_in_parallel(self, designs: dict[str, dict[str, Any]], workload: Any, cache_dir: str) -> None:
        """The columns are independent, so fill them on `max_workers` forked processes. Each worker starts from
        an empty cache and sends back only what it harvested; the parent merges."""
        global _WARM_UP_CONTEXT  # noqa: PLW0603
        _WARM_UP_CONTEXT = (self, workload, cache_dir)
        mp_context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=mp_context) as executor:
            for harvested in executor.map(_warm_up_column_worker, list(designs.items())):
                self.costs.update(harvested)

    def _estimate(self, workload: Any, accelerator: Any, cost_lut_path: str):
        """Run ZigZag on every unique tile of `workload` on `accelerator`, through the same stage the regular
        pipeline uses, and return its cost LUT (loaded from `cost_lut_path` instead when it already exists)."""
        estimation = ZigZagCoreMappingEstimationStage(
            [_UnusedLeafStage],
            workload=workload,
            accelerator=accelerator,
            cost_lut_path=cost_lut_path,
            **self.estimation_kwargs,
        )
        estimation.update_cost_lut()
        return estimation.cost_lut

    def required_cme_keys(self) -> list[tuple[int, str]]:
        """The `(shape class, design key)` cells the searches will look up: the whole matrix after a full
        warm-up, else every tile on the designs its own decision class carries."""
        if self.full_matrix:
            shapes = sorted(set(self.shape_class_of_tile.values()))
            return [(shape, design_key) for shape in shapes for design_key in self.distinct_designs()]
        return [
            (shape_class, self.design_key(design_class, rank))
            for shape_class, design_class in self.cost_pairs
            for rank in range(len(self.designs_per_class[design_class]))
        ]

    def _check_complete(self) -> None:
        missing = [key for key in self.required_cme_keys() if key not in self.costs.runtime]
        if missing:
            raise ValueError(f"ScheduleSpace: warm-up missed {len(missing)} (tile, design) pair(s): {missing[:5]}")
        logger.info(
            f"ScheduleSpace: cached {len(self.costs.cme)} (tile, design) cost model result(s) over "
            f"{len(self.costs.area)} distinct design(s)."
        )

    # ------------------------------------------------------------------------------- assemble and measure ----

    def _yaml_path(self, save_yaml: bool) -> str | None:
        return os.path.join(os.path.dirname(self.tiled_workload_path), "accelerator.yaml") if save_yaml else None

    def build_accelerator(self, node_core_dicts: dict[int, dict[str, Any]], workload: Any, save_yaml: bool = False):
        """Assemble one combination's unrolled accelerator, pinning every tile in `workload` to its core as a side
        effect of `derive_group_dedicated_topology` (identical in every combination -- property 1)."""
        return assemble_custom_accelerator(
            workload,
            self.original_workload,
            self.accelerator,
            node_core_dicts,
            yaml_path=self._yaml_path(save_yaml),
        )

    def assemble_unrolled(self, ranks: tuple[int, ...], save_yaml: bool = False):
        """`(workload, accelerator, cost_lut)` of the unrolled schedule with one design rank per decision
        class, ready to be scheduled."""
        workload = pickle_deepcopy(self.workload)
        node_core_dicts = self.node_core_dicts(ranks)
        accelerator = self.build_accelerator(node_core_dicts, workload, save_yaml=save_yaml)
        cost_lut = self.costs.assemble_cost_lut(
            workload, accelerator, self.design_key_of_core(node_core_dicts), self.shape_class_of_tile
        )
        return workload, accelerator, cost_lut

    def assemble_rolled(
        self,
        merge_of_core: dict[int, Any],
        core_dict_of_core: dict[int, dict[str, Any]],
        design_key_of_core: dict[int, str],
        save_yaml: bool = False,
    ):
        """`(workload, accelerator, cost_lut, topology)` of the unrolled cores folded as `merge_of_core` says,
        each folded core on the design `core_dict_of_core` gives it (both keyed as `assemble_rolled_accelerator`
        documents), with a cost LUT assembled from the full matrix."""
        workload = pickle_deepcopy(self.workload)
        accelerator, topology = assemble_rolled_accelerator(
            workload,
            self.original_workload,
            self.accelerator,
            merge_of_core,
            core_dict_of_core,
            yaml_path=self._yaml_path(save_yaml),
        )
        cost_lut = self.costs.assemble_cost_lut(workload, accelerator, design_key_of_core, self.shape_class_of_tile)
        return workload, accelerator, cost_lut, topology

    def assemble_record(self, record: dict[str, Any], save_yaml: bool = False):
        """`(workload, accelerator, cost_lut)` of a measured record of either family."""
        if record.get("kind") == "rolled":
            workload, accelerator, cost_lut, _topology = self.assemble_rolled(
                record["merge_of_core"], record["core_dict_of_core"], record["design_key_of_core"], save_yaml
            )
            return workload, accelerator, cost_lut
        return self.assemble_unrolled(record["ranks"], save_yaml=save_yaml)

    def measure(self, workload: Any, accelerator: Any, cost_lut: Any):
        """Schedule an assembled accelerator with the measurement pipeline."""
        return self.pipeline.measure(workload, accelerator, cost_lut)

    def payload(self, record: dict[str, Any]) -> dict[str, Any]:
        """One Pareto entry as plain data, carrying whatever is needed to rebuild its accelerator: an unrolled
        entry gets its `node_core_dicts`, a rolled one already carries its whole fold and topology (the unrolled
        `node_core_dicts` would describe the accelerator it was folded from, not the one it runs on)."""
        if record.get("kind") == "rolled":
            return strip_unpicklable(record)
        return {**strip_unpicklable(record), "node_core_dicts": self.node_core_dicts(record["ranks"])}


@dataclass
class ScheduleSearchState:
    """What a schedule search hands to the next stage (as the `schedule_search` kwarg): the space it searched,
    the Pareto archive and every measured record so far, its statistics, and the plain-data context the
    visualization stages need (`report`, which ends up in `extra_info["schedule_report"]`)."""

    space: ScheduleSpace
    archive: ParetoArchive
    records: list[dict[str, Any]]
    stats: dict[str, Any]
    report: dict[str, Any] = field(default_factory=dict)
