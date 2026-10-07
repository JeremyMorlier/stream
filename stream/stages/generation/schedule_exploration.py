"""Enumerate whole-accelerator schedules over *every* per-tile Pareto core design, instead of collapsing each
tile's front to a single design.

`CoreArchitectureExplorationStage` ends by picking, per unique tile, the one front design with the best
`sort_key` and evaluating that single accelerator (see its `run`). That throws the multi-objective work away:
the front's other designs might combine into an accelerator that is slower per tile but far cheaper in area, or
one whose slow tiles sit off the critical path and cost nothing at all. This stage keeps every design and
searches the *product* of the fronts -- one choice per decision class -- measuring combinations with the real
scheduler and returning the Pareto front over `(latency, energy, area)` of whole schedules.

Done naively that is `prod(len(front) for front in fronts)` full pipeline runs. Three properties of how a
schedule is constructed make it tractable:

1. **The topology is design-independent.** `derive_group_dedicated_topology` assigns cores from the workload and
   its tiling alone, so the core count per node, every tile's core pinning and the tile-to-tile links are
   identical in all combinations. All of it is derived once, in `_build_plan`.

2. **Per-(tile, design) cost is separable and reusable.** A tile's `CostModelEvaluation` depends only on its own
   core's design (plus the shared offchip core), never on the rest of the accelerator. So every distinct
   `(tile class, design)` pair is evaluated by ZigZag exactly once, by a warm-up over the "diagonal" of the
   fronts (`_warm_up_cme_cache`), and each combination's cost LUT is then assembled by copy. The enumeration
   itself never calls ZigZag -- all that is left per combination is the scheduler run.

3. **Two objectives are exact without scheduling, the third is boundable.**
   - `area` is a plain sum over cores of each design's area -- exact.
   - schedule `energy` is the sum of per-tile CME energies *plus* non-negative transfer/eviction energy, so that
     sum is a lower bound.
   - schedule `latency` is bounded below by both the busiest core's serial load (a core runs its tiles one at a
     time) and the tiled DAG's critical path weighted by per-tile runtimes (transfers only ever delay a node).
   Each bound is monotone non-decreasing in the per-class costs, so filling the unassigned classes with their
   per-objective minima bounds *every* completion of a partial assignment.

Together these give a sound branch and bound: if a (partial or complete) assignment's lower-bound triple is
dominated by an already-*measured* schedule, every completion of it has a true triple at least as large and is
dominated too, so the subtree is skipped without ever running the scheduler. Only measured points are ever used
as dominators, so the front returned is the one exhaustive enumeration would return -- `prune_schedules=False`
runs the exhaustive search for exactly that comparison.
"""

import csv
import hashlib
import logging
import math
import os
from typing import Any

import networkx as nx
from zigzag.utils import pickle_deepcopy, pickle_load, pickle_save

from stream.hardware.architecture.core_generator import build_core_from_dict
from stream.stages.estimation.zigzag_core_mapping_estimation import (
    ZigZagCoreMappingEstimationStage,
    lowest_memory_level_violations,
)
from stream.stages.generation.core_architecture_exploration import (
    CoreArchitectureExplorationStage,
    ParetoCoreSearchResult,
    assemble_custom_accelerator,
)
from stream.stages.generation.tiled_workload_accelerator_generation import (
    derive_group_dedicated_topology,
    get_original_nodes,
    get_tiles_by_original_node,
)
from stream.stages.stage import MainStage, Stage, StageCallable
from stream.utils import CostModelEvaluationLUT
from stream.visualization.hardware_graph import plot_hardware_graph, topology_from_connectivity
from stream.visualization.schedule_tradeoff import plot_schedule_tradeoffs
from stream.visualization.topology_report import design_payload, write_topology_report
from stream.workload.computation.computation_node import ComputationNode

logger = logging.getLogger(__name__)

OBJECTIVES = ("latency", "energy", "area")
TILE_KEY_T = tuple[int, int]


def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """True if `a` is at least as good as `b` on every objective and strictly better on at least one."""
    return all(x <= y for x, y in zip(a, b, strict=True)) and any(x < y for x, y in zip(a, b, strict=True))


def core_dict_key(core_dict: dict[str, Any]) -> str:
    """Content-addressed identity of a core design, ignoring its name: two front designs that decode to the same
    hardware are the same accelerator, and must not be searched (or cost-modelled) twice."""
    without_name = {key: value for key, value in core_dict.items() if key != "name"}
    return hashlib.sha1(repr(_freeze(without_name)).encode()).hexdigest()[:16]


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


# Record fields that exist only while the search runs (big, and not plain data), stripped before anything is
# pickled or written.
TRANSIENT_RECORD_KEYS = ("occupancy",)


def strip_unpicklable(record: dict[str, Any]) -> dict[str, Any]:
    """A record without its transient in-memory fields -- what gets persisted."""
    return {key: value for key, value in record.items() if key not in TRANSIENT_RECORD_KEYS}


def format_ranks(ranks: tuple[int, ...]) -> str:
    """Stable text form of a rank tuple, shared by record ids, `schedules.csv` and topology filenames so a row,
    an id and a figure can be matched by eye."""
    return "-".join(str(rank) for rank in ranks)


def tile_key(tile: ComputationNode) -> TILE_KEY_T:
    """Identity of a tile that survives `pickle_deepcopy`, unlike `id(...)`: every combination is measured on
    its own copy of the workload, but `(id, sub_id)` names the same tile in all of them."""
    return (tile.id, tile.sub_id)


class _UnusedLeafStage(Stage):
    """Placeholder sub-stage for stages this module instantiates only to call one of their methods (e.g.
    `ZigZagCoreMappingEstimationStage.update_cost_lut`). `Stage.__init__` rejects an empty `list_of_callables`
    on a non-leaf stage, and these are never `run()`."""

    def is_leaf(self) -> bool:
        return True

    def run(self):
        yield from ()


class ParetoArchive:
    """Non-dominated set of *measured* schedules, serving as both the search's result and its pruning oracle.

    Only measured points are ever stored, which is what makes the pruning sound: a bound dominated by one of
    these can never beat it, whatever the bounded schedule's true cost turns out to be."""

    def __init__(self) -> None:
        self.entries: list[tuple[tuple[float, float, float], dict[str, Any]]] = []

    def is_dominated(self, point: tuple[float, float, float]) -> bool:
        return any(dominates(stored, point) for stored, _ in self.entries)

    def add(self, point: tuple[float, float, float], payload: dict[str, Any]) -> bool:
        """Insert a measured point, dropping every point it dominates. Returns whether it was kept."""
        if self.is_dominated(point):
            return False
        self.entries = [(p, load) for p, load in self.entries if not dominates(point, p)]
        self.entries.append((point, payload))
        return True

    def sorted_by(self, objective: str) -> list[tuple[tuple[float, float, float], dict[str, Any]]]:
        index = OBJECTIVES.index(objective)
        return sorted(self.entries, key=lambda entry: entry[0][index])


class ScheduleExplorationStage(CoreArchitectureExplorationStage):
    """Searches the per-tile core Pareto fronts (inherited from `CoreArchitectureExplorationStage`), then
    explores the whole-schedule Pareto front over combinations of those designs rather than collapsing each
    front to a single design. See the module docstring for the properties the search exploits.

    Yields the best schedule by `sort_key` as this stage's answer; the full Pareto set -- metrics plus the
    `node_core_dicts` needed to rebuild any of them -- goes to `schedule_search/` and into `extra_info`, since
    holding one `StreamCostModelEvaluation` per Pareto schedule does not scale with workload size.

    Artifacts, all under `schedule_search/`:
        - `schedules.csv`: every *measured* schedule with its ranks, measured `(latency, energy, area)`, the
          bounds it was admitted on, and whether it ended up on the front. Pruned combinations are not listed
          (there can be arbitrarily many); they are counted in the search stats.
        - `pareto_schedules.pickle`: `{stats, decision_classes, front_sizes, pareto, all}`, where each pareto
          entry carries the `node_core_dicts` needed to rebuild that accelerator.
        - `schedule_tradeoffs.png`: the three pairwise projections plus a 3D view (see
          `stream.visualization.schedule_tradeoff`).
        - `cost_lut_warmup_<k>.pickle`: the warm-up cost LUTs, one per diagonal accelerator.
    """

    schedule_search_time_fraction: float = 1.0

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        schedule_search_dir: str | None = None,
        max_schedule_evaluations: int = 1000,
        max_search_nodes: int = 1_000_000,
        max_topology_figures: int = 12,
        prune_schedules: bool = True,
        reuse_pareto_fronts: bool = True,
        schedule_search_time_fraction: float = 1.0,
        **kwargs: Any,
    ):
        """
        Args:
            schedule_search_dir: where this stage's artifacts go. Defaults to `schedule_search/` next to the
                tiled workload.
            max_schedule_evaluations: stop after this many *measured* schedules. Everything measured is still
                exact, but the reported front is then only the front of what was measured before the cut-off.
            max_search_nodes: budget on branch-and-bound tree nodes, a guard for fronts where pruning cannot
                keep up with the combinatorics.
            max_topology_figures: how many Pareto designs get a hardware-topology figure. The unrolled topology
                does not depend on the designs, so past a handful the figures differ only in their labels --
                the selection spans the front rather than drawing all of it.
            prune_schedules: False measures every combination exhaustively -- exponentially slower, and used to
                verify that the pruned search returns the same front.
            reuse_pareto_fronts: reuse `core_search/pareto_fronts.pickle` from an earlier run instead of
                re-running the per-tile core GA, when it matches this workload's tile count.
            schedule_search_time_fraction: only with a `search_budget`: stop measuring unrolled schedules once
                this fraction of the budget has elapsed, leaving the rest to subclasses (the rolled search).
        """
        super().__init__(list_of_callables, **kwargs)
        self.schedule_search_dir = schedule_search_dir or os.path.join(
            os.path.dirname(self.tiled_workload_path), "schedule_search"
        )
        self.max_schedule_evaluations = max_schedule_evaluations
        self.max_search_nodes = max_search_nodes
        self.max_topology_figures = max_topology_figures
        self.prune_schedules = prune_schedules
        self.reuse_pareto_fronts = reuse_pareto_fronts
        self.schedule_search_time_fraction = schedule_search_time_fraction
        self.latency_attr: str = kwargs.get("latency_attr", "latency_total2")
        # Where the warm-up's cost LUTs go. Its own attribute so the rolled stage can keep its (much larger)
        # full-matrix cache under `rolled_search/` instead of mixing it into `schedule_search/`.
        self.cme_cache_dir: str = self.schedule_search_dir

    # --------------------------------------------------------------------- per-tile fronts (inherited GA) ----

    def _load_pareto_fronts(self) -> list[list[dict[str, Any]]] | None:
        fronts_path = os.path.join(os.path.dirname(self.tiled_workload_path), "core_search", "pareto_fronts.pickle")
        if not (self.reuse_pareto_fronts and os.path.exists(fronts_path)):
            return None
        fronts = pickle_load(fronts_path)
        unique_tiles, _ = self._unique_tiles()
        if not isinstance(fronts, list) or len(fronts) != len(unique_tiles):
            logger.warning(
                f"ScheduleExplorationStage: ignoring {fronts_path} -- it holds {len(fronts)} front(s) but this "
                f"workload has {len(unique_tiles)} unique tile(s)."
            )
            return None
        logger.info(f"ScheduleExplorationStage: reusing {sum(map(len, fronts))} core design(s) from {fronts_path}.")
        return fronts

    def _get_search_result(self) -> ParetoCoreSearchResult:
        fronts = self._load_pareto_fronts()
        if fronts is None:
            return self.search_pareto_fronts()
        unique_tiles, tile_to_unique_index = self._unique_tiles()
        original_nodes = get_original_nodes(self.original_workload)
        return ParetoCoreSearchResult(
            designs_per_tile=fronts,
            unique_tiles=unique_tiles,
            tile_to_unique_index=tile_to_unique_index,
            original_nodes=original_nodes,
            tiles_by_original=get_tiles_by_original_node(self.workload, original_nodes),
        )

    # ------------------------------------------------------------------- design-independent plan (prop. 1) ---

    def _build_plan(self, search: ParetoCoreSearchResult, workload: Any) -> None:
        """Derive everything that does not depend on which designs get picked: which front drives each node's
        cores, how many cores carry each front, every tile's core pinning and cost key, and the topological
        order the critical-path bound walks."""
        self.designs_per_class = search.designs_per_tile
        self.unique_tiles = search.unique_tiles

        topology = derive_group_dedicated_topology(workload, self.original_workload)
        # Design-independent (property 1), so the node -> cores map is derived once and reused by every
        # combination's `_design_key_of_core`.
        self.core_ids_of_node: dict[int, list[int]] = topology.node_core_ids
        self.offchip_core_id: int = topology.offchip_core_id
        self.bus_connection: dict[str, Any] = topology.bus_connection
        self.link_connections: list[dict[str, Any]] = topology.link_connections
        # `original_nodes` is already topologically sorted, so it doubles as the left-to-right column order.
        self.layer_names: dict[int, str] = {node.id: node.short_name for node in search.original_nodes}
        self.layer_order: list[int] = [node.id for node in search.original_nodes]
        # Same representative-tile match `_build_node_core_dicts` uses, so plan and assembled accelerator agree.
        self.design_class_of_node: dict[int, int] = {
            node.id: self._shape_class_of(search.tiles_by_original[node][0]) for node in search.original_nodes
        }
        self.decision_classes: list[int] = sorted(set(self.design_class_of_node.values()))
        self.depth_of_class = {class_index: depth for depth, class_index in enumerate(self.decision_classes)}
        unused = [i for i in range(len(self.designs_per_class)) if i not in self.depth_of_class]
        if unused:
            logger.info(
                f"ScheduleExplorationStage: {len(unused)} unique tile class(es) never drive a core design (their "
                f"node is represented by another class), so their fronts are not search dimensions."
            )

        # Cores carrying each decision class -> exact area per combination.
        self.cores_per_class = dict.fromkeys(self.decision_classes, 0)
        for node_id, class_index in self.design_class_of_node.items():
            self.cores_per_class[class_index] += len(topology.node_core_ids[node_id])

        # Each tile's cost key: (its own shape class, the class whose design its core carries). The two differ
        # whenever a node's tiles aren't all the same shape -- the core still comes from the node's
        # representative tile, so the CME needed is "this tile, on that other class's design".
        tiles = [tile for tile in workload.node_list if isinstance(tile, ComputationNode)]
        # Each tile's shape class, resolved once: `_shape_class_of` is a scan over the unique tiles, and both
        # the cost pairs below and every cost-LUT assembly need it per tile.
        self.shape_class_of_tile: dict[TILE_KEY_T, int] = {tile_key(tile): self._shape_class_of(tile) for tile in tiles}
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
        # Which core each tile is pinned to. Design-independent (property 1), and the rolled search needs it to
        # map a fold's core partition back onto tiles.
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
        logger.info(
            f"ScheduleExplorationStage: plan built -- {len(tiles)} tile(s), {len(self.cost_pairs)} cost pair(s), "
            f"{len(self.counts_per_core)} core(s), {len(self.decision_classes)} decision class(es)."
        )

    def _shape_class_of(self, tile: ComputationNode) -> int:
        return next(i for i, unique in enumerate(self.unique_tiles) if tile.has_same_performance(unique))

    # -------------------------------------------------------------------------- infeasible-design filter -----

    def _shapes_needing_design(self, design_class: int) -> list[int]:
        """Which tile shape classes will be cost-modelled on `design_class`'s designs. Only the shapes its own
        cores actually run: the unrolled search never puts a tile on a core its class did not supply."""
        return sorted({shape for shape, design in self.cost_pairs if design == design_class})

    def _drop_infeasible_designs(self) -> None:
        """Drop front designs that cannot run every tile shape they will be asked to.

        The per-tile core GA only ever guarantees a design maps *its own* tile (see
        `lowest_memory_level_violations`), but the schedule search cost-models each design against every shape
        `_shapes_needing_design` lists. A pair ZigZag cannot map raises `NoValidLoopOrderingFoundException`
        inside `update_cost_lut`, which aborts the whole warm-up -- and with it the run -- rather than leaving
        one cell empty, so such designs are removed from the fronts here, before `_warm_up_cme_cache`. Ranks
        are only read after this point (`_build_rank_order` onwards), so renumbering them is safe."""
        dropped = 0
        for class_index in self.decision_classes:
            shapes = self._shapes_needing_design(class_index)
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
                        f"{type(self).__name__}: dropping design {core_dict_key(design['core_dict'])} of class "
                        f"{class_index} -- it cannot run shape {shape} ({self.unique_tiles[shape]}): {violation}"
                    )
                else:
                    kept.append(design)
            if not kept:
                raise ValueError(
                    f"{type(self).__name__}: every design on class {class_index}'s front fails to map at least "
                    f"one of the tile shapes {shapes} it has to run. Re-run the core search (its Pareto fronts "
                    f"may predate this check) or widen `core_param_ranges` so larger innermost memories are "
                    f"reachable."
                )
            self.designs_per_class[class_index] = kept
        if dropped:
            logger.warning(
                f"{type(self).__name__}: dropped {dropped} core design(s) that cannot map every tile shape "
                f"they would have to run; front sizes are now "
                f"{[len(self.designs_per_class[c]) for c in self.decision_classes]}."
            )

    # -------------------------------------------------------------------------- design/cost lookups ----------

    def _design(self, class_index: int, rank: int) -> dict[str, Any]:
        return self.designs_per_class[class_index][rank]

    def _design_key(self, class_index: int, rank: int) -> str:
        return core_dict_key(self._design(class_index, rank)["core_dict"])

    def _node_core_dicts(self, ranks: tuple[int, ...]) -> dict[int, dict[str, Any]]:
        return {
            node_id: self._design(class_index, ranks[self.depth_of_class[class_index]])["core_dict"]
            for node_id, class_index in self.design_class_of_node.items()
        }

    def _design_key_of_core(self, node_core_dicts: dict[int, dict[str, Any]]) -> dict[int, str]:
        """Which design each core carries, keyed by *core id*. Both the CME harvest and the cost-LUT assembly
        need this rather than a node -> design map: once a core can host tiles of several layers (see
        `RolledScheduleExplorationStage`) the node no longer identifies the design, but the core always does."""
        return {
            core_id: core_dict_key(node_core_dicts[node_id])
            for node_id, core_ids in self.core_ids_of_node.items()
            for core_id in core_ids
        }

    # ------------------------------------------------------------------------------ bounds (property 3) ------

    def _index_costs(self) -> None:
        """Tabulate, per cost pair and per rank, the runtime/energy the warm-up measured, plus each pair's and
        class's per-objective minimum -- the minima are what make a partial assignment boundable."""
        self.pair_runtime: list[list[float]] = []
        self.pair_energy: list[list[float]] = []
        for (shape_class, design_class), _depth in zip(self.cost_pairs, self.pair_depth, strict=True):
            runtimes, energies = [], []
            for rank in range(len(self.designs_per_class[design_class])):
                key = (shape_class, self._design_key(design_class, rank))
                runtimes.append(self.runtime_cache[key])
                energies.append(self.energy_cache[key])
            self.pair_runtime.append(runtimes)
            self.pair_energy.append(energies)
        self.pair_min_runtime = [min(runtimes) for runtimes in self.pair_runtime]
        self.pair_min_energy = [min(energies) for energies in self.pair_energy]

        self.class_area: list[list[float]] = [
            [
                self.cores_per_class[class_index] * self.area_cache[self._design_key(class_index, rank)]
                for rank in range(len(self.designs_per_class[class_index]))
            ]
            for class_index in self.decision_classes
        ]
        self.class_min_area = [min(areas) for areas in self.class_area]

    def _bounds(self, ranks: tuple[int, ...], depth: int | None = None) -> tuple[float, float, float]:
        """Lower bound on `(latency, energy, area)` for every completion of `ranks[:depth]`. With `depth` at its
        default (a complete assignment) latency/energy stay bounds and area is exact."""
        depth = len(self.decision_classes) if depth is None else depth
        runtimes = [
            self.pair_runtime[index][ranks[pair_depth]] if pair_depth < depth else self.pair_min_runtime[index]
            for index, pair_depth in enumerate(self.pair_depth)
        ]
        energies = [
            self.pair_energy[index][ranks[pair_depth]] if pair_depth < depth else self.pair_min_energy[index]
            for index, pair_depth in enumerate(self.pair_depth)
        ]
        area = sum(
            self.class_area[depth_][ranks[depth_]] if depth_ < depth else self.class_min_area[depth_]
            for depth_ in range(len(self.decision_classes))
        )
        return self._latency_bound(runtimes), self._energy_bound(energies), area

    def _latency_bound(self, runtimes: list[float]) -> float:
        """max(busiest core's serial load, DAG critical path): a core runs its tiles one at a time, and a tile
        never starts before its predecessors finish -- transfers only add to both."""
        resource_bound = max(
            (sum(count * runtimes[index] for index, count in counts) for counts in self.counts_per_core),
            default=0.0,
        )
        finish = [0.0] * len(self.topo_pair_index)
        critical_path = 0.0
        for position, pair_index in enumerate(self.topo_pair_index):
            predecessors = self.topo_predecessors[position]
            start = max((finish[p] for p in predecessors), default=0.0)
            finish[position] = start + runtimes[pair_index]
            critical_path = max(critical_path, finish[position])
        return max(resource_bound, critical_path)

    def _energy_bound(self, energies: list[float]) -> float:
        """Sum of per-tile compute/memory energy; the schedule only adds non-negative transfer and eviction
        energy on top."""
        return sum(multiplicity * energy for multiplicity, energy in zip(self.pair_multiplicity, energies, strict=True))

    # ---------------------------------------------------------------------- ZigZag warm-up (property 2) ------

    def _build_accelerator(self, node_core_dicts: dict[int, dict[str, Any]], workload: Any, save_yaml: bool = False):
        """Assemble one combination's accelerator, pinning every tile in `workload` to its core as a side effect
        of `derive_group_dedicated_topology` (identical in every combination -- property 1)."""
        return assemble_custom_accelerator(
            workload,
            self.original_workload,
            self.accelerator,
            node_core_dicts,
            yaml_path=os.path.join(os.path.dirname(self.tiled_workload_path), "accelerator.yaml")
            if save_yaml
            else None,
        )

    def _warm_up_cme_cache(self, workload: Any) -> None:
        """Evaluate every distinct `(tile class, design)` pair with ZigZag exactly once.

        Walking the fronts diagonally -- accelerator k gives every class its rank-k design (clamped to the last
        rank for shorter fronts) -- covers all pairs in `max(len(front))` accelerators, because a tile's CME
        depends only on its own core. The enumeration then assembles every combination's LUT from this cache."""
        self.cme_cache: dict[tuple[int, str], Any] = {}
        self.runtime_cache: dict[tuple[int, str], float] = {}
        self.energy_cache: dict[tuple[int, str], float] = {}
        self.area_cache: dict[str, float] = {}

        nb_ranks = max(len(self.designs_per_class[class_index]) for class_index in self.decision_classes)
        os.makedirs(self.cme_cache_dir, exist_ok=True)
        logger.info(
            f"ScheduleExplorationStage: warming up the cost model for {len(self.cost_pairs)} (tile, design) "
            f"pair(s) with {nb_ranks} accelerator(s)."
        )
        for rank in range(nb_ranks):
            ranks = tuple(
                min(rank, len(self.designs_per_class[class_index]) - 1) for class_index in self.decision_classes
            )
            node_core_dicts = self._node_core_dicts(ranks)
            accelerator = self._build_accelerator(node_core_dicts, workload)
            # A fresh LUT per warm-up: sharing one would evict entries through `remove_cores_with_same_id`,
            # since every accelerator reuses the same core ids with different designs.
            estimation = ZigZagCoreMappingEstimationStage(
                [_UnusedLeafStage],
                workload=workload,
                accelerator=accelerator,
                loma_lpf_limit=self.kwargs["loma_lpf_limit"],
                cost_lut_path=os.path.join(self.cme_cache_dir, f"cost_lut_warmup_{rank}.pickle"),
                layer_stacks=self.kwargs["layer_stacks"],
                temporal_mapping_type=self.kwargs["temporal_mapping_type"],
            )
            estimation.update_cost_lut()
            self._harvest_cmes(estimation.cost_lut, workload, accelerator, self._design_key_of_core(node_core_dicts))

        self._check_cme_cache_complete()

    def _required_cme_keys(self) -> list[tuple[int, str]]:
        """The `(shape class, design key)` cells the search will look up. Only the cells the *diagonal* covers:
        every tile on the design its own decision class carries."""
        return [
            (shape_class, self._design_key(design_class, rank))
            for shape_class, design_class in self.cost_pairs
            for rank in range(len(self.designs_per_class[design_class]))
        ]

    def _check_cme_cache_complete(self) -> None:
        missing = [key for key in self._required_cme_keys() if key not in self.runtime_cache]
        if missing:
            raise ValueError(
                f"{type(self).__name__}: warm-up missed {len(missing)} (tile, design) pair(s): {missing[:5]}"
            )
        logger.info(
            f"{type(self).__name__}: cached {len(self.cme_cache)} (tile, design) cost model result(s) over "
            f"{len(self.area_cache)} distinct design(s)."
        )

    def _harvest_cmes(
        self, cost_lut: CostModelEvaluationLUT, workload: Any, accelerator: Any, design_key_of_core: dict[int, str]
    ) -> None:
        """Cache one `(tile shape class, design)` cost model result per pair this accelerator covers. Keyed off
        the *core* each tile sits on rather than its node, so it works unchanged when a core hosts several
        layers (the rolled case) or when every core carries the same design (the full-matrix warm-up)."""
        for tile in workload.node_list:
            if not isinstance(tile, ComputationNode):
                continue
            core_id = tile.chosen_core_allocation
            design_key = design_key_of_core[core_id]
            key = (self.shape_class_of_tile[tile_key(tile)], design_key)
            if key in self.cme_cache:
                continue
            core = accelerator.get_core(core_id)
            equal_node = cost_lut.get_equal_node(tile)
            if equal_node is None or not cost_lut.has_cme(equal_node, core):
                continue
            cme = cost_lut.get_cme(equal_node, core)
            self.cme_cache[key] = cme
            self.runtime_cache[key] = float(getattr(cme, self.latency_attr))
            self.energy_cache[key] = float(cme.energy_total)
            self.area_cache[design_key] = core.get_area()

    def _assemble_cost_lut(
        self, workload: Any, accelerator: Any, design_key_of_core: dict[int, str]
    ) -> CostModelEvaluationLUT:
        """Build one combination's LUT by copying cached CMEs onto its cores -- the same `core_id` fix-up
        `ZigZagCoreMappingEstimationStage` applies when it finds an equal-performance entry, minus the ZigZag
        run.

        The LUT must be shaped the way `update_cost_lut` shapes it: **one key per unique node, holding every
        core that node can run on**. Keying by tile instead would give each tile its own entry with a single
        core, because `ComputationNode.__hash__` includes `sub_id` while `has_same_performance` excludes it --
        and `SetFixedAllocationPerformanceStage` looks a node up via `get_equal_node` (the first same-shape
        tile) and then asks for *its own* core, which would be missing as soon as two same-shape tiles sit on
        different cores, i.e. whenever `inter_core_tiling > 1`.

        Every core in `tile.possible_core_allocation` is filled, not just the chosen one: that is the shape
        `GeneticAlgorithmAllocationStage` reads back via `cost_lut.get_cores(...)`, and the entries are copies
        from `cme_cache`, so the extra fill is cheap."""
        cost_lut = CostModelEvaluationLUT(None, load=False)
        for tile in workload.node_list:
            if not isinstance(tile, ComputationNode):
                continue
            shape_class = self.shape_class_of_tile[tile_key(tile)]
            representative = cost_lut.get_equal_node(tile) or tile
            for core_id in dict.fromkeys([tile.chosen_core_allocation, *tile.possible_core_allocation]):
                design_key = design_key_of_core.get(core_id)
                if design_key is None:  # not a compute core of this accelerator (e.g. a stale offchip id)
                    continue
                core = accelerator.get_core(core_id)
                if cost_lut.has_cme(representative, core):
                    continue
                cme = pickle_deepcopy(self.cme_cache[(shape_class, design_key)])
                cme.layer.core_allocation = [core_id]
                cme.core_id = core_id
                cost_lut.add_cme(representative, core, cme)
        return cost_lut

    def _measured_extras(self, scme: Any) -> dict[str, Any]:
        """Extra fields to attach to a measured record. Empty here; `RolledScheduleExplorationStage` uses it to
        keep each schedule's per-core busy windows without holding on to the `StreamCostModelEvaluation`."""
        return {}

    def _measure_accelerator(self, workload: Any, accelerator: Any, cost_lut: CostModelEvaluationLUT):
        """Run the stages after this one on an already-built accelerator and cost LUT. Split out of `_measure`
        so the rolled search can measure an accelerator it assembled itself rather than one named by `ranks`."""
        kwargs = self.kwargs.copy()
        kwargs["workload"] = workload
        kwargs["accelerator"] = accelerator
        kwargs["cost_lut"] = cost_lut
        scme, _ = MainStage(list(self.list_of_callables), **kwargs).run()[0]
        return scme

    def _measure(self, ranks: tuple[int, ...], save_yaml: bool = False):
        """Measure one combination with the real pipeline: build its accelerator, hand the remaining stages a
        cost LUT assembled from the warm-up cache, and let them produce the schedule."""
        workload = pickle_deepcopy(self.workload)
        node_core_dicts = self._node_core_dicts(ranks)
        accelerator = self._build_accelerator(node_core_dicts, workload, save_yaml=save_yaml)
        cost_lut = self._assemble_cost_lut(workload, accelerator, self._design_key_of_core(node_core_dicts))
        return self._measure_accelerator(workload, accelerator, cost_lut)

    # ------------------------------------------------------------------------------ branch and bound ---------

    def _class_aggregate(self, depth: int, rank: int, objective_index: int) -> float:
        """One class's own contribution on one objective, used only to order and seed the search."""
        if objective_index == OBJECTIVES.index("area"):
            return self.class_area[depth][rank]
        table = self.pair_runtime if objective_index == OBJECTIVES.index("latency") else self.pair_energy
        return sum(
            self.pair_multiplicity[index] * table[index][rank]
            for index, pair_depth in enumerate(self.pair_depth)
            if pair_depth == depth
        )

    def _search(self) -> tuple[ParetoArchive, dict[str, Any]]:
        archive = ParetoArchive()
        stats = {
            "combinations": self.nb_combinations,
            "measured": 0,
            "pruned_subtrees": 0,
            "visited": 0,
            "hit_evaluation_budget": False,
            "hit_node_budget": False,
            "hit_time_budget": False,
        }
        self.all_results: list[dict[str, Any]] = []
        measured_ranks: set[tuple[int, ...]] = set()

        def measure(ranks: tuple[int, ...]) -> None:
            if ranks in measured_ranks:
                return
            if stats["measured"] >= self.max_schedule_evaluations:
                stats["hit_evaluation_budget"] = True
                return
            # Past its time share the unrolled search hands over to subclasses (folding) -- but only once it has
            # a schedule to hand over.
            if self.search_budget is not None and (
                self.search_budget.should_stop()
                or (
                    stats["measured"] > 0
                    and self.search_budget.elapsed()
                    >= self.schedule_search_time_fraction * self.search_budget.max_time_s
                )
            ):
                stats["hit_time_budget"] = True
                return
            measured_ranks.add(ranks)
            bounds = self._bounds(ranks)
            scme = self._measure(ranks)
            point = (float(scme.latency), float(scme.energy), bounds[2])
            stats["measured"] += 1
            if self.search_budget is not None:
                self.search_budget.record(*point, info=f"unrolled {format_ranks(ranks)}")
            record = {
                "id": f"u{format_ranks(ranks)}",
                "kind": "unrolled",
                "ranks": ranks,
                "nb_cores": sum(self.cores_per_class.values()),
                "design_keys": tuple(
                    self._design_key(class_index, ranks[depth])
                    for depth, class_index in enumerate(self.decision_classes)
                ),
                "latency": point[0],
                "energy": point[1],
                "area": point[2],
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
                    for depth in range(len(self.decision_classes))
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
            if stats["hit_time_budget"]:
                return
            if depth == len(self.decision_classes):
                measure(tuple(ranks))
                return
            for rank in self.rank_order[depth]:
                ranks[depth] = rank
                if self.prune_schedules and archive.is_dominated(self._bounds(tuple(ranks), depth + 1)):
                    stats["pruned_subtrees"] += 1
                    continue
                descend(depth + 1, ranks)
            ranks[depth] = self.rank_order[depth][0]

        descend(0, [self.rank_order[depth][0] for depth in range(len(self.decision_classes))])
        return archive, stats

    # -------------------------------------------------------------------------------------------- run --------

    def _prepare_search(self) -> Any:
        """Everything the search needs before the first measurement: the per-tile fronts, the design-independent
        plan, the cost-model warm-up, the cost tables and the per-class rank order. Returns the pinned reference
        workload. Split out of `run` so subclasses can reuse the whole set-up without copying it."""
        search = self._get_search_result()
        os.makedirs(self.schedule_search_dir, exist_ok=True)
        os.makedirs(self.cme_cache_dir, exist_ok=True)

        # One pinned workload copy drives both the plan and the warm-up: the pinning is design-independent, so
        # it only has to be derived once.
        reference_workload = pickle_deepcopy(self.workload)
        self.unique_tiles = search.unique_tiles
        self._build_plan(search, reference_workload)
        self._drop_infeasible_designs()
        self._warm_up_cme_cache(reference_workload)
        self._index_costs()
        self._build_rank_order()
        return reference_workload

    def _build_rank_order(self) -> None:
        """Designs that decode to the same core are the same accelerator: deduplicate each class's ranks before
        taking the product, and order them cheapest-runtime-first so good schedules are measured early."""
        self.rank_order: list[list[int]] = []
        for depth, class_index in enumerate(self.decision_classes):
            unique_ranks: dict[str, int] = {}
            for rank in range(len(self.designs_per_class[class_index])):
                unique_ranks.setdefault(self._design_key(class_index, rank), rank)
            self.rank_order.append(
                sorted(unique_ranks.values(), key=lambda rank, d=depth: self._class_aggregate(d, rank, 0))
            )
        self.nb_combinations = math.prod(len(ranks) for ranks in self.rank_order)
        logger.info(
            f"{type(self).__name__}: {self.nb_combinations} schedule combination(s) over "
            f"{len(self.decision_classes)} decision class(es), front sizes {[len(r) for r in self.rank_order]}."
        )

    def _explore(self) -> tuple[ParetoArchive, dict[str, Any]]:
        self._prepare_search()
        archive, stats = self._search()
        if not archive.entries:
            raise ValueError(f"{type(self).__name__}: no schedule could be measured.")
        logger.info(
            f"{type(self).__name__}: measured {stats['measured']}/{stats['combinations']} combination(s), "
            f"pruned {stats['pruned_subtrees']} subtree(s), {len(archive.entries)} schedule(s) on the front."
        )
        return archive, stats

    def _report(self, archive: ParetoArchive, stats: dict[str, Any], out_dir: str) -> None:
        self._save_results(archive, stats, out_dir)
        plot_schedule_tradeoffs(
            self.all_results,
            [entry[1] for entry in archive.entries],
            os.path.join(out_dir, "schedule_tradeoffs.png"),
        )
        self._plot_pareto_topologies(archive, out_dir)

    # ------------------------------------------------------------------------------ topology figures ---------

    def _topology_of_record(self, record: dict[str, Any]):
        """The hardware graph a Pareto record describes, built from its connectivity rather than by assembling
        an `Accelerator` -- which would construct every `Core` and can invoke CACTI just to draw a picture."""
        design_key_of_core = self._design_key_of_core(self._node_core_dicts(record["ranks"]))
        return topology_from_connectivity(
            name=f"{self.accelerator.name} [{record['id']}]",
            node_core_ids=self.core_ids_of_node,
            offchip_core_id=self.offchip_core_id,
            bus_connection=self.bus_connection,
            link_connections=self.link_connections,
            design_keys=design_key_of_core,
            areas={core: self.area_cache[key] for core, key in design_key_of_core.items()},
            layer_names=self.layer_names,
            layer_order=self.layer_order,
        )

    def _select_for_figures(self, archive: ParetoArchive) -> list[dict[str, Any]]:
        """Which Pareto records get a figure: the per-objective extremes first, then a stride through the front
        so the sample spans it instead of clustering at one end."""
        records = [entry[1] for entry in archive.sorted_by(self.sort_key)]
        if len(records) <= self.max_topology_figures:
            return records
        chosen: dict[str, dict[str, Any]] = {}
        for objective in OBJECTIVES:
            extreme = archive.sorted_by(objective)[0][1]
            chosen[extreme["id"]] = extreme
        remaining = [record for record in records if record["id"] not in chosen]
        slots = max(0, self.max_topology_figures - len(chosen))
        if slots and remaining:
            stride = max(1, len(remaining) // slots)
            for record in remaining[::stride][:slots]:
                chosen[record["id"]] = record
        return [record for record in records if record["id"] in chosen]

    def _plot_pareto_topologies(self, archive: ParetoArchive, out_dir: str) -> None:
        figures_dir = os.path.join(out_dir, "topologies")
        os.makedirs(figures_dir, exist_ok=True)
        selected = self._select_for_figures(archive)
        payloads: list[dict[str, Any]] = []
        for index, record in enumerate(selected):
            try:
                topology = self._topology_of_record(record)
            except Exception:
                logger.warning(f"{type(self).__name__}: could not build a topology for {record['id']}.", exc_info=True)
                continue
            plot_hardware_graph(
                topology,
                os.path.join(figures_dir, f"topology_{index}_{record['id']}.png"),
                title=(
                    f"{record['id']}: {record['nb_cores']} core(s), latency={record['latency']:.4g}, "
                    f"energy={record['energy']:.4g}, area={record['area']:.4g}"
                ),
            )
            payloads.append(design_payload(record, topology))
        logger.info(
            f"{type(self).__name__}: saved {len(selected)} of {len(archive.entries)} pareto topology figure(s) "
            f"to {figures_dir}."
        )
        if payloads:
            write_topology_report(
                payloads,
                os.path.join(out_dir, "topologies.html"),
                json_path=os.path.join(out_dir, "topologies.json"),
                title=f"{self.accelerator.name}: schedule / topology trade-off",
                measured=self.all_results,
            )

    def _best_record(self, archive: ParetoArchive) -> dict[str, Any]:
        best_record = archive.sorted_by(self.sort_key)[0][1]
        logger.info(
            f"{type(self).__name__}: best schedule by {self.sort_key} is {best_record['id']} "
            f"(latency={best_record['latency']:.6g}, energy={best_record['energy']:.6g}, "
            f"area={best_record['area']:.6g}, cores={best_record['nb_cores']})."
        )
        return best_record

    def _measure_record(self, record: dict[str, Any], save_yaml: bool = False):
        """Re-measure a record's accelerator. Dispatches on `record["kind"]` so the rolled subclass can rebuild
        a folded accelerator from the same front."""
        return self._measure(record["ranks"], save_yaml=save_yaml)

    def _extra_info(self, archive: ParetoArchive, stats: dict[str, Any]) -> dict[str, Any]:
        best_record = archive.sorted_by(self.sort_key)[0][1]
        return {
            "schedule_ranks": best_record["ranks"],
            # `schedule_ranks` alone is ambiguous once folded schedules share the front: they inherit their
            # source's ranks but run on a different accelerator.
            "best_schedule": strip_unpicklable(best_record),
            "pareto_schedules": [self._pareto_payload(entry[1]) for entry in archive.entries],
            "all_schedules": [strip_unpicklable(record) for record in self.all_results],
            "search_stats": stats,
        }

    def run(self):
        archive, stats = self._explore()
        self._report(archive, stats, self.schedule_search_dir)
        best_record = self._best_record(archive)
        # Re-measured once so the winning accelerator is the one written to accelerator.yaml.
        best_scme = self._measure_record(best_record, save_yaml=True)
        yield best_scme, self._extra_info(archive, stats)

    # ------------------------------------------------------------------------------------- artifacts ---------

    CSV_COLUMNS = (
        "schedule_index",
        "id",
        "kind",
        "ranks",
        "nb_cores",
        "design_keys",
        "latency",
        "energy",
        "area",
        "latency_bound",
        "energy_bound",
        "pareto",
    )

    def _csv_row(self, index: int, record: dict[str, Any], pareto_ids: set[str]) -> list[Any]:
        return [
            index,
            record["id"],
            record["kind"],
            format_ranks(record["ranks"]),
            record["nb_cores"],
            "-".join(record["design_keys"]),
            record["latency"],
            record["energy"],
            record["area"],
            record["latency_bound"],
            record["energy_bound"],
            record["id"] in pareto_ids,
        ]

    def _pareto_payload(self, record: dict[str, Any]) -> dict[str, Any]:
        """One Pareto entry as plain data, carrying whatever is needed to rebuild its accelerator."""
        return {**strip_unpicklable(record), "node_core_dicts": self._node_core_dicts(record["ranks"])}

    def _save_results(self, archive: ParetoArchive, stats: dict[str, Any], out_dir: str) -> None:
        pareto_ids = {entry[1]["id"] for entry in archive.entries}
        csv_path = os.path.join(out_dir, "schedules.csv")
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self.CSV_COLUMNS)
            for index, record in enumerate(self.all_results):
                writer.writerow(self._csv_row(index, record, pareto_ids))
        summary_path = os.path.join(out_dir, "pareto_schedules.pickle")
        pickle_save(
            {
                "stats": stats,
                "decision_classes": self.decision_classes,
                "front_sizes": [len(ranks) for ranks in self.rank_order],
                "pareto": [self._pareto_payload(record) for _point, record in archive.entries],
                "all": [strip_unpicklable(record) for record in self.all_results],
            },
            summary_path,
        )  # type: ignore
        logger.info(
            f"{type(self).__name__}: saved {len(self.all_results)} measured schedule(s) to {csv_path} and "
            f"{len(archive.entries)} pareto schedule(s) to {summary_path}."
        )
