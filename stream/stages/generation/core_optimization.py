"""Core optimization: the pool of core designs the schedule searches choose from, covering every case of core
specialization of `docs/source/schedule_optimization_formulation.md` (section 3.1).

- Case 1, dedicated: one Pareto front per tile shape class, the search `CoreArchitectureExplorationStage` already
  runs.
- Case 2, dedicated with several shapes: one front per operator whose tiles span several shape classes.
- Case 3, shared: one front per cluster of similar shape classes, the classes being clustered on their loop sizes.
- Case 4, fused: one front per chain of consecutive operators of a layer stack.

Cases 2 to 4 search one front per *target*, a set of shape classes each weighted by its tile count `n_sigma`, with
the fitness `(sum n_sigma l_sigma, sum n_sigma e_sigma, area)` of `_evaluate_design_on_tiles`. Each target's designs
are added to the front of every class it covers, so the pool reaches the next stage through the same
`core_pareto_fronts` kwarg as the per-tile search, and nothing downstream changes.
"""

import copy
import csv
import functools
import itertools
import logging
import math
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from deap import base
from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.utils import pickle_load, pickle_save

from stream.stages.generation.core_architecture_exploration import (
    CoreArchitectureExplorationStage,
    CoreGeneLayout,
    ParetoCoreSearchResult,
    _evaluate_design_on_tiles,
    _make_core_toolbox,
    _register_dedup_map,
    _round_population,
    _run_nsga2,
    _write_all_evaluated_cores_csv,
    decode_designs,
)
from stream.stages.generation.layer_stacks_generation import STACK_T
from stream.stages.generation.schedule_evaluation import core_dict_key
from stream.stages.stage import StageCallable
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ONNXWorkload

logger = logging.getLogger(__name__)

# Loop dimensions describing a shape class for the clustering of case 3; products are taken of the pairs.
FEATURE_DIMS: tuple[tuple[str, ...], ...] = (("K",), ("C",), ("OX", "OY"), ("FX", "FY"), ("G",))
# A target of fewer classes (or a chain of fewer operators) is case 1.
MIN_TARGET_SIZE = 2


@dataclass
class SpecializationTarget:
    """A set of shape classes one core is specialized for, each weighted by its tile count."""

    case: Literal["operator", "shared", "fused"]
    label: str
    member_classes: list[int]  # indices into `ParetoCoreSearchResult.unique_tiles`
    counts: list[int]  # n_sigma, aligned with `member_classes`

    def tiles_and_counts(self, search: ParetoCoreSearchResult) -> list[tuple[ComputationNode, int]]:
        return [(search.unique_tiles[sigma], n) for sigma, n in zip(self.member_classes, self.counts, strict=True)]


def _class_counts(tiles: list[ComputationNode], tile_to_unique_index: dict[int, int]) -> dict[int, int]:
    """n_sigma of every shape class among `tiles`, in order of first appearance."""
    return dict(Counter(tile_to_unique_index[id(tile)] for tile in tiles))


def operator_targets(search: ParetoCoreSearchResult) -> list[SpecializationTarget]:
    """Case 2: one target per operator whose tiles span several shape classes (boundary or padded tiles)."""
    targets = []
    for node in search.original_nodes:
        counts = _class_counts(search.tiles_by_original[node], search.tile_to_unique_index)
        if len(counts) > 1:
            targets.append(SpecializationTarget("operator", f"op{node.id}", list(counts), list(counts.values())))
    return targets


def class_features(tile: ComputationNode) -> list[float]:
    """`log2` of the loop sizes in `FEATURE_DIMS` (a missing dimension counts as 1) and of the MAC count."""
    sizes = {dim.name: size for dim, size in tile.layer_dim_sizes.items()}
    features = [math.log2(math.prod(sizes.get(name, 1) for name in dims)) for dims in FEATURE_DIMS]
    return [*features, math.log2(math.prod(sizes.values()))]


def ward_clusters(features: list[list[float]], nb_clusters: int) -> list[list[int]]:
    """Agglomerative clustering with Ward's linkage of the standardized `features` into `nb_clusters` clusters.

    Returns the clusters as sorted lists of row indices. Quadratic merges over a handful of shape classes, so no
    need for scipy."""
    points = np.asarray(features, dtype=float)
    std = points.std(axis=0)
    points = (points - points.mean(axis=0)) / np.where(std > 0, std, 1.0)
    clusters = [[i] for i in range(len(points))]
    centroids = [points[i] for i in range(len(points))]
    while len(clusters) > max(nb_clusters, 1):
        # Ward's cost of merging a and b: the increase of the within-cluster sum of squares.
        sizes = [len(cluster) for cluster in clusters]
        _cost, a, b = min(
            (sizes[a] * sizes[b] / (sizes[a] + sizes[b]) * float(np.sum((centroids[a] - centroids[b]) ** 2)), a, b)
            for a, b in itertools.combinations(range(len(clusters)), 2)
        )
        size_a, size_b = len(clusters[a]), len(clusters[b])
        centroids[a] = (size_a * centroids[a] + size_b * centroids[b]) / (size_a + size_b)
        clusters[a] = sorted(clusters[a] + clusters[b])
        del clusters[b], centroids[b]
    return clusters


def shared_targets(search: ParetoCoreSearchResult, nb_clusters: int | None) -> list[SpecializationTarget]:
    """Case 3: one target per cluster of at least two similar shape classes, each weighted by its tile count in
    the whole workload. `nb_clusters` defaults to `ceil(sqrt(|Sigma|))`; 0 disables the case."""
    nb_classes = len(search.unique_tiles)
    if nb_clusters is None:
        nb_clusters = math.ceil(math.sqrt(nb_classes))
    if nb_clusters == 0 or nb_classes < MIN_TARGET_SIZE:
        return []
    counts = Counter(search.tile_to_unique_index.values())
    clusters = ward_clusters([class_features(tile) for tile in search.unique_tiles], nb_clusters)
    return [
        SpecializationTarget("shared", f"cluster{index}", members, [counts[sigma] for sigma in members])
        for index, members in enumerate(cluster for cluster in clusters if len(cluster) >= MIN_TARGET_SIZE)
    ]


def _computation_successors(workload: ONNXWorkload, node: Any) -> list[ComputationNode]:
    """The computation nodes fed by `node`, looking through the non-computation nodes in between."""
    successors, frontier, seen = [], list(workload.successors(node)), set()
    while frontier:
        successor = frontier.pop()
        if id(successor) in seen:
            continue
        seen.add(id(successor))
        if isinstance(successor, ComputationNode):
            successors.append(successor)
        else:
            frontier.extend(workload.successors(successor))
    return successors


def fused_chains(
    original_workload: ONNXWorkload, layer_stacks: list[STACK_T], max_length: int
) -> list[list[ComputationNode]]:
    """Every chain of 2..`max_length` consecutive computation nodes along the DAG that stays in one layer stack."""
    if max_length < MIN_TARGET_SIZE:
        return []
    stack_of = {node_id: index for index, stack in enumerate(layer_stacks) for node_id in stack}
    nodes = [node for node in original_workload.topological_sort() if isinstance(node, ComputationNode)]
    chains: list[list[ComputationNode]] = []

    def extend(chain: list[ComputationNode]) -> None:
        if len(chain) >= MIN_TARGET_SIZE:
            chains.append(chain)
        if len(chain) == max_length:
            return
        for successor in _computation_successors(original_workload, chain[-1]):
            if successor.id in stack_of and stack_of[successor.id] == stack_of.get(chain[0].id):
                extend([*chain, successor])

    for node in nodes:
        extend([node])
    return chains


def fused_targets(
    search: ParetoCoreSearchResult, original_workload: ONNXWorkload, layer_stacks: list[STACK_T], max_length: int
) -> list[SpecializationTarget]:
    """Case 4: one target per chain of `fused_chains`, covering the classes of all its operators."""
    tiles_by_id = {node.id: tiles for node, tiles in search.tiles_by_original.items()}
    targets = []
    for chain in fused_chains(original_workload, layer_stacks, max_length):
        counts = _class_counts([tile for node in chain for tile in tiles_by_id[node.id]], search.tile_to_unique_index)
        label = "stack_" + "_".join(str(node.id) for node in chain)
        targets.append(SpecializationTarget("fused", label, list(counts), list(counts.values())))
    return targets


def dedup_targets(targets: list[SpecializationTarget]) -> list[SpecializationTarget]:
    """Drop single-class targets (case 1 covers them) and every target whose classes and counts repeat an earlier
    one, e.g. a cluster holding exactly the classes of one operator."""
    kept, seen = [], set()
    for target in targets:
        key = tuple(sorted(zip(target.member_classes, target.counts, strict=True)))
        if len(target.member_classes) >= MIN_TARGET_SIZE and key not in seen:
            seen.add(key)
            kept.append(target)
    return kept


def merge_into_fronts(
    search: ParetoCoreSearchResult, target: SpecializationTarget, designs: list[dict[str, Any]]
) -> int:
    """Add `designs` to the front of every class `target` covers, tagged with their origin, skipping designs a
    front already holds. Returns the number of designs added."""
    added = 0
    for sigma in target.member_classes:
        front = search.designs_per_tile[sigma]
        keys = {core_dict_key(design["core_dict"]) for design in front}
        for design in designs:
            if core_dict_key(design["core_dict"]) not in keys:
                front.append({**design, "origin": target.case, "target": target.label})
                added += 1
    return added


class CoreOptimizationStage(CoreArchitectureExplorationStage):
    """Builds the pool of core designs for every case of core specialization, and hands it down as
    `core_pareto_fronts`, like `CoreArchitectureExplorationStage`, which it can replace in any chain.

    It first runs the per-tile search of its parent (case 1), then one NSGA2 per specialization target of cases 2
    to 4, seeded with the case 1 designs of the classes the target covers. The targets share the parent's
    budget settings (`nb_core_ga_generations`, `nb_core_ga_individuals`, convergence and `search_budget`)."""

    def __init__(  # noqa: PLR0913
        self,
        list_of_callables: list[StageCallable],
        *,
        core_operator_targets: bool = True,
        nb_shared_clusters: int | None = None,
        max_fused_stack_length: int = 1,
        shared_core_pareto_points: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            core_operator_targets: search case 2, one front per operator with several shape classes.
            nb_shared_clusters: number of clusters of shape classes for case 3; `ceil(sqrt(|Sigma|))` when None,
                and 0 disables the case.
            max_fused_stack_length: longest chain of consecutive operators searched for case 4; the default 1
                disables the case. Fused designs only pay off once the system model checks memory capacity
                (section 9 of the formulation), so leave it off until then.
            shared_core_pareto_points: designs kept per target front; `core_pareto_points` when None.
        """
        super().__init__(list_of_callables, **kwargs)
        self.core_operator_targets = core_operator_targets
        self.nb_shared_clusters = nb_shared_clusters
        self.max_fused_stack_length = max_fused_stack_length
        self.shared_core_pareto_points = shared_core_pareto_points or self.core_pareto_points
        self.specialized_designs_path = os.path.join(self.core_search_dir, "specialized_designs.pickle")
        self.targets_csv_path = os.path.join(self.core_search_dir, "specialization_targets.csv")

    def enumerate_targets(self, search: ParetoCoreSearchResult) -> list[SpecializationTarget]:
        """The targets of cases 2, 3 and 4, in that order, without duplicates."""
        targets = operator_targets(search) if self.core_operator_targets else []
        targets += shared_targets(search, self.nb_shared_clusters)
        layer_stacks = self.kwargs.get("layer_stacks") or [tuple(node.id for node in search.original_nodes)]
        targets += fused_targets(search, self.original_workload, layer_stacks, self.max_fused_stack_length)
        return dedup_targets(targets)

    def _run_target_ga(  # noqa: PLR0913
        self,
        target_index: int,
        target: SpecializationTarget,
        search: ParetoCoreSearchResult,
        gene_layout: CoreGeneLayout,
        toolbox: base.Toolbox,
        mu: int,
        loma_lpf_limit: int,
        temporal_mapping_type: TemporalMappingType,
    ) -> tuple[list[Any], list[tuple[Any, ...]]]:
        """One target's NSGA2, the weighted counterpart of `_run_tile_ga`, seeded with the case 1 designs of the
        classes it covers (designs from fronts saved before genes were kept give no seed)."""
        # Copied before registering `evaluate`, as concurrent target threads share it (see `_run_tile_ga`).
        toolbox = copy.copy(toolbox)
        toolbox.register(
            "evaluate",
            functools.partial(
                _evaluate_design_on_tiles,
                gene_layout=gene_layout,
                tiles_and_counts=target.tiles_and_counts(search),
                name=target.label,
                loma_lpf_limit=loma_lpf_limit,
                temporal_mapping_type=temporal_mapping_type,
            ),
        )
        seeds = [
            design["genes"]
            for sigma in target.member_classes
            for design in search.designs_per_tile[sigma]
            if design.get("origin", "tile") == "tile" and "genes" in design
        ]
        return _run_nsga2(
            toolbox,
            mu,
            label=f"target {target.label}",
            history_index=-1 - target_index,
            nb_generations=self.nb_core_ga_generations,
            nb_points=self.shared_core_pareto_points,
            search_budget=self.search_budget,
            convergence_patience=self.core_convergence_patience,
            rel_tol=self.core_rel_tol,
            initial=seeds,
        )

    def _load_specialized_designs(self, targets: list[SpecializationTarget]) -> dict[str, list[dict[str, Any]]]:
        """Designs of an earlier run, for the targets whose label and classes are unchanged."""
        if not (self.reuse_pareto_fronts and os.path.exists(self.specialized_designs_path)):
            return {}
        saved = pickle_load(self.specialized_designs_path)
        reused = {
            target.label: saved[target.label]["designs"]
            for target in targets
            if target.label in saved and saved[target.label]["member_classes"] == target.member_classes
        }
        logger.info(f"CoreOptimizationStage: reusing the designs of {len(reused)} target(s) from an earlier run.")
        return reused

    def search_targets(
        self, targets: list[SpecializationTarget], search: ParetoCoreSearchResult
    ) -> dict[str, list[dict[str, Any]]]:
        """Each target's front, decoded, keyed by target label. Targets saved by an earlier run are reused when
        `reuse_pareto_fronts` is set; the others run concurrently on one shared process pool, like the tiles of
        `search_pareto_fronts`."""
        designs = self._load_specialized_designs(targets)
        pending = [(index, target) for index, target in enumerate(targets) if target.label not in designs]
        if pending:
            gene_layout = CoreGeneLayout(self.gene_ranges)
            toolbox = _make_core_toolbox(gene_layout)
            mu = _round_population(self.nb_core_ga_individuals)
            logger.info(f"CoreOptimizationStage: searching {len(pending)} specialization target(s).")
            with ProcessPoolExecutor(
                max_workers=self.max_workers, mp_context=multiprocessing.get_context("fork")
            ) as executor:
                _register_dedup_map(toolbox, executor)
                with ThreadPoolExecutor(max_workers=len(pending)) as target_executor:
                    futures = [
                        target_executor.submit(
                            self._run_target_ga,
                            index,
                            target,
                            search,
                            gene_layout,
                            toolbox,
                            mu,
                            self.kwargs["loma_lpf_limit"],
                            self.kwargs["temporal_mapping_type"],
                        )
                        for index, target in pending
                    ]
                    results = [future.result() for future in futures]
            _write_all_evaluated_cores_csv(
                os.path.join(self.core_search_dir, "all_evaluated_specialized_cores.csv"),
                gene_layout.names,
                [row for _top, history in results for row in history],
            )
            for (_index, target), (top_designs, _history) in zip(pending, results, strict=True):
                finite = [d for d in top_designs if all(math.isfinite(v) for v in d.fitness.values)]
                designs[target.label] = decode_designs(finite, gene_layout, f"core_{target.label}")
        os.makedirs(self.core_search_dir, exist_ok=True)
        pickle_save(  # type: ignore
            {t.label: {"member_classes": t.member_classes, "designs": designs[t.label]} for t in targets},
            self.specialized_designs_path,
        )
        return designs

    def _write_targets_csv(
        self, targets: list[SpecializationTarget], designs: dict[str, list[dict[str, Any]]], added: dict[str, int]
    ) -> None:
        os.makedirs(self.core_search_dir, exist_ok=True)
        with open(self.targets_csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["label", "case", "classes", "counts", "designs_kept", "designs_added_to_fronts"])
            for target in targets:
                writer.writerow(
                    [
                        target.label,
                        target.case,
                        " ".join(map(str, target.member_classes)),
                        " ".join(map(str, target.counts)),
                        len(designs[target.label]),
                        added[target.label],
                    ]
                )
        logger.info(f"CoreOptimizationStage: saved {len(targets)} specialization target(s) to {self.targets_csv_path}.")

    def get_pareto_fronts(self) -> ParetoCoreSearchResult:
        """The per-tile fronts of case 1, extended with the designs of every specialization target."""
        search = super().get_pareto_fronts()
        for front in search.designs_per_tile:
            for design in front:
                design.setdefault("origin", "tile")
        targets = self.enumerate_targets(search)
        designs = self.search_targets(targets, search) if targets else {}
        added = {target.label: merge_into_fronts(search, target, designs[target.label]) for target in targets}
        self._write_targets_csv(targets, designs, added)
        self.specialization_targets = targets

        nb_designs = len({core_dict_key(d["core_dict"]) for front in search.designs_per_tile for d in front})
        max_cells = self.kwargs.get("max_cme_matrix_cells", 2000)
        logger.info(
            f"CoreOptimizationStage: pool of {nb_designs} distinct design(s) over {len(search.unique_tiles)} shape "
            f"class(es), {sum(added.values())} front entr(ies) added by {len(targets)} target(s)."
        )
        if nb_designs * len(search.unique_tiles) > max_cells:
            logger.warning(
                f"CoreOptimizationStage: the full cost matrix has {nb_designs * len(search.unique_tiles)} cells, "
                f"above max_cme_matrix_cells={max_cells}; the warm-up will fall back to the diagonal and cost the "
                f"specialized designs only on the classes they cover."
            )
        return search

    def run(self):
        """Build the pool and hand it down like the parent, with the targets for reporting."""
        kwargs = self.kwargs.copy()
        kwargs["core_pareto_fronts"] = self.get_pareto_fronts()
        kwargs["core_specialization_targets"] = self.specialization_targets
        # Forwarded by reference: `tile_to_unique_index` is keyed by `id()` of this very workload's tiles.
        kwargs["workload"] = self.workload
        kwargs["original_workload"] = self.original_workload
        kwargs["accelerator"] = self.accelerator
        kwargs["tiled_workload_path"] = self.tiled_workload_path
        kwargs["core_max_workers"] = self.max_workers
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        yield from sub_stage.run()
