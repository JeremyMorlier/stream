import array
import csv
import functools
import hashlib
import logging
import multiprocessing
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing.managers import AcquirerProxy
from typing import Any, Literal

from deap import algorithms, base, creator, tools
from zigzag.datatypes import LayerDim
from zigzag.utils import pickle_deepcopy, pickle_load, pickle_save

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.stage import MainStage, Stage, StageCallable
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.mapping import TILING_T, TILING_WILDCARD_T
from stream.workload.onnx_workload import ONNXWorkload

logger = logging.getLogger(__name__)

INTRA_CANDIDATE_T = int
INTER_CANDIDATE_T = "tuple[LayerDim, int | Literal['*']] | None"
MAX_TILES_PER_LAYER = 1000


@dataclass
class TilingGene:
    """One tiling decision point. Two kinds:
    - `kind="intra"`: one gene per (layer, dimension); `dim` is fixed; `candidates[i]` is a factor -- `1`
      (no split) or an integer prime divisor of that dimension's size. Independent per dimension: a layer can
      have any number of intra-tiled dimensions simultaneously (already supported by the framework).
    - `kind="inter"`: one gene per *layer* (not per dimension); `dim` is `None` since the dimension itself is
      part of each candidate; `candidates[i]` is `None` (no inter split for this layer) or `(dim, factor)`
      where `factor` is an integer prime divisor or `"*"` (wildcard, CO-only). Combining the dimension choice
      into a single per-layer gene guarantees at most one inter-tiled dimension per layer -- required because
      the framework (`GroupIdManager`, `stream.utils.get_inter_core_tiling_size`) only ever considers a
      single `inter_core_tiling` entry; giving a layer more than one causes group-id/core-allocation
      mismatches downstream."""

    node_id: int
    kind: Literal["intra", "inter"]
    dim: LayerDim | None
    candidates: "list[INTRA_CANDIDATE_T] | list[INTER_CANDIDATE_T]"


def prime_divisors(n: int) -> list[int]:
    """All divisors of `n` greater than 1, computed from its prime factorization."""
    remainder = n
    prime_factors: dict[int, int] = {}
    d = 2
    while d * d <= remainder:
        while remainder % d == 0:
            prime_factors[d] = prime_factors.get(d, 0) + 1
            remainder //= d
        d += 1
    if remainder > 1:
        prime_factors[remainder] = prime_factors.get(remainder, 0) + 1

    divisors = {1}
    for prime, exponent in prime_factors.items():
        divisors = {d * prime**power for d in divisors for power in range(exponent + 1)}
    return sorted(d for d in divisors if d > 1)


def prime_factors(n: int) -> set[int]:
    factors: set[int] = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def _candidate_key(individual: "array.array | list[int]") -> str:
    """Deterministic, content-addressed identifier for a GA individual: identical gene values always map to
    the same key regardless of object identity, so it stays valid even after DEAP mutates an individual's
    array in place (an attribute stored on the individual itself could go stale after mutation; this can't)."""
    return hashlib.sha1(",".join(str(v) for v in individual).encode()).hexdigest()[:16]


def _decode_individual(
    individual: "array.array | list[int]", genes: list[TilingGene]
) -> dict[int, dict[str, TILING_T | TILING_WILDCARD_T]]:
    """Group each gene's chosen value by the layer it belongs to, building each layer's
    intra_core_tiling/inter_core_tiling lists. An "intra" gene's factor of 1, or an "inter" gene's `None`
    choice, is a no-op: it contributes nothing to either list."""
    per_node: dict[int, dict[str, TILING_T | TILING_WILDCARD_T]] = defaultdict(
        lambda: {"intra_core_tiling": [], "inter_core_tiling": []}
    )
    for gene, value in zip(genes, individual, strict=True):
        choice = gene.candidates[value]
        if gene.kind == "intra":
            if choice == 1:
                continue
            per_node[gene.node_id]["intra_core_tiling"].append((gene.dim, choice))
        else:
            if choice is None:
                continue
            dim, factor = choice
            per_node[gene.node_id]["inter_core_tiling"].append((dim, factor))
    return dict(per_node)


def _candidate_tiling_factor(gene: TilingGene, candidate_index: int) -> int:
    """Return the numeric tile factor contributed by a selected gene candidate.
    A no-op candidate (`1` for intra or `None` for inter) contributes 1. CO wildcard inter-core choices are not
    concrete during tiling exploration, so they do not increase the fixed tile count checked here."""
    choice = gene.candidates[candidate_index]
    if gene.kind == "intra":
        return int(choice)
    if choice is None:
        return 1
    _dim, factor = choice
    return factor if isinstance(factor, int) else 1


def _tile_counts_by_node(individual: "array.array | list[int]", genes: list[TilingGene]) -> dict[int, int]:
    tile_counts: dict[int, int] = defaultdict(lambda: 1)
    for gene, value in zip(genes, individual, strict=True):
        tile_counts[gene.node_id] *= _candidate_tiling_factor(gene, value)
    return dict(tile_counts)


def _intra_tile_counts_by_node(individual: "array.array | list[int]", genes: list[TilingGene]) -> dict[int, int]:
    intra_tile_counts: dict[int, int] = defaultdict(lambda: 1)
    for gene, value in zip(genes, individual, strict=True):
        if gene.kind == "intra":
            intra_tile_counts[gene.node_id] *= _candidate_tiling_factor(gene, value)
    return dict(intra_tile_counts)


def _respects_tile_limit(individual: "array.array | list[int]", genes: list[TilingGene]) -> bool:
    return all(tile_count <= MAX_TILES_PER_LAYER for tile_count in _tile_counts_by_node(individual, genes).values())


def _has_divisible_minimum_intra_core_product(individual: "array.array | list[int]", genes: list[TilingGene]) -> bool:
    intra_tile_counts = _intra_tile_counts_by_node(individual, genes)
    if not intra_tile_counts:
        return True

    min_tile_count = min(intra_tile_counts.values())
    return min_tile_count > 1 and all(tile_count % min_tile_count == 0 for tile_count in intra_tile_counts.values())


def _respects_tiling_constraints(individual: "array.array | list[int]", genes: list[TilingGene]) -> bool:
    return _respects_tile_limit(individual, genes) and _has_divisible_minimum_intra_core_product(individual, genes)


def _repair_tile_limit(individual: "array.array | list[int]", genes: list[TilingGene]):
    """Reduce selected tiling factors until every layer stays within MAX_TILES_PER_LAYER.
    Candidate index 0 is the no-op candidate for both intra and inter genes."""
    while True:
        tile_counts = _tile_counts_by_node(individual, genes)
        invalid_node_ids = [node_id for node_id, tile_count in tile_counts.items() if tile_count > MAX_TILES_PER_LAYER]
        if not invalid_node_ids:
            return individual

        for node_id in invalid_node_ids:
            selected_gene_indices = [
                i
                for i, gene in enumerate(genes)
                if gene.node_id == node_id and _candidate_tiling_factor(gene, individual[i]) > 1
            ]
            if not selected_gene_indices:
                return individual

            gene_index = max(
                selected_gene_indices,
                key=lambda i: _candidate_tiling_factor(genes[i], individual[i]),
            )
            individual[gene_index] = 0


def _common_available_intra_primes(genes: list[TilingGene]) -> set[int]:
    primes_by_node: dict[int, set[int]] = defaultdict(set)
    for gene in genes:
        if gene.kind != "intra":
            continue
        for candidate_index in range(len(gene.candidates)):
            factor = _candidate_tiling_factor(gene, candidate_index)
            primes_by_node[gene.node_id].update(prime_factors(factor))

    if not primes_by_node:
        return set()
    return set.intersection(*primes_by_node.values())


def _repair_divisible_minimum_intra_core_product(individual: "array.array | list[int]", genes: list[TilingGene]):
    """Try to make every layer's intra-core tiling product equal to a shared factor larger than 1.
    Equal products satisfy the stricter condition that the minimum product divides every other product."""
    if _has_divisible_minimum_intra_core_product(individual, genes):
        return individual

    common_primes = _common_available_intra_primes(genes)
    if not common_primes:
        return individual

    target_prime = random.choice(sorted(common_primes))
    node_ids = {gene.node_id for gene in genes if gene.kind == "intra"}
    for node_id in node_ids:
        for gene_index, gene in enumerate(genes):
            if gene.kind == "intra" and gene.node_id == node_id:
                individual[gene_index] = 0

        tile_counts = _tile_counts_by_node(individual, genes)
        valid_replacements: list[tuple[int, int]] = []
        for gene_index, gene in enumerate(genes):
            if gene.kind != "intra" or gene.node_id != node_id:
                continue

            old_factor = _candidate_tiling_factor(gene, individual[gene_index])
            for candidate_index in range(len(gene.candidates)):
                new_factor = _candidate_tiling_factor(gene, candidate_index)
                if new_factor % target_prime != 0:
                    continue
                new_tile_count = tile_counts[node_id] // old_factor * new_factor
                if new_tile_count <= MAX_TILES_PER_LAYER:
                    valid_replacements.append((gene_index, candidate_index))

        if not valid_replacements:
            continue

        gene_index, candidate_index = min(
            valid_replacements,
            key=lambda replacement: _candidate_tiling_factor(genes[replacement[0]], replacement[1]),
        )
        individual[gene_index] = candidate_index
    return individual


def _repair_tiling_constraints(individual: "array.array | list[int]", genes: list[TilingGene]):
    """Best-effort repair for the hard tiling constraints used by the GA."""
    for _ in range(len(genes) + 1):
        if _respects_tiling_constraints(individual, genes):
            return individual
        _repair_tile_limit(individual, genes)
        _repair_divisible_minimum_intra_core_product(individual, genes)
    return individual


def _load_cached_result(tiling_candidates_dir: str, key: str) -> tuple[Any, dict[str, Any]] | None:
    result_path = f"{tiling_candidates_dir}/candidate_{key}/result.pickle"
    if not os.path.exists(result_path):
        return None
    return pickle_load(result_path)


def _write_candidate_results_csv(
    csv_path: str,
    candidate_records: dict[str, tuple[tuple[int, ...], float]],
    genes: list[TilingGene],
    tiling_candidates_dir: str,
) -> None:
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["candidate_index", "tiling_config", "fitness", "latency", "energy", "status"])
        for key, (individual, fitness) in candidate_records.items():
            cached = _load_cached_result(tiling_candidates_dir, key)
            tiling_config = _decode_individual(list(individual), genes)
            if cached is None:
                status = (
                    "invalid_constraints" if not _respects_tiling_constraints(list(individual), genes) else "failed"
                )
                writer.writerow([key, tiling_config, fitness, "", "", status])
                continue

            scme, extra_info = cached
            writer.writerow(
                [
                    key,
                    extra_info.get("tiling_config", tiling_config),
                    fitness,
                    scme.latency,
                    scme.energy,
                    "ok",
                ]
            )
    logger.info(f"TilingExplorationStage: saved all candidate results to {csv_path}.")


def _evaluate_tiling_individual(  # noqa: PLR0913
    individual: "array.array | list[int]",
    genes: list[TilingGene],
    workload: ONNXWorkload,
    list_of_callables: list[StageCallable],
    kwargs_template: dict[str, Any],
    tiling_candidates_dir: str,
    scme_path: str,
    sort_key: str,
    scme_lock: "AcquirerProxy | nullcontext[None]",
) -> tuple[float]:
    """Runs in a worker process: decodes one GA individual into per-layer tiling, evaluates it through the
    full downstream pipeline, and atomically updates the shared best-so-far `scme_path` under `scme_lock` --
    the same interruption-safety mechanism the previous exhaustive grid search used. Module-level (rather than
    a method) so it can be pickled and sent to worker processes. Results are cached on disk keyed by a hash of
    the individual's gene values, so re-evaluating an identical individual (common across GA generations via
    elitism/crossover) is free."""
    key = _candidate_key(individual)
    candidate_dir = f"{tiling_candidates_dir}/candidate_{key}"
    result_path = f"{candidate_dir}/result.pickle"

    if not _respects_tiling_constraints(individual, genes):
        logger.info(
            f"TilingExplorationStage: candidate {key} violates tiling constraints "
            f"(max {MAX_TILES_PER_LAYER} tiles/layer and minimum intra-core product divides all others), "
            "penalizing it."
        )
        return (float("inf"),)

    if os.path.exists(result_path):
        cached_scme, _cached_extra_info = pickle_load(result_path)
        return (getattr(cached_scme, sort_key),)

    per_node = _decode_individual(individual, genes)
    candidate_workload = pickle_deepcopy(workload)
    for node in candidate_workload.node_list:
        if isinstance(node, ComputationNode) and node.id in per_node:
            node.intra_core_tiling = per_node[node.id]["intra_core_tiling"]
            node.inter_core_tiling = per_node[node.id]["inter_core_tiling"]

    kwargs = kwargs_template.copy()
    kwargs["workload"] = candidate_workload
    os.makedirs(candidate_dir, exist_ok=True)
    kwargs["tiled_workload_path"] = f"{candidate_dir}/tiled_workload.pickle"
    kwargs["cost_lut_path"] = f"{candidate_dir}/cost_lut.pickle"
    if "allocations_path" in kwargs_template:  # CO-only, present when allocation_strategy == "co"
        kwargs["allocations_path"] = f"{candidate_dir}/waco/"
        os.makedirs(kwargs["allocations_path"], exist_ok=True)
        kwargs["tiled_workload_post_co_path"] = f"{candidate_dir}/tiled_workload_post_co.pickle"
        kwargs["cost_lut_post_co_path"] = f"{candidate_dir}/cost_lut_post_co.pickle"

    try:
        sub_mainstage = MainStage(list_of_callables, **kwargs)
        # The pipeline's real result is always the first yielded answer (see e.g. TiledWorkloadGenerationStage,
        # which yields a trailing `(None, None)` after forwarding the chain).
        scme = sub_mainstage.run()[0][0]
    except Exception:
        logger.exception(f"TilingExplorationStage: candidate {key} failed, penalizing it.")
        return (float("inf"),)

    extra_info = {"tiling_config": per_node, "candidate_index": key}
    pickle_save((scme, extra_info), result_path)  # type: ignore

    with scme_lock:
        current_best = pickle_load(scme_path) if os.path.exists(scme_path) else None
        if current_best is None or getattr(scme, sort_key) < getattr(current_best, sort_key):
            pickle_save(scme, scme_path)  # type: ignore

    return (getattr(scme, sort_key),)


class TilingExplorationStage(Stage):
    """
    Searches per-layer, per-dimension intra-/inter-core tiling assignments on a fixed hardware and workload
    using a genetic algorithm. Every dimension of every layer (excluding batch `B` and group `G` dims) gets
    its own independent intra-core-tiling gene (factor of 1, i.e. no split, or one of that dimension's prime
    divisors -- see `prime_divisors`); each layer additionally gets a single combined inter-core-tiling gene
    choosing at most one of its dimensions to split across cores (see `TilingGene` for why inter-core tiling
    must stay single-dimension). Re-runs the remaining pipeline (tiling generation onward) once per evaluated
    individual.
    """

    LARGE_SEARCH_SPACE_WARNING_THRESHOLD = 1000

    def __init__(  # noqa: PLR0913
        self,
        list_of_callables: list[StageCallable],
        *,
        accelerator: Accelerator,
        workload: ONNXWorkload,
        layer_stacks: list[tuple[int, ...]],
        tiling_candidates_dir: str,
        scme_path: str,
        sort_key: str = "latency",
        max_workers: int | None = None,
        nb_tiling_ga_generations: int = 4,
        nb_tiling_ga_individuals: int = 8,
        candidate_results_csv_path: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.workload = workload
        self.layer_stacks = layer_stacks
        self.tiling_candidates_dir = tiling_candidates_dir
        self.scme_path = scme_path
        self.sort_key = sort_key
        self.max_workers = max_workers
        self.nb_tiling_ga_generations = nb_tiling_ga_generations
        self.nb_tiling_ga_individuals = nb_tiling_ga_individuals
        self.candidate_results_csv_path = candidate_results_csv_path or os.path.join(
            tiling_candidates_dir, "candidate_results.csv"
        )

    def run(self):  # noqa: PLR0915
        # CO resolves inter-core tiling itself (via a wildcard) to match whatever core split it settles on, and
        # asserts an exact match against the mapping's core_allocation length for any non-wildcard value — so a
        # swept fixed inter-core factor is only safe under GA. See ConstraintOptimizationAllocationStage.
        is_co_pipeline = "allocations_path" in self.kwargs
        genes = self._build_genes(sweep_inter_core_tiling=not is_co_pipeline)

        kwargs_template = self.kwargs.copy()
        kwargs_template["accelerator"] = self.accelerator
        kwargs_template["layer_stacks"] = self.layer_stacks

        if not genes:
            logger.info("TilingExplorationStage: no tiling decision points found; running the pipeline unchanged.")
            empty_individual: list[int] = []
            _evaluate_tiling_individual(
                empty_individual,
                genes,
                self.workload,
                self.list_of_callables,
                kwargs_template,
                self.tiling_candidates_dir,
                self.scme_path,
                self.sort_key,
                nullcontext(),
            )
            cached = _load_cached_result(self.tiling_candidates_dir, _candidate_key(empty_individual))
            fitness = float("inf") if cached is None else getattr(cached[0], self.sort_key)
            _write_candidate_results_csv(
                self.candidate_results_csv_path,
                {_candidate_key(empty_individual): (tuple(empty_individual), fitness)},
                genes,
                self.tiling_candidates_dir,
            )
            if cached is not None:
                yield cached
            return

        logger.info(f"TilingExplorationStage: {len(genes)} tiling decision point(s) (genes) built.")
        logger.info(
            f"TilingExplorationStage: limiting tiled workload generation to {MAX_TILES_PER_LAYER} tiles/layer "
            "and requiring the minimum intra-core tiling product to divide every other layer's product."
        )
        total_evaluations = self.nb_tiling_ga_individuals * self.nb_tiling_ga_generations
        if total_evaluations > self.LARGE_SEARCH_SPACE_WARNING_THRESHOLD:
            logger.warning(
                f"Up to {total_evaluations} full-pipeline evaluations planned across the tiling-search GA "
                f"({self.nb_tiling_ga_individuals} individuals x {self.nb_tiling_ga_generations} generations) "
                "-- each reruns full cost estimation and allocation search."
            )

        if not hasattr(creator, "TilingFitnessMin"):
            creator.create("TilingFitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "TilingIndividual"):
            creator.create("TilingIndividual", array.array, typecode="i", fitness=creator.TilingFitnessMin)

        def _random_individual():
            individual = creator.TilingIndividual(random.randrange(len(gene.candidates)) for gene in genes)
            return _repair_tiling_constraints(individual, genes)

        def _mate(individual_1, individual_2):
            if len(individual_1) < 2:
                return individual_1, individual_2
            tools.cxTwoPoint(individual_1, individual_2)
            _repair_tiling_constraints(individual_1, genes)
            _repair_tiling_constraints(individual_2, genes)
            return individual_1, individual_2

        def _mutate(individual):
            for i, gene in enumerate(genes):
                if random.random() < 1 / len(individual):
                    valid_replacements = [v for v in range(len(gene.candidates)) if v != individual[i]]
                    if valid_replacements:
                        individual[i] = random.choice(valid_replacements)
            _repair_tiling_constraints(individual, genes)
            return (individual,)

        toolbox = base.Toolbox()
        toolbox.register("individual", _random_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", _mate)
        toolbox.register("mutate", _mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Force the "fork" start method rather than relying on the platform default: "spawn"/"forkserver"
        # re-import this process' __main__ module in each worker, which requires every caller of optimize_tiling
        # to guard its top-level code with `if __name__ == "__main__":` — "fork" instead clones this already-
        # running process, so it has no such requirement on the caller.
        mp_context = multiprocessing.get_context("fork")
        # A Manager lock (rather than a plain multiprocessing.Lock) so it can be safely pickled and sent to
        # worker processes via ProcessPoolExecutor, guarding concurrent writes to the shared `scme_path` pickle.
        with mp_context.Manager() as manager:
            scme_lock = manager.Lock()
            with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=mp_context) as executor:
                candidate_records: dict[str, tuple[tuple[int, ...], float]] = {}
                toolbox.register(
                    "evaluate",
                    functools.partial(
                        _evaluate_tiling_individual,
                        genes=genes,
                        workload=self.workload,
                        list_of_callables=self.list_of_callables,
                        kwargs_template=kwargs_template,
                        tiling_candidates_dir=self.tiling_candidates_dir,
                        scme_path=self.scme_path,
                        sort_key=self.sort_key,
                        scme_lock=scme_lock,
                    ),
                )

                def _dedup_map(func, individuals):
                    """DEAP's eaSimple calls toolbox.map(toolbox.evaluate, invalid_ind) once per generation
                    (and once for the initial population) -- registering this wrapper around the executor's
                    map here is what parallelizes fitness evaluation across worker processes, mirroring the
                    previous grid search's ProcessPoolExecutor usage but driven by DEAP instead of a manual
                    submit/as_completed loop. Deduplicating by content key before submitting is necessary,
                    not just an optimization: two individuals with identical gene values landing in the same
                    batch would otherwise both try to write/read the same on-disk candidate_{key} directory
                    concurrently, racing (one worker could read a tiled_workload.pickle the other is still
                    writing). Submitting each unique individual only once avoids that race entirely."""
                    individuals = list(individuals)
                    keys = [_candidate_key(ind) for ind in individuals]
                    first_index_for_key: dict[str, int] = {}
                    unique_individuals = []
                    for ind, key in zip(individuals, keys, strict=True):
                        if key not in first_index_for_key:
                            first_index_for_key[key] = len(unique_individuals)
                            unique_individuals.append(ind)
                    unique_results = list(executor.map(func, unique_individuals))
                    for individual, result in zip(unique_individuals, unique_results, strict=True):
                        key = _candidate_key(individual)
                        candidate_records[key] = (tuple(individual), result[0])
                    return [unique_results[first_index_for_key[key]] for key in keys]

                toolbox.register("map", _dedup_map)

                population = toolbox.population(n=self.nb_tiling_ga_individuals)
                hall_of_fame = tools.HallOfFame(min(10, self.nb_tiling_ga_individuals))
                statistics = tools.Statistics(lambda ind: ind.fitness.values)
                statistics.register("min", lambda values: min(v[0] for v in values))
                statistics.register("avg", lambda values: sum(v[0] for v in values) / len(values))

                algorithms.eaSimple(
                    population,
                    toolbox,
                    cxpb=0.5,
                    mutpb=0.3,
                    ngen=self.nb_tiling_ga_generations,
                    stats=statistics,
                    halloffame=hall_of_fame,
                    verbose=True,
                )

                _write_candidate_results_csv(
                    self.candidate_results_csv_path,
                    candidate_records,
                    genes,
                    self.tiling_candidates_dir,
                )

                seen_keys: set[str] = set()
                for individual in hall_of_fame:
                    key = _candidate_key(individual)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    cached = _load_cached_result(self.tiling_candidates_dir, key)
                    if cached is None:
                        continue
                    scme, extra_info = cached
                    logger.info(
                        f"TilingExplorationStage: hall-of-fame candidate {key}: "
                        f"{self.sort_key}={getattr(scme, self.sort_key)}"
                    )
                    yield scme, extra_info

    def _build_genes(self, sweep_inter_core_tiling: bool) -> list[TilingGene]:
        """Build tiling decision points for every layer: one independent "intra" gene per dimension (every
        dimension of every layer except batch `B` and group `G` — matching the exclusion convention already
        used elsewhere, e.g. the fallback dimension pick in `TilingGenerationStage.get_fusion_partition_dim`),
        plus a single combined per-layer "inter" gene choosing at most one of those dimensions to split
        inter-core (see `TilingGene`'s docstring for why inter must stay single-dimension). Dimensions of
        size <= 1 have nothing to tile and are excluded from both."""
        genes: list[TilingGene] = []
        for node in self.workload.node_list:
            if not isinstance(node, ComputationNode):
                continue
            nb_cores = len(node.possible_core_allocation) or 1
            relevant_dims = [
                dim
                for dim in node.layer_dim_sizes
                if dim not in (LayerDim("B"), LayerDim("G")) and node.layer_dim_sizes[dim] > 1
            ]
            if not relevant_dims:
                continue

            for dim in relevant_dims:
                size = node.layer_dim_sizes[dim]
                intra_candidates: list[INTRA_CANDIDATE_T] = [1, *prime_divisors(size)]
                if len(intra_candidates) > 1:
                    genes.append(TilingGene(node_id=node.id, kind="intra", dim=dim, candidates=intra_candidates))

            inter_candidates: list[INTER_CANDIDATE_T] = [None]
            for dim in relevant_dims:
                size = node.layer_dim_sizes[dim]
                if sweep_inter_core_tiling:
                    inter_candidates += [(dim, factor) for factor in prime_divisors(size) if factor <= nb_cores]
                else:
                    # Let the allocation stage (CO) resolve the actual split itself.
                    inter_candidates.append((dim, "*"))
            if len(inter_candidates) > 1:
                genes.append(TilingGene(node_id=node.id, kind="inter", dim=None, candidates=inter_candidates))
        return genes
