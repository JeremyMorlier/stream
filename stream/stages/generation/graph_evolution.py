"""Evolving-graph genetic search: an independent hardware graph that evolves alongside the workload mapping, in
two steps.

A genome is a hardware graph plus a mapping onto it:

- **nodes**: physical cores, each a real-valued `CoreGeneLayout` design;
- **links**: an explicit set of point-to-point links between nodes. It is part of the genome, not derived from
  the traffic, so the search decides the topology. A shared bus to the DRAM controller always exists, so a
  missing link slows a transfer down rather than making it impossible;
- **tiling**: the intra-/inter-core tiling genes of `TilingExplorationStage` (`build_tiling_genes`). The
  all-zero vector keeps the mapping file's tiling, which is the seed;
- **alloc**: for every layer, the node each of its inter-core groups runs on. Its length follows the layer's
  inter-core factor, so it is resized whenever the tiling changes.

The search runs in two steps:

1. **Hardware graph.** The tiling stays at the seed; NSGA2 evolves the graph (add/remove/perturb/share a node
   design, add/remove a link) together with the allocation onto it. Unused nodes are pruned. The step ends when
   the best EDP gains less than `rel_tol` over `step1_patience` generations, or after `graph_step_fraction` of
   the budget.
2. **Mapping.** The `nb_frozen_graphs` best distinct graphs of the step-1 front are frozen (every node keeps
   counting toward area); NSGA2 evolves the tiling and the allocation on them -- and which frozen graph a
   mapping runs on -- until the budget stops.

Fitness is `(latency, energy, area)`, measured with the real scheduler. Each tiling is generated once
(`TilingGenerationStage` + `TiledWorkloadGenerationStage`) and cached; a design's cost on a tiling's shapes is
learned once with a uniform accelerator, as `RolledScheduleExplorationStage` fills its cost matrix, and shared
across tilings through global tile-shape ids. A design must map every shape of the genome's tiling (the cheap
`lowest_memory_level_violations` pre-check), so any node can host any group.
"""

import copy
import csv
import json
import logging
import math
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import yaml
from deap import base, creator, tools
from zigzag.utils import pickle_deepcopy

from stream.hardware.architecture.core_generator import build_core_from_dict
from stream.opt.search_budget import SearchBudget, map_until_deadline
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.stages.estimation.zigzag_core_mapping_estimation import (
    ZigZagCoreMappingEstimationStage,
    lowest_memory_level_violations,
)
from stream.stages.generation.core_architecture_exploration import CoreGeneLayout
from stream.stages.generation.schedule_exploration import (
    ScheduleExplorationStage,
    _UnusedLeafStage,
    core_dict_key,
    tile_key,
)
from stream.stages.generation.tiled_workload_accelerator_generation import (
    derive_group_dedicated_topology,
    load_offchip_core_data,
)
from stream.stages.generation.tiled_workload_generation import TiledWorkloadGenerationStage
from stream.stages.generation.tiling_exploration import (
    TilingGene,
    _candidate_key,
    _decode_individual,
    _repair_tiling_constraints,
    _respects_tiling_constraints,
    build_tiling_genes,
)
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.stage import MainStage, Stage, StageCallable
from stream.workload.computation.computation_node import ComputationNode

logger = logging.getLogger(__name__)

INF3 = (math.inf, math.inf, math.inf)
LINK_BANDWIDTH = 32  # as tpu_like_quad_core's point-to-point links
BUS_BANDWIDTH = 128  # as tpu_like_quad_core's shared bus
# Chance that a reassigned group opens a node of its own instead of joining an existing one.
NEW_NODE_PROBABILITY = 0.2
# Step-2 mutation: chance to move the mapping onto another frozen graph, and to mutate the tiling (else the
# allocation).
SWITCH_GRAPH_PROBABILITY = 0.1
TILING_MUTATION_PROBABILITY = 0.5
# Merge, share-design and link mutations act on a pair of nodes.
PAIR = 2

# The stage a forked worker process evaluates against. Set right before each pool is forked, so workers inherit
# the current caches and tiling contexts without pickling the stage per task.
_WORKER_STAGE: "GraphEvolutionStage | None" = None


if not hasattr(creator, "GraphFitness"):
    creator.create("GraphFitness", base.Fitness, weights=(-1.0, -1.0, -1.0))


@dataclass
class GraphGenome:
    nodes: dict[int, list[float]]  # node label -> design gene vector
    links: set[frozenset[int]]  # node label pairs joined by a point-to-point link
    tiling: tuple[int, ...]  # tiling-gene indices (all zero = the mapping file's tiling)
    alloc: dict[int, list[int]]  # layer id -> node label of each inter-core group
    graph_id: int | None = None  # step 2: which frozen graph `nodes`/`links` are
    fitness: Any = field(default_factory=creator.GraphFitness)

    def prune(self) -> "GraphGenome":
        """Step 1 only: drop nodes no group runs on, and links to them."""
        used = {label for labels in self.alloc.values() for label in labels}
        self.nodes = {label: values for label, values in self.nodes.items() if label in used}
        self.links = {pair for pair in self.links if pair <= used}
        return self

    def new_label(self) -> int:
        return max(self.nodes, default=-1) + 1

    @property
    def nb_cores(self) -> int:
        return len(self.nodes)


@dataclass
class TilingContext:
    """Everything that depends on the tiling only: the tiled workload, the matching ONNX workload, the
    group-dedicated core ids (used only by the cost-model warm-up), every tile's global shape id and the number of
    inter-core groups per layer."""

    key: str
    workload: Any
    original_workload: Any
    core_ids_of_node: dict[int, list[int]]
    shape_class_of_tile: dict[tuple[int, int], int]
    groups_per_layer: dict[int, int]
    shapes: set[int]


class _CaptureStage(Stage):
    """Leaf that hands the tiled workload back instead of scheduling it."""

    def is_leaf(self) -> bool:
        return True

    def run(self):
        yield self.kwargs["workload"], self.kwargs["original_workload"]


def _warm_up_worker(job: tuple[str, list[float]]) -> tuple[dict[tuple[int, str], Any] | None, float]:
    assert _WORKER_STAGE is not None
    return _WORKER_STAGE._warm_up_design(*job)


def _measure_worker(genome: "GraphGenome") -> tuple[float, float]:
    assert _WORKER_STAGE is not None
    try:
        scme = _WORKER_STAGE._measure_genome(genome)
        return float(scme.latency), float(scme.energy)
    except Exception:
        logger.warning("GraphEvolutionStage: genome failed to schedule, penalizing it.", exc_info=True)
        return math.inf, math.inf


class GraphEvolutionStage(ScheduleExplorationStage):
    """Two-step evolving-graph NSGA2; see the module docstring. Reuses `ScheduleExplorationStage` for the cost-model
    harvest, the cost-LUT assembly and the downstream measurement, but runs no per-tile core GA."""

    def __init__(  # noqa: PLR0913
        self,
        list_of_callables: list[StageCallable],
        *,
        population_size: int = 16,
        graph_ga_generations: int = 10,
        max_workers: int | None = None,
        graph_search_dir: str | None = None,
        crossover_probability: float = 0.7,
        design_sampling_attempts: int = 2000,
        graph_step_fraction: float = 0.5,
        step1_patience: int = 5,
        nb_frozen_graphs: int = 4,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        # NSGA2's crowded tournament needs a multiple of 4.
        self.population_size = max(4, -(-population_size // 4) * 4)
        self.graph_ga_generations = graph_ga_generations
        self.max_workers = max_workers
        self.graph_search_dir = graph_search_dir or os.path.join(
            os.path.dirname(self.tiled_workload_path), "graph_search"
        )
        self.crossover_probability = crossover_probability
        self.design_sampling_attempts = design_sampling_attempts
        self.graph_step_fraction = graph_step_fraction
        self.step1_patience = step1_patience
        self.nb_frozen_graphs = nb_frozen_graphs
        self.search_budget: SearchBudget | None = kwargs.get("search_budget")

    # ------------------------------------------------------------------------------------------ set-up -------

    def _prepare(self) -> None:
        os.makedirs(self.graph_search_dir, exist_ok=True)
        # The untiled workload every tiling is re-derived from, and the tiling genes over it.
        self.base_workload = pickle_deepcopy(self.original_workload)
        self.tiling_genes: list[TilingGene] = build_tiling_genes(self.base_workload)
        self.layers: list[int] = [node.id for node in self.base_workload.node_list if isinstance(node, ComputationNode)]
        self.seed_tiling: tuple[int, ...] = (0,) * len(self.tiling_genes)

        self.unique_tiles: list[ComputationNode] = []  # global shape registry, across every tiling
        self.contexts: dict[str, TilingContext | None] = {}
        self.cme_cache: dict[tuple[int, str], Any] = {}
        self.runtime_cache: dict[tuple[int, str], float] = {}
        self.energy_cache: dict[tuple[int, str], float] = {}
        self.area_cache: dict[str, float] = {}
        self.failed_cells: set[tuple[str, str]] = set()  # (tiling key, design key) whose warm-up failed
        self.mappable: dict[tuple[str, str], bool] = {}
        self.key_memo: dict[tuple[float, ...], str] = {}
        self.gene_layout = CoreGeneLayout(self.gene_ranges)
        self.fitness_of_key: dict[tuple, tuple[float, float, float]] = {}
        self.evaluated: list[dict[str, Any]] = []
        self.frozen_graphs: list[GraphGenome] = []
        self.step = "step1"

        seed = self._context(self.seed_tiling)
        if seed is None:
            raise ValueError("GraphEvolutionStage: the mapping file's tiling could not be generated.")
        logger.info(
            f"GraphEvolutionStage: {len(self.layers)} layer(s), {len(self.tiling_genes)} tiling gene(s), seed "
            f"tiling has {sum(seed.groups_per_layer.values())} group(s) over {len(seed.shapes)} tile shape(s)."
        )

    def _global_shape_id(self, tile: ComputationNode) -> int:
        for index, unique in enumerate(self.unique_tiles):
            if tile.has_same_performance(unique):
                return index
        self.unique_tiles.append(tile)
        return len(self.unique_tiles) - 1

    def _context(self, tiling: tuple[int, ...]) -> TilingContext | None:
        """The tiled workload of a tiling, generated once and cached (None if the tiling cannot be generated)."""
        key = _candidate_key(tiling)
        if key in self.contexts:
            return self.contexts[key]
        workload = pickle_deepcopy(self.base_workload)
        per_node = _decode_individual(tiling, self.tiling_genes)
        for node in workload.node_list:
            if isinstance(node, ComputationNode) and node.id in per_node:
                node.intra_core_tiling = per_node[node.id]["intra_core_tiling"]
                node.inter_core_tiling = per_node[node.id]["inter_core_tiling"]
        kwargs = self.kwargs.copy()
        kwargs.update(
            workload=workload,
            accelerator=self.accelerator,
            tiled_workload_path=os.path.join(self.graph_search_dir, "tilings", key, "tiled_workload.pickle"),
        )
        os.makedirs(os.path.dirname(kwargs["tiled_workload_path"]), exist_ok=True)
        try:
            tiled, original = MainStage(
                [TilingGenerationStage, TiledWorkloadGenerationStage, _CaptureStage], **kwargs
            ).run()[0]
            reference = pickle_deepcopy(tiled)
            topology = derive_group_dedicated_topology(reference, original)
        except Exception:
            logger.warning(f"GraphEvolutionStage: tiling {key} could not be generated.", exc_info=True)
            self.contexts[key] = None
            return None
        shape_class_of_tile = {
            tile_key(tile): self._global_shape_id(tile)
            for tile in reference.node_list
            if isinstance(tile, ComputationNode)
        }
        context = TilingContext(
            key=key,
            workload=tiled,
            original_workload=original,
            core_ids_of_node=topology.node_core_ids,
            shape_class_of_tile=shape_class_of_tile,
            groups_per_layer={layer: len(cores) for layer, cores in topology.node_core_ids.items()},
            shapes=set(shape_class_of_tile.values()),
        )
        self.contexts[key] = context
        return context

    def _use(self, context: TilingContext) -> None:
        """Point the inherited helpers (`_build_accelerator`, `_harvest_cmes`, `_assemble_cost_lut`) at a tiling."""
        self.workload = context.workload
        self.original_workload = context.original_workload
        self.core_ids_of_node = context.core_ids_of_node
        self.shape_class_of_tile = context.shape_class_of_tile

    # ------------------------------------------------------------------------------------- designs -------------

    def _decode(self, values: list[float], name: str = "graph_core") -> dict[str, Any] | None:
        try:
            return self.gene_layout.decode_node(values, name=name)
        except ValueError:
            return None

    def _key_of(self, values: list[float]) -> str:
        """Content key of the core a gene vector decodes to (memoized: canonical keys ask for it constantly)."""
        memo_key = tuple(values)
        if memo_key not in self.key_memo:
            core_dict = self._decode(values)
            self.key_memo[memo_key] = "invalid" if core_dict is None else core_dict_key(core_dict)
        return self.key_memo[memo_key]

    def _maps(self, values: list[float], context: TilingContext) -> bool:
        """Cheap pre-check that a design can host any group of a tiling: its innermost memories fit every shape."""
        cache_key = (context.key, self._key_of(values))
        if cache_key not in self.mappable:
            core_dict = self._decode(values)
            ok = core_dict is not None
            if ok:
                try:
                    core = build_core_from_dict(core_dict, core_id=0)
                    ok = not any(
                        lowest_memory_level_violations(self.unique_tiles[shape], core) for shape in context.shapes
                    )
                except Exception:
                    ok = False
            self.mappable[cache_key] = ok
        return self.mappable[cache_key]

    def _random_design(self, context: TilingContext) -> list[float]:
        for _ in range(self.design_sampling_attempts):
            values = [
                random.uniform(lo, hi) for lo, hi in zip(self.gene_layout.low, self.gene_layout.high, strict=True)
            ]
            if self._maps(values, context):
                return values
        raise ValueError(
            f"GraphEvolutionStage: no design in {self.design_sampling_attempts} random sample(s) maps every tile "
            "shape; widen `core_param_ranges`."
        )

    def _perturbed(self, values: list[float], attempts: int = 5) -> list[float]:
        """Polynomial-bounded mutation, retried until the design still maps the seed tiling (else unchanged)."""
        context = self.contexts[_candidate_key(self.seed_tiling)]
        assert context is not None
        n_genes = len(values)
        for _ in range(attempts):
            (candidate,) = tools.mutPolynomialBounded(
                list(values), eta=20.0, low=self.gene_layout.low, up=self.gene_layout.high, indpb=max(1 / n_genes, 0.1)
            )
            if self._maps(candidate, context):
                return list(candidate)
        return list(values)

    def _warm_up_design(self, context_key: str, design_values: list[float]):
        """Cost-model every tile shape of one tiling on one design, with a uniform accelerator (every layer on that
        design): ZigZag then runs once per shape. Returns `({(shape, design key): cme}, area)`, or `(None, inf)`."""
        context = self.contexts[context_key]
        assert context is not None
        self._use(context)
        core_dict = self._decode(design_values)
        design_key = core_dict_key(core_dict)
        workload = pickle_deepcopy(context.workload)
        node_core_dicts = dict.fromkeys(context.core_ids_of_node, core_dict)
        lut_path = os.path.join(self.graph_search_dir, f"cost_lut_{design_key}_{os.getpid()}.pickle")
        try:
            accelerator = self._build_accelerator(node_core_dicts, workload)
            estimation = ZigZagCoreMappingEstimationStage(
                [_UnusedLeafStage],
                workload=workload,
                accelerator=accelerator,
                loma_lpf_limit=self.kwargs["loma_lpf_limit"],
                cost_lut_path=lut_path,
                layer_stacks=self.kwargs["layer_stacks"],
                temporal_mapping_type=self.kwargs["temporal_mapping_type"],
            )
            estimation.update_cost_lut()
            self.cme_cache, self.runtime_cache, self.energy_cache, self.area_cache = {}, {}, {}, {}
            self._harvest_cmes(estimation.cost_lut, workload, accelerator, self._design_key_of_core(node_core_dicts))
            return dict(self.cme_cache), self.area_cache.get(design_key, math.inf)
        except Exception:
            logger.warning(f"GraphEvolutionStage: design {design_key} failed its cost-model warm-up.", exc_info=True)
            return None, math.inf
        finally:
            if os.path.exists(lut_path):
                os.remove(lut_path)

    def _missing_cells(self, context: TilingContext, design_key: str) -> bool:
        return any((shape, design_key) not in self.cme_cache for shape in context.shapes)

    def _warm_up(self, genomes: list[GraphGenome]) -> None:
        """Cost-model, in parallel, every (tiling, design) pair these genomes need that the cache cannot cover."""
        jobs: dict[tuple[str, str], tuple[str, list[float]]] = {}
        for genome in genomes:
            context = self._context(genome.tiling)
            if context is None:
                continue
            for values in genome.nodes.values():
                design_key = self._key_of(values)
                pair = (context.key, design_key)
                if (
                    pair not in jobs
                    and pair not in self.failed_cells
                    and self._maps(values, context)
                    and self._missing_cells(context, design_key)
                ):
                    jobs[pair] = (context.key, values)
        if not jobs:
            return
        logger.info(f"GraphEvolutionStage: cost-modelling {len(jobs)} new (tiling, design) pair(s).")
        results = self._parallel(_warm_up_worker, list(jobs.values()), fallback=(None, math.inf))
        for pair, (cells, area) in zip(jobs, results, strict=True):
            if cells is None:
                self.failed_cells.add(pair)
                continue
            for cell_key, cme in cells.items():
                self.cme_cache[cell_key] = cme
                self.runtime_cache[cell_key] = float(getattr(cme, self.latency_attr))
                self.energy_cache[cell_key] = float(cme.energy_total)
            self.area_cache[pair[1]] = area

    # ------------------------------------------------------------------------------------ measuring ----------

    def _assemble(self, genome: GraphGenome, workload: Any, yaml_path: str | None = None):
        """Build the genome's accelerator (nodes renumbered densely, offchip last, its own links plus the bus) and
        pin every tile of `workload` to the node its group is allocated to."""
        core_of_label = {label: index for index, label in enumerate(sorted(genome.nodes))}
        offchip = len(core_of_label)
        cores: dict[int, Any] = {
            core_of_label[label]: self._decode(values, name=f"graph_core_{label}")
            for label, values in sorted(genome.nodes.items())
        }
        cores[offchip] = load_offchip_core_data()
        links = [
            {"type": "link", "cores": sorted(core_of_label[label] for label in pair), "bandwidth": LINK_BANDWIDTH}
            for pair in sorted(genome.links, key=sorted)
        ]
        accelerator_data = {
            "name": f"{self.accelerator.name}_graph",
            "cores": cores,
            "offchip_core_id": offchip,
            "unit_energy_cost": 0,
            "core_memory_sharing": [],
            "core_connectivity": [
                {"type": "bus", "cores": list(range(offchip + 1)), "bandwidth": BUS_BANDWIDTH},
                *links,
            ],
        }
        if yaml_path:
            with open(yaml_path, "w") as f:
                yaml.safe_dump(accelerator_data, f, sort_keys=False)
        for tile in workload.node_list:
            if isinstance(tile, ComputationNode):
                core_ids = [core_of_label[label] for label in genome.alloc[tile.id]]
                tile.possible_core_allocation = core_ids
                tile.set_chosen_core_allocation(core_ids[tile.group])
        design_key_of_core = {core: core_dict_key(core_dict) for core, core_dict in cores.items() if core != offchip}
        return AcceleratorFactory(accelerator_data).create(), design_key_of_core

    def _measure_genome(self, genome: GraphGenome, save_yaml: bool = False):
        context = self._context(genome.tiling)
        assert context is not None
        self._use(context)
        workload = pickle_deepcopy(context.workload)
        yaml_path = os.path.join(self.graph_search_dir, "accelerator.yaml") if save_yaml else None
        accelerator, design_key_of_core = self._assemble(genome, workload, yaml_path)
        cost_lut = self._assemble_cost_lut(workload, accelerator, design_key_of_core)
        return self._measure_accelerator(workload, accelerator, cost_lut)

    def _area(self, genome: GraphGenome) -> float:
        return sum(self.area_cache[self._key_of(values)] for values in genome.nodes.values())

    def _is_feasible(self, genome: GraphGenome) -> bool:
        context = self._context(genome.tiling)
        if context is None:
            return False
        if genome.tiling != self.seed_tiling and not _respects_tiling_constraints(
            list(genome.tiling), self.tiling_genes
        ):
            return False
        if any(len(genome.alloc.get(layer, [])) != groups for layer, groups in context.groups_per_layer.items()):
            return False
        return all(
            self._maps(values, context) and not self._missing_cells(context, self._key_of(values))
            for values in genome.nodes.values()
        )

    def _canonical_key(self, genome: GraphGenome) -> tuple:
        """Identity of the accelerator and mapping a genome describes, independent of its label numbering."""
        relabel: dict[int, int] = {}
        for layer in self.layers:
            for label in genome.alloc.get(layer, []):
                relabel.setdefault(label, len(relabel))
        for label in sorted(genome.nodes, key=lambda label: self._key_of(genome.nodes[label])):
            relabel.setdefault(label, len(relabel))
        return (
            genome.tiling,
            tuple(tuple(relabel[label] for label in genome.alloc.get(layer, [])) for layer in self.layers),
            tuple(self._key_of(genome.nodes[label]) for label in sorted(relabel, key=relabel.get)),
            tuple(sorted(tuple(sorted(relabel[label] for label in pair)) for pair in genome.links)),
        )

    def _graph_key(self, genome: GraphGenome) -> tuple:
        """Identity of the hardware alone (nodes and links), for picking distinct graphs to freeze."""
        order = sorted(genome.nodes, key=lambda label: self._key_of(genome.nodes[label]))
        index = {label: position for position, label in enumerate(order)}
        return (
            tuple(self._key_of(genome.nodes[label]) for label in order),
            tuple(sorted(tuple(sorted(index[label] for label in pair)) for pair in genome.links)),
        )

    def _parallel(self, func, items: list[Any], fallback: Any, on_result=None) -> list[Any]:
        global _WORKER_STAGE  # noqa: PLW0603
        _WORKER_STAGE = self
        mp_context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=mp_context) as executor:
            return map_until_deadline(executor, func, items, self.search_budget, fallback, on_result=on_result)

    def _evaluate(self, genomes: list[GraphGenome]) -> None:
        """Set every genome's fitness: generate new tilings, warm up new (tiling, design) pairs, then schedule each
        new genome once (cached by canonical key)."""
        self._warm_up(genomes)
        to_measure: dict[tuple, GraphGenome] = {}
        for genome in genomes:
            key = self._canonical_key(genome)
            if key in self.fitness_of_key or key in to_measure:
                continue
            if not self._is_feasible(genome):
                self.fitness_of_key[key] = INF3
                continue
            to_measure[key] = genome
        keys = list(to_measure)
        budget = self.search_budget

        def on_result(index: int, result: tuple[float, float]) -> None:
            genome = to_measure[keys[index]]
            area = self._area(genome)
            latency, energy = result
            self.fitness_of_key[keys[index]] = (latency, energy, area) if math.isfinite(latency) else INF3
            self.evaluated.append(
                {
                    "step": self.step,
                    "nb_cores": genome.nb_cores,
                    "nb_links": len(genome.links),
                    "tiling": _candidate_key(genome.tiling),
                    "latency": latency,
                    "energy": energy,
                    "area": area,
                }
            )
            if budget is not None:
                budget.record(latency, energy, area, info=f"{self.step} cores={genome.nb_cores}")

        if keys:
            self._parallel(
                _measure_worker, [to_measure[key] for key in keys], fallback=(math.inf, math.inf), on_result=on_result
            )
        for genome in genomes:
            genome.fitness.values = self.fitness_of_key.get(self._canonical_key(genome), INF3)

    # ------------------------------------------------------------------------------ step 1: graph -------------

    def _random_graph(self, design_pool: list[list[float]], context: TilingContext) -> GraphGenome:
        """Spread the initial population over node counts (one per group, a single node, and in between) and over
        topologies (no links, a ring, random links)."""
        nb_groups = sum(context.groups_per_layer.values())
        nb_nodes = random.choice([nb_groups, 1, random.randint(1, nb_groups)])
        alloc = {
            layer: [random.randrange(nb_nodes) for _ in range(groups)]
            for layer, groups in context.groups_per_layer.items()
        }
        nodes = {label: list(random.choice(design_pool)) for label in range(nb_nodes)}
        topology = random.choice(("none", "ring", "random"))
        links: set[frozenset[int]] = set()
        if nb_nodes >= PAIR and topology == "ring":
            links = {frozenset((label, (label + 1) % nb_nodes)) for label in range(nb_nodes)}
        elif nb_nodes >= PAIR and topology == "random":
            links = {frozenset(random.sample(range(nb_nodes), PAIR)) for _ in range(nb_nodes)}
        return GraphGenome(nodes=nodes, links=links, tiling=self.seed_tiling, alloc=alloc).prune()

    def _groups(self, genome: GraphGenome) -> list[tuple[int, int]]:
        return [(layer, group) for layer, labels in genome.alloc.items() for group in range(len(labels))]

    def _mutate_graph(self, genome: GraphGenome) -> GraphGenome:
        operators = [
            self._mut_reassign,
            self._mut_split,
            self._mut_merge,
            self._mut_design,
            self._mut_share_design,
            self._mut_add_link,
            self._mut_remove_link,
        ]
        for _ in range(random.choice((1, 1, 2))):
            random.choice(operators)(genome)
            genome.prune()
        return genome

    def _mut_reassign(self, genome: GraphGenome) -> None:
        layer, group = random.choice(self._groups(genome))
        if random.random() < NEW_NODE_PROBABILITY:
            label = genome.new_label()
            genome.nodes[label] = self._perturbed(genome.nodes[genome.alloc[layer][group]])
        else:
            label = random.choice(list(genome.nodes))
        genome.alloc[layer][group] = label

    def _mut_split(self, genome: GraphGenome) -> None:
        """Add a node: move some of a busy node's groups onto a perturbed copy of it."""
        members: dict[int, list[tuple[int, int]]] = {}
        for layer, group in self._groups(genome):
            members.setdefault(genome.alloc[layer][group], []).append((layer, group))
        splittable = [label for label, groups in members.items() if len(groups) > 1]
        if not splittable:
            return
        label = random.choice(splittable)
        new = genome.new_label()
        genome.nodes[new] = self._perturbed(genome.nodes[label])
        for layer, group in random.sample(members[label], k=random.randint(1, len(members[label]) - 1)):
            genome.alloc[layer][group] = new

    def _mut_merge(self, genome: GraphGenome) -> None:
        """Remove a node: its groups move onto another one."""
        if len(genome.nodes) < PAIR:
            return
        keep, drop = random.sample(list(genome.nodes), PAIR)
        for labels in genome.alloc.values():
            labels[:] = [keep if label == drop else label for label in labels]

    def _mut_design(self, genome: GraphGenome) -> None:
        label = random.choice(list(genome.nodes))
        genome.nodes[label] = self._perturbed(genome.nodes[label])

    def _mut_share_design(self, genome: GraphGenome) -> None:
        if len(genome.nodes) < PAIR:
            return
        source, target = random.sample(list(genome.nodes), PAIR)
        genome.nodes[target] = list(genome.nodes[source])

    def _mut_add_link(self, genome: GraphGenome) -> None:
        if len(genome.nodes) < PAIR:
            return
        genome.links.add(frozenset(random.sample(list(genome.nodes), PAIR)))

    def _mut_remove_link(self, genome: GraphGenome) -> None:
        if genome.links:
            genome.links.discard(random.choice(sorted(genome.links, key=sorted)))

    def _crossover_graph(self, first: GraphGenome, second: GraphGenome) -> tuple[GraphGenome, GraphGenome]:
        """Graft the allocation of a random half of the layers -- with the nodes hosting them and the links among
        those nodes -- from one parent onto the other, in both directions."""
        layers = set(random.sample(self.layers, k=max(1, len(self.layers) // 2)))
        return self._graft(first, second, layers), self._graft(second, first, layers)

    def _graft(self, receiver: GraphGenome, donor: GraphGenome, layers: set[int]) -> GraphGenome:
        child = GraphGenome(
            nodes={label: list(values) for label, values in receiver.nodes.items()},
            links=set(receiver.links),
            tiling=receiver.tiling,
            alloc={layer: list(labels) for layer, labels in receiver.alloc.items()},
        )
        offset = child.new_label()
        grafted: set[int] = set()
        for layer in layers:
            child.alloc[layer] = [offset + label for label in donor.alloc[layer]]
            grafted.update(donor.alloc[layer])
        for label in grafted:
            child.nodes[offset + label] = list(donor.nodes[label])
        child.links |= {frozenset(offset + label for label in pair) for pair in donor.links if pair <= grafted}
        return child.prune()

    # ---------------------------------------------------------------------------- step 2: mapping -------------

    def _on_graph(self, graph_id: int, tiling: tuple[int, ...], alloc: dict[int, list[int]]) -> GraphGenome:
        graph = self.frozen_graphs[graph_id]
        genome = GraphGenome(nodes=graph.nodes, links=graph.links, tiling=tiling, alloc=alloc, graph_id=graph_id)
        self._fit_alloc(genome)
        return genome

    def _fit_alloc(self, genome: GraphGenome) -> None:
        """Resize every layer's allocation to its group count under the genome's tiling, and keep its labels on the
        genome's graph."""
        context = self._context(genome.tiling)
        if context is None:
            return
        labels = sorted(genome.nodes)
        for layer, groups in context.groups_per_layer.items():
            current = [
                label if label in genome.nodes else labels[label % len(labels)] for label in genome.alloc.get(layer, [])
            ]
            genome.alloc[layer] = (current + [random.choice(labels) for _ in range(groups)])[:groups]

    def _mutate_mapping(self, genome: GraphGenome) -> GraphGenome:
        if len(self.frozen_graphs) > 1 and random.random() < SWITCH_GRAPH_PROBABILITY:
            graph_id = random.choice([g for g in range(len(self.frozen_graphs)) if g != genome.graph_id])
            genome = self._on_graph(graph_id, genome.tiling, genome.alloc)
        if self.tiling_genes and random.random() < TILING_MUTATION_PROBABILITY:
            tiling = list(genome.tiling)
            for index, gene in enumerate(self.tiling_genes):
                if random.random() < 1 / len(tiling):
                    replacements = [v for v in range(len(gene.candidates)) if v != tiling[index]]
                    if replacements:
                        tiling[index] = random.choice(replacements)
            genome.tiling = tuple(_repair_tiling_constraints(tiling, self.tiling_genes))
        else:
            layer, group = random.choice(self._groups(genome))
            genome.alloc[layer][group] = random.choice(sorted(genome.nodes))
        self._fit_alloc(genome)
        return genome

    def _crossover_mapping(self, first: GraphGenome, second: GraphGenome) -> tuple[GraphGenome, GraphGenome]:
        """Two-point crossover on the tiling plus a per-layer allocation swap, between mappings on one graph."""
        if first.graph_id != second.graph_id:
            return first, second
        tiling_1, tiling_2 = list(first.tiling), list(second.tiling)
        if len(tiling_1) >= PAIR:
            tools.cxTwoPoint(tiling_1, tiling_2)
        first.tiling = tuple(_repair_tiling_constraints(tiling_1, self.tiling_genes))
        second.tiling = tuple(_repair_tiling_constraints(tiling_2, self.tiling_genes))
        for layer in random.sample(self.layers, k=max(1, len(self.layers) // 2)):
            first.alloc[layer], second.alloc[layer] = second.alloc.get(layer, []), first.alloc.get(layer, [])
        self._fit_alloc(first)
        self._fit_alloc(second)
        return first, second

    def _freeze(self, front: list[GraphGenome]) -> None:
        """Keep the best distinct graphs of the step-1 front: lowest EDP first."""
        self.frozen_graphs = []
        seen: set[tuple] = set()
        for genome in sorted(front, key=lambda g: g.fitness.values[0] * g.fitness.values[1]):
            key = self._graph_key(genome)
            if key in seen:
                continue
            seen.add(key)
            self.frozen_graphs.append(copy.deepcopy(genome))
            if len(self.frozen_graphs) == self.nb_frozen_graphs:
                break
        with open(os.path.join(self.graph_search_dir, "frozen_graphs.json"), "w") as f:
            json.dump(
                [
                    {
                        "graph_id": index,
                        "nb_cores": graph.nb_cores,
                        "links": sorted(sorted(pair) for pair in graph.links),
                        "design_keys": {label: self._key_of(values) for label, values in graph.nodes.items()},
                        "step1_fitness": list(graph.fitness.values),
                    }
                    for index, graph in enumerate(self.frozen_graphs)
                ],
                f,
                indent=2,
            )
        logger.info(
            f"GraphEvolutionStage: froze {len(self.frozen_graphs)} graph(s) with "
            f"{[graph.nb_cores for graph in self.frozen_graphs]} core(s)."
        )

    # ------------------------------------------------------------------------------------------ loop ---------

    def _clone(self, genome: GraphGenome) -> GraphGenome:
        """Copy the mapping; step-2 genomes keep sharing their frozen graph's nodes and links."""
        frozen = genome.graph_id is not None
        return GraphGenome(
            nodes=genome.nodes if frozen else copy.deepcopy(genome.nodes),
            links=genome.links if frozen else set(genome.links),
            tiling=genome.tiling,
            alloc={layer: list(labels) for layer, labels in genome.alloc.items()},
            graph_id=genome.graph_id,
        )

    def _generation(self, population, mutate, crossover) -> tuple[list[GraphGenome], list[GraphGenome]]:
        mu = self.population_size
        offspring = [self._clone(genome) for genome in tools.selTournamentDCD(population, mu)]
        children: list[GraphGenome] = []
        for parent_1, parent_2 in zip(offspring[::2], offspring[1::2], strict=True):
            pair = (
                crossover(parent_1, parent_2) if random.random() < self.crossover_probability else (parent_1, parent_2)
            )
            children += [mutate(child) for child in pair]
        self._evaluate(children)
        return tools.selNSGA2(population + children, mu), children

    def _budget_stops(self, generation: int) -> bool:
        if self.search_budget is not None:
            return self.search_budget.should_stop()
        return generation >= self.graph_ga_generations

    def _step1_done(self, generation: int, best_edps: list[float]) -> bool:
        if self._budget_stops(generation):
            return True
        budget = self.search_budget
        if budget is not None and budget.elapsed() >= self.graph_step_fraction * budget.max_time_s:
            logger.info("GraphEvolutionStage: step 1 used its share of the budget.")
            return True
        if budget is None and generation >= self.graph_ga_generations // 2:
            return True
        rel_tol = budget.rel_tol if budget is not None else 0.005
        if len(best_edps) > self.step1_patience:
            previous = best_edps[-1 - self.step1_patience]
            if math.isfinite(best_edps[-1]) and previous / best_edps[-1] - 1 < rel_tol:
                logger.info(f"GraphEvolutionStage: step 1 converged after {generation} generation(s).")
                return True
        return False

    @staticmethod
    def _best_edp(front) -> float:
        return min((g.fitness.values[0] * g.fitness.values[1] for g in front), default=math.inf)

    def _evolve(self) -> tools.ParetoFront:
        mu = self.population_size
        front = tools.ParetoFront(similar=lambda a, b: self._canonical_key(a) == self._canonical_key(b))

        # Step 1: the hardware graph, on the seed tiling.
        seed = self.contexts[_candidate_key(self.seed_tiling)]
        assert seed is not None
        design_pool = [self._random_design(seed) for _ in range(mu)]
        population = [self._random_graph(design_pool, seed) for _ in range(mu)]
        self._evaluate(population)
        population = tools.selNSGA2(population, mu)
        front.update([g for g in population if math.isfinite(g.fitness.values[0])])
        best_edps = [self._best_edp(front)]
        generation = 0
        while not self._step1_done(generation, best_edps):
            generation += 1
            population, children = self._generation(population, self._mutate_graph, self._crossover_graph)
            front.update([g for g in children if math.isfinite(g.fitness.values[0])])
            best_edps.append(self._best_edp(front))
            self._log_generation(generation, front)
        if not len(front) or self._budget_stops(generation):
            return front

        # Step 2: the mapping, on the frozen graphs. Each graph first carries its own step-1 mapping.
        self.step = "step2"
        self._freeze(list(front))
        population = []
        for index in range(mu):
            graph_id = index % len(self.frozen_graphs)
            graph = self.frozen_graphs[graph_id]
            genome = self._on_graph(graph_id, graph.tiling, {k: list(v) for k, v in graph.alloc.items()})
            population.append(genome if index < len(self.frozen_graphs) else self._mutate_mapping(genome))
        self._evaluate(population)
        population = tools.selNSGA2(population, mu)
        front.update([g for g in population if math.isfinite(g.fitness.values[0])])
        while not self._budget_stops(generation):
            generation += 1
            population, children = self._generation(population, self._mutate_mapping, self._crossover_mapping)
            front.update([g for g in children if math.isfinite(g.fitness.values[0])])
            self._log_generation(generation, front)
        return front

    def _log_generation(self, generation: int, front) -> None:
        logger.info(
            f"GraphEvolutionStage: {self.step} generation {generation}: {len(self.fitness_of_key)} distinct "
            f"genome(s), {len(self.contexts)} tiling(s), {len(self.area_cache)} design(s) cost-modelled, "
            f"front={len(front)}, best EDP={self._best_edp(front):.4g}, "
            f"core counts on front={sorted({g.nb_cores for g in front})}."
        )

    def _save(self, front) -> None:
        with open(os.path.join(self.graph_search_dir, "evaluated.csv"), "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step", "nb_cores", "nb_links", "tiling", "latency", "energy", "area"]
            )
            writer.writeheader()
            writer.writerows(self.evaluated)
        with open(os.path.join(self.graph_search_dir, "pareto.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["graph_id", "nb_cores", "nb_links", "tiling", "latency", "energy", "area"])
            for genome in front:
                writer.writerow(
                    [genome.graph_id, genome.nb_cores, len(genome.links), _candidate_key(genome.tiling)]
                    + list(genome.fitness.values)
                )

    def run(self):
        self._prepare()
        front = self._evolve()
        self._save(front)
        if not len(front):
            raise ValueError("GraphEvolutionStage: no graph could be scheduled.")
        best = min(front, key=lambda genome: genome.fitness.values[0] * genome.fitness.values[1])
        logger.info(
            f"GraphEvolutionStage: best genome by EDP has {best.nb_cores} core(s), {len(best.links)} link(s): "
            f"(latency, energy, area)={best.fitness.values}; {len(front)} genome(s) on the front."
        )
        best_scme = self._measure_genome(best, save_yaml=True)
        yield (
            best_scme,
            {
                "pareto_graphs": [
                    {
                        "step": "step1" if genome.graph_id is None else "step2",
                        "nb_cores": genome.nb_cores,
                        "nb_links": len(genome.links),
                        "latency": genome.fitness.values[0],
                        "energy": genome.fitness.values[1],
                        "area": genome.fitness.values[2],
                    }
                    for genome in front
                ],
                "nb_evaluated": len(self.evaluated),
            },
        )
