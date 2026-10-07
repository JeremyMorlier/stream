"""Evolving-graph genetic search over the accelerator's core graph and the workload's allocation onto it.

The rolled schedule exploration builds an accelerator in fixed phases: one core design per tile (NSGA2), one
schedule per design combination (branch and bound), then folding onto fewer cores. This stage searches the same
space -- which cores exist, what each one is, which layer groups share a core, which links connect them -- in a
single NSGA2 loop over *variable-size graphs*:

- **nodes** are physical cores, each a real-valued `CoreGeneLayout` design;
- **assignment** maps every core of the group-dedicated (fully unrolled) topology -- one per `(layer, inter-core
  group)` -- onto a node. That is exactly the `merge_of_core` labelling `derive_rolled_topology` folds, so a
  genome *is* a rolled accelerator: nodes sharing groups host them, traffic between them becomes free, and the
  links between nodes are re-derived from the traffic;
- **dropped links**: node pairs whose derived point-to-point link is removed (the transfer falls back to the
  shared bus), so the search can trade link cost against bandwidth.

Mutations change the graph's structure (move a group, split a node, merge two nodes, share a design) as well as
its contents (perturb a design, toggle a link); crossover grafts the assignment of a random subset of layers
from one parent onto the other. Fitness is `(latency, energy, area)`, measured with the real scheduler.

Every node design must map every tile shape of the workload (checked with the cheap
`lowest_memory_level_violations` pre-check before any ZigZag run), so any design can host any group and the
cost of a design is learned once: a new design is cost-modelled on all shapes with a uniform accelerator, the
same way `RolledScheduleExplorationStage` fills its cost matrix.
"""

import copy
import csv
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
    derive_rolled_topology,
    load_offchip_core_data,
)
from stream.stages.stage import StageCallable
from stream.workload.computation.computation_node import ComputationNode

logger = logging.getLogger(__name__)

INF3 = (math.inf, math.inf, math.inf)
# Chance that a reassigned group opens a node of its own instead of joining an existing one.
NEW_NODE_PROBABILITY = 0.2
# Merge, share-design and link mutations act on a pair of nodes.
PAIR = 2

# The stage a forked worker process evaluates against. Set right before each pool is forked, so workers inherit
# the current cost-model cache without pickling the stage (and its workload) per task.
_WORKER_STAGE: "GraphEvolutionStage | None" = None


if not hasattr(creator, "GraphFitness"):
    creator.create("GraphFitness", base.Fitness, weights=(-1.0, -1.0, -1.0))


@dataclass
class GraphGenome:
    designs: dict[int, list[float]]  # node label -> design gene vector
    assign: dict[int, int]  # unrolled core id -> node label
    dropped_links: set[frozenset[int]] = field(default_factory=set)  # node label pairs without a direct link
    fitness: Any = field(default_factory=creator.GraphFitness)

    def prune(self) -> "GraphGenome":
        """Drop nodes no group is assigned to, and link drops that no longer name two live nodes."""
        used = set(self.assign.values())
        self.designs = {label: values for label, values in self.designs.items() if label in used}
        self.dropped_links = {pair for pair in self.dropped_links if pair <= used}
        return self

    def new_label(self) -> int:
        return max(self.designs, default=-1) + 1

    @property
    def nb_cores(self) -> int:
        return len(set(self.assign.values()))


def _warm_up_worker(core_dict: dict[str, Any]) -> tuple[str, dict[tuple[int, str], Any] | None, float]:
    assert _WORKER_STAGE is not None
    return _WORKER_STAGE._warm_up_design(core_dict)


def _measure_worker(genome: GraphGenome) -> tuple[float, float]:
    assert _WORKER_STAGE is not None
    try:
        scme = _WORKER_STAGE._measure_genome(genome)
        return float(scme.latency), float(scme.energy)
    except Exception:
        logger.warning("GraphEvolutionStage: genome failed to schedule, penalizing it.", exc_info=True)
        return math.inf, math.inf


class GraphEvolutionStage(ScheduleExplorationStage):
    """NSGA2 over variable-size core graphs; see the module docstring. Reuses `ScheduleExplorationStage` for
    the cost-model cache, the cost-LUT assembly and the downstream measurement, but runs no per-tile core GA."""

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
        self.search_budget: SearchBudget | None = kwargs.get("search_budget")

    # ------------------------------------------------------------------------------------------ set-up -------

    def _prepare(self) -> None:
        os.makedirs(self.graph_search_dir, exist_ok=True)
        self.unique_tiles, _ = self._unique_tiles()
        reference = pickle_deepcopy(self.workload)
        topology = derive_group_dedicated_topology(reference, self.original_workload)
        self.core_ids_of_node: dict[int, list[int]] = topology.node_core_ids
        self.unrolled_core_ids: list[int] = list(range(topology.offchip_core_id))
        self.shape_class_of_tile = {
            tile_key(tile): self._shape_class_of(tile)
            for tile in reference.node_list
            if isinstance(tile, ComputationNode)
        }
        self.cme_cache: dict[tuple[int, str], Any] = {}
        self.runtime_cache: dict[tuple[int, str], float] = {}
        self.energy_cache: dict[tuple[int, str], float] = {}
        self.area_cache: dict[str, float] = {}
        self.infeasible_designs: set[str] = set()
        self.gene_layout = CoreGeneLayout(self.gene_ranges)
        self.fitness_of_key: dict[tuple, tuple[float, float, float]] = {}
        self.key_memo: dict[tuple[float, ...], str] = {}
        self.evaluated: list[dict[str, Any]] = []
        logger.info(
            f"GraphEvolutionStage: {len(self.unrolled_core_ids)} unrolled core(s) over {len(self.core_ids_of_node)} "
            f"layer(s), {len(self.unique_tiles)} tile shape(s)."
        )

    # ------------------------------------------------------------------------------------- designs -------------

    def _decode(self, values: list[float], name: str = "graph_core") -> dict[str, Any] | None:
        try:
            return self.gene_layout.decode_node(values, name=name)
        except ValueError:
            return None

    def _maps_every_shape(self, values: list[float]) -> bool:
        """Cheap pre-check that a design can host any group: its innermost memories fit every tile shape."""
        core_dict = self._decode(values)
        if core_dict is None:
            return False
        try:
            core = build_core_from_dict(core_dict, core_id=0)
        except Exception:
            return False
        return not any(lowest_memory_level_violations(tile, core) for tile in self.unique_tiles)

    def _random_design(self) -> list[float]:
        for _ in range(self.design_sampling_attempts):
            values = [
                random.uniform(lo, hi) for lo, hi in zip(self.gene_layout.low, self.gene_layout.high, strict=True)
            ]
            if self._maps_every_shape(values):
                return values
        raise ValueError(
            f"GraphEvolutionStage: no design in {self.design_sampling_attempts} random sample(s) maps every tile "
            "shape; widen `core_param_ranges`."
        )

    def _perturbed(self, values: list[float], attempts: int = 5) -> list[float]:
        """Polynomial-bounded mutation, retried until the design still maps every shape (else unchanged)."""
        n_genes = len(values)
        for _ in range(attempts):
            (candidate,) = tools.mutPolynomialBounded(
                list(values), eta=20.0, low=self.gene_layout.low, up=self.gene_layout.high, indpb=max(1 / n_genes, 0.1)
            )
            if self._maps_every_shape(candidate):
                return list(candidate)
        return list(values)

    def _key_of(self, values: list[float]) -> str:
        """Content key of the core a gene vector decodes to (memoized: canonical keys ask for it constantly)."""
        memo_key = tuple(values)
        if memo_key not in self.key_memo:
            core_dict = self._decode(values)
            self.key_memo[memo_key] = "invalid" if core_dict is None else core_dict_key(core_dict)
        return self.key_memo[memo_key]

    def _warm_up_design(self, core_dict: dict[str, Any]) -> tuple[str, dict[tuple[int, str], Any] | None, float]:
        """Cost-model every tile shape on one design, with a uniform accelerator (every layer on that design):
        ZigZag then runs once per shape. Returns `(design key, {(shape, key): cme}, area)`, with `None` cells
        when the design fails to map some shape after all."""
        design_key = core_dict_key(core_dict)
        workload = pickle_deepcopy(self.workload)
        node_core_dicts = dict.fromkeys(self.core_ids_of_node, core_dict)
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
            return design_key, dict(self.cme_cache), self.area_cache.get(design_key, math.inf)
        except Exception:
            logger.warning(f"GraphEvolutionStage: design {design_key} failed its cost-model warm-up.", exc_info=True)
            return design_key, None, math.inf
        finally:
            if os.path.exists(lut_path):
                os.remove(lut_path)

    def _warm_up_designs(self, genomes: list[GraphGenome]) -> None:
        """Cost-model, in parallel, every design these genomes use that the cache has not seen yet."""
        pending: dict[str, dict[str, Any]] = {}
        for genome in genomes:
            for values in genome.designs.values():
                core_dict = self._decode(values)
                if core_dict is None:
                    continue
                key = core_dict_key(core_dict)
                if key not in self.area_cache and key not in self.infeasible_designs:
                    pending[key] = core_dict
        if not pending:
            return
        logger.info(f"GraphEvolutionStage: cost-modelling {len(pending)} new design(s).")
        results = self._parallel(_warm_up_worker, list(pending.values()), fallback=(None, None, math.inf))
        for (key, cells, area), expected_key in zip(results, pending, strict=True):
            if cells is None or key is None:
                self.infeasible_designs.add(expected_key)
                continue
            shapes = {shape for shape in self.shape_class_of_tile.values()}
            if {shape for shape, _ in cells} != shapes:
                self.infeasible_designs.add(key)
                continue
            for cell_key, cme in cells.items():
                self.cme_cache[cell_key] = cme
                self.runtime_cache[cell_key] = float(getattr(cme, self.latency_attr))
                self.energy_cache[cell_key] = float(cme.energy_total)
            self.area_cache[key] = area

    # ------------------------------------------------------------------------------------ measuring ----------

    def _assemble(self, genome: GraphGenome, workload: Any, yaml_path: str | None = None):
        """Fold the unrolled topology by the genome's assignment, give each node its design, drop its links."""
        topology = derive_rolled_topology(workload, self.original_workload, genome.assign)
        label_of_core = {
            rolled_id: genome.assign[members[0]] for rolled_id, members in topology.members_of_core.items()
        }
        core_dicts = {
            rolled_id: self._decode(genome.designs[label], name=f"graph_core_{label}")
            for rolled_id, label in label_of_core.items()
        }
        cores: dict[int, Any] = {rolled_id: core_dicts[rolled_id] for rolled_id in range(topology.offchip_core_id)}
        cores[topology.offchip_core_id] = load_offchip_core_data()
        links = [
            link
            for link in topology.link_connections
            if frozenset(label_of_core[core] for core in link["cores"]) not in genome.dropped_links
        ]
        accelerator_data = {
            "name": f"{self.accelerator.name}_graph",
            "cores": cores,
            "offchip_core_id": topology.offchip_core_id,
            "unit_energy_cost": 0,
            "core_memory_sharing": [],
            "core_connectivity": [topology.bus_connection, *links],
        }
        if yaml_path:
            with open(yaml_path, "w") as f:
                yaml.safe_dump(accelerator_data, f, sort_keys=False)
        design_key_of_core = {rolled_id: core_dict_key(core_dict) for rolled_id, core_dict in core_dicts.items()}
        return AcceleratorFactory(accelerator_data).create(), design_key_of_core

    def _measure_genome(self, genome: GraphGenome, save_yaml: bool = False):
        workload = pickle_deepcopy(self.workload)
        yaml_path = os.path.join(self.graph_search_dir, "accelerator.yaml") if save_yaml else None
        accelerator, design_key_of_core = self._assemble(genome, workload, yaml_path)
        cost_lut = self._assemble_cost_lut(workload, accelerator, design_key_of_core)
        return self._measure_accelerator(workload, accelerator, cost_lut)

    def _area(self, genome: GraphGenome) -> float:
        return sum(self.area_cache[self._key_of(values)] for values in genome.designs.values())

    def _is_feasible(self, genome: GraphGenome) -> bool:
        return all(
            (key := self._key_of(values)) in self.area_cache and key not in self.infeasible_designs
            for values in genome.designs.values()
        )

    def _canonical_key(self, genome: GraphGenome) -> tuple:
        """Identity of the accelerator a genome describes, independent of how its labels are numbered."""
        relabel: dict[int, int] = {}
        for core in self.unrolled_core_ids:
            relabel.setdefault(genome.assign[core], len(relabel))
        return (
            tuple(relabel[genome.assign[core]] for core in self.unrolled_core_ids),
            tuple(self._key_of(genome.designs[label]) for label in sorted(relabel, key=relabel.get)),
            tuple(sorted(tuple(sorted(relabel[label] for label in pair)) for pair in genome.dropped_links)),
        )

    def _parallel(self, func, items: list[Any], fallback: Any, on_result=None) -> list[Any]:
        global _WORKER_STAGE  # noqa: PLW0603
        _WORKER_STAGE = self
        mp_context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=mp_context) as executor:
            return map_until_deadline(executor, func, items, self.search_budget, fallback, on_result=on_result)

    def _evaluate(self, genomes: list[GraphGenome]) -> None:
        """Set every genome's fitness: warm up new designs, then schedule each new graph once (cached by key)."""
        self._warm_up_designs(genomes)
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
            fitness = (latency, energy, area) if math.isfinite(latency) else INF3
            self.fitness_of_key[keys[index]] = fitness
            self.evaluated.append({"nb_cores": genome.nb_cores, "latency": latency, "energy": energy, "area": area})
            if budget is not None:
                budget.record(latency, energy, area, info=f"graph cores={genome.nb_cores}")

        if keys:
            self._parallel(
                _measure_worker, [to_measure[key] for key in keys], fallback=(math.inf, math.inf), on_result=on_result
            )
        for genome in genomes:
            genome.fitness.values = self.fitness_of_key.get(self._canonical_key(genome), INF3)

    # ------------------------------------------------------------------------------------ operators ----------

    def _random_genome(self, design_pool: list[list[float]]) -> GraphGenome:
        """Spread the initial population over core counts: fully unrolled, a single shared core, and random
        partitions in between."""
        n_unrolled = len(self.unrolled_core_ids)
        nb_nodes = random.choice([n_unrolled, 1, random.randint(1, n_unrolled)])
        assign = {
            core: (core if nb_nodes == n_unrolled else random.randrange(nb_nodes)) for core in self.unrolled_core_ids
        }
        designs = {label: list(random.choice(design_pool)) for label in set(assign.values())}
        return GraphGenome(designs=designs, assign=assign).prune()

    def _mutate(self, genome: GraphGenome) -> GraphGenome:
        operators = [
            self._mut_reassign,
            self._mut_split,
            self._mut_merge,
            self._mut_design,
            self._mut_share_design,
            self._mut_toggle_link,
        ]
        for _ in range(random.choice((1, 1, 2))):
            random.choice(operators)(genome)
            genome.prune()
        return genome

    def _mut_reassign(self, genome: GraphGenome) -> None:
        core = random.choice(self.unrolled_core_ids)
        if random.random() < NEW_NODE_PROBABILITY:
            label = genome.new_label()
            genome.designs[label] = self._perturbed(genome.designs[genome.assign[core]])
        else:
            label = random.choice(list(genome.designs))
        genome.assign[core] = label

    def _mut_split(self, genome: GraphGenome) -> None:
        members: dict[int, list[int]] = {}
        for core, label in genome.assign.items():
            members.setdefault(label, []).append(core)
        splittable = [label for label, cores in members.items() if len(cores) > 1]
        if not splittable:
            return
        label = random.choice(splittable)
        moved = random.sample(members[label], k=random.randint(1, len(members[label]) - 1))
        new = genome.new_label()
        genome.designs[new] = self._perturbed(genome.designs[label])
        for core in moved:
            genome.assign[core] = new

    def _mut_merge(self, genome: GraphGenome) -> None:
        if len(genome.designs) < PAIR:
            return
        keep, drop = random.sample(list(genome.designs), PAIR)
        for core, label in genome.assign.items():
            if label == drop:
                genome.assign[core] = keep

    def _mut_design(self, genome: GraphGenome) -> None:
        label = random.choice(list(genome.designs))
        genome.designs[label] = self._perturbed(genome.designs[label])

    def _mut_share_design(self, genome: GraphGenome) -> None:
        if len(genome.designs) < PAIR:
            return
        source, target = random.sample(list(genome.designs), PAIR)
        genome.designs[target] = list(genome.designs[source])

    def _mut_toggle_link(self, genome: GraphGenome) -> None:
        if len(genome.designs) < PAIR:
            return
        pair = frozenset(random.sample(list(genome.designs), PAIR))
        genome.dropped_links ^= {pair}

    def _crossover(self, first: GraphGenome, second: GraphGenome) -> tuple[GraphGenome, GraphGenome]:
        """Graft the assignment of a random subset of layers (with the nodes and designs hosting them) from one
        parent onto the other, in both directions."""
        layers = list(self.core_ids_of_node)
        graft = set(random.sample(layers, k=random.randint(1, max(1, len(layers) // 2))))
        return self._graft(first, second, graft), self._graft(second, first, graft)

    def _graft(self, receiver: GraphGenome, donor: GraphGenome, layers: set[int]) -> GraphGenome:
        child = GraphGenome(
            designs={label: list(values) for label, values in receiver.designs.items()},
            assign=dict(receiver.assign),
            dropped_links=set(receiver.dropped_links),
        )
        offset = child.new_label()
        for layer in layers:
            for core in self.core_ids_of_node[layer]:
                donor_label = donor.assign[core]
                child.assign[core] = offset + donor_label
                child.designs[offset + donor_label] = list(donor.designs[donor_label])
        return child.prune()

    # ------------------------------------------------------------------------------------------ loop ---------

    def _should_stop(self, generation: int) -> bool:
        if self.search_budget is not None:
            return self.search_budget.should_stop()
        return generation >= self.graph_ga_generations

    def _clone(self, genome: GraphGenome) -> GraphGenome:
        clone = copy.deepcopy(genome)
        del clone.fitness.values
        return clone

    def _evolve(self) -> tuple[list[GraphGenome], tools.ParetoFront]:
        mu = self.population_size
        design_pool = [self._random_design() for _ in range(mu)]
        population = [self._random_genome(design_pool) for _ in range(mu)]
        self._evaluate(population)
        population = tools.selNSGA2(population, mu)
        front = tools.ParetoFront(similar=lambda a, b: self._canonical_key(a) == self._canonical_key(b))
        front.update([genome for genome in population if math.isfinite(genome.fitness.values[0])])

        generation = 0
        while not self._should_stop(generation):
            generation += 1
            offspring = [self._clone(genome) for genome in tools.selTournamentDCD(population, mu)]
            children: list[GraphGenome] = []
            for parent_1, parent_2 in zip(offspring[::2], offspring[1::2], strict=True):
                pair = (
                    self._crossover(parent_1, parent_2)
                    if random.random() < self.crossover_probability
                    else (parent_1, parent_2)
                )
                children += [self._mutate(child) for child in pair]
            self._evaluate(children)
            population = tools.selNSGA2(population + children, mu)
            front.update([genome for genome in children if math.isfinite(genome.fitness.values[0])])
            best_edp = min((g.fitness.values[0] * g.fitness.values[1] for g in front), default=math.inf)
            logger.info(
                f"GraphEvolutionStage: generation {generation}: {len(self.fitness_of_key)} distinct graph(s), "
                f"{len(self.area_cache)} design(s) cost-modelled, front={len(front)}, best EDP={best_edp:.4g}, "
                f"core counts on front={sorted({g.nb_cores for g in front})}."
            )
        return population, front

    def _save(self, front: tools.ParetoFront) -> None:
        with open(os.path.join(self.graph_search_dir, "evaluated.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["nb_cores", "latency", "energy", "area"])
            writer.writeheader()
            writer.writerows(self.evaluated)
        with open(os.path.join(self.graph_search_dir, "pareto.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nb_cores", "latency", "energy", "area"])
            for genome in front:
                writer.writerow([genome.nb_cores, *genome.fitness.values])

    def run(self):
        self._prepare()
        _population, front = self._evolve()
        self._save(front)
        if not len(front):
            raise ValueError("GraphEvolutionStage: no graph could be scheduled.")
        best = min(front, key=lambda genome: genome.fitness.values[0] * genome.fitness.values[1])
        logger.info(
            f"GraphEvolutionStage: best graph by EDP has {best.nb_cores} core(s): (latency, energy, area)="
            f"{best.fitness.values}; {len(front)} graph(s) on the front."
        )
        best_scme = self._measure_genome(best, save_yaml=True)
        yield (
            best_scme,
            {
                "pareto_graphs": [
                    {
                        "nb_cores": genome.nb_cores,
                        "latency": genome.fitness.values[0],
                        "energy": genome.fitness.values[1],
                        "area": genome.fitness.values[2],
                    }
                    for genome in front
                ],
                "nb_evaluated": len(self.evaluated),
            },
        )
