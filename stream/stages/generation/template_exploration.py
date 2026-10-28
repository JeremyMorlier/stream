"""NSGA2 over a homogeneous accelerator built from one parameterized core template.

The accelerator is a `rows x cols` mesh of identical copies of a template core (e.g. `edge_tpu_like.yaml`), plus a
shared SIMD core and the off-chip DRAM, laid out as `grid_generator.template_grid_dict`. A genome sets:

- the grid: `rows` and `cols` (the number of PEs along x and y);
- the core: the size of every operational-array dimension (powers of two), and the size of every template memory
  (a log grid around its template size). Each memory's read/write energy and area come from CACTI
  (`auto_cost_extraction`), served from the precomputed lookup (`TemplateGeneLayout.cacti_configs`);
- the inter-core tiling: one `TilingExplorationStage` inter gene per layer (at most one split dimension), its
  factor clamped to the grid's core count.

The intra-core tiling stays the mapping's. Genes are real values in `[0, 1)`, each the index of a discrete choice,
so the core NSGA2 operators (`_make_core_toolbox`, `_run_nsga2`) apply unchanged. Every distinct design is
scheduled once with the stages after this one (tiling generation, ZigZag cost estimation and the measurement
chain) in a forked worker; fitness is `(latency, energy, area)`, area being the compute cores' (SIMD included,
off-chip excluded) plus the interconnect's, as in `TilingExplorationStage`.

The CACTI lookup is keyed on `(mem_type, size, read bandwidth)` with one read and one write port, so a template
memory with more ports (edge_tpu's `rf_2B`) is costed as a 1R/1W one.
"""

import copy
import csv
import logging
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import yaml
from zigzag.utils import pickle_deepcopy

from stream.hardware.architecture.core_generator import _align_to_byte
from stream.hardware.architecture.grid_generator import template_grid_dict
from stream.hardware.architecture.noc.interconnect_area import InterconnectAreaModel
from stream.opt.search_budget import SearchBudget, map_recording
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.stages.generation.core_architecture_exploration import (
    SIZE_GRID_STEPS,
    _make_core_toolbox,
    _round_population,
    _run_nsga2,
    _validate_core_dict,
)
from stream.stages.generation.tiled_workload_accelerator_generation import load_offchip_core_data, validate_core_yaml
from stream.stages.generation.tiling_exploration import TilingGene, _candidate_key, build_tiling_genes
from stream.stages.stage import MainStage, Stage, StageCallable
from stream.visualization.design_record import design_record
from stream.workload.computation.computation_node import ComputationNode

logger = logging.getLogger(__name__)

INF3 = (math.inf, math.inf, math.inf)
CORES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "inputs", "examples", "hardware", "cores")
SIMD_CORE_YAML_PATH = os.path.normpath(os.path.join(CORES_DIR, "simd.yaml"))

# The stage a forked worker evaluates against, set right before the pool is created (as in `graph_evolution`).
_WORKER_STAGE: "TemplateExplorationStage | None" = None

# The fewest rows (memory size over port width) CACTI builds an SRAM with: 16 rows fail, 32 work.
CACTI_MIN_SRAM_ROWS = 32


@dataclass
class TemplateParamRanges:
    """What the template search may change. A memory's sizes are `SIZE_GRID_STEPS` log-spaced, byte-aligned values
    from `template size / memory_shrink` to `template size * memory_growth`; an array dimension's are the powers of
    two from 1 to `template size * oa_growth`."""

    max_rows: int = 4
    max_cols: int = 4
    oa_growth: int = 4
    memory_shrink: int = 16
    memory_growth: int = 4


@dataclass(frozen=True)
class TemplateDesign:
    """A decoded genome. `inter` holds, per inter gene, the chosen candidate index after clamping to the grid."""

    rows: int
    cols: int
    oa_sizes: tuple[int, ...]
    memory_sizes: tuple[int, ...]
    inter: tuple[int, ...]

    @property
    def nb_cores(self) -> int:
        return self.rows * self.cols

    @property
    def key(self) -> str:
        return _candidate_key([self.rows, self.cols, *self.oa_sizes, *self.memory_sizes, *self.inter])


@dataclass
class TemplateGeneLayout:
    """The genome of a template search: grid, array sizes, memory sizes, then the inter-core tiling genes."""

    template: dict[str, Any]
    inter_genes: list[TilingGene]
    ranges: TemplateParamRanges = field(default_factory=TemplateParamRanges)

    def __post_init__(self):
        array = self.template["operational_array"]
        self.oa_dims: list[str] = list(array["dimensions"])
        self.memory_names: list[str] = list(self.template["memories"])
        self.choices: list[list[Any]] = [
            list(range(1, self.ranges.max_rows + 1)),
            list(range(1, self.ranges.max_cols + 1)),
        ]
        for size in array["sizes"]:
            top = size * self.ranges.oa_growth
            self.choices.append([2**k for k in range(int(math.log2(top)) + 1)])
        for name in self.memory_names:
            self.choices.append(self._memory_sizes(self.template["memories"][name]))
        self.choices += [list(range(len(gene.candidates))) for gene in self.inter_genes]
        self.low = [0.0] * len(self.choices)
        self.high = [1.0] * len(self.choices)

    @property
    def memory_choices(self) -> list[list[int]]:
        """The size choices of each template memory, in `memory_names` order."""
        offset = 2 + len(self.oa_dims)
        return self.choices[offset : offset + len(self.memory_names)]

    @staticmethod
    def _read_bandwidth(memory: dict[str, Any]) -> int:
        """The bandwidth `MemoryInstance` hands CACTI: its first read (else read-write) port's maximum."""
        ports = memory["ports"]
        port = next((p for p in ports if p["type"] == "read"), None) or next(
            p for p in ports if p["type"] == "read_write"
        )
        return int(port["bandwidth_max"])

    def _memory_sizes(self, memory: dict[str, Any]) -> list[int]:
        size = memory["size"]
        bandwidth = max(int(port["bandwidth_max"]) for port in memory["ports"])
        # CACTI cannot build an SRAM of fewer than `CACTI_MIN_SRAM_ROWS` rows of its port width.
        floor = max(8, bandwidth * (CACTI_MIN_SRAM_ROWS if memory["mem_type"] == "sram" else 1))
        low = max(size // self.ranges.memory_shrink, floor)
        high = max(size * self.ranges.memory_growth, low)
        if high == low:
            return [_align_to_byte(low, min_size=floor)]
        steps = [low * (high / low) ** (step / (SIZE_GRID_STEPS - 1)) for step in range(SIZE_GRID_STEPS)]
        return sorted({_align_to_byte(round(value), min_size=floor) for value in steps})

    @staticmethod
    def _pick(value: float, choices: list[Any]) -> Any:
        return choices[min(max(int(value * len(choices)), 0), len(choices) - 1)]

    def encode(self, choice_values: list[Any]) -> list[float]:
        """The gene vector at the middle of each choice's bin (choices given by value, e.g. the template's own)."""
        return [
            (choices.index(value) + 0.5) / len(choices)
            for value, choices in zip(choice_values, self.choices, strict=True)
        ]

    def template_values(self, rows: int, cols: int) -> list[Any]:
        """The template core itself on a `rows x cols` grid, with no inter-core split."""
        array = self.template["operational_array"]
        return [
            rows,
            cols,
            *array["sizes"],
            *(
                min(choices, key=lambda size, name=name: abs(size - self.template["memories"][name]["size"]))
                for name, choices in zip(self.memory_names, self.memory_choices, strict=True)
            ),
            *([0] * len(self.inter_genes)),
        ]

    def decode(self, values: list[float]) -> TemplateDesign:
        picked = [self._pick(value, choices) for value, choices in zip(values, self.choices, strict=True)]
        rows, cols = picked[0], picked[1]
        nb_oa = len(self.oa_dims)
        oa_sizes = tuple(picked[2 : 2 + nb_oa])
        memory_sizes = tuple(picked[2 + nb_oa : 2 + nb_oa + len(self.memory_names)])
        inter = tuple(
            self._clamp(gene, index, rows * cols)
            for gene, index in zip(self.inter_genes, picked[2 + nb_oa + len(self.memory_names) :], strict=True)
        )
        return TemplateDesign(rows, cols, oa_sizes, memory_sizes, inter)

    @staticmethod
    def _clamp(gene: TilingGene, index: int, nb_cores: int) -> int:
        """The candidate itself if its factor fits on `nb_cores`, else the largest one of the same dimension that
        does, else no split (index 0)."""
        choice = gene.candidates[index]
        if choice is None or choice[1] <= nb_cores:
            return index
        fitting = [
            (candidate[1], i)
            for i, candidate in enumerate(gene.candidates)
            if candidate is not None and candidate[0] == choice[0] and candidate[1] <= nb_cores
        ]
        return max(fitting)[1] if fitting else 0

    def core_dict(self, design: TemplateDesign, name: str = "template_core") -> dict[str, Any]:
        """The validated core dict of a design: the template with its array and memory sizes replaced."""
        core = copy.deepcopy(self.template)
        core["name"] = name
        core["operational_array"]["sizes"] = list(design.oa_sizes)
        for memory_name, size in zip(self.memory_names, design.memory_sizes, strict=True):
            core["memories"][memory_name]["size"] = size
        return _validate_core_dict(core)

    def cacti_configs(self) -> set[tuple[str, int, int]]:
        """Every `(mem_type, size, r_bw)` a decoded design can hand CACTI, for `precompute_cacti_configs`."""
        configs: set[tuple[str, int, int]] = set()
        for name, sizes in zip(self.memory_names, self.memory_choices, strict=True):
            memory = self.template["memories"][name]
            if not memory.get("auto_cost_extraction"):
                continue
            mem_type = memory.get("mem_type", "sram")
            configs |= {(mem_type, size, self._read_bandwidth(memory)) for size in sizes}
        return configs

    def describe(self, design: TemplateDesign) -> dict[str, Any]:
        """A design's parameters, for the CSV and the design records."""
        return {
            "rows": design.rows,
            "cols": design.cols,
            **{f"oa_{dim}": size for dim, size in zip(self.oa_dims, design.oa_sizes, strict=True)},
            **{f"mem_{name}": size for name, size in zip(self.memory_names, design.memory_sizes, strict=True)},
            "inter_core_tiling": {
                gene.node_id: gene.candidates[index]
                for gene, index in zip(self.inter_genes, design.inter, strict=True)
                if index
            },
        }


def load_template_core(path: str) -> dict[str, Any]:
    """A template core file, as written (validation fills defaults in later, per design)."""
    with open(path) as f:
        return yaml.safe_load(f)


def _measure_worker(design: TemplateDesign) -> tuple[float, float, float, dict[str, Any] | None]:
    assert _WORKER_STAGE is not None
    return _WORKER_STAGE.measure(design)


class TemplateExplorationStage(Stage):
    """NSGA2 over the template grid; see the module docstring. Comes after the workload parser and the layer stacks,
    on a seed accelerator laid out by `template_grid_dict` (compute cores, then the SIMD core, then the off-chip
    core): a layer allocated to the seed's SIMD core stays on the SIMD core, every other layer may use every grid
    core. The stages after this one schedule each design; the best design by EDP is yielded."""

    def __init__(  # noqa: PLR0913
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: Any,
        accelerator: Any,
        layer_stacks: list[tuple[int, ...]],
        template_core_path: str,
        template_search_dir: str,
        template_param_ranges: dict[str, Any] | None = None,
        population_size: int = 16,
        nb_template_generations: int = 10,
        max_workers: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.layer_stacks = layer_stacks
        self.template_search_dir = template_search_dir
        self.population_size = _round_population(population_size)
        self.nb_template_generations = nb_template_generations
        self.max_workers = max_workers
        self.search_budget: SearchBudget | None = kwargs.get("search_budget")
        self.interconnect_model = InterconnectAreaModel.from_config(kwargs.get("interconnect_area_model"))
        self.seed_simd_id: int = accelerator.offchip_core_id - 1
        inter_genes = [gene for gene in build_tiling_genes(workload) if gene.kind == "inter"]
        self.layout = TemplateGeneLayout(
            load_template_core(template_core_path),
            inter_genes,
            TemplateParamRanges(**(template_param_ranges or {})),
        )
        self.simd_core = validate_core_yaml(SIMD_CORE_YAML_PATH)
        self.offchip_core = load_offchip_core_data()
        self.fitness_of_key: dict[str, tuple[float, float, float]] = {}
        self.evaluated: list[dict[str, Any]] = []

    # ------------------------------------------------------------------------------------ measuring ----------

    def build(self, design: TemplateDesign) -> tuple[Any, Any]:
        """A design's accelerator, and the workload allocated and inter-core tiled onto it."""
        accelerator_data = template_grid_dict(
            self.layout.core_dict(design),
            design.rows,
            design.cols,
            name=f"template_grid_{design.rows}x{design.cols}",
            simd_core=self.simd_core,
            offchip_core=self.offchip_core,
        )
        accelerator_data["core_memory_sharing"] = []
        accelerator = AcceleratorFactory(accelerator_data).create()

        workload = pickle_deepcopy(self.workload)
        inter_of_node = {
            gene.node_id: gene.candidates[index]
            for gene, index in zip(self.layout.inter_genes, design.inter, strict=True)
        }
        simd_id = design.nb_cores
        for node in workload.node_list:
            if not isinstance(node, ComputationNode):
                continue
            # Tiles take their allocation from `core_allocation` (the mapping's), the cost estimation from
            # `possible_core_allocation`: both move to the design's grid.
            on_simd = node.possible_core_allocation == [self.seed_simd_id]
            node.core_allocation = [simd_id] if on_simd else list(range(design.nb_cores))
            node.possible_core_allocation = list(node.core_allocation)
            if not on_simd and node.id in inter_of_node:
                choice = inter_of_node[node.id]
                # No split: an empty tiling, which `TilingGenerationStage` turns into a factor of 1.
                node.inter_core_tiling = [] if choice is None else [choice]
        return accelerator, workload

    def area(self, accelerator: Any) -> float:
        return (
            sum(core.get_area() for core in accelerator.core_list if core.id != accelerator.offchip_core_id)
            + accelerator.interconnect_area(self.interconnect_model).total
        )

    def schedule(self, design: TemplateDesign) -> tuple[Any, float]:
        """Run the downstream stages on a design. Returns its schedule and its area."""
        accelerator, workload = self.build(design)
        design_dir = os.path.join(self.template_search_dir, "designs", design.key)
        os.makedirs(design_dir, exist_ok=True)
        kwargs = self.kwargs.copy()
        kwargs.update(
            workload=workload,
            accelerator=accelerator,
            layer_stacks=self.layer_stacks,
            tiled_workload_path=os.path.join(design_dir, "tiled_workload.pickle"),
            cost_lut_path=os.path.join(design_dir, "cost_lut.pickle"),
        )
        scme, _ = MainStage(self.list_of_callables, **kwargs).run()[0]
        return scme, self.area(accelerator)

    def measure(self, design: TemplateDesign) -> tuple[float, float, float, dict[str, Any] | None]:
        """Runs in a worker: `(latency, energy, area, design record or None)`, infinite if it cannot be scheduled."""
        try:
            scme, area = self.schedule(design)
        except Exception:
            logger.warning(f"TemplateExplorationStage: design {design.key} could not be scheduled.", exc_info=True)
            return (*INF3, None)
        latency, energy = float(scme.latency), float(scme.energy)
        artifact = None
        budget = self.search_budget
        if budget is not None and budget.artifact_dir and not budget.is_dominated(latency, energy, area):
            try:
                artifact = design_record(scme, template=self.layout.describe(design))
            except Exception:
                logger.warning("TemplateExplorationStage: could not build a design record.", exc_info=True)
        return latency, energy, area, artifact

    def _map(self, executor: ProcessPoolExecutor):
        """The toolbox's `map`: decode, measure each new distinct design once in the pool (recording it on the
        budget as it completes), and hand NSGA2 the cached fitnesses."""

        def _template_map(_evaluate, individuals):
            designs = [self.layout.decode(list(ind)) for ind in individuals]
            new: dict[str, TemplateDesign] = {}
            for design in designs:
                if design.key not in self.fitness_of_key:
                    new.setdefault(design.key, design)
            keys = list(new)

            def on_result(index: int, result: tuple[float, float, float, dict[str, Any] | None]) -> None:
                design = new[keys[index]]
                latency, energy, area, artifact = result
                fitness = (latency, energy, area) if math.isfinite(latency) else INF3
                self.fitness_of_key[design.key] = fitness
                self.evaluated.append(
                    {
                        "key": design.key,
                        **self.layout.describe(design),
                        **dict(zip(("latency", "energy", "area"), fitness, strict=True)),
                    }
                )
                if self.search_budget is not None:
                    self.search_budget.record(
                        latency, energy, area, info=f"template {design.rows}x{design.cols}", artifact=artifact
                    )

            if keys:
                map_recording(executor, _measure_worker, [new[key] for key in keys], (*INF3, None), on_result)
            return [self.fitness_of_key[design.key] for design in designs]

        return _template_map

    # ------------------------------------------------------------------------------------------ loop ---------

    def run(self):
        global _WORKER_STAGE  # noqa: PLW0603
        os.makedirs(self.template_search_dir, exist_ok=True)
        logger.info(
            f"TemplateExplorationStage: {len(self.layout.choices)} gene(s) "
            f"({len(self.layout.inter_genes)} inter-core tiling), population {self.population_size}."
        )
        toolbox = _make_core_toolbox(self.layout)
        # The template itself on a 2x2 grid seeds the population.
        seed = self.layout.encode(
            self.layout.template_values(min(2, self.layout.ranges.max_rows), min(2, self.layout.ranges.max_cols))
        )
        _WORKER_STAGE = self
        mp_context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=mp_context) as executor:
            toolbox.register("evaluate", _measure_worker)  # unused: `_template_map` measures designs itself
            toolbox.register("map", self._map(executor))
            nb_generations = self.nb_template_generations if self.search_budget is None else 10**9
            front, _history = _run_nsga2(
                toolbox,
                self.population_size,
                label="template search",
                history_index=0,
                nb_generations=nb_generations,
                nb_points=10**9,
                search_budget=self.search_budget,
                initial=[seed],
            )
        self._save()
        front = [ind for ind in front if math.isfinite(ind.fitness.values[0])]
        if not front:
            raise ValueError("TemplateExplorationStage: no design could be scheduled.")
        best = min(front, key=lambda ind: ind.fitness.values[0] * ind.fitness.values[1])
        best_design = self.layout.decode(list(best))
        logger.info(
            f"TemplateExplorationStage: best design by EDP {self.layout.describe(best_design)}: "
            f"(latency, energy, area)={best.fitness.values}; {len(front)} design(s) on the front."
        )
        best_scme, _area = self.schedule(best_design)
        yield (
            best_scme,
            {
                "pareto_designs": [
                    {
                        **self.layout.describe(self.layout.decode(list(ind))),
                        **dict(zip(("latency", "energy", "area"), ind.fitness.values, strict=True)),
                    }
                    for ind in front
                ],
                "nb_evaluated": len(self.evaluated),
            },
        )

    def _save(self) -> None:
        if not self.evaluated:
            return
        path = os.path.join(self.template_search_dir, "evaluated_designs.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.evaluated[0]))
            writer.writeheader()
            writer.writerows(self.evaluated)
        logger.info(f"TemplateExplorationStage: {len(self.evaluated)} design(s) written to {path}.")
