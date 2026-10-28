"""Per-tile core-architecture search: one NSGA2 Pareto front of core designs per unique tile shape, and the
stages that turn those fronts into an accelerator.

- `CoreArchitectureExplorationStage` searches the fronts (or reuses saved ones) and hands them down as
  `core_pareto_fronts`.
- `CoreDesignSelectionStage` collapses each front to its single best design, and
  `CustomCoreAcceleratorGenerationStage` assembles the group-dedicated accelerator from those designs -- the
  single-accelerator path `optimize_tiling` uses.
- The schedule searches (`ScheduleExplorationStage` and the stages after it) instead keep every design and
  search their combinations.
"""

import array
import copy
import csv
import functools
import itertools
import logging
import math
import multiprocessing
import os
import random
import time as _time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import yaml
from deap import base, creator, tools
from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.utils import pickle_deepcopy, pickle_load, pickle_save

from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core_generator import (
    OPERANDS,
    build_core_from_dict,
    build_rf_memory_dict,
    build_shared_memory_dict,
    core_dict_from_params,
)
from stream.opt.search_budget import SearchBudget, hypervolume_3d
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.core_validator import CoreValidator
from stream.stages.estimation.zigzag_core_mapping_estimation import (
    evaluate_node_on_core,
    lowest_memory_level_violations,
)
from stream.stages.generation.schedule_evaluation import downstream_requires
from stream.stages.generation.tiled_workload_accelerator_generation import (
    RolledTopology,
    derive_group_dedicated_topology,
    derive_rolled_topology,
    get_original_nodes,
    get_tiles_by_original_node,
    load_offchip_core_data,
    with_link_widths,
)
from stream.stages.generation.tiling_exploration import _candidate_key
from stream.stages.stage import Stage, StageCallable
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload, ONNXWorkload

logger = logging.getLogger(__name__)


@dataclass
class CoreParamRanges:
    """Search ranges for the per-tile core-architecture GA, mirroring `core_generator.random_core_dict`'s
    tunable knobs (same names) so the GA searches the parameter space that generator can randomize, just
    deliberately (via NSGA2) instead of uniformly at random. Cost ranges (`*_cost_range`) are
    intentionally absent: memories always use `auto_cost_extraction` now, so read/write cost and area are
    inferred by ZigZag/CACTI rather than chosen.

    The memory floors are low enough for a core well under 1 mm^2 (CACTI puts a 1 MiB memory at about 2.4 mm^2),
    and the levels of `optional_levels` ("rf", "sram", "upper") may also be left out (see `CoreGeneLayout`)."""

    operands: tuple[str, ...] = OPERANDS
    oa_dims: tuple[str, ...] = ("D1", "D2", "D3")
    oa_size_ranges: tuple[tuple[int, int], ...] = ((2, 32), (2, 32), (1, 8))
    rf_size_range: tuple[int, int] = (8, 1024)
    rf_bandwidth_choices: tuple[int, ...] = (16, 32, 64, 128)
    sram_size_range: tuple[int, int] = (2**16, 2**24)
    sram_bandwidth_choices: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)
    upper_size_range: tuple[int, int] = (2**20, 2**27)
    upper_bandwidth_choices: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    unit_energy_range: tuple[float, float] = (0.01, 0.1)
    optional_levels: tuple[str, ...] = ("upper",)


def _snap(value: float, choices: tuple[int, ...]) -> int:
    """Round a continuous gene value to the nearest allowed discrete choice."""
    return min(choices, key=lambda choice: abs(choice - value))


SIZE_GRID_STEPS = 24


def _size_grid_values(size_range: tuple[int, int]) -> list[int]:
    """All `SIZE_GRID_STEPS` log-spaced, byte-aligned grid points `_snap_size_to_grid` can snap a nonzero value
    onto for `size_range` (memory sizes span orders of magnitude -- RF in the hundreds of bits, SRAM/upper in
    the millions -- so a *linear* grid would either waste steps at the low end or be too coarse at the high
    end). Does not include the 0/omit sentinel `_snap_size_to_grid` also returns for small values. Small ranges
    (e.g. `rf_size_range`) can collapse many steps onto the same byte-aligned value -- callers that need the
    distinct set should dedupe (e.g. via `set(...)`)."""
    low, high = size_range
    low = max(low, 1)
    if high <= low:
        return [max(8, round(low / 8) * 8)]
    log_low, log_high = math.log2(low), math.log2(high)
    return [
        max(8, round(2 ** (log_low + step * (log_high - log_low) / (SIZE_GRID_STEPS - 1)) / 8) * 8)
        for step in range(SIZE_GRID_STEPS)
    ]


def _snap_size_to_grid(value: float, size_range: tuple[int, int]) -> int:
    """Snap a continuous size gene to one of `_size_grid_values(size_range)`'s grid points.

    This exists purely to keep the GA tractable: `auto_cost_extraction` means every *new* `(size, bandwidth,
    ports, mem_type)` combination triggers a real CACTI circuit-simulator subprocess call (roughly 1-2s); a
    real-valued NSGA2 gene mutated/crossed continuously would produce a virtually unique size per individual,
    so the population would almost never hit CACTI's own result cache. Snapping to a small, bounded set of
    grid points means the population converges onto a handful of distinct memory configs that get reused (and
    cached) across individuals and generations. Values at or below half the range's floor snap to 0, preserving
    the "0 means omit this memory level" convention used throughout `core_generator.py`."""
    low, high = size_range
    if value <= low / 2:
        return 0
    low = max(low, 1)
    value = min(max(value, low), high)
    grid_values = _size_grid_values(size_range)
    if high <= low:
        return grid_values[0]
    log_low, log_high = math.log2(low), math.log2(high)
    step = round((math.log2(value) - log_low) / (log_high - log_low) * (SIZE_GRID_STEPS - 1))
    step = max(0, min(step, SIZE_GRID_STEPS - 1))
    return grid_values[step]


def _size_gene_bounds(size_range: tuple[int, int], optional: bool) -> tuple[float, float]:
    """Bounds of a size gene, which holds log2 of the size: the range spans orders of magnitude, and a linear gene
    would almost never sample its low end. An optional level's gene reaches one unit lower, its omit band."""
    log_low = math.log2(max(size_range[0], 8))
    return (log_low - 1 if optional else log_low, max(log_low, math.log2(max(size_range[1], 1))))


def _decode_size_gene(value: float, size_range: tuple[int, int], optional: bool) -> int:
    """A size gene's memory size: 0 (omit the level) in an optional level's omit band, else the grid point."""
    if optional and value < math.log2(max(size_range[0], 8)) - 0.5:
        return 0
    return _snap_size_to_grid(2**value, size_range)


def _validate_core_dict(core_dict: dict[str, Any]) -> dict[str, Any]:
    """`core_dict_from_params` only builds the schema-*shaped* dict; it never runs Cerberus normalization, so
    fields the schema fills in with defaults are missing until this runs -- exactly like
    `TiledWorkloadAcceleratorGenerationStage.validate_core_yaml` does for its fixed template before handing it
    to `AcceleratorFactory`."""
    validator = CoreValidator(core_dict)
    if not validator.validate():
        raise ValueError(f"Generated core '{core_dict.get('name')}' failed schema validation: {validator.errors}")
    return validator.normalized_data


@dataclass
class CoreGeneLayout:
    """Flattens `CoreParamRanges` into a fixed-length, per-tile real-valued gene segment (bounds + decode), so a
    DEAP real-coded NSGA2 individual directly encodes one tile's core design (unlike the old whole-accelerator
    search, there is exactly one segment per individual now -- the search unit is a single tile). Size genes hold
    log2 of the size (`_size_gene_bounds`)."""

    ranges: CoreParamRanges
    array_dims: list[str] = field(init=False)
    served_choices: list[list[str]] = field(init=False)
    low: list[float] = field(init=False)
    high: list[float] = field(init=False)
    names: list[str] = field(init=False)

    def __post_init__(self):
        ranges = self.ranges
        self.array_dims = list(ranges.oa_dims[:-1])
        self.served_choices = [[], *[[d] for d in self.array_dims], self.array_dims]

        low: list[float] = []
        high: list[float] = []
        names: list[str] = []
        for operand in ranges.operands:
            size_low, size_high = _size_gene_bounds(ranges.rf_size_range, self.is_optional("rf"))
            low += [size_low, min(ranges.rf_bandwidth_choices), 0]
            high += [size_high, max(ranges.rf_bandwidth_choices), len(self.served_choices) - 1]
            names += [f"rf_{operand}_size", f"rf_{operand}_bandwidth", f"rf_{operand}_served"]

        for prefix, size_range, bandwidth_choices in (
            ("sram", ranges.sram_size_range, ranges.sram_bandwidth_choices),
            ("upper", ranges.upper_size_range, ranges.upper_bandwidth_choices),
        ):
            size_low, size_high = _size_gene_bounds(size_range, self.is_optional(prefix))
            low += [size_low, min(bandwidth_choices), min(bandwidth_choices)]
            high += [size_high, max(bandwidth_choices), max(bandwidth_choices)]
            names += [f"{prefix}_size", f"{prefix}_bandwidth_min", f"{prefix}_bandwidth_max"]

        for dim, size_range in zip(ranges.oa_dims, ranges.oa_size_ranges, strict=True):
            low.append(size_range[0])
            high.append(size_range[1])
            names.append(f"oa_{dim}_size")

        low.append(ranges.unit_energy_range[0])
        high.append(ranges.unit_energy_range[1])
        names.append("unit_energy")

        self.low = low
        self.high = high
        self.names = names

    def is_optional(self, level: str) -> bool:
        """Whether the search may leave out memory level `level` ("rf", "sram" or "upper")."""
        size_range = getattr(self.ranges, f"{level}_size_range")
        return level in self.ranges.optional_levels or size_range[0] <= 0

    @property
    def n_genes_per_node(self) -> int:
        """Length of one core's gene segment."""
        return len(self.low)

    def decode_node(self, values: "array.array | list[float] | tuple[float, ...]", name: str) -> dict[str, Any]:
        """Turn one gene segment into a schema-valid core dict. Raises `ValueError` (propagated from
        `core_dict_from_params`/`_validate_core_dict`) if the decoded memories leave some operand with no
        memory level anywhere, or otherwise fail schema validation."""
        ranges = self.ranges
        idx = 0
        rf_memories: dict[str, dict[str, Any] | None] = {}
        for operand in ranges.operands:
            size = _decode_size_gene(values[idx], ranges.rf_size_range, self.is_optional("rf"))
            bandwidth = _snap(values[idx + 1], ranges.rf_bandwidth_choices)
            served_index = max(0, min(int(round(values[idx + 2])), len(self.served_choices) - 1))
            rf_memories[operand] = build_rf_memory_dict(operand, size, bandwidth, self.served_choices[served_index])
            idx += 3

        sram_size = _decode_size_gene(values[idx], ranges.sram_size_range, self.is_optional("sram"))
        sram_bandwidth_min = _snap(values[idx + 1], ranges.sram_bandwidth_choices)
        sram_bandwidth_choices_at_least_min = tuple(b for b in ranges.sram_bandwidth_choices if b >= sram_bandwidth_min)
        sram_bandwidth_max = _snap(
            max(values[idx + 2], sram_bandwidth_min), sram_bandwidth_choices_at_least_min or (sram_bandwidth_min,)
        )
        sram_memory = build_shared_memory_dict(
            list(ranges.operands), sram_size, sram_bandwidth_min, sram_bandwidth_max, self.array_dims
        )
        idx += 3

        upper_size = _decode_size_gene(values[idx], ranges.upper_size_range, self.is_optional("upper"))
        upper_bandwidth_min = _snap(values[idx + 1], ranges.upper_bandwidth_choices)
        upper_bandwidth_choices_at_least_min = tuple(
            b for b in ranges.upper_bandwidth_choices if b >= upper_bandwidth_min
        )
        upper_bandwidth_max = _snap(
            max(values[idx + 2], upper_bandwidth_min), upper_bandwidth_choices_at_least_min or (upper_bandwidth_min,)
        )
        upper_memory = build_shared_memory_dict(
            list(ranges.operands), upper_size, upper_bandwidth_min, upper_bandwidth_max, list(ranges.oa_dims)
        )
        idx += 3

        oa_sizes = [max(1, round(values[idx + i])) for i in range(len(ranges.oa_dims))]
        idx += len(ranges.oa_dims)

        unit_energy = float(values[idx])

        core_dict = core_dict_from_params(
            name,
            operands=ranges.operands,
            oa_dims=ranges.oa_dims,
            oa_sizes=oa_sizes,
            unit_energy=unit_energy,
            rf_memories=rf_memories,
            sram_memory=sram_memory,
            upper_memory=upper_memory,
        )
        return _validate_core_dict(core_dict)


@dataclass
class ParetoCoreSearchResult:
    """Everything the per-tile core search produced, in plain data.

    Handed down the chain as `core_pareto_fronts`: `CoreDesignSelectionStage` keeps one design per tile,
    `ScheduleExplorationStage` searches combinations of all of them."""

    designs_per_tile: list[list[dict[str, Any]]]  # per unique tile: [{"core_dict", "fitness": (l, e, a), "genes"}]
    unique_tiles: list[ComputationNode]
    tile_to_unique_index: dict[int, int]  # id(tile) -> index into `unique_tiles`
    original_nodes: list[ComputationNode]
    tiles_by_original: dict[ComputationNode, list[ComputationNode]]


class CustomCoreAcceleratorGenerationStage(Stage):
    """Builds an accelerator with the same group-dedicated topology as `TiledWorkloadAcceleratorGenerationStage`
    (see `derive_group_dedicated_topology`), but instead of copying one fixed core template to every dedicated
    core, each original workload node's dedicated cores get their own core dict from `node_core_dicts`."""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ComputationNodeWorkload,
        original_workload: ONNXWorkload,
        accelerator: Accelerator,
        node_core_dicts: dict[int, dict[str, Any]],
        tiled_workload_path: str,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.original_workload = original_workload
        self.accelerator = accelerator
        self.node_core_dicts = node_core_dicts
        self.tiled_workload_path = tiled_workload_path

    def run(self):
        kwargs = self.kwargs.copy()
        kwargs["workload"] = self.workload
        kwargs["original_workload"] = self.original_workload
        kwargs["accelerator"] = self.build_custom_accelerator()
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        yield from sub_stage.run()

    def build_custom_accelerator(self, save_yaml: bool = True) -> Accelerator:
        """Assemble the accelerator (pinning every tile to its core), saving it to `accelerator.yaml` next to the
        tiled workload unless `save_yaml` is False."""
        return assemble_custom_accelerator(
            self.workload,
            self.original_workload,
            self.accelerator,
            self.node_core_dicts,
            yaml_path=os.path.join(os.path.dirname(self.tiled_workload_path), "accelerator.yaml")
            if save_yaml
            else None,
        )


def assemble_custom_accelerator(
    workload: ComputationNodeWorkload,
    original_workload: ONNXWorkload,
    accelerator: Accelerator,
    node_core_dicts: dict[int, dict[str, Any]],
    yaml_path: str | None = None,
    link_widths: dict[tuple[int, int], int] | None = None,
    timed_widths: bool = False,
) -> Accelerator:
    """Assemble the group-dedicated accelerator that gives each original node's cores its own design, optionally
    dumping the assembled description to `yaml_path`. `link_widths` (unordered core pair -> bits) are the links'
    physical widths, for the interconnect area and, with `timed_widths`, for timing too (see `with_link_widths`).

    A plain function rather than only a stage method because callers that build *many* accelerators without
    running a sub-pipeline (see `ScheduleExplorationStage`) have no list of callables to hand a stage. Note it
    pins every tile in `workload` to its core as a side effect of `derive_group_dedicated_topology`."""
    topology = derive_group_dedicated_topology(workload, original_workload)
    offchip_core_id = topology.offchip_core_id

    cores: dict[int, dict[str, Any]] = {}
    for original_node in topology.original_nodes:
        core_dict = node_core_dicts[original_node.id]
        for core_id in topology.node_core_ids[original_node.id]:
            cores[core_id] = core_dict
    cores[offchip_core_id] = load_offchip_core_data()

    accelerator_data = {
        "name": f"{accelerator.name}_custom",
        "cores": cores,
        "offchip_core_id": offchip_core_id,
        "unit_energy_cost": 0,
        "core_memory_sharing": [],
        "core_connectivity": [
            topology.bus_connection,
            *with_link_widths(topology.link_connections, link_widths, timed_widths),
        ],
    }
    if yaml_path:
        with open(yaml_path, "w") as f:
            yaml.safe_dump(accelerator_data, f, sort_keys=False)
        logger.info(f"Saved custom-core accelerator to {yaml_path}.")
    return AcceleratorFactory(accelerator_data).create()


def assemble_rolled_accelerator(
    workload: ComputationNodeWorkload,
    original_workload: ONNXWorkload,
    accelerator: Accelerator,
    merge_of_core: dict[int, Any],
    core_dict_of_core: dict[int, dict[str, Any]],
    yaml_path: str | None = None,
    link_widths: dict[tuple[int, int], int] | None = None,
    timed_widths: bool = False,
) -> tuple[Accelerator, "RolledTopology"]:
    """Assemble a *rolled* accelerator: the group-dedicated topology folded onto fewer physical cores by
    `derive_rolled_topology`, each rolled core getting the design named in `core_dict_of_core` (keyed by rolled
    core id, since a rolled core can host several layers and so is no longer identified by a node).

    Returns the topology alongside the accelerator -- the caller needs `members_of_core`/`hosted_node_groups`
    for its artifacts and topology figures, and re-deriving it would re-pin the workload a second time.

    Note this must be used *instead of* `assemble_custom_accelerator` on the rolled path: that one re-derives
    the unrolled topology and would silently re-pin every tile back to an unrolled core id.

    `link_widths` are the *rolled* links' physical widths, timed at with `timed_widths` (see `with_link_widths`);
    the returned topology's `link_connections` carry them.
    """
    topology = derive_rolled_topology(workload, original_workload, merge_of_core)
    topology.link_connections = with_link_widths(topology.link_connections, link_widths, timed_widths)
    offchip_core_id = topology.offchip_core_id

    # Insertion order matters: `AcceleratorFactory.create` walks this dict and `create_core_graph` asserts that
    # each core's id equals its index, so build it densely in ascending id order with offchip last.
    cores: dict[int, dict[str, Any]] = {core_id: core_dict_of_core[core_id] for core_id in range(offchip_core_id)}
    cores[offchip_core_id] = load_offchip_core_data()

    accelerator_data = {
        "name": f"{accelerator.name}_rolled",
        "cores": cores,
        "offchip_core_id": offchip_core_id,
        "unit_energy_cost": 0,
        "core_memory_sharing": [],
        "core_connectivity": [topology.bus_connection, *topology.link_connections],
    }
    if yaml_path:
        with open(yaml_path, "w") as f:
            yaml.safe_dump(accelerator_data, f, sort_keys=False)
        logger.info(f"Saved rolled accelerator to {yaml_path}.")
    return AcceleratorFactory(accelerator_data).create(), topology


def _evaluate_design_on_tile(
    values: "array.array | tuple[float, ...]",
    gene_layout: CoreGeneLayout,
    tile: ComputationNode,
    name: str,
    loma_lpf_limit: int,
    temporal_mapping_type: TemporalMappingType,
) -> tuple[float, float, float]:
    """Runs in a worker process: decodes one gene vector into a core dict, builds a standalone `Core` from it,
    and evaluates `tile` directly on it via `evaluate_node_on_core` -- no accelerator assembly, no full
    downstream pipeline. Returns `(latency, energy, area)`, or `(inf, inf, inf)` if the decoded core is invalid,
    if CACTI itself rejects the synthesized memory configuration (`auto_cost_extraction` shells out to a real
    CACTI subprocess per novel `(size, bandwidth, ports, mem_type)` combination, which can fail with a
    `ChildProcessError`/`FileNotFoundError` for configurations it considers physically invalid -- not just a
    `ValueError` from our own schema validation), or if ZigZag fails to map the tile onto it. Module-level
    (rather than a method) so it can be pickled and sent to worker processes; reused both as the per-tile GA's
    own fitness function and as the cross-tile evaluation's worker (see `_evaluate_cross_job`)."""
    try:
        core_dict = gene_layout.decode_node(values, name=name)
        core = build_core_from_dict(core_dict, core_id=0)
        # Cores whose innermost memories cannot hold one spatially unrolled operand slice of this tile are
        # ruled out before ZigZag ever sees them: LOMA would find no valid loop ordering at all, so they are
        # worthless as designs, and the check is far cheaper than the mapping search that would reject them.
        # It also keeps them out of the per-tile Pareto fronts, where a design that is fine for its own tile
        # but not for another one later aborts the schedule search's cost-model warm-up.
        violations = lowest_memory_level_violations(tile, core)
        if violations:
            logger.debug(
                f"CoreArchitectureExplorationStage: candidate '{name}' cannot map tile {tile} "
                f"({violations[0]}), penalizing it."
            )
            return (float("inf"), float("inf"), float("inf"))
        # Deep-copy before handing the tile to ZigZag: it may set temporary attributes (spatial mapping, chosen
        # core allocation, ...) on the node it's given, and this same tile object is reused across many
        # evaluations (its own GA, plus every other tile's top-K cross-evaluation).
        node = pickle_deepcopy(tile)
        cme = evaluate_node_on_core(
            node, core, loma_lpf_limit=loma_lpf_limit, temporal_mapping_type=temporal_mapping_type
        )
    except Exception:
        logger.exception(f"CoreArchitectureExplorationStage: candidate '{name}' failed on tile {tile}, penalizing it.")
        return (float("inf"), float("inf"), float("inf"))

    return (cme.latency_total2, cme.energy_total, core.get_area())


def _evaluate_cross_job(
    job: tuple[tuple[float, ...], int],
    gene_layout: CoreGeneLayout,
    unique_tiles: list[ComputationNode],
    loma_lpf_limit: int,
    temporal_mapping_type: TemporalMappingType,
) -> tuple[float, float, float]:
    """Cross-tile evaluation worker: unpacks one `(design_values, target_tile_index)` job and delegates to
    `_evaluate_design_on_tile`."""
    values, target_index = job
    return _evaluate_design_on_tile(
        values,
        gene_layout,
        unique_tiles[target_index],
        name=f"cross_tile_{target_index}",
        loma_lpf_limit=loma_lpf_limit,
        temporal_mapping_type=temporal_mapping_type,
    )


def _evaluate_design_on_tiles(
    values: "array.array | tuple[float, ...]",
    gene_layout: CoreGeneLayout,
    tiles_and_counts: list[tuple[ComputationNode, int]],
    name: str,
    loma_lpf_limit: int,
    temporal_mapping_type: TemporalMappingType,
) -> tuple[float, float, float]:
    """Fitness of one design that must run *several* tile shapes, as a folded core does: `(sum of count x
    latency, sum of count x energy, area)` -- the same weighting `ScheduleRollingStage._merged_design_key`
    picks merged designs with. A design that cannot run one of the shapes is infeasible for all of them."""
    total_latency = total_energy = 0.0
    area = float("inf")
    for tile_index, (tile, count) in enumerate(tiles_and_counts):
        latency, energy, area = _evaluate_design_on_tile(
            values, gene_layout, tile, f"{name}_tile_{tile_index}", loma_lpf_limit, temporal_mapping_type
        )
        if not math.isfinite(latency):
            return (float("inf"), float("inf"), float("inf"))
        total_latency += count * latency
        total_energy += count * energy
    return (total_latency, total_energy, area)


def _make_core_toolbox(gene_layout: CoreGeneLayout) -> base.Toolbox:
    """The NSGA2 operators over a core gene vector, without `evaluate` or `map` (both depend on the search)."""
    low_bounds, up_bounds = gene_layout.low, gene_layout.high
    if not hasattr(creator, "CoreTileFitness"):
        creator.create("CoreTileFitness", base.Fitness, weights=(-1.0, -1.0, -1.0))
    if not hasattr(creator, "CoreTileIndividual"):
        creator.create("CoreTileIndividual", array.array, typecode="d", fitness=creator.CoreTileFitness)

    def _random_individual():
        return creator.CoreTileIndividual(random.uniform(lo, hi) for lo, hi in zip(low_bounds, up_bounds, strict=True))

    toolbox = base.Toolbox()
    toolbox.register("individual", _random_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=low_bounds, up=up_bounds)
    toolbox.register(
        "mutate", tools.mutPolynomialBounded, eta=20.0, low=low_bounds, up=up_bounds, indpb=1 / len(low_bounds)
    )
    toolbox.register("select", tools.selNSGA2)
    return toolbox


def _register_dedup_map(toolbox: base.Toolbox, executor: ProcessPoolExecutor) -> None:
    """Evaluate a batch on `executor`, submitting each distinct gene vector once: two individuals with identical
    values landing in the same batch would otherwise re-run CACTI/ZigZag for nothing (same rationale as
    `tiling_exploration._dedup_map`)."""

    def _dedup_map(func, individuals):
        individuals = list(individuals)
        keys = [_candidate_key(ind) for ind in individuals]
        first_index_for_key: dict[str, int] = {}
        unique_individuals = []
        for ind, key in zip(individuals, keys, strict=True):
            if key not in first_index_for_key:
                first_index_for_key[key] = len(unique_individuals)
                unique_individuals.append(ind)
        logger.debug(
            f"CoreArchitectureExplorationStage: batch of {len(individuals)} individual(s), "
            f"{len(unique_individuals)} unique after dedup."
        )
        unique_results = list(executor.map(func, unique_individuals))
        return [unique_results[first_index_for_key[key]] for key in keys]

    toolbox.register("map", _dedup_map)


def _round_population(mu: int) -> int:
    """DEAP's selTournamentDCD requires its population to be a multiple of 4; round up rather than erroring out
    on an otherwise-reasonable population size."""
    if mu % 4 == 0:
        return mu
    rounded_mu = ((mu // 4) + 1) * 4
    logger.warning(
        f"CoreArchitectureExplorationStage: nb_core_ga_individuals={mu} is not a multiple of 4 "
        f"(required by NSGA2's crowded-tournament selection); rounding up to {rounded_mu}."
    )
    return rounded_mu


def _run_nsga2(  # noqa: PLR0913
    toolbox: base.Toolbox,
    mu: int,
    *,
    label: str,
    history_index: int,
    nb_generations: int,
    nb_points: int,
    search_budget: SearchBudget | None = None,
    convergence_patience: int | None = None,
    rel_tol: float = 0.005,
    initial: list[list[float]] | None = None,
) -> tuple[list[Any], list[tuple[Any, ...]]]:
    """One NSGA2 search over core gene vectors with the toolbox's `evaluate`.

    `initial` seeds the first individuals of the population (the rest stay random). Returns up to `nb_points`
    representative individuals from the Pareto front (each with `.fitness.values` set), plus the history of
    every individual evaluated, for `all_evaluated_cores.csv`."""
    history: list[tuple[Any, ...]] = []
    population = toolbox.population(n=mu)
    for slot, values in enumerate((initial or [])[:mu]):
        population[slot] = creator.CoreTileIndividual(values)
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses, strict=True):
        ind.fitness.values = fit
    _record_evaluated_designs(history, history_index, 0, population)
    # Canonical DEAP NSGA2 pattern (see deap/examples/ga/nsga2.py): this initial select assigns
    # crowding-distance info onto `population`, which `selTournamentDCD` needs for the mating pool below.
    population = toolbox.select(population, mu)

    pareto_front = tools.ParetoFront()
    pareto_front.update(population)

    # Convergence (when `convergence_patience` is set): stop once the front's (latency, energy, area)
    # hypervolume has grown by less than `rel_tol` over the last `convergence_patience` generations.
    # The reference point is fixed from the initial population so successive volumes are comparable.
    finite = [ind.fitness.values for ind in population if all(math.isfinite(v) for v in ind.fitness.values)]
    reference = [1.1 * max(values[i] for values in finite) for i in range(3)] if finite else None
    hv_history: list[float] = []

    for generation in range(nb_generations):
        # The budget only keeps a new generation from starting; the one under way always completes.
        if search_budget is not None and search_budget.should_stop():
            logger.info(f"CoreArchitectureExplorationStage: {label} stopped by the search budget.")
            break
        if convergence_patience is not None and reference is not None:
            hv_history.append(hypervolume_3d([ind.fitness.values for ind in pareto_front], reference))
            if len(hv_history) > convergence_patience:
                previous = hv_history[-1 - convergence_patience]
                if previous > 0 and hv_history[-1] / previous - 1 < rel_tol:
                    logger.info(
                        f"CoreArchitectureExplorationStage: {label} front converged after "
                        f"{generation} generation(s) (hypervolume {hv_history[-1]:.4g})."
                    )
                    break
        offspring = tools.selTournamentDCD(population, mu)
        offspring = [toolbox.clone(ind) for ind in offspring]
        for ind1, ind2 in zip(offspring[::2], offspring[1::2], strict=False):
            if random.random() <= 0.9:
                toolbox.mate(ind1, ind2)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values
            del ind2.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses, strict=True):
            ind.fitness.values = fit
        _record_evaluated_designs(history, history_index, generation + 1, invalid_ind)

        population = toolbox.select(population + offspring, mu)
        pareto_front.update(population)
        best_per_objective = [min(ind.fitness.values[i] for ind in pareto_front) for i in range(3)]
        logger.info(
            f"CoreArchitectureExplorationStage: {label} generation {generation}/{nb_generations} (mu={mu}): "
            f"front={len(pareto_front)} pareto point(s), best (latency, energy, area)={tuple(best_per_objective)}."
        )

    logger.info(f"CoreArchitectureExplorationStage: {label} search done ({len(pareto_front)} pareto point(s)).")
    n_points = min(nb_points, len(pareto_front))
    return (tools.selNSGA2(list(pareto_front), n_points) if n_points else []), history


def search_shared_core_front(  # noqa: PLR0913
    tiles_and_counts: list[tuple[ComputationNode, int]],
    *,
    loma_lpf_limit: int,
    temporal_mapping_type: TemporalMappingType,
    nb_generations: int,
    nb_individuals: int,
    nb_points: int,
    core_param_ranges: dict[str, Any] | None = None,
    max_workers: int | None = None,
    initial: list[list[float]] | None = None,
    search_budget: SearchBudget | None = None,
    convergence_patience: int | None = None,
    rel_tol: float = 0.005,
) -> tuple[list[dict[str, Any]], list[tuple[Any, ...]]]:
    """Search one Pareto front of core designs for a core that runs *all* of `tiles_and_counts` -- a generalist
    for a folded core, rather than the specialist each per-tile front holds.

    Same gene space and NSGA2 as the per-tile search, but scored with `_evaluate_design_on_tiles`. Returns the
    decoded designs (`{"core_dict", "fitness", "genes"}`) and the history of every evaluated individual. The
    CACTI design space must already be precomputed and patched in, as `stream.api` does before its searches."""
    gene_layout = CoreGeneLayout(CoreParamRanges(**(core_param_ranges or {})))
    toolbox = _make_core_toolbox(gene_layout)
    toolbox.register(
        "evaluate",
        functools.partial(
            _evaluate_design_on_tiles,
            gene_layout=gene_layout,
            tiles_and_counts=tiles_and_counts,
            name="shared",
            loma_lpf_limit=loma_lpf_limit,
            temporal_mapping_type=temporal_mapping_type,
        ),
    )
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("fork")) as executor:
        _register_dedup_map(toolbox, executor)
        top_designs, history = _run_nsga2(
            toolbox,
            _round_population(nb_individuals),
            label="shared core",
            history_index=-1,
            nb_generations=nb_generations,
            nb_points=nb_points,
            search_budget=search_budget,
            convergence_patience=convergence_patience,
            rel_tol=rel_tol,
            initial=initial,
        )
    designs = decode_designs(
        [design for design in top_designs if all(math.isfinite(v) for v in design.fitness.values)],
        gene_layout,
        "core_shared",
    )
    return designs, history


def decode_designs(top_designs: list[Any], gene_layout: CoreGeneLayout, name_prefix: str) -> list[dict[str, Any]]:
    """Decode NSGA2 individuals into front entries `{"core_dict", "fitness", "genes"}`. The genes are kept so a
    later search can be seeded with the design."""
    return [
        {
            "core_dict": gene_layout.decode_node(design, name=f"{name_prefix}_rank_{rank}"),
            "fitness": tuple(design.fitness.values),
            "genes": list(design),
        }
        for rank, design in enumerate(top_designs)
    ]


def _record_evaluated_designs(
    rows: list[tuple[Any, ...]],
    tile_index: int,
    generation: int,
    individuals: list[Any],
) -> None:
    """Append one row per individual the GA evaluated.

    Raw gene values are kept, not the decoded core: `CoreGeneLayout.decode_node` can rebuild any of them, so
    every design the search ever looked at stays re-evaluatable -- including the ones NSGA2 discarded, which
    nothing else persists (`pareto_fronts.pickle` only ever holds the final front's top `core_pareto_points`)."""
    for individual in individuals:
        latency, energy, area = individual.fitness.values
        rows.append((tile_index, generation, _candidate_key(individual), latency, energy, area, *tuple(individual)))


def _write_all_evaluated_cores_csv(csv_path: str, gene_names: list[str], rows: list[tuple[Any, ...]]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["tile_index", "generation", "design_key", "latency", "energy", "area", *gene_names])
        writer.writerows(rows)
    logger.info(f"CoreArchitectureExplorationStage: saved {len(rows)} evaluated core design(s) to {csv_path}.")


def _write_cross_tile_results_csv(csv_path: str, rows: list[tuple[int, int, int, float, float, float]]) -> None:
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["origin_tile", "design_rank", "evaluated_on_tile", "latency", "energy", "area"])
        for row in rows:
            writer.writerow(row)
    logger.info(f"CoreArchitectureExplorationStage: saved {len(rows)} cross-tile evaluation result(s) to {csv_path}.")


class CoreArchitectureExplorationStage(Stage):
    """
    Searches core hardware designs (memory sizes, bandwidths, operational-array sizes) per *unique tile* -- the
    same shape-dedup granularity `stream.utils.get_unique_nodes`/`ZigZagCoreMappingEstimationStage` already use
    -- with a genuine multi-objective (NSGA2) genetic algorithm over `(latency, energy, area)`. `area` comes
    from `Core.get_area()` (inherited from ZigZag's own `Accelerator.get_area()`, summing operational-array
    area and each memory level's CACTI-derived area); `latency`/`energy` come straight from the
    `CostModelEvaluation` ZigZag returns for that tile on that candidate core. The interconnect area the schedule
    searches add (`interconnect_area`) is left out here: a core's router and links depend on the topology, not
    on its design, except through the wire length along its side, so it would not change which designs a front
    keeps by much.

    Each unique tile gets its own independent NSGA2 run, evaluated directly via
    `stream.stages.estimation.zigzag_core_mapping_estimation.evaluate_node_on_core` -- no accelerator assembly,
    no full downstream pipeline per candidate. Once a tile's search finishes, up to `core_pareto_points`
    representative designs are selected from its Pareto front (crowding-distance-diverse, via
    `tools.selNSGA2`). Unless the chain later cost-models the full shape x design matrix itself (a stage with
    `requires_full_cost_matrix`, see `ScheduleSearchStage`), every one of them is also evaluated on *every other*
    unique tile -- this surfaces designs that generalize well across tile shapes, not just ones overfit to their
    own tile. Every evaluated `(origin_tile, design, evaluated_on_tile)` triple is written to
    `candidate_results_csv_path`.

    The fronts are handed to the next stage as `core_pareto_fronts` (a `ParetoCoreSearchResult`), together with
    the `workload`, `original_workload`, `accelerator`, `tiled_workload_path` and `core_max_workers` this stage
    consumed: `CoreDesignSelectionStage` to build one accelerator from them, or `ScheduleExplorationStage` to
    search their combinations.

    Under a joint tiling+hardware search this stage runs once per tiling-GA individual, *inside*
    `TilingExplorationStage`'s own worker processes -- but since each evaluation here is a single direct ZigZag
    call (not a full nested pipeline run), that nesting stays affordable.
    """

    # Class-level defaults so stages built without `__init__` (the synthetic tests) run the unbudgeted search.
    search_budget: "SearchBudget | None" = None
    core_convergence_patience: int | None = None
    cross_tile_evaluation: bool = True

    def __init__(  # noqa: PLR0913
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ComputationNodeWorkload,
        original_workload: ONNXWorkload,
        accelerator: Accelerator,
        tiled_workload_path: str,
        core_max_workers: int | None = 1,
        nb_core_ga_generations: int = 4,
        nb_core_ga_individuals: int = 8,
        core_pareto_points: int = 10,
        core_max_pareto_schedules: int = 10_000,
        core_param_ranges: dict[str, Any] | None = None,
        candidate_results_csv_path: str | None = None,
        all_evaluated_cores_csv_path: str | None = None,
        core_convergence_patience: int | None = None,
        core_rel_tol: float = 0.005,
        reuse_pareto_fronts: bool = False,
        cross_tile_evaluation: bool | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            core_max_workers: worker processes for the GA's design evaluations.
            nb_core_ga_generations: hard cap on each tile's NSGA2 generations.
            nb_core_ga_individuals: NSGA2 population (rounded up to a multiple of 4).
            core_pareto_points: designs kept per tile front.
            core_max_pareto_schedules: cap on the per-tile design combinations enumerated into
                `core_search/pareto_schedules.pickle` for offline evaluation.
            core_param_ranges: overrides for `CoreParamRanges`.
            candidate_results_csv_path, all_evaluated_cores_csv_path: where the cross-tile results and every
                evaluated design go; default to `core_search/` next to the tiled workload.
            core_convergence_patience: stop a tile's GA once its front hypervolume gained less than `core_rel_tol`
                over this many generations (no convergence stop when None).
            core_rel_tol: relative hypervolume gain counted as progress.
            reuse_pareto_fronts: reuse `core_search/pareto_fronts.pickle` from an earlier run instead of
                re-running the GA, when it matches this workload's unique tile count.
            cross_tile_evaluation: evaluate every kept design on every other tile shape. Defaults to doing so
                unless a later stage cost-models the full shape x design matrix itself.
        """
        super().__init__(list_of_callables, **kwargs)
        # Each tile's NSGA2 stops on convergence (front hypervolume gaining < `core_rel_tol` over
        # `core_convergence_patience` generations) when a patience is given, or before a new generation once the
        # optional `SearchBudget` (see stream.opt.search_budget) has stopped; `nb_core_ga_generations` stays a hard
        # cap either way.
        self.search_budget = kwargs.get("search_budget")
        self.core_convergence_patience = core_convergence_patience
        self.core_rel_tol = core_rel_tol
        self.workload = workload
        self.original_workload = original_workload
        self.accelerator = accelerator
        self.tiled_workload_path = tiled_workload_path
        self.max_workers = core_max_workers
        self.nb_core_ga_generations = nb_core_ga_generations
        self.nb_core_ga_individuals = nb_core_ga_individuals
        self.core_pareto_points = core_pareto_points
        self.max_pareto_schedules = core_max_pareto_schedules
        self.gene_ranges = CoreParamRanges(**(core_param_ranges or {}))
        self.core_search_dir = os.path.join(os.path.dirname(tiled_workload_path), "core_search")
        self.candidate_results_csv_path = candidate_results_csv_path or os.path.join(
            self.core_search_dir, "candidate_results.csv"
        )
        self.all_evaluated_cores_csv_path = all_evaluated_cores_csv_path or os.path.join(
            self.core_search_dir, "all_evaluated_cores.csv"
        )
        self.reuse_pareto_fronts = reuse_pareto_fronts
        # The full-matrix warm-up of a later stage runs every design on every tile shape with the complete
        # ZigZag stage, so the lighter cross-tile pass would only recompute the same pairs.
        self.cross_tile_evaluation = (
            not downstream_requires(list_of_callables, "requires_full_cost_matrix")
            if cross_tile_evaluation is None
            else cross_tile_evaluation
        )

    def _unique_tiles(self) -> tuple[list[ComputationNode], dict[int, int]]:
        """Same shape-dedup as `stream.utils.get_unique_nodes`, done in one pass so we also get a mapping from
        each concrete tile (by object identity) to the index of its representative unique tile."""
        tiles = [node for node in self.workload.node_list if isinstance(node, ComputationNode)]
        unique_tiles: list[ComputationNode] = []
        tile_to_unique_index: dict[int, int] = {}
        for tile in tiles:
            match_index = next((i for i, u in enumerate(unique_tiles) if tile.has_same_performance(u)), None)
            if match_index is None:
                match_index = len(unique_tiles)
                unique_tiles.append(tile)
            tile_to_unique_index[id(tile)] = match_index
        return unique_tiles, tile_to_unique_index

    def _run_tile_ga(  # noqa: PLR0913
        self,
        tile_index: int,
        tile: ComputationNode,
        gene_layout: CoreGeneLayout,
        toolbox: base.Toolbox,
        mu: int,
        loma_lpf_limit: int,
        temporal_mapping_type: TemporalMappingType,
    ) -> tuple[list[Any], list[tuple[Any, ...]]]:
        """Run one tile's independent NSGA2 search.

        Returns up to `core_pareto_points` representative individuals from its Pareto front (each with
        `.fitness.values` already set to `(latency, energy, area)`), plus the history of every individual it
        evaluated, for `all_evaluated_cores.csv`."""
        # Tiles run concurrently (see `run()`), each on its own thread sharing the same underlying
        # ProcessPoolExecutor -- copy the toolbox before registering this tile's "evaluate" so concurrent
        # threads don't race on (and clobber) each other's registration on one shared toolbox instance.
        toolbox = copy.copy(toolbox)
        toolbox.register(
            "evaluate",
            functools.partial(
                _evaluate_design_on_tile,
                gene_layout=gene_layout,
                tile=tile,
                name=f"tile_{tile_index}",
                loma_lpf_limit=loma_lpf_limit,
                temporal_mapping_type=temporal_mapping_type,
            ),
        )

        return _run_nsga2(
            toolbox,
            mu,
            label=f"tile {tile_index}",
            history_index=tile_index,
            nb_generations=self.nb_core_ga_generations,
            nb_points=self.core_pareto_points,
            search_budget=self.search_budget,
            convergence_patience=self.core_convergence_patience,
            rel_tol=self.core_rel_tol,
        )

    def search_pareto_fronts(self) -> ParetoCoreSearchResult:  # noqa: PLR0915
        """Search every unique tile's Pareto front of core designs, saving them to
        `core_search/pareto_fronts.pickle`."""
        loma_lpf_limit = self.kwargs["loma_lpf_limit"]
        temporal_mapping_type = self.kwargs["temporal_mapping_type"]

        unique_tiles, tile_to_unique_index = self._unique_tiles()
        original_nodes = get_original_nodes(self.original_workload)
        tiles_by_original = get_tiles_by_original_node(self.workload, original_nodes)
        gene_layout = CoreGeneLayout(self.gene_ranges)
        n_genes = gene_layout.n_genes_per_node

        logger.info(
            f"CoreArchitectureExplorationStage: searching {len(unique_tiles)} unique tile design(s) "
            f"({n_genes} genes each)."
        )

        toolbox = _make_core_toolbox(gene_layout)
        mu = _round_population(self.nb_core_ga_individuals)

        # Force "fork" (not the platform default) for the same reason as TilingExplorationStage: it clones this
        # already-running process instead of re-importing __main__.
        mp_context = multiprocessing.get_context("fork")
        csv_rows: list[tuple[int, int, int, float, float, float]] = []

        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=mp_context) as executor:
            _register_dedup_map(toolbox, executor)

            # Run every tile's NSGA2 search concurrently rather than one-at-a-time: each tile's search is
            # otherwise independent, and driving them from separate threads lets their individual evaluation
            # jobs share the same `executor` queue -- when one tile's batch has a slow straggler (e.g. a slow
            # CACTI call), idle workers can immediately pick up another tile's jobs instead of sitting idle
            # until that tile's generation finishes. Threads (not processes) for the orchestration keeps real
            # parallelism capped at `core_max_workers` actual worker processes -- no oversubscription.
            with ThreadPoolExecutor(max_workers=len(unique_tiles)) as tile_executor:
                futures = [
                    tile_executor.submit(
                        self._run_tile_ga,
                        tile_index,
                        tile,
                        gene_layout,
                        toolbox,
                        mu,
                        loma_lpf_limit,
                        temporal_mapping_type,
                    )
                    for tile_index, tile in enumerate(unique_tiles)
                ]
                tile_results = [future.result() for future in futures]
            top_designs_per_tile = [top_designs for top_designs, _history in tile_results]
            _write_all_evaluated_cores_csv(
                self.all_evaluated_cores_csv_path,
                gene_layout.names,
                [row for _top_designs, history in tile_results for row in history],
            )

            for tile_index, top_designs in enumerate(top_designs_per_tile):
                for rank, design in enumerate(top_designs):
                    latency, energy, area = design.fitness.values
                    csv_rows.append((tile_index, rank, tile_index, latency, energy, area))

            # Cross-tile evaluation: every tile's top-K designs, evaluated on every OTHER unique tile.
            cross_jobs: list[tuple[int, int, int, tuple[float, ...]]] = [
                (origin_index, rank, target_index, tuple(design))
                for origin_index, top_designs in enumerate(top_designs_per_tile)
                for rank, design in enumerate(top_designs)
                for target_index in range(len(unique_tiles))
                if target_index != origin_index
            ]
            if not self.cross_tile_evaluation:
                cross_jobs = []
            if cross_jobs:
                # Dedup identical (design values, target tile) pairs -- distinct origin tiles can independently
                # land on the same design, especially once sizes/bandwidths snap to their discrete choices.
                job_keys = [(values, target_index) for _o, _r, target_index, values in cross_jobs]
                first_index_for_job: dict[tuple[Any, int], int] = {}
                unique_jobs: list[tuple[tuple[float, ...], int]] = []
                for job_key in job_keys:
                    if job_key not in first_index_for_job:
                        first_index_for_job[job_key] = len(unique_jobs)
                        unique_jobs.append(job_key)

                logger.info(
                    f"CoreArchitectureExplorationStage: cross-tile evaluation: {len(cross_jobs)} job(s), "
                    f"{len(unique_jobs)} unique after dedup."
                )
                t_cross_start = _time.perf_counter()
                cross_results = list(
                    executor.map(
                        functools.partial(
                            _evaluate_cross_job,
                            gene_layout=gene_layout,
                            unique_tiles=unique_tiles,
                            loma_lpf_limit=loma_lpf_limit,
                            temporal_mapping_type=temporal_mapping_type,
                        ),
                        unique_jobs,
                    )
                )
                logger.info(
                    f"CoreArchitectureExplorationStage: cross-tile evaluation done in "
                    f"{_time.perf_counter() - t_cross_start:.1f}s."
                )
                for (origin_index, rank, target_index, _values), job_key in zip(cross_jobs, job_keys, strict=True):
                    latency, energy, area = cross_results[first_index_for_job[job_key]]
                    csv_rows.append((origin_index, rank, target_index, latency, energy, area))

        _write_cross_tile_results_csv(self.candidate_results_csv_path, csv_rows)

        # Decode every retained Pareto-front design once (plain core dict + fitness) -- reused below both to
        # persist the full per-tile front and to build every schedule combination, instead of re-decoding
        # per schedule (decode_node can also raise ValueError on an invalid design, so decoding once here
        # rather than nb_schedules times avoids redundant repeated work/failure surface).
        decoded_designs_per_tile: list[list[dict[str, Any]]] = []
        for tile_index, top_designs in enumerate(top_designs_per_tile):
            if not top_designs:
                raise ValueError(f"CoreArchitectureExplorationStage: tile {tile_index} produced no valid core design.")
            decoded_designs_per_tile.append(decode_designs(top_designs, gene_layout, f"core_tile_{tile_index}"))

        os.makedirs(self.core_search_dir, exist_ok=True)
        pareto_fronts_path = os.path.join(self.core_search_dir, "pareto_fronts.pickle")
        pickle_save(decoded_designs_per_tile, pareto_fronts_path)  # type: ignore
        logger.info(
            f"CoreArchitectureExplorationStage: saved {sum(len(f) for f in decoded_designs_per_tile)} pareto-front "
            f"design(s) across {len(decoded_designs_per_tile)} unique tile(s) to {pareto_fronts_path}."
        )

        self._save_pareto_schedules(decoded_designs_per_tile, original_nodes, tiles_by_original, tile_to_unique_index)

        return ParetoCoreSearchResult(
            designs_per_tile=decoded_designs_per_tile,
            unique_tiles=unique_tiles,
            tile_to_unique_index=tile_to_unique_index,
            original_nodes=original_nodes,
            tiles_by_original=tiles_by_original,
        )

    def _load_pareto_fronts(self) -> list[list[dict[str, Any]]] | None:
        fronts_path = os.path.join(self.core_search_dir, "pareto_fronts.pickle")
        if not (self.reuse_pareto_fronts and os.path.exists(fronts_path)):
            return None
        fronts = pickle_load(fronts_path)
        unique_tiles, _ = self._unique_tiles()
        if not isinstance(fronts, list) or len(fronts) != len(unique_tiles):
            logger.warning(
                f"CoreArchitectureExplorationStage: ignoring {fronts_path} -- it holds {len(fronts)} front(s) but "
                f"this workload has {len(unique_tiles)} unique tile(s)."
            )
            return None
        logger.info(
            f"CoreArchitectureExplorationStage: reusing {sum(map(len, fronts))} core design(s) from {fronts_path}."
        )
        return fronts

    def get_pareto_fronts(self) -> ParetoCoreSearchResult:
        """The per-tile fronts: reused from an earlier run when `reuse_pareto_fronts` allows it, else searched."""
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

    def run(self):
        """Search (or reuse) every unique tile's Pareto front and hand it to the next stage."""
        kwargs = self.kwargs.copy()
        kwargs["core_pareto_fronts"] = self.get_pareto_fronts()
        # Forwarded by reference: `tile_to_unique_index` is keyed by `id()` of this very workload's tiles.
        kwargs["workload"] = self.workload
        kwargs["original_workload"] = self.original_workload
        kwargs["accelerator"] = self.accelerator
        kwargs["tiled_workload_path"] = self.tiled_workload_path
        kwargs["core_max_workers"] = self.max_workers
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        yield from sub_stage.run()

    def _save_pareto_schedules(
        self,
        decoded_designs_per_tile: list[list[dict[str, Any]]],
        original_nodes: list[ComputationNode],
        tiles_by_original: dict[ComputationNode, list[ComputationNode]],
        tile_to_unique_index: dict[int, int],
    ) -> None:
        """Enumerate every combination of (one Pareto-front design per unique tile) as a candidate "schedule"
        for later offline evaluation, gated by cost: skip enumeration if the combination count would exceed
        `self.max_pareto_schedules`. `itertools.product` runs over per-tile design *indices*, never
        materializing designs, so checking the count first is cheap even when enumeration itself is skipped."""
        front_sizes = [len(designs) for designs in decoded_designs_per_tile]
        nb_schedules = math.prod(front_sizes) if front_sizes else 0
        schedules_path = os.path.join(
            os.path.dirname(self.tiled_workload_path), "core_search", "pareto_schedules.pickle"
        )
        if not (0 < nb_schedules <= self.max_pareto_schedules):
            logger.warning(
                f"CoreArchitectureExplorationStage: skipping full schedule enumeration -- {nb_schedules} "
                f"combination(s) (per-tile front sizes {front_sizes}) exceeds core_max_pareto_schedules="
                f"{self.max_pareto_schedules}; only the single best-per-tile design will be used downstream."
            )
            return

        schedules: list[dict[str, Any]] = []
        for rank_combo in itertools.product(*(range(n) for n in front_sizes)):
            design_per_tile = [decoded_designs_per_tile[t][r]["core_dict"] for t, r in enumerate(rank_combo)]
            schedules.append(
                {
                    "tile_choices": [
                        {"tile_index": t, "rank": r, "fitness": decoded_designs_per_tile[t][r]["fitness"]}
                        for t, r in enumerate(rank_combo)
                    ],
                    "node_core_dicts": build_node_core_dicts(
                        design_per_tile, original_nodes, tiles_by_original, tile_to_unique_index
                    ),
                }
            )
        pickle_save(schedules, schedules_path)  # type: ignore
        logger.info(
            f"CoreArchitectureExplorationStage: generated and saved {len(schedules)} full pareto-schedule "
            f"combination(s) to {schedules_path}."
        )


def build_node_core_dicts(
    design_per_tile: list[dict[str, Any]],
    original_nodes: list[ComputationNode],
    tiles_by_original: dict[ComputationNode, list[ComputationNode]],
    tile_to_unique_index: dict[int, int],
) -> dict[int, dict[str, Any]]:
    """Map one core-dict choice per unique tile onto every original workload node (via a representative-tile-shape
    match). Shared by the single-best-design path and every enumerated schedule (see `_save_pareto_schedules`) so
    the assembly logic isn't duplicated."""
    node_core_dicts: dict[int, dict[str, Any]] = {}
    for original_node in original_nodes:
        representative_tile = tiles_by_original[original_node][0]
        unique_index = tile_to_unique_index[id(representative_tile)]
        node_core_dicts[original_node.id] = design_per_tile[unique_index]
    return node_core_dicts


def extra_shapes_per_unique_tile(search: ParetoCoreSearchResult) -> dict[int, set[int]]:
    """Which *other* tile shape classes each unique tile's chosen design will also have to run.

    `build_node_core_dicts` gives every tile of an original node the design its *representative* tile picked, so a
    node whose tiles are not all the same shape makes one design responsible for several shapes -- and the per-tile
    GA only ever checked it against its own. Empty for every unique tile whose node is uniformly shaped, which is
    the usual case."""
    extra: dict[int, set[int]] = {}
    for tiles in search.tiles_by_original.values():
        representative = search.tile_to_unique_index[id(tiles[0])]
        others = {search.tile_to_unique_index[id(tile)] for tile in tiles} - {representative}
        if others:
            extra.setdefault(representative, set()).update(others)
    return extra


class CoreDesignSelectionStage(Stage):
    """Collapses each per-tile front (`core_pareto_fronts`) to its single best design by `sort_key` (`latency`,
    `energy` or `area`, on the design's own tile), maps those designs onto the original nodes and hands
    `node_core_dicts` to the next stage -- normally `CustomCoreAcceleratorGenerationStage`, which builds the
    accelerator the rest of the pipeline runs on."""

    def __init__(  # noqa: PLR0913
        self,
        list_of_callables: list[StageCallable],
        *,
        core_pareto_fronts: ParetoCoreSearchResult,
        workload: ComputationNodeWorkload,
        original_workload: ONNXWorkload,
        accelerator: Accelerator,
        tiled_workload_path: str,
        sort_key: str = "latency",
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.core_pareto_fronts = core_pareto_fronts
        self.workload = workload
        self.original_workload = original_workload
        self.accelerator = accelerator
        self.tiled_workload_path = tiled_workload_path
        self.sort_key = sort_key

    @staticmethod
    def _designs_running(
        unique_index: int,
        designs: list[dict[str, Any]],
        unique_tiles: list[ComputationNode],
        extra_shapes: dict[int, set[int]],
    ) -> list[dict[str, Any]]:
        """`designs` restricted to the ones that can also map the other shapes they will have to run (see
        `extra_shapes_per_unique_tile`). A design that cannot raises `NoValidLoopOrderingFoundException` in the
        downstream `ZigZagCoreMappingEstimationStage`, which aborts the whole pipeline."""
        shapes = extra_shapes.get(unique_index)
        if not shapes:
            return designs
        usable = [
            design
            for design in designs
            if not any(
                lowest_memory_level_violations(
                    unique_tiles[shape], build_core_from_dict(design["core_dict"], core_id=0)
                )
                for shape in sorted(shapes)
            )
        ]
        if not usable:
            raise ValueError(
                f"CoreDesignSelectionStage: no design on tile {unique_index}'s front can map the other tile shapes "
                f"{sorted(shapes)} of its node. Widen `core_param_ranges` so larger innermost memories are "
                f"reachable."
            )
        if len(usable) < len(designs):
            logger.info(
                f"CoreDesignSelectionStage: {len(designs) - len(usable)} of tile {unique_index}'s design(s) cannot "
                f"map the other shapes {sorted(shapes)} its node runs; picking among the rest."
            )
        return usable

    def run(self):
        search = self.core_pareto_fronts
        sort_index = {"latency": 0, "energy": 1, "area": 2}[self.sort_key]
        extra_shapes = extra_shapes_per_unique_tile(search)
        best_design_per_tile: list[dict[str, Any]] = []
        for unique_index, designs in enumerate(search.designs_per_tile):
            usable = self._designs_running(unique_index, designs, search.unique_tiles, extra_shapes)
            best_design_per_tile.append(min(usable, key=lambda design: design["fitness"][sort_index])["core_dict"])

        kwargs = self.kwargs.copy()
        kwargs["workload"] = self.workload
        kwargs["original_workload"] = self.original_workload
        kwargs["accelerator"] = self.accelerator
        kwargs["node_core_dicts"] = build_node_core_dicts(
            best_design_per_tile, search.original_nodes, search.tiles_by_original, search.tile_to_unique_index
        )
        kwargs["tiled_workload_path"] = self.tiled_workload_path
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        yield from sub_stage.run()
