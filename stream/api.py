import csv
import math
import logging as _logging
import os
import time as _time
from typing import Any, Literal

import gurobipy as gp
from onnx import ModelProto
from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.utils import pickle_load, pickle_save

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.hardware.architecture.cacti_precompute import install_cacti_monkeypatch, precompute_cacti_design_space
from stream.opt.search_budget import SearchBudget
from stream.stages.allocation.constraint_optimization_allocation import ConstraintOptimizationAllocationStage
from stream.stages.allocation.genetic_algorithm_allocation import GeneticAlgorithmAllocationStage
from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.core_architecture_exploration import CoreArchitectureExplorationStage, CoreParamRanges
from stream.stages.generation.graph_evolution import GraphEvolutionStage
from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.rolled_schedule_exploration import RolledScheduleExplorationStage
from stream.stages.generation.schedule_exploration import ScheduleExplorationStage
from stream.stages.generation.scheduling_order_generation import SchedulingOrderGenerationStage
from stream.stages.generation.tiled_workload_generation import TiledWorkloadGenerationStage
from stream.stages.generation.tiling_exploration import TilingExplorationStage, scme_objective
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage as StreamONNXModelParserStage
from stream.stages.set_fixed_allocation_performance import SetFixedAllocationPerformanceStage
from stream.stages.stage import MainStage
from stream.utils import get_exclusive_stage_times, wrap_stages_with_timing

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


def _sanity_check_inputs(
    hardware: str, workload: str, mapping: str, mode: Literal["lbl"] | Literal["fused"], output_path: str
):
    assert os.path.exists(hardware), f"Hardware file {hardware} does not exist"
    assert isinstance(workload, ModelProto) or os.path.exists(workload), f"Workload file {workload} does not exist"
    assert os.path.exists(mapping), f"Mapping file {mapping} does not exist"
    assert mode in ["lbl", "fused"], "Mode must be either 'lbl' or 'fused'"
    if not os.path.exists(output_path):
        os.makedirs(output_path)


def _sanity_check_gurobi_license():
    try:
        # Try to create a simple optimization model
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        # Check if the model was successfully created (license check)
        model.optimize()
        # If model.optimize() runs without a license issue, return
        return
    except gp.GurobiError as exc:
        # Catch any Gurobi errors, especially licensing errors
        if exc.errno == gp.GRB.Error.NO_LICENSE:
            error_message = "No valid Gurobi license found. Get an academic WLS license at https://www.gurobi.com/academia/academic-program-and-licenses/"
        else:
            error_message = f"An unexpected Gurobi error occurred: {exc.message}"
        raise ValueError(error_message) from exc


def optimize_allocation_ga(  # noqa: PLR0913
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    nb_ga_generations: int,
    nb_ga_individuals: int,
    experiment_id: str,
    output_path: str,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
    profile: bool = False,
) -> StreamCostModelEvaluation:
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)

    # Create experiment_id path
    os.makedirs(f"{output_path}/{experiment_id}", exist_ok=True)

    # Output paths
    tiled_workload_path = f"{output_path}/{experiment_id}/tiled_workload.pickle"
    cost_lut_path = f"{output_path}/{experiment_id}/cost_lut.pickle"
    scme_path = f"{output_path}/{experiment_id}/scme.pickle"

    # Get logger
    logger = _logging.getLogger(__name__)

    # Determine temporal mapping type for ZigZag
    if temporal_mapping_type == "uneven":
        temporal_mapping_type = TemporalMappingType.UNEVEN
    elif temporal_mapping_type == "even":
        temporal_mapping_type = TemporalMappingType.EVEN
    else:
        raise ValueError(f"Invalid temporal mapping type: {temporal_mapping_type}. Must be 'uneven' or 'even'.")

    # Load SCME if it exists and skip_if_exists is True
    if os.path.exists(scme_path) and skip_if_exists:
        scme = pickle_load(scme_path)
        logger.info(f"Loaded SCME from {scme_path}")
    else:
        stage_classes = [  # Initializes the MainStage as entry point
            AcceleratorParserStage,  # Parses the accelerator
            StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
            LayerStacksGenerationStage,
            TilingGenerationStage,
            TiledWorkloadGenerationStage,
            ZigZagCoreMappingEstimationStage,
            SetFixedAllocationPerformanceStage,
            SchedulingOrderGenerationStage,
            GeneticAlgorithmAllocationStage,
        ]
        timings: dict[str, list[float]] = {}
        list_of_callables = stage_classes
        if profile:
            list_of_callables, timings = wrap_stages_with_timing(stage_classes)

        mainstage = MainStage(
            list_of_callables,
            accelerator=hardware,  # required by AcceleratorParserStage
            workload_path=workload,  # required by ModelParserStage
            mapping_path=mapping,  # required by ModelParserStage
            loma_lpf_limit=6,  # required by LomaEngine
            nb_ga_generations=nb_ga_generations,  # number of genetic algorithm (ga) generations
            nb_ga_individuals=nb_ga_individuals,  # number of individuals in each ga generation
            mode=mode,
            layer_stacks=layer_stacks,
            tiled_workload_path=tiled_workload_path,
            cost_lut_path=cost_lut_path,
            temporal_mapping_type=temporal_mapping_type,  # required by ZigZagCoreMappingEstimationStage
            operands_to_prefetch=[],  # required by GeneticAlgorithmAllocationStage
        )
        # Launch the MainStage
        t_start = _time.perf_counter()
        answers = mainstage.run()
        total_time = _time.perf_counter() - t_start
        scme = answers[0][0]
        pickle_save(scme, scme_path)  # type: ignore

        if profile:
            exclusive_times = get_exclusive_stage_times(stage_classes, timings)
            logger.info("optimize_allocation_ga stage timing breakdown (total %.2fs):", total_time)
            for label, exclusive_time in exclusive_times.items():
                percent = 100 * exclusive_time / total_time if total_time else 0.0
                logger.info("  %-38s %8.3fs (%5.1f%%)", label, exclusive_time, percent)
    return scme


def optimize_allocation_co(  # noqa: PLR0913
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    experiment_id: str,
    output_path: str,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
) -> StreamCostModelEvaluation:
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)
    _sanity_check_gurobi_license()

    # Create experiment_id path
    os.makedirs(f"{output_path}/{experiment_id}", exist_ok=True)

    # Output paths
    tiled_workload_path = f"{output_path}/{experiment_id}/tiled_workload.pickle"
    cost_lut_path = f"{output_path}/{experiment_id}/cost_lut.pickle"
    allocations_path = f"{output_path}/{experiment_id}/waco/"
    tiled_workload_post_co_path = f"{output_path}/{experiment_id}/tiled_workload_post_co.pickle"
    cost_lut_post_co_path = f"{output_path}/{experiment_id}/cost_lut_post_co.pickle"
    scme_path = f"{output_path}/{experiment_id}/scme.pickle"

    # Get logger
    logger = _logging.getLogger(__name__)

    # Determine temporal mapping type for ZigZag
    if temporal_mapping_type == "uneven":
        temporal_mapping_type = TemporalMappingType.UNEVEN
    elif temporal_mapping_type == "even":
        temporal_mapping_type = TemporalMappingType.EVEN
    else:
        raise ValueError(f"Invalid temporal mapping type: {temporal_mapping_type}. Must be 'uneven' or 'even'.")

    # Load SCME if it exists and skip_if_exists is True
    if os.path.exists(scme_path) and skip_if_exists:
        scme = pickle_load(scme_path)
        logger.info(f"Loaded SCME from {scme_path}")
    else:
        mainstage = MainStage(
            [  # Initializes the MainStage as entry point
                AcceleratorParserStage,  # Parses the accelerator
                StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
                LayerStacksGenerationStage,
                TilingGenerationStage,
                TiledWorkloadGenerationStage,
                ZigZagCoreMappingEstimationStage,
                ConstraintOptimizationAllocationStage,
            ],
            accelerator=hardware,  # required by AcceleratorParserStage
            workload_path=workload,  # required by ModelParserStage
            mapping_path=mapping,  # required by ModelParserStage
            loma_lpf_limit=6,  # required by LomaEngine
            mode=mode,
            layer_stacks=layer_stacks,
            tiled_workload_path=tiled_workload_path,
            cost_lut_path=cost_lut_path,
            allocations_path=allocations_path,
            tiled_workload_post_co_path=tiled_workload_post_co_path,
            cost_lut_post_co_path=cost_lut_post_co_path,
            temporal_mapping_type=temporal_mapping_type,  # required by ZigZagCoreMappingEstimationStage
            operands_to_prefetch=[],  # required by ConstraintOptimizationAllocationStage
        )
        # Launch the MainStage
        answers = mainstage.run()
        scme = answers[0][0]
        pickle_save(scme, scme_path)  # type: ignore
    return scme


def optimize_tiling(  # noqa: PLR0913, PLR0915
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    experiment_id: str,
    output_path: str,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
    nb_ga_generations: int = 4,
    nb_ga_individuals: int = 4,
    nb_tiling_ga_generations: int = 4,
    nb_tiling_ga_individuals: int = 8,
    sort_key: Literal["latency", "energy", "edp"] = "latency",
    max_workers: int | None = None,
    profile: bool = False,
    nb_core_ga_generations: int = 4,
    nb_core_ga_individuals: int = 8,
    core_max_workers: int | None = 1,
    core_pareto_points: int = 10,
    core_max_pareto_schedules: int = 10_000,
    core_param_ranges: dict[str, Any] | None = None,
    cacti_precompute_workers: int | None = None,
    explore_cores: bool = True,
    search_budget: SearchBudget | None = None,
) -> tuple[StreamCostModelEvaluation, list[tuple[StreamCostModelEvaluation, Any]]]:
    """Search per-layer, per-dimension intra-/inter-core tiling assignments on a fixed hardware and workload
    using a genetic algorithm, running the full pipeline (tiling -> cost estimation -> allocation) once per
    evaluated individual, and returning the best result by `sort_key` together with a sample of other good
    candidates found (the search's hall of fame).

    Candidates are derived by `TilingExplorationStage`: every dimension of every layer (excluding batch/group
    dims) is a gene assigned to either intra-core or inter-core tiling with a factor of 1 (no split) or one of
    that dimension's prime divisors (see `stream.stages.generation.tiling_exploration.prime_divisors`).
    Individuals are evaluated in parallel worker processes (see `TilingExplorationStage`); each worker
    atomically updates the shared `scme.pickle` under a lock as soon as it finds a result better than the
    current best, so the best-so-far survives even if the search is interrupted partway through.

    For each candidate, the fixed-template accelerator generation is replaced by
    `CoreArchitectureExplorationStage`, which searches a customized hardware design *per unique tile* (memory
    sizes, bandwidths, operational-array sizes) with a genuine multi-objective (NSGA2) genetic algorithm over
    `(latency, energy, area)` -- `area` via `Core.get_area()`, `latency`/`energy` from the
    `CostModelEvaluation` ZigZag returns for that tile on that candidate core. Each unique tile's top
    `core_pareto_points` Pareto-front designs are also evaluated on every *other* unique tile and everything is
    logged to a candidate-results CSV (see `CoreArchitectureExplorationStage`'s docstring). Runs once per
    tiling-GA individual, nested inside `TilingExplorationStage`'s own worker processes.

    Args:
        nb_tiling_ga_generations: number of generations for the outer tiling-search genetic algorithm run by
            `TilingExplorationStage`. Independent of `nb_ga_generations`, which only sizes the inner
            per-candidate `GeneticAlgorithmAllocationStage` core-allocation search.
        nb_tiling_ga_individuals: population size for the outer tiling-search genetic algorithm. Independent of
            `nb_ga_individuals`, for the same reason as above.
        max_workers: number of worker processes to evaluate candidates in parallel. Defaults to
            `os.process_cpu_count()` (via `ProcessPoolExecutor`'s own default) when None.
        nb_core_ga_generations: number of generations for the inner per-tile core-architecture NSGA2 search.
        nb_core_ga_individuals: population size for the inner per-tile core-architecture NSGA2 search (rounded
            up to a multiple of 4 if needed).
        core_max_workers: number of worker processes for the inner core-architecture search's own
            `ProcessPoolExecutor`. Defaults to 1 since this search already runs nested inside each tiling
            candidate's own worker process; raise it only if `max_workers`/`nb_tiling_ga_individuals` are kept
            low enough to avoid oversubscribing the machine.
        core_pareto_points: number of representative designs selected from each unique tile's Pareto front for
            the cross-tile evaluation sweep.
        core_max_pareto_schedules: cap on the number of full per-tile-design combinations
            (`prod(len(front) for front in per_tile_pareto_fronts)`) `CoreArchitectureExplorationStage` will
            enumerate and persist as candidate "schedules" for later offline evaluation. If the actual
            combination count would exceed this, enumeration is skipped (logged) and only the single best
            design per tile is used to build the real accelerator, same as always.
        core_param_ranges: optional overrides for the core-architecture search's per-parameter bounds, using the
            same keyword names as `core_generator.random_core_dict` (e.g. `rf_size_range`, `sram_bandwidth_choices`).
        cacti_precompute_workers: number of worker processes used to precompute the entire CACTI memory design
            space `core_param_ranges` can produce, once, before the search starts (see
            `stream.hardware.architecture.cacti_precompute`). Defaults to every available core when None, since
            this precompute runs once, serially before any GA parallelism, unlike `core_max_workers`.
        explore_cores: False keeps `hardware` fixed: the per-candidate core-architecture search (and its CACTI
            precompute) is dropped, so only the intra-/inter-core tiling is searched on the given accelerator.
        search_budget: optional wall-clock budget (see `stream.opt.search_budget`). When given, the tiling GA runs
            generations until the budget stops it instead of for `nb_tiling_ga_generations`, and records every
            new candidate on it.
    """
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)

    # Create experiment_id path
    os.makedirs(f"{output_path}/{experiment_id}", exist_ok=True)

    # Output paths
    tiling_candidates_dir = f"{output_path}/{experiment_id}/tiling_search"
    scme_path = f"{output_path}/{experiment_id}/scme.pickle"
    results_csv_path = f"{output_path}/{experiment_id}/tiling_search_results.csv"
    candidate_results_csv_path = f"{output_path}/{experiment_id}/tiling_search_all_candidates.csv"

    # Get logger
    logger = _logging.getLogger(__name__)

    # Determine temporal mapping type for ZigZag
    if temporal_mapping_type == "uneven":
        temporal_mapping_type = TemporalMappingType.UNEVEN
    elif temporal_mapping_type == "even":
        temporal_mapping_type = TemporalMappingType.EVEN
    else:
        raise ValueError(f"Invalid temporal mapping type: {temporal_mapping_type}. Must be 'uneven' or 'even'.")

    # Load SCME if it exists and skip_if_exists is True
    if os.path.exists(scme_path) and skip_if_exists:
        scme = pickle_load(scme_path)
        logger.info(f"Loaded SCME from {scme_path}")
        return scme, []

    # GA needs a fixed scheduling order and pre-fixed single-core allocations before searching (see
    # optimize_allocation_ga).
    allocation_stage_kwargs: dict[str, Any] = {
        "nb_ga_generations": nb_ga_generations,
        "nb_ga_individuals": nb_ga_individuals,
    }
    allocation_stage_classes = [
        SetFixedAllocationPerformanceStage,
        SchedulingOrderGenerationStage,
        GeneticAlgorithmAllocationStage,
    ]

    # Stages up to and including TilingExplorationStage run once in this process; everything after it
    # (TilingGenerationStage onward) runs per-candidate inside worker processes spawned by
    # TilingExplorationStage, so those classes must stay picklable and are never timing-wrapped below.
    outer_stage_classes = [
        AcceleratorParserStage,  # Parses the accelerator
        StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
        LayerStacksGenerationStage,
        TilingExplorationStage,  # Sweeps candidate intra-/inter-core tiling configurations
    ]
    # CoreArchitectureExplorationStage replaces the fixed-template accelerator generation: it runs once per
    # tiling-GA individual (inside TilingExplorationStage's own worker processes) and searches per-layer core
    # designs with its own nested NSGA2 GA before handing off to ZigZagCoreMappingEstimationStage. See its
    # docstring for the resulting nested-multiprocessing caveat (core_max_workers defaults to 1 to avoid
    # oversubscription).
    per_candidate_stage_classes = [
        TilingGenerationStage,
        TiledWorkloadGenerationStage,
        *([CoreArchitectureExplorationStage] if explore_cores else []),
        ZigZagCoreMappingEstimationStage,
        *allocation_stage_classes,
    ]
    timings: dict[str, list[float]] = {}
    if profile:
        outer_stage_classes, timings = wrap_stages_with_timing(outer_stage_classes)
    list_of_callables = outer_stage_classes + per_candidate_stage_classes

    core_search_kwargs: dict[str, Any] = {
        "nb_core_ga_generations": nb_core_ga_generations,  # required by CoreArchitectureExplorationStage
        "nb_core_ga_individuals": nb_core_ga_individuals,  # required by CoreArchitectureExplorationStage
        "core_max_workers": core_max_workers,  # required by CoreArchitectureExplorationStage
        "core_pareto_points": core_pareto_points,  # required by CoreArchitectureExplorationStage
        "core_max_pareto_schedules": core_max_pareto_schedules,  # required by CoreArchitectureExplorationStage
        "core_param_ranges": core_param_ranges,  # required by CoreArchitectureExplorationStage
    }

    # Precompute the entire CACTI memory design space `core_param_ranges` can produce, once, before any GA work
    # starts (and before TilingExplorationStage's first fork below) -- see cacti_precompute's module docstring
    # for why this makes every CACTI lookup during the search hit memory instead of a subprocess call or a
    # CactiBatch pool-file read.
    if explore_cores:
        ranges = CoreParamRanges(**(core_param_ranges or {}))
        logger.info("optimize_tiling: precomputing CACTI design space for core_search...")
        t_precompute_start = _time.perf_counter()
        design_space = precompute_cacti_design_space(ranges, max_workers=cacti_precompute_workers)
        install_cacti_monkeypatch(design_space)
        logger.info(
            f"optimize_tiling: precomputed {len(design_space)} CACTI config(s) in "
            f"{_time.perf_counter() - t_precompute_start:.1f}s."
        )

    mainstage = MainStage(
        list_of_callables,
        accelerator=hardware,  # required by AcceleratorParserStage
        workload_path=workload,  # required by ModelParserStage
        mapping_path=mapping,  # required by ModelParserStage
        loma_lpf_limit=6,  # required by LomaEngine
        mode=mode,
        layer_stacks=layer_stacks,
        tiling_candidates_dir=tiling_candidates_dir,  # required by TilingExplorationStage
        scme_path=scme_path,  # required by TilingExplorationStage
        sort_key=sort_key,  # required by TilingExplorationStage (and forwarded on to CoreArchitectureExplorationStage)
        max_workers=max_workers,  # required by TilingExplorationStage
        nb_tiling_ga_generations=nb_tiling_ga_generations,  # required by TilingExplorationStage
        nb_tiling_ga_individuals=nb_tiling_ga_individuals,  # required by TilingExplorationStage
        candidate_results_csv_path=candidate_results_csv_path,  # required by TilingExplorationStage
        temporal_mapping_type=temporal_mapping_type,  # required by ZigZagCoreMappingEstimationStage
        operands_to_prefetch=[],  # required by GeneticAlgorithmAllocationStage/ConstraintOptimizationAllocationStage
        profile=profile,  # required by TilingExplorationStage, to time its own per-candidate pipeline stages
        search_budget=search_budget,  # optional, read by TilingExplorationStage
        **allocation_stage_kwargs,
        **core_search_kwargs,
    )
    # Launch the MainStage
    t_start = _time.perf_counter()
    answers = mainstage.run()
    total_time = _time.perf_counter() - t_start

    if not answers:
        raise ValueError("No candidate tiling configuration produced a result.")

    best_scme, best_extra_info = min(answers, key=lambda answer: scme_objective(answer[0], sort_key))
    pickle_save(best_scme, scme_path)  # type: ignore
    logger.info(
        f"Best tiling candidate {best_extra_info['candidate_index']} out of {len(answers)}: "
        f"{best_extra_info['tiling_config']} ({sort_key}={scme_objective(best_scme, sort_key)})."
    )

    with open(results_csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["candidate_index", "tiling_config", "latency", "energy"])
        for scme, extra_info in answers:
            writer.writerow([extra_info["candidate_index"], extra_info["tiling_config"], scme.latency, scme.energy])

    if profile:
        exclusive_times = get_exclusive_stage_times(outer_stage_classes, timings)
        logger.info("optimize_tiling stage timing breakdown (total %.2fs):", total_time)
        for label, exclusive_time in exclusive_times.items():
            percent = 100 * exclusive_time / total_time if total_time else 0.0
            logger.info("  %-38s %8.3fs (%5.1f%%)", label, exclusive_time, percent)

    return best_scme, answers


def optimize_schedules(  # noqa: PLR0913
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    experiment_id: str,
    output_path: str,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
    sort_key: Literal["latency", "energy"] = "latency",
    nb_ga_generations: int = 4,
    nb_ga_individuals: int = 4,
    nb_core_ga_generations: int = 4,
    nb_core_ga_individuals: int = 8,
    core_max_workers: int | None = None,
    core_pareto_points: int = 10,
    core_param_ranges: dict[str, Any] | None = None,
    cacti_precompute_workers: int | None = None,
    max_schedule_evaluations: int = 1000,
    max_search_nodes: int = 1_000_000,
    prune_schedules: bool = True,
    reuse_pareto_fronts: bool = True,
) -> tuple[StreamCostModelEvaluation, list[dict[str, Any]]]:
    """Search the `(latency, energy, area)` Pareto front of whole *schedules* on a fixed workload and tiling,
    over every combination of the per-tile Pareto core designs -- the multi-objective counterpart of
    `optimize_allocation_ga`, which evaluates a single fixed accelerator.

    `CoreArchitectureExplorationStage` (used by `optimize_tiling`) searches a Pareto front of core designs per
    unique tile and then collapses each front to its single best design by `sort_key`, so only one accelerator
    is ever scheduled. Here `ScheduleExplorationStage` keeps every design instead and searches the product of
    the fronts, measuring combinations through the real scheduler; the result is the set of schedules that are
    non-dominated on all three objectives at once -- e.g. the one that gives up 5% latency for half the area.

    The enumeration is a branch and bound, not a sweep: the accelerator topology is derived once (it does not
    depend on the designs), each distinct `(tile, design)` pair is cost-modelled by ZigZag exactly once, and
    area/energy/latency lower bounds computed without scheduling prune whole subtrees whose every completion is
    already dominated by a measured schedule. See `stream.stages.generation.schedule_exploration`'s module
    docstring for why each bound holds.

    Args:
        core_pareto_points: designs retained per unique tile -- this is the branching factor of the schedule
            search, so it drives the combination count directly (`core_pareto_points ** nb_decision_classes`).
        max_schedule_evaluations: cap on schedules actually measured. The search stays exact up to the cap; past
            it, the reported front is the front of what was measured.
        max_search_nodes: branch-and-bound node budget, a guard against fronts where pruning cannot keep up.
        prune_schedules: False measures every combination (exhaustive, exponentially slower) -- used to verify
            the pruned search returns the same front.
        reuse_pareto_fronts: reuse `core_search/pareto_fronts.pickle` from an earlier run of this experiment
            instead of re-running the per-tile core GA.

    Returns:
        The best schedule by `sort_key` as a `StreamCostModelEvaluation`, and the full Pareto set as plain
        records (ranks, latency, energy, area, bounds). The same records, the trade-off figure and the measured
        schedules' CSV are written under `{output_path}/{experiment_id}/schedule_search/` -- see
        `ScheduleExplorationStage`'s docstring for the artifact list.
    """
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)
    os.makedirs(f"{output_path}/{experiment_id}", exist_ok=True)

    # Output paths
    tiled_workload_path = f"{output_path}/{experiment_id}/tiled_workload.pickle"
    cost_lut_path = f"{output_path}/{experiment_id}/cost_lut.pickle"
    scme_path = f"{output_path}/{experiment_id}/scme.pickle"
    schedule_search_dir = f"{output_path}/{experiment_id}/schedule_search"

    logger = _logging.getLogger(__name__)

    if temporal_mapping_type == "uneven":
        temporal_mapping_type = TemporalMappingType.UNEVEN
    elif temporal_mapping_type == "even":
        temporal_mapping_type = TemporalMappingType.EVEN
    else:
        raise ValueError(f"Invalid temporal mapping type: {temporal_mapping_type}. Must be 'uneven' or 'even'.")

    if os.path.exists(scme_path) and skip_if_exists:
        scme = pickle_load(scme_path)
        logger.info(f"Loaded SCME from {scme_path}")
        return scme, []

    # ScheduleExplorationStage measures each combination by running the stages after it itself, with a cost LUT
    # it assembles from its own warm-up -- hence no ZigZagCoreMappingEstimationStage in this chain.
    stage_classes = [
        AcceleratorParserStage,
        StreamONNXModelParserStage,
        LayerStacksGenerationStage,
        TilingGenerationStage,
        TiledWorkloadGenerationStage,
        ScheduleExplorationStage,
        SetFixedAllocationPerformanceStage,
        SchedulingOrderGenerationStage,
        GeneticAlgorithmAllocationStage,
    ]

    # Same one-off CACTI precompute as optimize_tiling: the core GA and every accelerator the schedule search
    # assembles hit memory instead of a CACTI subprocess.
    ranges = CoreParamRanges(**(core_param_ranges or {}))
    logger.info("optimize_schedules: precomputing CACTI design space...")
    t_precompute_start = _time.perf_counter()
    design_space = precompute_cacti_design_space(ranges, max_workers=cacti_precompute_workers)
    install_cacti_monkeypatch(design_space)
    logger.info(
        f"optimize_schedules: precomputed {len(design_space)} CACTI config(s) in "
        f"{_time.perf_counter() - t_precompute_start:.1f}s."
    )

    mainstage = MainStage(
        stage_classes,
        accelerator=hardware,  # required by AcceleratorParserStage
        workload_path=workload,  # required by ModelParserStage
        mapping_path=mapping,  # required by ModelParserStage
        loma_lpf_limit=6,  # required by LomaEngine
        mode=mode,
        layer_stacks=layer_stacks,
        tiled_workload_path=tiled_workload_path,
        cost_lut_path=cost_lut_path,
        temporal_mapping_type=temporal_mapping_type,
        operands_to_prefetch=[],  # required by GeneticAlgorithmAllocationStage
        nb_ga_generations=nb_ga_generations,  # required by GeneticAlgorithmAllocationStage
        nb_ga_individuals=nb_ga_individuals,  # required by GeneticAlgorithmAllocationStage
        sort_key=sort_key,  # required by ScheduleExplorationStage (inherited from CoreArchitectureExplorationStage)
        nb_core_ga_generations=nb_core_ga_generations,  # required by the inherited per-tile core search
        nb_core_ga_individuals=nb_core_ga_individuals,
        core_max_workers=core_max_workers,
        core_pareto_points=core_pareto_points,
        core_param_ranges=core_param_ranges,
        schedule_search_dir=schedule_search_dir,  # required by ScheduleExplorationStage
        max_schedule_evaluations=max_schedule_evaluations,
        max_search_nodes=max_search_nodes,
        prune_schedules=prune_schedules,
        reuse_pareto_fronts=reuse_pareto_fronts,
    )

    t_start = _time.perf_counter()
    answers = [answer for answer in mainstage.run() if answer[0] is not None]
    total_time = _time.perf_counter() - t_start
    if not answers:
        raise ValueError("No schedule could be evaluated.")

    best_scme, extra_info = answers[0]
    pickle_save(best_scme, scme_path)  # type: ignore
    pareto_schedules: list[dict[str, Any]] = extra_info["pareto_schedules"]
    stats = extra_info["search_stats"]
    logger.info(
        f"optimize_schedules: {len(pareto_schedules)} pareto schedule(s) from {stats['measured']} measured of "
        f"{stats['combinations']} combination(s) in {total_time:.1f}s; best by {sort_key}: "
        f"latency={best_scme.latency}, energy={best_scme.energy}."
    )
    return best_scme, pareto_schedules


def optimize_rolled_schedules(  # noqa: PLR0913
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    experiment_id: str,
    output_path: str,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
    sort_key: Literal["latency", "energy"] = "latency",
    nb_ga_generations: int = 4,
    nb_ga_individuals: int = 4,
    nb_core_ga_generations: int = 4,
    nb_core_ga_individuals: int = 8,
    core_max_workers: int | None = None,
    core_pareto_points: int = 10,
    core_param_ranges: dict[str, Any] | None = None,
    cacti_precompute_workers: int | None = None,
    max_schedule_evaluations: int = 1000,
    max_search_nodes: int = 1_000_000,
    prune_schedules: bool = True,
    reuse_pareto_fronts: bool = True,
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
    search_budget: SearchBudget | None = None,
    core_convergence_patience: int | None = None,
    core_rel_tol: float = 0.005,
    schedule_search_time_fraction: float = 0.6,
) -> tuple[StreamCostModelEvaluation, list[dict[str, Any]]]:
    """Search schedules as `optimize_schedules` does, then *roll* them: reuse cores that are idle later in the
    schedule so the accelerator needs fewer of them, and return one Pareto front over both families.

    `optimize_schedules` only ever searches which core *design* each layer gets. The topology underneath is
    fully unrolled -- one dedicated core per `(layer, inter-core group)` -- so the core count grows with the
    workload and every core idles outside its layer's window. This adds the complementary axis: each measured
    schedule is folded onto fewer physical cores by merging cores whose busy windows don't collide, the
    communication graph is re-derived (merged traffic becomes free, and new links appear between merged
    groups), and the folded accelerator is re-measured with the real scheduler.

    Both families share one `(latency, energy, area)` archive, so the returned front shows the whole trade-off:
    unrolled designs at the fast end, heavily folded ones at the cheap end. Records are tagged `kind` and
    carry `nb_cores`; rolled ones also carry their full fold, enough to rebuild the accelerator.

    Args:
        fold_overlap: "hull" merges only cores whose whole busy spans are disjoint (optimal colouring);
            "exact" compares the true busy sets and can fold further when layers interleave.
        fold_tolerances: the aggressiveness ladder -- a pair conflicts when it is busy at once for more than
            `tolerance * min(load)`. 0.0 merges only genuinely idle cores; 1.0 merges almost anything and pays
            serialization for area. Under layer fusion the strict setting often admits no fold at all, which is
            why the sweep matters.
        merged_design_objectives: one folded variant per objective, each giving merged cores the design that is
            cheapest for the layers they now host.
        max_cme_matrix_cells: guard on the full `shape x design` cost-model warm-up that folding needs (a
            folded core runs layers that were never meant for its design). Above it the run falls back to the
            unrolled diagonal warm-up.
        prune_rolled: skip folds whose lower bound is already dominated by a measured schedule.
        core_convergence_patience: when set, each tile's core GA stops once its front hypervolume grew by less
            than `core_rel_tol` over that many generations (`nb_core_ga_generations` stays a hard cap).
        search_budget: optional wall-clock budget (see `stream.opt.search_budget`). When given, unrolled
            schedules are measured until `schedule_search_time_fraction` of it and folding uses the rest; every
            measured schedule is recorded on it. Pass large iteration caps alongside it.

    Returns:
        The best schedule by `sort_key`, and the combined Pareto set as plain records. Artifacts are written
        under `{output_path}/{experiment_id}/rolled_search/` -- see `RolledScheduleExplorationStage`.
    """
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)
    os.makedirs(f"{output_path}/{experiment_id}", exist_ok=True)

    tiled_workload_path = f"{output_path}/{experiment_id}/tiled_workload.pickle"
    cost_lut_path = f"{output_path}/{experiment_id}/cost_lut.pickle"
    scme_path = f"{output_path}/{experiment_id}/scme.pickle"
    schedule_search_dir = f"{output_path}/{experiment_id}/schedule_search"
    rolled_search_dir = f"{output_path}/{experiment_id}/rolled_search"

    logger = _logging.getLogger(__name__)

    if temporal_mapping_type == "uneven":
        temporal_mapping_type = TemporalMappingType.UNEVEN
    elif temporal_mapping_type == "even":
        temporal_mapping_type = TemporalMappingType.EVEN
    else:
        raise ValueError(f"Invalid temporal mapping type: {temporal_mapping_type}. Must be 'uneven' or 'even'.")

    if os.path.exists(scme_path) and skip_if_exists:
        scme = pickle_load(scme_path)
        logger.info(f"Loaded SCME from {scme_path}")
        return scme, []

    # As in `optimize_schedules`, the exploration stage supplies its own cost LUT, so no
    # ZigZagCoreMappingEstimationStage in this chain.
    stage_classes = [
        AcceleratorParserStage,
        StreamONNXModelParserStage,
        LayerStacksGenerationStage,
        TilingGenerationStage,
        TiledWorkloadGenerationStage,
        RolledScheduleExplorationStage,
        SetFixedAllocationPerformanceStage,
        SchedulingOrderGenerationStage,
        GeneticAlgorithmAllocationStage,
    ]

    # Must happen before `mainstage.run()` forks anything, so every worker inherits the patch.
    ranges = CoreParamRanges(**(core_param_ranges or {}))
    logger.info("optimize_rolled_schedules: precomputing CACTI design space...")
    t_precompute_start = _time.perf_counter()
    design_space = precompute_cacti_design_space(ranges, max_workers=cacti_precompute_workers)
    install_cacti_monkeypatch(design_space)
    logger.info(
        f"optimize_rolled_schedules: precomputed {len(design_space)} CACTI config(s) in "
        f"{_time.perf_counter() - t_precompute_start:.1f}s."
    )

    mainstage = MainStage(
        stage_classes,
        accelerator=hardware,
        workload_path=workload,
        mapping_path=mapping,
        loma_lpf_limit=6,
        mode=mode,
        layer_stacks=layer_stacks,
        tiled_workload_path=tiled_workload_path,
        cost_lut_path=cost_lut_path,
        temporal_mapping_type=temporal_mapping_type,
        operands_to_prefetch=[],
        nb_ga_generations=nb_ga_generations,
        nb_ga_individuals=nb_ga_individuals,
        sort_key=sort_key,
        nb_core_ga_generations=nb_core_ga_generations,
        nb_core_ga_individuals=nb_core_ga_individuals,
        core_max_workers=core_max_workers,
        core_pareto_points=core_pareto_points,
        core_param_ranges=core_param_ranges,
        schedule_search_dir=schedule_search_dir,
        max_schedule_evaluations=max_schedule_evaluations,
        max_search_nodes=max_search_nodes,
        prune_schedules=prune_schedules,
        reuse_pareto_fronts=reuse_pareto_fronts,
        rolled_search_dir=rolled_search_dir,
        fold_overlap=fold_overlap,
        fold_tolerances=fold_tolerances,
        rolled_core_count_targets=rolled_core_count_targets,
        fold_allow_sibling_merge=fold_allow_sibling_merge,
        merged_design_pool=merged_design_pool,
        merged_design_objectives=merged_design_objectives,
        keep_singleton_designs=keep_singleton_designs,
        max_rolled_variants_per_schedule=max_rolled_variants_per_schedule,
        max_rolled_evaluations=max_rolled_evaluations,
        fold_source=fold_source,
        prune_rolled=prune_rolled,
        max_cme_matrix_cells=max_cme_matrix_cells,
        search_budget=search_budget,
        core_convergence_patience=core_convergence_patience,
        core_rel_tol=core_rel_tol,
        schedule_search_time_fraction=schedule_search_time_fraction if search_budget is not None else 1.0,
    )

    t_start = _time.perf_counter()
    answers = [answer for answer in mainstage.run() if answer[0] is not None]
    total_time = _time.perf_counter() - t_start
    if not answers:
        raise ValueError("No schedule could be evaluated.")

    best_scme, extra_info = answers[0]
    pickle_save(best_scme, scme_path)  # type: ignore
    pareto_schedules: list[dict[str, Any]] = extra_info["pareto_schedules"]
    stats = extra_info["search_stats"]
    rolled = stats.get("rolled", {})
    nb_rolled = sum(1 for record in pareto_schedules if record.get("kind") == "rolled")
    logger.info(
        f"optimize_rolled_schedules: {len(pareto_schedules)} pareto schedule(s) ({nb_rolled} rolled) from "
        f"{stats['measured']} unrolled + {rolled.get('measured', 0)} rolled measurement(s) in {total_time:.1f}s; "
        f"best by {sort_key}: latency={best_scme.latency}, energy={best_scme.energy}."
    )
    return best_scme, pareto_schedules


def optimize_graph_evolution(  # noqa: PLR0913
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    experiment_id: str,
    output_path: str,
    temporal_mapping_type: str = "uneven",
    nb_ga_generations: int = 4,
    nb_ga_individuals: int = 4,
    population_size: int = 16,
    graph_ga_generations: int = 10,
    max_workers: int | None = None,
    core_param_ranges: dict[str, Any] | None = None,
    cacti_precompute_workers: int | None = None,
    search_budget: SearchBudget | None = None,
    graph_step_fraction: float = 0.5,
    step1_patience: int = 5,
    nb_frozen_graphs: int = 4,
) -> tuple[StreamCostModelEvaluation, list[dict[str, Any]]]:
    """Evolve an independent hardware graph alongside the workload mapping, in two steps (see
    `GraphEvolutionStage`): first the graph -- core designs and explicit links -- with the allocation onto it, on
    the mapping file's tiling; then, on the best graphs frozen, the mapping -- tiling and allocation. Scored on
    `(latency, energy, area)` by the real scheduler.

    `hardware` and `mapping` only seed the search: the mapping's tiling is step 1's tiling and the step-2 starting
    point; the hardware itself is replaced by the evolved graphs.

    Args:
        population_size: NSGA2 population (rounded up to a multiple of 4).
        graph_ga_generations: generations to run when no `search_budget` is given (half per step).
        max_workers: worker processes for the cost-model warm-ups and the schedule measurements.
        search_budget: optional wall-clock budget (see `stream.opt.search_budget`); when given, the search runs
            until it stops and every measured genome is recorded on it.
        graph_step_fraction: step 1 ends after this fraction of the budget at the latest.
        step1_patience: step 1 ends earlier once the best EDP gained less than the budget's `rel_tol` over this
            many generations.
        nb_frozen_graphs: how many of step 1's best distinct graphs step 2 maps onto.

    Returns:
        The best genome by EDP as a `StreamCostModelEvaluation`, and the final Pareto front as plain records.
        Artifacts go to `{output_path}/{experiment_id}/graph_search/`.
    """
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)
    os.makedirs(f"{output_path}/{experiment_id}", exist_ok=True)
    logger = _logging.getLogger(__name__)

    if temporal_mapping_type == "uneven":
        temporal_mapping_type = TemporalMappingType.UNEVEN
    elif temporal_mapping_type == "even":
        temporal_mapping_type = TemporalMappingType.EVEN
    else:
        raise ValueError(f"Invalid temporal mapping type: {temporal_mapping_type}. Must be 'uneven' or 'even'.")

    stage_classes = [
        AcceleratorParserStage,
        StreamONNXModelParserStage,
        LayerStacksGenerationStage,
        TilingGenerationStage,
        TiledWorkloadGenerationStage,
        GraphEvolutionStage,
        SetFixedAllocationPerformanceStage,
        SchedulingOrderGenerationStage,
        GeneticAlgorithmAllocationStage,
    ]

    # Must happen before anything forks, so every worker inherits the patch.
    ranges = CoreParamRanges(**(core_param_ranges or {}))
    t_precompute_start = _time.perf_counter()
    design_space = precompute_cacti_design_space(ranges, max_workers=cacti_precompute_workers)
    install_cacti_monkeypatch(design_space)
    logger.info(
        f"optimize_graph_evolution: precomputed {len(design_space)} CACTI config(s) in "
        f"{_time.perf_counter() - t_precompute_start:.1f}s."
    )

    mainstage = MainStage(
        stage_classes,
        accelerator=hardware,
        workload_path=workload,
        mapping_path=mapping,
        loma_lpf_limit=6,
        mode=mode,
        layer_stacks=layer_stacks,
        tiled_workload_path=f"{output_path}/{experiment_id}/tiled_workload.pickle",
        cost_lut_path=f"{output_path}/{experiment_id}/cost_lut.pickle",
        temporal_mapping_type=temporal_mapping_type,
        operands_to_prefetch=[],
        nb_ga_generations=nb_ga_generations,
        nb_ga_individuals=nb_ga_individuals,
        core_param_ranges=core_param_ranges,
        population_size=population_size,
        graph_ga_generations=graph_ga_generations,
        max_workers=max_workers,
        graph_search_dir=f"{output_path}/{experiment_id}/graph_search",
        search_budget=search_budget,
        graph_step_fraction=graph_step_fraction,
        step1_patience=step1_patience,
        nb_frozen_graphs=nb_frozen_graphs,
    )
    t_start = _time.perf_counter()
    answers = [answer for answer in mainstage.run() if answer[0] is not None]
    if not answers:
        raise ValueError("No graph could be evaluated.")
    best_scme, extra_info = answers[0]
    pickle_save(best_scme, f"{output_path}/{experiment_id}/scme.pickle")  # type: ignore
    logger.info(
        f"optimize_graph_evolution: {extra_info['nb_evaluated']} graph(s) measured in "
        f"{_time.perf_counter() - t_start:.1f}s; best by EDP: latency={best_scme.latency}, "
        f"energy={best_scme.energy}."
    )
    return best_scme, extra_info["pareto_graphs"]


def optimize_rolled_tiling(  # noqa: PLR0913
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    experiment_id: str,
    output_path: str,
    search_budget: SearchBudget,
    candidate_time_cap_s: float,
    temporal_mapping_type: str = "uneven",
    nb_tiling_ga_individuals: int = 4,
    nb_ga_generations: int = 4,
    nb_ga_individuals: int = 4,
    nb_core_ga_generations: int = 1000,
    nb_core_ga_individuals: int = 16,
    core_convergence_patience: int = 5,
    core_rel_tol: float = 0.005,
    core_time_fraction: float = 0.4,
    core_max_workers: int | None = None,
    core_pareto_points: int = 10,
    core_param_ranges: dict[str, Any] | None = None,
    cacti_precompute_workers: int | None = None,
    schedule_search_time_fraction: float = 0.6,
    fold_overlap: Literal["hull", "exact"] = "exact",
    fold_tolerances: tuple[float, ...] = (0.0, 0.25, 1.0),
) -> tuple[StreamCostModelEvaluation, list[tuple[StreamCostModelEvaluation, Any]]]:
    """The full rolled methodology: a GA over intra-/inter-core tilings, where every tiling candidate generates
    its own hardware graph through `RolledScheduleExplorationStage` -- a per-tile core NSGA2 (stopping on front
    hypervolume convergence), the unrolled schedule search over the per-tile fronts, then rolling onto fewer
    cores -- and is scored by the best EDP on its (latency, energy, area) front.

    Candidates run one at a time in this process (their inner searches use `core_max_workers`), each under
    `search_budget.sub_budget(candidate_time_cap_s)`: unrolled schedules are measured until
    `schedule_search_time_fraction` of that cap, folding uses the rest. The outer GA keeps generating
    candidates until `search_budget` stops. Every schedule measured anywhere is recorded on `search_budget`.
    The core GAs stop on convergence, or after `core_time_fraction` of the candidate's cap as a safety net, and
    each candidate measures at least one schedule even if it overruns its cap.

    Returns:
        The best candidate's schedule and every candidate's `(scme, extra_info)`, as `optimize_tiling` does; the
        extra info of each candidate carries its Pareto set (`pareto_schedules`). Artifacts go to
        `{output_path}/{experiment_id}/tiling_search/candidate_<key>/`.
    """
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)
    os.makedirs(f"{output_path}/{experiment_id}", exist_ok=True)
    logger = _logging.getLogger(__name__)
    if temporal_mapping_type == "uneven":
        temporal_mapping_type = TemporalMappingType.UNEVEN
    elif temporal_mapping_type == "even":
        temporal_mapping_type = TemporalMappingType.EVEN
    else:
        raise ValueError(f"Invalid temporal mapping type: {temporal_mapping_type}. Must be 'uneven' or 'even'.")

    outer_stage_classes = [
        AcceleratorParserStage,
        StreamONNXModelParserStage,
        LayerStacksGenerationStage,
        TilingExplorationStage,
    ]
    per_candidate_stage_classes = [
        TilingGenerationStage,
        TiledWorkloadGenerationStage,
        RolledScheduleExplorationStage,
        SetFixedAllocationPerformanceStage,
        SchedulingOrderGenerationStage,
        GeneticAlgorithmAllocationStage,
    ]

    # Must happen before anything forks, so every worker inherits the patch.
    ranges = CoreParamRanges(**(core_param_ranges or {}))
    t_precompute_start = _time.perf_counter()
    design_space = precompute_cacti_design_space(ranges, max_workers=cacti_precompute_workers)
    install_cacti_monkeypatch(design_space)
    logger.info(
        f"optimize_rolled_tiling: precomputed {len(design_space)} CACTI config(s) in "
        f"{_time.perf_counter() - t_precompute_start:.1f}s."
    )

    tiling_candidates_dir = f"{output_path}/{experiment_id}/tiling_search"
    mainstage = MainStage(
        outer_stage_classes + per_candidate_stage_classes,
        accelerator=hardware,
        workload_path=workload,
        mapping_path=mapping,
        loma_lpf_limit=6,
        mode=mode,
        layer_stacks=layer_stacks,
        temporal_mapping_type=temporal_mapping_type,
        operands_to_prefetch=[],
        # Outer tiling GA.
        tiling_candidates_dir=tiling_candidates_dir,
        scme_path=f"{output_path}/{experiment_id}/scme.pickle",
        sort_key="edp",
        nb_tiling_ga_generations=10**9,  # the budget stops it
        nb_tiling_ga_individuals=nb_tiling_ga_individuals,
        candidate_results_csv_path=f"{output_path}/{experiment_id}/tiling_search_all_candidates.csv",
        evaluate_in_process=True,
        candidate_time_cap_s=candidate_time_cap_s,
        search_budget=search_budget,
        # Per-candidate rolled search.
        nb_ga_generations=nb_ga_generations,
        nb_ga_individuals=nb_ga_individuals,
        nb_core_ga_generations=nb_core_ga_generations,
        nb_core_ga_individuals=nb_core_ga_individuals,
        core_convergence_patience=core_convergence_patience,
        core_rel_tol=core_rel_tol,
        core_time_fraction=core_time_fraction,
        core_max_workers=core_max_workers,
        core_pareto_points=core_pareto_points,
        core_param_ranges=core_param_ranges,
        max_schedule_evaluations=10**9,
        reuse_pareto_fronts=False,
        schedule_search_time_fraction=schedule_search_time_fraction,
        fold_overlap=fold_overlap,
        fold_tolerances=fold_tolerances,
        max_rolled_evaluations=10**9,
    )
    answers = mainstage.run()
    if not answers:
        raise ValueError("No tiling candidate produced a rolled schedule.")
    best_scme, best_extra_info = min(answers, key=lambda answer: answer[1].get("fitness", math.inf))
    logger.info(
        f"optimize_rolled_tiling: best tiling candidate {best_extra_info['candidate_index']} of {len(answers)} "
        f"(best EDP on its front={best_extra_info.get('fitness')})."
    )
    return best_scme, answers
