import csv
import logging as _logging
import os
import time as _time
from typing import Any, Literal

import gurobipy as gp
from onnx import ModelProto
from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.utils import pickle_load, pickle_save

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.stages.allocation.constraint_optimization_allocation import ConstraintOptimizationAllocationStage
from stream.stages.allocation.genetic_algorithm_allocation import GeneticAlgorithmAllocationStage
from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.scheduling_order_generation import SchedulingOrderGenerationStage
from stream.stages.generation.tiled_workload_accelerator_generation import TiledWorkloadAcceleratorGenerationStage
from stream.stages.generation.tiled_workload_generation import TiledWorkloadGenerationStage
from stream.stages.generation.tiling_exploration import TilingExplorationStage
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


def optimize_tiling(  # noqa: PLR0912, PLR0913, PLR0915
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    allocation_strategy: Literal["ga", "co"],
    experiment_id: str,
    output_path: str,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
    nb_ga_generations: int = 4,
    nb_ga_individuals: int = 4,
    nb_tiling_ga_generations: int = 4,
    nb_tiling_ga_individuals: int = 8,
    sort_key: Literal["latency", "energy"] = "latency",
    max_workers: int | None = None,
    profile: bool = False,
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

    Args:
        allocation_strategy: which allocation stage to run per candidate, "ga" for GeneticAlgorithmAllocationStage
            or "co" for ConstraintOptimizationAllocationStage. Note: inter-core tiling is only swept under "ga" —
            under "co" it is left as a wildcard for `ConstraintOptimizationAllocationStage` to resolve itself,
            since it asserts an exact match against the mapping's core_allocation length for any fixed value.
        nb_tiling_ga_generations: number of generations for the outer tiling-search genetic algorithm run by
            `TilingExplorationStage`. Independent of `nb_ga_generations`, which only sizes the inner
            per-candidate `GeneticAlgorithmAllocationStage` core-allocation search.
        nb_tiling_ga_individuals: population size for the outer tiling-search genetic algorithm. Independent of
            `nb_ga_individuals`, for the same reason as above.
        max_workers: number of worker processes to evaluate candidates in parallel. Defaults to
            `os.process_cpu_count()` (via `ProcessPoolExecutor`'s own default) when None.
    """
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)
    if allocation_strategy == "co":
        _sanity_check_gurobi_license()

    # Create experiment_id path
    os.makedirs(f"{output_path}/{experiment_id}", exist_ok=True)

    # Output paths
    tiling_candidates_dir = f"{output_path}/{experiment_id}/tiling_search"
    scme_path = f"{output_path}/{experiment_id}/scme.pickle"
    results_csv_path = f"{output_path}/{experiment_id}/tiling_search_results.csv"

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

    allocation_stage_kwargs: dict[str, Any]
    # NOTE: GA and CO use different downstream stage chains (see optimize_allocation_ga/optimize_allocation_co):
    # GA needs a fixed scheduling order and pre-fixed single-core allocations before searching; CO resolves
    # allocation (and any inter-core tiling wildcards) itself and builds its own internal sub-pipeline.
    if allocation_strategy == "ga":
        allocation_stage_kwargs = {
            "nb_ga_generations": nb_ga_generations,
            "nb_ga_individuals": nb_ga_individuals,
        }
        allocation_stage_classes = [
            SetFixedAllocationPerformanceStage,
            SchedulingOrderGenerationStage,
            GeneticAlgorithmAllocationStage,
        ]
    elif allocation_strategy == "co":
        allocation_stage_kwargs = {
            "allocations_path": f"{output_path}/{experiment_id}/waco/",
            "tiled_workload_post_co_path": f"{output_path}/{experiment_id}/tiled_workload_post_co.pickle",
            "cost_lut_post_co_path": f"{output_path}/{experiment_id}/cost_lut_post_co.pickle",
        }
        allocation_stage_classes = [ConstraintOptimizationAllocationStage]
    else:
        raise ValueError(f"Invalid allocation strategy: {allocation_strategy}. Must be 'ga' or 'co'.")

    # Stages up to and including TilingExplorationStage run once in this process; everything after it
    # (TilingGenerationStage onward) runs per-candidate inside worker processes spawned by
    # TilingExplorationStage, so those classes must stay picklable and are never timing-wrapped below.
    outer_stage_classes = [
        AcceleratorParserStage,  # Parses the accelerator
        StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
        LayerStacksGenerationStage,
        TilingExplorationStage,  # Sweeps candidate intra-/inter-core tiling configurations
    ]
    per_candidate_stage_classes = [
        TilingGenerationStage,
        TiledWorkloadGenerationStage,
        TiledWorkloadAcceleratorGenerationStage,
        ZigZagCoreMappingEstimationStage,
        *allocation_stage_classes,
    ]
    timings: dict[str, list[float]] = {}
    if profile:
        outer_stage_classes, timings = wrap_stages_with_timing(outer_stage_classes)
    list_of_callables = outer_stage_classes + per_candidate_stage_classes

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
        sort_key=sort_key,  # required by TilingExplorationStage
        max_workers=max_workers,  # required by TilingExplorationStage
        nb_tiling_ga_generations=nb_tiling_ga_generations,  # required by TilingExplorationStage
        nb_tiling_ga_individuals=nb_tiling_ga_individuals,  # required by TilingExplorationStage
        temporal_mapping_type=temporal_mapping_type,  # required by ZigZagCoreMappingEstimationStage
        operands_to_prefetch=[],  # required by GeneticAlgorithmAllocationStage/ConstraintOptimizationAllocationStage
        **allocation_stage_kwargs,
    )
    # Launch the MainStage
    t_start = _time.perf_counter()
    answers = mainstage.run()
    total_time = _time.perf_counter() - t_start

    if not answers:
        raise ValueError("No candidate tiling configuration produced a result.")

    best_scme, best_extra_info = min(answers, key=lambda answer: getattr(answer[0], sort_key))
    pickle_save(best_scme, scme_path)  # type: ignore
    logger.info(
        f"Best tiling candidate {best_extra_info['candidate_index']} out of {len(answers)}: "
        f"{best_extra_info['tiling_config']} ({sort_key}={getattr(best_scme, sort_key)})."
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
