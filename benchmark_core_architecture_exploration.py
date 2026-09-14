"""Fine-grained subpart timing for `CoreArchitectureExplorationStage`'s per-candidate evaluation.

Default mode picks one unique tile (one part of the neural network) and one random core design (one candidate
architecture), then times each subpart of evaluating that design on that tile: gene decode/validate, core
building (including any CACTI subprocess calls), tile deep-copy, and -- broken open, since it's normally one
opaque `evaluate_node_on_core` call -- ZigZag's own spatial-mapping search, LOMA temporal-mapping search, and
cost model evaluation.

`--aggregate` instead precomputes the whole CACTI design space up front (see
`stream.hardware.architecture.cacti_precompute`, matching how `optimize_tiling(core_search=True)` now runs)
and samples several random candidates per unique tile, reporting the same phase breakdown aggregated across all
of them -- a statistically meaningful picture of where `CoreArchitectureExplorationStage` spends its time,
rather than one data point.

Usage: `python benchmark_core_architecture_exploration.py [options]` (see --help).
"""

import argparse
import logging as _logging
import os
import random
import time
from collections import defaultdict
from math import prod
from typing import Any

import onnx
import torch
from onnx.shape_inference import infer_shapes_path
from torch import nn
from zigzag.cacti.cacti_parser import CactiParser
from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.stages.evaluation.cost_model_evaluation import CostModelStage
from zigzag.stages.mapping.spatial_mapping_generation import SpatialMappingGeneratorStage
from zigzag.stages.mapping.temporal_mapping_generator_stage import TemporalMappingGeneratorStage
from zigzag.utils import pickle_deepcopy

from stream.hardware.architecture.cacti_precompute import install_cacti_monkeypatch, precompute_cacti_design_space
from stream.hardware.architecture.core_generator import build_core_from_dict
from stream.stages.estimation.zigzag_core_mapping_estimation import MinimalBandwidthLatencyStage
from stream.stages.generation.core_architecture_exploration import CoreArchitectureExplorationStage, CoreGeneLayout
from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.tiled_workload_generation import TiledWorkloadGenerationStage
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage as StreamONNXModelParserStage
from stream.stages.stage import MainStage, Stage, StageCallable
from stream.utils import contains_wildcard

_logging.basicConfig(level=_logging.INFO, format="%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s")
logger = _logging.getLogger(__name__)


class _CaptureStage(Stage):
    """Leaf stage that stashes `workload`/`original_workload`/`accelerator` into a shared dict passed via
    kwargs, so a minimal setup pipeline can hand them back to this script without running the full downstream
    pipeline (mirrors the `_stage_timings`-via-kwargs idiom used in `stream.utils`/`TilingExplorationStage`)."""

    def is_leaf(self) -> bool:
        return True

    def run(self):
        capture = self.kwargs["_capture"]
        capture["workload"] = self.kwargs["workload"]
        capture["original_workload"] = self.kwargs["original_workload"]
        capture["accelerator"] = self.kwargs["accelerator"]
        yield (None, None)


class _NullLeafStage(Stage):
    """Only exists to satisfy `Stage.__init__`'s non-empty-`list_of_callables` check when constructing a
    `CoreArchitectureExplorationStage` purely as a probe (its `.run()` is never called here)."""

    def is_leaf(self) -> bool:
        return True

    def run(self):
        yield (None, None)


def _ensure_default_workload(workload_path: str) -> None:
    """Regenerate the small ResNet50-first-bottleneck ONNX workload if it isn't there (same block as
    `main_stream_tiling_search.py`) -- 3 conv layers, already proven to work with `core_search=True`."""
    if os.path.exists(workload_path):
        return

    class ResNet50FirstBottleneck(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
            self.bn3 = nn.BatchNorm2d(256)
            self.downsample_conv = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
            self.downsample_bn = nn.BatchNorm2d(256)
            self.relu = nn.ReLU(inplace=False)

        def forward(self, input_tensor):
            identity = self.downsample_bn(self.downsample_conv(input_tensor))
            out = self.relu(self.bn1(self.conv1(input_tensor)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out = out + identity
            return self.relu(out)

    os.makedirs(os.path.dirname(workload_path), exist_ok=True)
    dummy_input = torch.randn(1, 64, 56, 56)
    torch.onnx.export(
        ResNet50FirstBottleneck().eval(),
        dummy_input,
        workload_path,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    infer_shapes_path(workload_path, workload_path)
    logger.info(f"Generated default benchmark workload at {workload_path}.")


def _build_inputs(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """Run the minimal prefix of `optimize_tiling`'s pipeline needed to produce a real `(workload,
    original_workload, accelerator)` triple, capturing them via `_CaptureStage` instead of hand-building a
    `ComputationNodeWorkload` from scratch."""
    _ensure_default_workload(args.workload)
    nb_nodes = len(onnx.load(args.workload).graph.node)
    layer_stacks = [tuple(range(nb_nodes))]

    os.makedirs(args.output_dir, exist_ok=True)
    capture: dict[str, Any] = {}
    setup_stages: list[StageCallable] = [
        AcceleratorParserStage,
        StreamONNXModelParserStage,
        LayerStacksGenerationStage,
        TilingGenerationStage,
        TiledWorkloadGenerationStage,
        _CaptureStage,
    ]
    MainStage(
        setup_stages,
        accelerator=args.hardware,
        workload_path=args.workload,
        mapping_path=args.mapping,
        mode=args.mode,
        layer_stacks=layer_stacks,
        tiled_workload_path=f"{args.output_dir}/tiled_workload.pickle",
        _capture=capture,
    ).run()
    return capture["workload"], capture["original_workload"], capture["accelerator"]


# Descriptive labels for `evaluate_node_on_core`'s own internal sub-pipeline (see that function): the two
# `MinimalBandwidthLatencyStage` occurrences get distinct labels here (unlike `stream.utils.wrap_stages_with_timing`,
# which would collapse same-named stages together) since they sit at different points in the mapping search.
_EVALUATE_STAGE_LABELS = (
    "bandwidth_latency_pre_spatial",
    "spatial_mapping",
    "bandwidth_latency_pre_temporal",
    "loma_mapping",
    "cost_model",
)


def _evaluate_node_on_core_with_timing(
    node: Any, core: Any, *, loma_lpf_limit: int, temporal_mapping_type: TemporalMappingType
) -> tuple[Any, dict[str, float]]:
    """Same sub-flow as `evaluate_node_on_core` (mirrored here rather than reused, so each of its 5 stages can
    be individually timed), returning `(cme, {stage_label: seconds})`."""
    nb_parallel_nodes = (
        1 if contains_wildcard(node.inter_core_tiling) else prod(size for _, size in node.inter_core_tiling)
    )
    stage_classes = [
        MinimalBandwidthLatencyStage,
        SpatialMappingGeneratorStage,
        MinimalBandwidthLatencyStage,
        TemporalMappingGeneratorStage,
        CostModelStage,
    ]
    timings: dict[str, float] = dict.fromkeys(_EVALUATE_STAGE_LABELS, 0.0)

    def _wrap(stage_cls: type, label: str) -> type:
        class TimedStage(stage_cls):  # type: ignore[misc,valid-type]
            def __init__(self, list_of_callables, **kwargs):
                t0 = time.perf_counter()
                super().__init__(list_of_callables, **kwargs)
                timings[label] += time.perf_counter() - t0

            def run(self):
                t0 = time.perf_counter()
                try:
                    yield from super().run()
                finally:
                    timings[label] += time.perf_counter() - t0

        return TimedStage

    wrapped = [_wrap(cls, label) for cls, label in zip(stage_classes, _EVALUATE_STAGE_LABELS, strict=True)]
    main_stage = MainStage(
        wrapped,
        layer=node,
        accelerator=core,  # Accelerator in zigzag corresponds to Core in stream
        loma_lpf_limit=loma_lpf_limit,
        loma_show_progress_bar=False,
        temporal_mapping_type=temporal_mapping_type,
        nb_parallel_nodes=nb_parallel_nodes,
        has_dram_level=False,
    )
    answers = main_stage.run()
    assert len(answers) == 1, "evaluate_node_on_core's subflow returned more than one CME"

    # `timings` so far holds each stage's INCLUSIVE time: stages are chained via `yield from sub_stage.run()`,
    # so e.g. `loma_mapping`'s timer stays on the stack for the whole nested `cost_model` evaluation too (and
    # some of these stages, like `spatial_mapping`, invoke that nested chain more than once -- once per
    # candidate they try -- with every invocation's time still accumulating into the same total). Subtract
    # each stage's own total from the one after it in pipeline order to get its exclusive (non-nested) time,
    # the same technique `stream.utils.get_exclusive_stage_times` uses.
    totals = [timings[label] for label in _EVALUATE_STAGE_LABELS]
    exclusive_timings = {
        label: totals[i] - (totals[i + 1] if i + 1 < len(totals) else 0.0)
        for i, label in enumerate(_EVALUATE_STAGE_LABELS)
    }
    return answers[0][0], exclusive_timings


def _profile_one_candidate(
    tile: Any,
    values: list[float],
    name: str,
    gene_layout: CoreGeneLayout,
    args: argparse.Namespace,
    temporal_mapping_type: TemporalMappingType,
) -> tuple[dict[str, float], dict[str, list[float]], bool, float, float, float]:
    """Times each subpart of evaluating one gene vector on one tile. Returns `(phases, cacti_calls,
    succeeded, latency, energy, area)`, where `cacti_calls` is `{"lookup": [...], "subprocess": [...]}` --
    `lookup` is every `CactiParser.get_item` call (cache hit or miss; a hit still means scanning/parsing the
    whole on-disk memory-pool YAML), `subprocess` is the subset of those that were a cache miss and had to
    shell out to the real CACTI simulator. On failure `phases` holds whichever subparts completed and the
    last three values are NaN."""
    cacti_calls: dict[str, list[float]] = {"lookup": [], "subprocess": []}
    original_get_item = CactiParser.get_item
    original_create_item = CactiParser.create_item

    def _timed_get_item(self: CactiParser, *call_args: Any, **call_kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return original_get_item(self, *call_args, **call_kwargs)
        finally:
            cacti_calls["lookup"].append(time.perf_counter() - t0)

    def _timed_create_item(self: CactiParser, *call_args: Any, **call_kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return original_create_item(self, *call_args, **call_kwargs)
        finally:
            cacti_calls["subprocess"].append(time.perf_counter() - t0)

    CactiParser.get_item = _timed_get_item  # type: ignore[method-assign]
    CactiParser.create_item = _timed_create_item  # type: ignore[method-assign]

    phases: dict[str, float] = {}
    latency = energy = area = float("nan")
    succeeded = False
    try:
        t0 = time.perf_counter()
        core_dict = gene_layout.decode_node(values, name=name)
        t1 = time.perf_counter()
        phases["decode"] = t1 - t0

        core = build_core_from_dict(core_dict, core_id=0)
        t2 = time.perf_counter()
        # `get_item` (cache lookup, hit or miss) runs synchronously inside `build_core_from_dict` -- carve its
        # time out into its own row rather than leaving it hidden inside "build_core".
        cacti_lookup_total = sum(cacti_calls["lookup"])
        phases["cacti_lookup"] = cacti_lookup_total
        phases["build_core"] = (t2 - t1) - cacti_lookup_total

        node = pickle_deepcopy(tile)
        t3 = time.perf_counter()
        phases["deepcopy"] = t3 - t2

        cme, evaluate_phases = _evaluate_node_on_core_with_timing(
            node, core, loma_lpf_limit=args.loma_lpf_limit, temporal_mapping_type=temporal_mapping_type
        )
        phases.update(evaluate_phases)

        t4 = time.perf_counter()
        area = core.get_area()
        phases["area"] = time.perf_counter() - t4

        latency, energy = cme.latency_total2, cme.energy_total
        succeeded = True
    except Exception:
        logger.exception(f"Evaluation of '{name}' failed; reporting partial timings only.")
    finally:
        CactiParser.get_item = original_get_item  # type: ignore[method-assign]
        CactiParser.create_item = original_create_item  # type: ignore[method-assign]

    return phases, cacti_calls, succeeded, latency, energy, area


def _print_report(
    tile_index: int,
    tile: Any,
    phases: dict[str, float],
    cacti_calls: dict[str, list[float]],
    succeeded: bool,
    latency: float,
    energy: float,
    area: float,
) -> None:
    total_time = sum(phases.values())
    print(f"\n=== Subpart timing: 1 design on unique tile {tile_index} ({'ok' if succeeded else 'FAILED'}) ===")
    print(f"tile: {tile}")
    if succeeded:
        print(f"result: latency={latency}, energy={energy}, area={area}")
    print(f"{'phase':>30} {'time_s':>10} {'pct':>6}")
    for phase, elapsed in sorted(phases.items(), key=lambda item: item[1], reverse=True):
        pct = 100 * elapsed / total_time if total_time else 0.0
        print(f"{phase:>30} {elapsed:>10.4f} {pct:>5.1f}%")
    print(f"{'total':>30} {total_time:>10.4f} {100.0 if total_time else 0.0:>5.1f}%")

    lookup_calls, subprocess_calls = cacti_calls["lookup"], cacti_calls["subprocess"]
    if lookup_calls:
        print(
            f"\ncacti_lookup: {len(lookup_calls)} call(s), total={sum(lookup_calls):.4f}s "
            f"(subset of build_core+cacti_lookup above; every CactiParser.get_item() call -- CactiParser now "
            f"memoizes per-process in memory, so a repeat of the same config within this run would be ~free "
            f"and wouldn't show up under cacti_subprocess below)"
        )
    if subprocess_calls:
        print(
            f"cacti_subprocess: {len(subprocess_calls)} call(s), total={sum(subprocess_calls):.4f}s "
            f"(subset of cacti_lookup above; only the calls that were new to this process and had to actually "
            f"run the CACTI simulator)"
        )
    elif lookup_calls:
        print("cacti_subprocess: 0 calls -- every memory config hit CACTI's on-disk cache.")


def _print_aggregate_report(
    agg_phases: dict[str, float], n_ok: int, n_fail: int, total_wall: float, design_space_size: int
) -> None:
    grand_total = sum(agg_phases.values())
    n_total = n_ok + n_fail
    print(f"\n=== Aggregate report: {n_ok} ok, {n_fail} failed, total wall time {total_wall:.2f}s ===")
    print(f"CACTI design space precomputed: {design_space_size} config(s) (see cacti_precompute_workers/-s below)")
    print(f"{'phase':>30} {'total_s':>10} {'pct':>6} {'mean_per_candidate_s':>22}")
    for phase, elapsed in sorted(agg_phases.items(), key=lambda item: item[1], reverse=True):
        pct = 100 * elapsed / grand_total if grand_total else 0.0
        mean = elapsed / n_ok if n_ok else 0.0
        print(f"{phase:>30} {elapsed:>10.4f} {pct:>5.1f}% {mean:>22.4f}")
    print(f"{'sum(all phases)':>30} {grand_total:>10.4f} {100.0 if grand_total else 0.0:>5.1f}%")
    if n_total:
        print(f"\nmean wall time per candidate (incl. decode/build_core/deepcopy): {total_wall / n_total:.4f}s")


def _run_aggregate(args: argparse.Namespace) -> None:
    """Precomputes the CACTI design space once (matching the fixed `optimize_tiling(core_search=True)`
    pipeline), then samples `args.samples_per_tile` random candidates on every unique tile and reports the
    aggregate phase breakdown -- a statistically meaningful picture instead of one data point."""
    if args.seed is not None:
        random.seed(args.seed)

    workload, original_workload, accelerator = _build_inputs(args)
    temporal_mapping_type = (
        TemporalMappingType.UNEVEN if args.temporal_mapping_type == "uneven" else TemporalMappingType.EVEN
    )

    probe = CoreArchitectureExplorationStage(
        [_NullLeafStage],
        workload=workload,
        original_workload=original_workload,
        accelerator=accelerator,
        tiled_workload_path=f"{args.output_dir}/tiled_workload.pickle",
        loma_lpf_limit=args.loma_lpf_limit,
        temporal_mapping_type=temporal_mapping_type,
        core_param_ranges=None,
    )
    unique_tiles, _ = probe._unique_tiles()  # noqa: SLF001
    gene_layout = CoreGeneLayout(probe.gene_ranges)
    logger.info(f"{len(unique_tiles)} unique tile(s) found.")

    logger.info("Precomputing CACTI design space...")
    t0 = time.perf_counter()
    design_space = precompute_cacti_design_space(probe.gene_ranges, max_workers=args.cacti_precompute_workers)
    install_cacti_monkeypatch(design_space)
    logger.info(f"Precomputed {len(design_space)} CACTI config(s) in {time.perf_counter() - t0:.1f}s.")

    agg_phases: dict[str, float] = defaultdict(float)
    n_ok = n_fail = 0
    total_wall = 0.0
    logger.info(f"Sampling {args.samples_per_tile} random candidate(s) per tile across {len(unique_tiles)} tile(s)...")
    for tile_index, tile in enumerate(unique_tiles):
        for sample in range(args.samples_per_tile):
            values = [random.uniform(lo, hi) for lo, hi in zip(gene_layout.low, gene_layout.high, strict=True)]
            name = f"bench_tile{tile_index}_s{sample}"
            t_start = time.perf_counter()
            phases, _cacti_calls, succeeded, _latency, _energy, _area = _profile_one_candidate(
                tile, values, name, gene_layout, args, temporal_mapping_type
            )
            total_wall += time.perf_counter() - t_start
            if succeeded:
                n_ok += 1
                for label, elapsed in phases.items():
                    agg_phases[label] += elapsed
            else:
                n_fail += 1
        logger.info(f"tile {tile_index}/{len(unique_tiles) - 1} done ({tile}).")

    _print_aggregate_report(agg_phases, n_ok, n_fail, total_wall, len(design_space))


def run(args: argparse.Namespace) -> None:
    if args.aggregate:
        _run_aggregate(args)
        return

    if args.seed is not None:
        random.seed(args.seed)

    workload, original_workload, accelerator = _build_inputs(args)
    temporal_mapping_type = (
        TemporalMappingType.UNEVEN if args.temporal_mapping_type == "uneven" else TemporalMappingType.EVEN
    )

    probe = CoreArchitectureExplorationStage(
        [_NullLeafStage],
        workload=workload,
        original_workload=original_workload,
        accelerator=accelerator,
        tiled_workload_path=f"{args.output_dir}/tiled_workload.pickle",
        loma_lpf_limit=args.loma_lpf_limit,
        temporal_mapping_type=temporal_mapping_type,
        core_param_ranges=None,
    )
    unique_tiles, _ = probe._unique_tiles()  # noqa: SLF001
    gene_layout = CoreGeneLayout(probe.gene_ranges)

    tile_index = args.tile_index if args.tile_index is not None else random.randrange(len(unique_tiles))
    tile = unique_tiles[tile_index]
    values = [random.uniform(lo, hi) for lo, hi in zip(gene_layout.low, gene_layout.high, strict=True)]
    name = f"bench_tile{tile_index}"

    logger.info(f"Profiling one random design on unique tile {tile_index}/{len(unique_tiles) - 1}: {tile}.")

    phases, cacti_calls, succeeded, latency, energy, area = _profile_one_candidate(
        tile, values, name, gene_layout, args, temporal_mapping_type
    )
    _print_report(tile_index, tile, phases, cacti_calls, succeeded, latency, energy, area)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hardware", default="stream/inputs/examples/hardware/tpu_like_quad_core.yaml")
    parser.add_argument("--workload", default="stream/inputs/examples/workload/resnet50_first_bottleneck.onnx")
    parser.add_argument("--mapping", default="stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml")
    parser.add_argument("--mode", default="fused", choices=["fused", "lbl"])
    parser.add_argument("--loma-lpf-limit", type=int, default=6)
    parser.add_argument("--temporal-mapping-type", default="uneven", choices=["uneven", "even"])
    parser.add_argument("--output-dir", default="outputs/core_architecture_benchmark")
    parser.add_argument("--tile-index", type=int, default=None, help="Which unique tile to profile; random if omitted.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for a reproducible tile/design pick.")
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help=(
            "Instead of profiling one random candidate, precompute the whole CACTI design space up front "
            "(matching optimize_tiling(core_search=True)'s fix) and sample --samples-per-tile random "
            "candidates on every unique tile, reporting the aggregate phase breakdown."
        ),
    )
    parser.add_argument(
        "--samples-per-tile", type=int, default=6, help="Only used with --aggregate: candidates sampled per tile."
    )
    parser.add_argument(
        "--cacti-precompute-workers",
        type=int,
        default=None,
        help="Only used with --aggregate: worker processes for the CACTI precompute pass (default: every core).",
    )
    return parser.parse_args()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
