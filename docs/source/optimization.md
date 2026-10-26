# Optimization flows

`stream.api` exposes one function per design-space exploration. Each one checks its inputs, precomputes what
must exist before any worker process forks (the CACTI memory design space for the core searches), assembles a
chain of [stages](stages.md) and runs it. This page lists those chains and what each search hands to the next
stage.

All functions take the same inputs -- `hardware` (accelerator YAML), `workload` (ONNX), `mapping` (mapping YAML),
`mode` (`"lbl"` or `"fused"`), `layer_stacks`, `experiment_id` and `output_path` -- and write their artifacts
under `{output_path}/{experiment_id}/`.

Two stage groups recur in the chains below:

- **measurement** = `SetFixedAllocationPerformanceStage` -> `SchedulingOrderGenerationStage` ->
  `GeneticAlgorithmAllocationStage`: schedules one fixed accelerator (`MEASUREMENT_STAGES` in `stream.api`);
- **report** = `TopologyReportStage` -> `ScheduleTradeoffPlotStage` -> `ScheduleResultsWriterStage`: draws a
  schedule search's results (`SCHEDULE_REPORT_STAGES`).

---

## Fixed hardware

### `optimize_allocation_ga` / `optimize_allocation_co`

```
AcceleratorParser -> ONNXModelParser -> LayerStacksGeneration -> TilingGeneration -> TiledWorkloadGeneration
  -> ZigZagCoreMappingEstimation -> measurement            (GA allocation)
  -> ZigZagCoreMappingEstimation -> ConstraintOptimizationAllocation   (CO allocation, needs Gurobi)
```

The original Stream flow: the layer-core allocation is searched on the given accelerator and tiling.

### `optimize_tiling`

```
AcceleratorParser -> ONNXModelParser -> LayerStacksGeneration -> TilingExploration
   | per tiling candidate (worker processes):
   +-> TilingGeneration -> TiledWorkloadGeneration
       [-> CoreArchitectureExploration -> CoreDesignSelection -> CustomCoreAcceleratorGeneration]
       -> ZigZagCoreMappingEstimation -> measurement
```

A genetic algorithm over every layer's intra-/inter-core tiling. With `explore_cores=True` (the default), each
candidate also gets its own accelerator: one Pareto front of core designs per unique tile, collapsed to its
best design by `sort_key`. Use `explore_cores=False` to keep the hardware fixed.

---

## Schedule searches over core designs

### `optimize_schedules`

```
... -> TiledWorkloadGeneration -> report
  -> CoreArchitectureExploration -> ScheduleExploration -> ScheduleSelection -> measurement
```

Keeps *every* design of the per-tile fronts and searches their combinations with a branch and bound. Each
combination is measured with the real scheduler, and the result is the `(latency, energy, area)` Pareto front
of whole schedules. Artifacts go to `schedule_search/`.

### `optimize_rolled_schedules`

```
... -> TiledWorkloadGeneration -> report
  -> CoreArchitectureExploration -> ScheduleExploration -> ScheduleRolling -> ScheduleSelection -> measurement
```

Adds the rolling step: each schedule found is folded onto fewer physical cores (cores whose busy windows do
not collide share one core), re-measured, and added to the same front. Artifacts go to `rolled_search/`.

### `optimize_rolled_tiling` (the rolled methodology)

```
AcceleratorParser -> ONNXModelParser -> LayerStacksGeneration -> TilingExploration
   | per tiling candidate (one at a time, in process):
   +-> TilingGeneration -> TiledWorkloadGeneration -> report
       -> CoreArchitectureExploration -> ScheduleExploration -> ScheduleRolling -> ScheduleSelection
       -> measurement
```

The tiling GA of `optimize_tiling` wrapped around the rolled schedule search. A candidate's fitness is the best
EDP on its front. Every measurement is recorded on the given `SearchBudget`, which stops the tiling GA (and the
per-tile core GAs) from starting a new generation once time runs out or the best EDP converges.

### `optimize_graph_evolution`

```
... -> TiledWorkloadGeneration -> GraphEvolution -> measurement
```

The alternative to the rolled methodology. Instead of deriving the topology from the tiling, an NSGA2 evolves
a free hardware graph (core designs, explicit links) together with the allocation onto it, then freezes the
best graphs and evolves the mapping (tiling and allocation) on them. Artifacts go to `graph_search/`.

### `optimize_placement_tiling`

```
... -> TilingExploration -> [TilingGeneration -> TiledWorkloadGeneration -> CoreArchitectureExploration
    -> ScheduleExploration -> {CpSat,Gurobi}PlacementOptimization -> ScheduleSelection -> measurement]
```

The chain of `optimize_model_tiling` with the placement optimization in place of the level's stage: per tiling
candidate, per-tile core fronts (core NSGA2 run to hypervolume convergence), the full cost matrix, then the
placement sweep solved with CP-SAT or Gurobi. `level` only sets the tasks per group. Reports go to
`tiling_search/candidate_<key>/placement_<solver>/`.

### `optimize_template_accelerator`

```
AcceleratorParser -> ONNXModelParser -> LayerStacksGeneration -> TemplateExploration
    -> [TilingGeneration -> TiledWorkloadGeneration -> ZigZagCoreMappingEstimation -> measurement]
```

NSGA2 over a homogeneous grid of one core template (e.g. `cores/edge_tpu_like.yaml`): grid rows and columns, the
template's array-dimension and memory sizes (memory costs from CACTI, precomputed), and each layer's inter-core
tiling. The intra-core tiling is the mapping's. The seed accelerator and mapping are written by
`grid_generator.write_template_grid`. Every evaluated design is listed in `template_search/evaluated_designs.csv`.

---

## What the schedule search stages hand down

| Produced by | Keyword | Content |
|---|---|---|
| `CoreArchitectureExplorationStage` | `core_pareto_fronts` | `ParetoCoreSearchResult`: the designs kept per unique tile, the unique tiles and how the workload's tiles map onto them. It also re-forwards `workload`, `original_workload`, `accelerator`, `tiled_workload_path` and `core_max_workers`. |
| `CoreDesignSelectionStage` | `node_core_dicts` | The core design of every original layer, for `CustomCoreAcceleratorGenerationStage`. |
| `ScheduleExplorationStage` | `schedule_search` | `ScheduleSearchState`: the `ScheduleSpace` (design-independent plan, cost-model cache, measurement), the Pareto archive, every measured record, the search statistics and the report context. |
| `ScheduleRollingStage` | `schedule_search` | The same state, with the folded schedules added to the archive and the records. |
| `ScheduleSelectionStage` | `extra_info` (yielded) | `schedule_ranks`, `best_schedule`, `pareto_schedules`, `all_schedules`, `search_stats` and `schedule_report` (plain data for the visualization stages). |

The measurement pipeline of a schedule search is whatever follows the last schedule search stage in the chain;
see [Stages](stages.md).

---

## Comparing the optimizers

`benchmarks/benchmark_optimizer_comparison.py` runs the rolled methodology, the graph GA and tiling GAs on fixed 2x2 and 4x4
grids under one wall-clock `SearchBudget`. `benchmarks/analyze_optimizer_comparison.py` then turns the runs into
convergence and Pareto-front figures (`stream.visualization.optimizer_comparison`), a comparison table and an
interactive page (`stream.visualization.comparison_report`). See the docstrings of both scripts for their
options.

`benchmarks/benchmark_accelerator_comparison.py` compares three accelerator design flows the same way: the
tiling GA with per-tile core search and the level-2 placement optimization (CP-SAT and Gurobi), the graph GA, and
the edge_tpu template NSGA2. Its output is read by the same analysis script (`--results-dir
outputs/accelerator_comparison`).
