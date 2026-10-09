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

`benchmark_optimizer_comparison.py` runs the rolled methodology, the graph GA and tiling GAs on fixed 2x2 and 4x4
grids under one wall-clock `SearchBudget`. `analyze_optimizer_comparison.py` then turns the runs into
convergence and Pareto-front figures (`stream.visualization.optimizer_comparison`), a comparison table and an
interactive page (`stream.visualization.comparison_report`). See the docstrings of both scripts for their
options.
