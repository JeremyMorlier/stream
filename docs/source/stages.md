# Stages

Stream runs as a chain of **stages**. Each stage receives keyword arguments from the one before it, does one
job (parse a file, tile a workload, search core designs, ...) and hands the result to the next stage as new
keyword arguments. Answers -- `(scme, extra_info)` pairs -- travel back up the chain, so a stage can also
look at, filter or re-rank what the stages after it produced.

---

## How a chain runs

A chain is a list of stage classes given to `MainStage` together with the initial keyword arguments:

```python
from stream.stages.stage import MainStage

answers = MainStage(
    [
        AcceleratorParserStage,
        StreamONNXModelParserStage,
        LayerStacksGenerationStage,
        TilingGenerationStage,
        TiledWorkloadGenerationStage,
        ZigZagCoreMappingEstimationStage,
        SetFixedAllocationPerformanceStage,
        SchedulingOrderGenerationStage,
        GeneticAlgorithmAllocationStage,
    ],
    accelerator="stream/inputs/examples/hardware/tpu_like_quad_core.yaml",
    workload_path="stream/inputs/examples/workload/resnet18.onnx",
    mapping_path="stream/inputs/examples/mapping/tpu_like_quad_core.yaml",
    ...
).run()
scme, extra_info = answers[0]
```

`MainStage` builds the first stage with the rest of the list. Every stage then builds the next one:

```python
def run(self):
    sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=accelerator, **self.kwargs)
    yield from sub_stage.run()
```

A stage declares the keyword arguments it uses as named parameters of its `__init__`; everything else lands in
`self.kwargs` and is passed on. A named parameter is *consumed*: if a later stage also needs it, the stage
must put it back into the kwargs it forwards.

The functions in `stream.api` (`optimize_allocation_ga`, `optimize_tiling`, `optimize_rolled_tiling`, ...)
assemble the usual chains; see [Optimization flows](optimization.md).

---

## Implemented stages

All stages live in [`stream/stages/`](https://github.com/KULeuven-MICAS/stream/tree/master/stream/stages),
grouped by what they do.

### Parsing -- `stream/stages/parsing/`

| Stage | Role |
|---|---|
| `AcceleratorParserStage` | Parses the hardware YAML into an `Accelerator`. |
| `ONNXModelParserStage` (imported as `StreamONNXModelParserStage`) | Parses the ONNX model and the mapping file into a workload graph. |
| `UserDefinedModelParserStage` | Same, for a workload described in Python instead of ONNX. |

### Generation -- `stream/stages/generation/`

| Stage | Role |
|---|---|
| `LayerStacksGenerationStage` | Groups layers into fused layer stacks (`layer_stacks`). |
| `TilingGenerationStage` | Resolves each layer's intra-/inter-core tiling. |
| `TiledWorkloadGenerationStage` | Splits every layer into tiles and builds the tile-level dependency graph. |
| `SchedulingOrderGenerationStage` | Fixes the order in which tiles are scheduled. |
| `TiledWorkloadAcceleratorGenerationStage` | Builds a group-dedicated accelerator (one core per layer group) from a fixed core template. |

### Exploration -- `stream/stages/generation/`

These search a design space. Each one runs the stages after it many times, once per candidate.

| Stage | Role |
|---|---|
| `TilingExplorationStage` | Genetic algorithm over every layer's intra-/inter-core tiling. Each individual runs the rest of the chain (in worker processes, or one at a time with `evaluate_in_process`), and its fitness is the best `sort_key` it returns. |
| `CoreArchitectureExplorationStage` | Per unique tile shape, an NSGA2 search over core designs (memory sizes and bandwidths, array size) on `(latency, energy, area)`. Hands the Pareto fronts down as `core_pareto_fronts`. |
| `CoreDesignSelectionStage` | Keeps each front's best design by `sort_key` and passes `node_core_dicts` on. |
| `CustomCoreAcceleratorGenerationStage` | Builds the group-dedicated accelerator from `node_core_dicts`. |
| `ScheduleExplorationStage` | Branch and bound over *combinations* of the front designs, measuring whole schedules with the real scheduler. Hands down a `ScheduleSearchState` (`schedule_search`). |
| `ScheduleRollingStage` | Folds the measured schedules onto fewer cores by reusing idle cores, and adds the folded schedules to the same Pareto front. |
| `ScheduleSelectionStage` | Ends a schedule search: re-measures the best schedule by `sort_key` and reports the whole front in `extra_info`. |
| `GraphEvolutionStage` | Two-step NSGA2 that evolves a free hardware graph (core designs and links) together with the mapping onto it. |

The schedule search stages (`ScheduleExplorationStage`, `ScheduleRollingStage`, `ScheduleSelectionStage`)
share one convention: **the stages after the last of them are the measurement pipeline**, run on every
candidate accelerator. A later stage can request extra work from an earlier one through class attributes:
`ScheduleRollingStage` sets `requires_full_cost_matrix` and `requires_core_occupancy`, so the core search
skips its cross-tile pass and the unrolled search warms up the full shape x design cost matrix and keeps each
schedule's per-core busy windows. Adding the stage to the chain is all it takes.

### Estimation -- `stream/stages/estimation/`

| Stage | Role |
|---|---|
| `ZigZagCoreMappingEstimationStage` | Runs ZigZag on every unique tile on every core it may run on, filling the cost LUT. |
| `StreamCostModelEvaluationStage` | Schedules a fixed allocation and returns its `StreamCostModelEvaluation`. |

### Allocation -- `stream/stages/allocation/` and `stream/stages/`

| Stage | Role |
|---|---|
| `SetFixedAllocationPerformanceStage` | Applies the fixed core allocations of the mapping and their costs. |
| `GeneticAlgorithmAllocationStage` | Genetic algorithm over the layer-core allocation; yields the best schedule. |
| `ConstraintOptimizationAllocationStage` | Allocation by constraint optimization (Gurobi). |

### Visualization -- `stream/stages/visualization/`

Pass-through stages that draw what the stages after them yield. They forward their kwargs untouched and yield
every answer unchanged, so they can be added to or removed from a chain freely. Place them *before* the
stage whose results they draw. A failure while drawing is logged and never fails the run.

| Stage | Writes |
|---|---|
| `ScheduleResultsWriterStage` | `schedules.csv`, `pareto_schedules.pickle` and, after rolling, `fold_log.csv`. |
| `ScheduleTradeoffPlotStage` | `schedule_tradeoffs.png`. |
| `TopologyReportStage` | `topologies/*.png`, `topologies.html`, `topologies.json`. |

They read `extra_info["schedule_report"]`, the plain-data context `ScheduleSelectionStage` attaches. New
visualization stages subclass `ExtraInfoReportStage` and implement `report(scme, extra_info)`.

---

You can also reuse [ZigZag's implemented stages](https://kuleuven-micas.github.io/zigzag/stages.html#implemented-stages) within Stream.

---

## Creating a custom stage

- Inherit from `stream.stages.stage.Stage`.
- Declare the keyword arguments you use as named `__init__` parameters, and call
  `super().__init__(list_of_callables, **kwargs)`.
- In `run()`, build the next stage with `self.list_of_callables[0](self.list_of_callables[1:], **kwargs)`,
  passing on `self.kwargs` plus whatever your stage produced (and any consumed argument a later stage needs).
- Yield the `(scme, extra_info)` pairs you get back, transformed or not. A reducing stage (one that picks the
  best of several answers) yields after its loop rather than inside it.
- A leaf stage (the last one of a chain) returns `True` from `is_leaf()` and yields its own answers.
