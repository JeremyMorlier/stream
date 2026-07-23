# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Stream is a HW architecture-mapping design space exploration (DSE) framework for multi-core deep
learning accelerators, built on top of the [ZigZag](https://zigzag-project.github.io/zigzag/) DSE
framework (single-core dataflow mapping). Stream adds multi-core allocation, fine-grained
layer-fused scheduling, and inter-core data transfer modeling on top of ZigZag's per-layer cost
model. This checkout (`stream_jeremy`, branch `NNToChips`) is a research fork actively extending
Stream with attention/transformer support (FuseMax-style hardware, Softmax decomposition, ONNX
attention parsing) — expect in-progress, sometimes rough, experimental code rather than a polished
library.

## Environment & commands

The project has both a `requirements.txt` (pip, used by CI and the devcontainer) and a `uv.lock`
(this working copy uses `uv`). The venv already exists at `.venv/` (Python 3.12).

```bash
# Run with the existing venv (preferred in this checkout)
uv run python main_stream_co.py       # constraint-optimization (Gurobi) allocation entry point
uv run python main_stream_ga.py       # genetic-algorithm allocation entry point
uv run python test_co.py              # CI's "test": runs a small end-to-end CO pipeline, no pytest

# Equivalent with pip (CI / devcontainer)
pip install -r requirements.txt
python main_stream_ga.py
```

There is **no pytest suite** — `test_co.py` and `main_stream_*.py` are runnable smoke-test scripts
that exercise the full pipeline end-to-end on small example workloads (see
`stream/inputs/testing/`). CI (`.github/workflows/python-can-run-*.yml`) just runs these scripts
and checks they don't raise. When validating a change, prefer running the smallest relevant
`main_stream_*.py`/`test_co.py`-style script over inventing new pytest tests, unless asked
otherwise.

Linting/formatting is `ruff` (config in `pyproject.toml`: line length 120, target py311, rules
`E,F,W,I,PL,N,UP,B`), wired through `.pre-commit-config.yaml`. Ruff is not installed in this venv by
default — install it (`uv pip install ruff` or via the `dev` extra) before relying on
`ruff check` / `ruff format`.

`optimize_allocation_co` (constraint optimization) requires a working **Gurobi license**; it hard-fails
via `_sanity_check_gurobi_license()` in `stream/api.py` if none is found. The GA path
(`optimize_allocation_ga`) has no such dependency.

## Architecture: the staged pipeline

Everything in Stream runs as a pipeline of composable **Stages** (`stream/stages/stage.py`).  Each
`Stage` is constructed with the *remaining* list of stage classes plus a shared `**kwargs` dict, and
recursively instantiates the next stage in `run()`, yielding `(StreamCostModelEvaluation, extra_info)`
tuples. `MainStage` is the non-generator entry point that drains this generator into a list of
results. This means:

- Adding a stage = inserting its class into the pipeline list passed to `MainStage(...)`, ordered
  so producer stages precede consumers.
- Every stage's `__init__` receives the *entire* shared kwargs dict — new inputs are threaded
  through by adding a new keyword, not by changing signatures.
- `stream/api.py` (`optimize_allocation_ga`, `optimize_allocation_co`) defines the two canonical
  pipelines end-to-end; read these first to see the concrete stage order and what kwargs each
  stage needs. Both pipelines share the same front half:
  `AcceleratorParserStage → ONNXModelParserStage → LayerStacksGenerationStage →
  TilingGenerationStage → TiledWorkloadGenerationStage → ZigZagCoreMappingEstimationStage`
  and diverge at the allocation step: `GeneticAlgorithmAllocationStage` (GA, stochastic
  core-allocation search) vs. `ConstraintOptimizationAllocationStage` (Gurobi MILP, exact/optimal
  under the modeled constraints).
- `stream/stages/*` are thin pipeline wrappers; the actual optimization algorithms live in
  `stream/opt/allocation/{genetic_algorithm,constraint_optimization}/` — read the `opt/` module for
  algorithmic changes, the `stages/` module for wiring/pipeline-order changes.

## Architecture: core data model

- **Hardware** (`stream/hardware/architecture/`): a Stream `Accelerator` is a graph
  (`CoreGraph`/`DiGraphWrapper`) of `Core` objects connected by `CommunicationLink`s.  Confusingly,
  a Stream `Core` *is* what ZigZag calls an "Accelerator" (one operational array + memory hierarchy
  + dataflow) — `Core` subclasses ZigZag's `Accelerator` (aliased `ZigZagCore`). So "multi-core" in
  Stream = a graph of ZigZag single-core accelerators. Hardware is defined in YAML
  (`stream/inputs/examples/hardware/*.yaml`, cores under `cores/`); topology (`2d-mesh` or `bus`)
  is declared in the same accelerator YAML (see `docs/source/hardware.md`).
- **Workload** (`stream/workload/`): ONNX models (or manual layer dicts) are parsed into a graph of
  `ComputationNode`s (extends ZigZag's `LayerNode`, the schedulable, tileable unit) plus
  non-compute `PropagationNode`s (reshape/transpose/concat/slice/etc. — pure tensor-shape bookkeeping,
  zero hardware cost) that wire the fine-grained dependency graph together.
  `stream/workload/steady_state/` is a separate, later representation (`SteadyStateWorkload` of
  `SteadyStateNode`s: computation/transfer/tensor/rolling-buffer) used specifically by the
  constraint-optimization allocator.
- **Mapping**: two independent specs are required — per-core *spatial* mapping/dataflow (declared
  on each core's YAML) and per-layer *core allocation* candidates (a separate mapping file/dict, see
  `docs/source/mapping.md`). Core allocation is resolved by layer name first, then layer type, then
  a `"default"` fallback.
- **Cost model** (`stream/cost_model/cost_model.py`): `StreamCostModelEvaluation` (SCME) is the
  central result object — runs a `CoalaScheduler` over the workload+accelerator+allocation and
  reports `latency` and itemized `energy` (compute, on/off-chip memory, eviction, sink-layer output,
  inter-core transfer). Almost every stage ultimately produces/consumes an SCME.
  `CostModelEvaluationLUT` (`stream/utils.py`) is a pickle-backed cache keyed by
  `(ComputationNode, Core)` with performance-equality-based lookup, used to avoid re-running ZigZag's
  per-layer estimation for equivalent node/core pairs.

## Adding a new ONNX operator

Follow the existing pattern in `stream/parser/onnx/`: subclass `OnnxOperatorParser`
(`OnnxComputeOperatorParser` for ops with a hardware cost), implement
`get_layer_node_user_format()` to emit the ZigZag-style layer dict (`equation`,
`dimension_relations`, `loop_dim_size`, `operand_precision`, `operand_source`,
`memory_operand_links`), and register the op in `OP_TYPE_TO_PARSER` in `stream/parser/onnx/model.py`.
For ops that don't map to a single hardware-costed layer (e.g. `softmax.py`), decompose into
multiple sub-parsers/`ComputationNode`s (e.g. Max → Exp → Sum → Div) and fix up
`input_operand_source`/`constant_operands` afterward so the fine-grained dependency graph stays
correct. Unsupported ops fall back to a zero-cost `DummyNode`.

## Outputs & visualization

Runs write to `outputs/<experiment_id>/` (tiled workload, cost LUT, and SCME pickles, keyed by an
`experiment_id` string the caller constructs — see the `main_stream_*.py` scripts for the naming
convention). `stream/visualization/` turns an SCME into: a Perfetto trace JSON
(`convert_scme_to_perfetto_json`, viewable at ui.perfetto.dev), a memory-usage plot
(`plot_memory_usage`), and Gantt-style schedule timelines (`schedule.py`).
