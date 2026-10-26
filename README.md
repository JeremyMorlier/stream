# Stream -- multi-core accelerator design-space exploration

Stream is a HW architecture-mapping design space exploration (DSE) framework for multi-core deep learning
accelerators. The mapping can be explored at different granularities, from classical layer-by-layer processing
to fine-grained layer-fused processing. Stream builds on top of the [ZigZag](https://zigzag-project.github.io/zigzag/)
DSE framework. See [the upstream documentation](https://kuleuven-micas.github.io/stream/) and the paper:

[A. Symons, L. Mei, S. Colleman, P. Houshmand, S. Karl and M. Verhelst, "Stream: Design Space Exploration of
Layer-Fused DNNs on Heterogeneous Dataflow Accelerators".](https://ieeexplore.ieee.org/abstract/document/10713407)

This fork adds a joint search of the tiling and of the hardware itself:

- **per-tile core search** -- an NSGA2 search of core designs (memory sizes and bandwidths, array size) for every
  unique tile shape, on `(latency, energy, area)`;
- **schedule search** -- a branch and bound over combinations of those designs, measured with Stream's
  scheduler, returning the Pareto front of whole schedules;
- **rolling** -- folding the resulting accelerators onto fewer cores by reusing cores that are idle later in the
  schedule;
- **tiling search** -- a genetic algorithm over every layer's intra-/inter-core tiling wrapped around all of the
  above (the *rolled methodology*), and an evolving-graph GA to compare it against;
- interactive HTML reports and TikZ figures of the fronts and of the accelerators behind them.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/); it pulls a ZigZag fork with batched CACTI calls.

```bash
uv sync
```

## Running

| Script | What it runs |
|---|---|
| `main_stream_ga.py`, `main_stream_co.py`, `main_stream_co_inter.py` | Layer-core allocation on a fixed accelerator, by genetic algorithm or constraint optimization. |
| `main_stream_tiling.py`, `example_tiling_exploration.py` | Tiling GA, optionally with a per-tile core search (`optimize_tiling`). |
| `example_schedule_exploration.py` | Schedule search over the per-tile core fronts (`optimize_schedules`). |
| `example_rolled_schedule_exploration.py` | Schedule search plus rolling, and TikZ export of the front (`optimize_rolled_schedules`). |
| `benchmarks/benchmark_optimizer_comparison.py`, `benchmarks/analyze_optimizer_comparison.py` | Rolled methodology vs. graph GA vs. grid baselines on one wall-clock budget, then the comparison report. |
| `benchmarks/benchmark_accelerator_comparison.py` | Tiling GA + per-tile cores + placement optimization (CP-SAT, Gurobi) vs. graph GA vs. edge_tpu template NSGA2, on the same budget and report. |
| `benchmarks/benchmark_schedule_optimization.py` | Rolled search vs. the schedule-optimization model levels and the placement optimization, from identical inputs. |
| `benchmarks/benchmark_core_architecture_exploration.py` | Profiles one core-design evaluation of the per-tile search. |

Outputs go to `outputs/<experiment id>/`. The stage chains behind each entry point are described in
[docs/source/optimization.md](docs/source/optimization.md), the stages themselves in
[docs/source/stages.md](docs/source/stages.md) and the files they write in
[docs/source/outputs.md](docs/source/outputs.md).

## Tests

```bash
python tests/test_schedule_exploration.py
python tests/test_schedule_rolling.py
python tests/test_optimizer_comparison.py
python tests/test_hardware_graph.py
python tests/test_generate_core.py
python tests/test_co.py   # needs a Gurobi license
```

Each test file also runs under `pytest`.
