"""Compare design-space optimizers on a common wall-clock budget instead of iteration counts.

Methods:
- `rolled`: the rolled methodology (`optimize_rolled_tiling`): a GA over intra-/inter-core tilings; each tiling
  candidate gets its hardware graph from a per-tile core NSGA2 (stopping on front hypervolume convergence), an
  unrolled schedule branch-and-bound over those cores, then rolling onto fewer cores.
- `graph_ga`: the two-step evolving-graph GA (`GraphEvolutionStage`): step 1 evolves an independent hardware graph
  (core designs, explicit links) with the allocation onto it; step 2 freezes the best graphs and evolves the
  mapping (tiling and allocation) on them.
- `grid_2x2` / `grid_4x4`: the intra-/inter-core tiling GA (`TilingExplorationStage`) on a fixed homogeneous
  grid of tpu_like cores, ranked by EDP.

`rolled` and `graph_ga` start from tpu_like_quad_core and its mapping only as a seed: the mapping's tiling is
where their tiling search starts, and the hardware is replaced by the generated graphs.

Workloads: `resnet` (ResNet-50 first bottleneck) and `attention` (QKV, multi-head attention, output projection
and residual of an LLM layer).

Every run gets the same `SearchBudget`: it stops at `--max-time` or once the best EDP improved by less than
`--rel-tol` over the last `--window` seconds. The budget is only checked before a new generation starts -- the
generation under way, and any search without generations, always finishes -- so runs can end past `--max-time`
(`total_wall_time` in `summary.json` is the real duration). Each (workload, method) pair runs in its own
subprocess (the CACTI monkeypatch and DEAP's `creator` are process-global) and writes `trace.csv` (every
measured design) and `summary.json` to `<output-dir>/<workload>/<method>/`. `analyze_optimizer_comparison.py`
turns those into convergence plots, Pareto fronts and a comparison table.

Run: python benchmark_optimizer_comparison.py --max-time 10800 --window 1800 --rel-tol 0.005 --workers 8
"""

# The stream imports are deferred to the per-run subprocess, so the orchestrator stays light.
# ruff: noqa: PLC0415

import argparse
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

WORKLOADS = ("resnet", "attention")
METHODS = ("rolled", "graph_ga", "grid_2x2", "grid_4x4")

QUAD_CORE = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
QUAD_CORE_MAPPINGS = {
    "resnet": "stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml",
    "attention": "stream/inputs/examples/mapping/tpu_like_quad_core_attention.yaml",
}
# Iteration caps handed to the searches alongside the budget: large enough that only time stops them.
NO_CAP = 10**9

logger = logging.getLogger(__name__)


def _export_workload(workload: str, path: str) -> str:
    from stream.inputs.examples.workload.comparison_workloads import export_attention_block, export_resnet_bottleneck

    return export_resnet_bottleneck(path) if workload == "resnet" else export_attention_block(path)


def _run_rolled(workload_path, mapping, layer_stacks, out_root, experiment_id, budget, args):
    from stream.api import optimize_rolled_tiling

    return optimize_rolled_tiling(
        hardware=QUAD_CORE,
        workload=workload_path,
        mapping=mapping,
        mode="fused",
        layer_stacks=layer_stacks,
        experiment_id=experiment_id,
        output_path=out_root,
        search_budget=budget,
        nb_tiling_ga_individuals=args.rolled_population,
        nb_core_ga_generations=NO_CAP,
        nb_core_ga_individuals=16,
        core_convergence_patience=args.core_patience,
        core_rel_tol=args.rel_tol,
        core_max_workers=args.workers,
        core_pareto_points=10,
        cacti_precompute_workers=args.workers,
    )


def _run_graph_ga(workload_path, mapping, layer_stacks, out_root, experiment_id, budget, args):
    from stream.api import optimize_graph_evolution

    return optimize_graph_evolution(
        hardware=QUAD_CORE,
        workload=workload_path,
        mapping=mapping,
        mode="fused",
        layer_stacks=layer_stacks,
        experiment_id=experiment_id,
        output_path=out_root,
        population_size=args.population,
        max_workers=args.workers,
        cacti_precompute_workers=args.workers,
        search_budget=budget,
        graph_step_fraction=args.graph_step_fraction,
        step1_patience=args.step1_patience,
        nb_frozen_graphs=args.frozen_graphs,
    )


def _run_grid(size, workload_path, layer_stacks, out_root, experiment_id, budget, args):
    from stream.api import optimize_tiling
    from stream.hardware.architecture.grid_generator import write_tpu_grid

    hardware, mapping = write_tpu_grid(size, size)
    return optimize_tiling(
        hardware=hardware,
        workload=workload_path,
        mapping=mapping,
        mode="fused",
        layer_stacks=layer_stacks,
        experiment_id=experiment_id,
        output_path=out_root,
        nb_ga_generations=40,
        nb_ga_individuals=10,
        nb_tiling_ga_generations=NO_CAP,
        nb_tiling_ga_individuals=args.population,
        sort_key="edp",
        max_workers=args.workers,
        explore_cores=False,
        search_budget=budget,
    )


def run_single(workload: str, method: str, args: argparse.Namespace) -> None:
    """One (workload, method) run, in this process."""
    from stream.inputs.examples.workload.comparison_workloads import single_stack
    from stream.opt.search_budget import SearchBudget

    out_root = os.path.join(args.output_dir, workload)
    run_dir = os.path.join(out_root, method)
    # Fresh directory: every search caches results on disk (per-tile fronts, tiling candidates), and a reused
    # cache would hand the method free evaluations.
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(run_dir)
    random.seed(args.seed)

    workload_path = _export_workload(workload, os.path.join(run_dir, "workload.onnx"))
    layer_stacks = single_stack(workload_path)
    # Started after the export (identical for every method) and before any method-specific set-up.
    budget = SearchBudget(args.max_time, args.window, args.rel_tol, os.path.join(run_dir, "trace.csv"))
    error = None
    try:
        if method == "rolled":
            _run_rolled(workload_path, QUAD_CORE_MAPPINGS[workload], layer_stacks, out_root, method, budget, args)
        elif method == "graph_ga":
            _run_graph_ga(workload_path, QUAD_CORE_MAPPINGS[workload], layer_stacks, out_root, method, budget, args)
        elif method.startswith("grid_"):
            size = int(method.split("_")[1].split("x", maxsplit=1)[0])
            _run_grid(size, workload_path, layer_stacks, out_root, method, budget, args)
        else:
            raise ValueError(f"Unknown method {method}")
    except Exception as exc:  # keep the trace and summary of a run that crashed late
        logger.exception(f"{workload}/{method} failed")
        error = repr(exc)
    budget.stop("exhausted")
    budget.save_summary(
        os.path.join(run_dir, "summary.json"),
        workload=workload,
        method=method,
        workers=args.workers,
        seed=args.seed,
        total_wall_time=budget.elapsed(),
        error=error,
    )
    summary = budget.summary()
    print(
        f"{workload}/{method}: stop={summary['stop_reason']} after {summary['time_to_stop']:.0f}s, "
        f"{summary['n_eval']} evaluation(s), best EDP={summary['best_edp']:.4g}"
    )


def _launch(workload: str, method: str, args: argparse.Namespace) -> int:
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{workload}_{method}.log")
    command = [
        sys.executable,
        __file__,
        "--single",
        workload,
        method,
        "--max-time",
        str(args.max_time),
        "--window",
        str(args.window),
        "--rel-tol",
        str(args.rel_tol),
        "--workers",
        str(args.workers),
        "--population",
        str(args.population),
        "--seed",
        str(args.seed),
        "--rolled-population",
        str(args.rolled_population),
        "--core-patience",
        str(args.core_patience),
        "--graph-step-fraction",
        str(args.graph_step_fraction),
        "--step1-patience",
        str(args.step1_patience),
        "--frozen-graphs",
        str(args.frozen_graphs),
        "--output-dir",
        args.output_dir,
    ]
    print(f"[{time.strftime('%H:%M:%S')}] start {workload}/{method} -> {log_path}", flush=True)
    with open(log_path, "w") as log:
        returncode = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}] done  {workload}/{method} (exit {returncode})", flush=True)
    return returncode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workloads", nargs="+", default=list(WORKLOADS), choices=WORKLOADS)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--max-time", type=float, default=3 * 3600, help="Wall-clock limit per run, in seconds.")
    parser.add_argument("--window", type=float, default=1800, help="Convergence window, in seconds.")
    parser.add_argument("--rel-tol", type=float, default=0.005, help="Converged below this EDP gain per window.")
    parser.add_argument("--workers", type=int, default=8, help="Worker processes, the same for every method.")
    parser.add_argument("--population", type=int, default=16, help="GA population (graph_ga and grid tiling).")
    parser.add_argument("--rolled-population", type=int, default=4, help="rolled: tiling GA population.")
    parser.add_argument("--core-patience", type=int, default=5, help="rolled: core GA hypervolume patience.")
    parser.add_argument("--graph-step-fraction", type=float, default=0.5, help="graph_ga: max budget for step 1.")
    parser.add_argument("--step1-patience", type=int, default=5, help="graph_ga: step 1 convergence patience.")
    parser.add_argument("--frozen-graphs", type=int, default=4, help="graph_ga: graphs frozen for step 2.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parallel-runs", type=int, default=1, help="Runs executed side by side.")
    parser.add_argument("--output-dir", default="outputs/optimizer_comparison")
    parser.add_argument("--log-dir", default="logs/optimizer_comparison")
    parser.add_argument("--single", nargs=2, metavar=("WORKLOAD", "METHOD"), help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.single:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
        )
        run_single(*args.single, args)
        return
    runs = [(workload, method) for workload in args.workloads for method in args.methods]
    with ThreadPoolExecutor(max_workers=args.parallel_runs) as executor:
        codes = list(executor.map(lambda run: _launch(*run, args), runs))
    failed = [run for run, code in zip(runs, codes, strict=True) if code]
    if failed:
        print(f"Failed runs: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
