"""Compare three accelerator design flows on a common wall-clock budget.

Methods:
- `tiling_placement_cpsat` / `tiling_placement_gurobi` (`optimize_placement_tiling`): a GA over intra-/inter-core
  tilings; each tiling candidate gets per-tile core fronts from a core NSGA2 (stopping on front hypervolume
  convergence), then the placement optimization at level 2 (`--chunks` tasks per layer group) picks the
  accelerator and schedule, solved with CP-SAT or as a Gurobi MILP -- the same model, epsilon grid and solve time,
  so the two differ by solver only. The tiling GA selects candidates by hypervolume contribution.
- `tiling_coreopt_cpsat` / `tiling_coreopt_gurobi`: the same, with the core pool of `CoreOptimizationStage`
  instead of the per-tile fronts alone: per-tile fronts plus designs specialized for operators with several tile
  shapes and for clusters of similar shapes (`--shared-clusters`).
- `graph_ga` (`optimize_graph_evolution`): the two-step evolving-graph GA of `benchmark_optimizer_comparison.py`
  (core designs and links with the allocation, then the mapping on the frozen best graphs).
- `edge_tpu_template` (`optimize_template_accelerator`): NSGA2 over a homogeneous grid of edge_tpu-like cores --
  grid rows and columns (up to `--max-grid`), the four array dimensions, and the four memory sizes (costed by
  CACTI) -- with every layer's inter-core tiling; the intra-core tiling is the grid mapping's.

`tiling_placement_*` and `graph_ga` start from tpu_like_quad_core and its mapping as a seed. Workloads: `resnet`
(ResNet-50 first bottleneck) and `attention` (an LLM layer's attention half).

The runs follow `benchmark_optimizer_comparison.py`: each (workload, method) pair runs in its own subprocess under
the same `SearchBudget` (`--max-time`, or the best EDP improving by less than `--rel-tol` over `--window` seconds;
the generation under way always finishes) and writes `trace.csv`, `summary.json` and `designs/` to
`<output-dir>/<workload>/<method>/`. Analyze them with
`python benchmarks/analyze_optimizer_comparison.py --results-dir <output-dir>`.

Run: python benchmarks/benchmark_accelerator_comparison.py --max-time 10800 --window 1800 --workers 8
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

from benchmark_optimizer_comparison import NO_CAP, QUAD_CORE, QUAD_CORE_MAPPINGS, _export_workload, _run_graph_ga

WORKLOADS = ("resnet", "attention")
METHODS = (
    "tiling_placement_cpsat",
    "tiling_placement_gurobi",
    "tiling_coreopt_cpsat",
    "tiling_coreopt_gurobi",
    "graph_ga",
    "edge_tpu_template",
)
# The options every run subprocess receives (the orchestrator-only ones are left out).
FORWARDED_OPTIONS = (
    "max_time",
    "window",
    "rel_tol",
    "workers",
    "seed",
    "output_dir",
    "population",
    "tiling_population",
    "core_patience",
    "hv_patience",
    "chunks",
    "solve_time",
    "max_cores",
    "mip_gap",
    "graph_step_fraction",
    "step1_patience",
    "frozen_graphs",
    "max_grid",
    "shared_clusters",
)

logger = logging.getLogger(__name__)


def _run_tiling_placement(  # noqa: PLR0913
    solver, workload_path, mapping, layer_stacks, out_root, experiment_id, budget, args, *, core_optimization=None
):
    from stream.api import optimize_placement_tiling

    return optimize_placement_tiling(
        hardware=QUAD_CORE,
        workload=workload_path,
        mapping=mapping,
        mode="fused",
        layer_stacks=layer_stacks,
        experiment_id=experiment_id,
        output_path=out_root,
        search_budget=budget,
        placement_solver=solver,
        level=2,
        chunks_per_group=args.chunks,
        max_cores=args.max_cores,
        mip_gap=args.mip_gap,
        solve_time_limit_s=args.solve_time,
        solver_workers=args.workers,
        hypervolume_patience=args.hv_patience,
        hypervolume_rel_tol=args.rel_tol,
        nb_tiling_ga_individuals=args.tiling_population,
        nb_core_ga_generations=NO_CAP,
        nb_core_ga_individuals=16,
        core_convergence_patience=args.core_patience,
        core_rel_tol=args.rel_tol,
        core_max_workers=args.workers,
        core_pareto_points=10,
        core_optimization=core_optimization,
        cacti_precompute_workers=args.workers,
    )


def _run_edge_tpu_template(workload_path, layer_stacks, out_root, experiment_id, budget, args):
    from stream.api import optimize_template_accelerator

    return optimize_template_accelerator(
        template_core="edge_tpu_like.yaml",
        workload=workload_path,
        mode="fused",
        layer_stacks=layer_stacks,
        experiment_id=experiment_id,
        output_path=out_root,
        search_budget=budget,
        template_param_ranges={"max_rows": args.max_grid, "max_cols": args.max_grid},
        population_size=args.population,
        max_workers=args.workers,
        cacti_precompute_workers=args.workers,
    )


def run_single(workload: str, method: str, args: argparse.Namespace) -> None:
    """One (workload, method) run, in this process."""
    from stream.inputs.examples.workload.comparison_workloads import single_stack
    from stream.opt.search_budget import SearchBudget

    out_root = os.path.join(args.output_dir, workload)
    run_dir = os.path.join(out_root, method)
    # Fresh directory: every search caches results on disk, and a reused cache would hand the method free
    # evaluations.
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(run_dir)
    random.seed(args.seed)

    workload_path = _export_workload(workload, os.path.join(run_dir, "workload.onnx"))
    layer_stacks = single_stack(workload_path)
    budget = SearchBudget(
        args.max_time,
        args.window,
        args.rel_tol,
        os.path.join(run_dir, "trace.csv"),
        artifact_dir=os.path.join(run_dir, "designs"),
    )
    error = None
    try:
        if method.startswith(("tiling_placement_", "tiling_coreopt_")):
            solver = method.rsplit("_", 1)[1]
            mapping = QUAD_CORE_MAPPINGS[workload]
            core_optimization = (
                {"nb_shared_clusters": args.shared_clusters} if method.startswith("tiling_coreopt_") else None
            )
            _run_tiling_placement(
                solver,
                workload_path,
                mapping,
                layer_stacks,
                out_root,
                method,
                budget,
                args,
                core_optimization=core_optimization,
            )
        elif method == "graph_ga":
            _run_graph_ga(workload_path, QUAD_CORE_MAPPINGS[workload], layer_stacks, out_root, method, budget, args)
        elif method == "edge_tpu_template":
            _run_edge_tpu_template(workload_path, layer_stacks, out_root, method, budget, args)
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
    command = [sys.executable, __file__, "--single", workload, method]
    for option in FORWARDED_OPTIONS:
        value = getattr(args, option)
        if value is not None:
            command += [f"--{option.replace('_', '-')}", str(value)]
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
    parser.add_argument("--population", type=int, default=16, help="GA population (graph_ga, edge_tpu_template).")
    parser.add_argument("--tiling-population", type=int, default=4, help="tiling_placement: tiling GA population.")
    parser.add_argument("--core-patience", type=int, default=5, help="tiling_placement: core GA HV patience.")
    parser.add_argument("--hv-patience", type=int, default=2, help="tiling_placement: tiling GA HV patience.")
    parser.add_argument("--chunks", type=int, default=8, help="tiling_placement: level-2 tasks per layer group.")
    parser.add_argument("--solve-time", type=float, default=20, help="tiling_placement: time limit per solve.")
    parser.add_argument("--max-cores", type=int, default=None, help="tiling_placement: core limit (default none).")
    parser.add_argument("--mip-gap", type=float, default=0.0, help="tiling_placement: Gurobi relative MIP gap.")
    parser.add_argument(
        "--shared-clusters", type=int, default=None, help="tiling_coreopt: clusters of shape classes (default sqrt)."
    )
    parser.add_argument("--graph-step-fraction", type=float, default=0.5, help="graph_ga: max budget for step 1.")
    parser.add_argument("--step1-patience", type=int, default=5, help="graph_ga: step 1 convergence patience.")
    parser.add_argument("--frozen-graphs", type=int, default=4, help="graph_ga: graphs frozen for step 2.")
    parser.add_argument("--max-grid", type=int, default=4, help="edge_tpu_template: max grid rows and columns.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parallel-runs", type=int, default=1, help="Runs executed side by side.")
    parser.add_argument("--output-dir", default="outputs/accelerator_comparison")
    parser.add_argument("--log-dir", default="logs/accelerator_comparison")
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
