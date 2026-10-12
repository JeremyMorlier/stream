"""Compare the rolled search with the constraint-model schedule optimization (levels 1 and 2), from identical inputs.

For each workload, all methods run on one experiment directory, in this order:

1. `rolled`: `optimize_rolled_schedules` with the settings of `example_rolled_schedule_exploration.py` (per-tile
   core GA, unrolled branch and bound over 60 schedules, folding). It searches and saves the per-tile Pareto fronts
   (`core_search/pareto_fronts.pickle`) and the full shape x design cost matrix (`rolled_search/cost_lut_*`).
2. `level1`: `optimize_schedule_model(level=1)` -- one task per layer group.
3. `level2`: `optimize_schedule_model(level=2, chunks_per_group=K)` -- K tasks per group.
4. `level3`: `optimize_schedule_model(level=3)` -- one task per tile (solved with `--level3-solve-time`).

Levels 1 and 2 reuse the fronts and the cost matrix, so all three choose from exactly the same designs with
exactly the same per-tile costs; only the search over accelerators and schedules differs. Each method runs in its
own subprocess (the CACTI monkeypatch and DEAP's `creator` are process-global) and writes `result_<method>.json`.

The comparison (`comparison.csv`, `comparison.json`, `fronts_<workload>.png`) reports per method:
- its own search time (rolled: unrolled search + folding; levels: model build + solves + measurements), next to
  the shared preparation time;
- how many accelerators it measured with the real scheduler and how many of them are on its front;
- best EDP, best latency, energy and area on its measured front;
- the normalized (latency, energy, area) hypervolume of its measured front, against a reference point shared by
  the three methods of a workload;
- for the levels, the model's prediction error (median and max relative error of latency and energy over every
  measured solution).

Run: python benchmark_schedule_optimization.py --workloads resnet attention --chunks 8 --solve-time 20 --workers 8
"""

# The stream imports are deferred to the per-run subprocess, so the orchestrator stays light.
# ruff: noqa: PLC0415

import argparse
import csv
import json
import logging
import math
import os
import statistics
import subprocess
import sys
import time

WORKLOADS = ("resnet", "attention")
METHODS = ("rolled", "level1", "level2", "level3")
QUAD_CORE = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
MAPPINGS = {
    "resnet": "stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml",
    "attention": "stream/inputs/examples/mapping/tpu_like_quad_core_attention.yaml",
}
# Validated categorical slots 1-3 of the dataviz reference palette (all-pairs safe).
# Level 3 reuses level 2's hue (aqua) with its own marker: no fourth hue clears the all-pairs color-vision floors.
COLORS = {"rolled": "#2a78d6", "level1": "#eb6834", "level2": "#1baf7a", "level3": "#1baf7a"}
MARKERS = {"rolled": "o", "level1": "^", "level2": "s", "level3": "D"}
EXPERIMENT_ID = "schedule_search"

logger = logging.getLogger(__name__)


def _common_kwargs(workload_path: str, workload: str, out_root: str, args: argparse.Namespace) -> dict:
    from stream.inputs.examples.workload.comparison_workloads import single_stack

    return {
        "hardware": QUAD_CORE,
        "workload": workload_path,
        "mapping": MAPPINGS[workload],
        "mode": "fused",
        "layer_stacks": single_stack(workload_path),
        "experiment_id": EXPERIMENT_ID,
        "output_path": out_root,
        "sort_key": "latency",
        "nb_core_ga_generations": 16,
        "nb_core_ga_individuals": 16,
        "core_pareto_points": 10,
        "core_max_workers": args.workers,
        "cacti_precompute_workers": args.workers,
        "reuse_pareto_fronts": True,
    }


def _read_schedules_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def run_single(workload: str, method: str, args: argparse.Namespace) -> None:
    """One method on one workload, in this process; writes `result_<method>.json`."""
    from stream.inputs.examples.workload.comparison_workloads import export_attention_block, export_resnet_bottleneck

    out_root = os.path.join(args.output_dir, workload)
    os.makedirs(out_root, exist_ok=True)
    workload_path = os.path.join(out_root, "workload.onnx")
    if not os.path.exists(workload_path):
        (export_resnet_bottleneck if workload == "resnet" else export_attention_block)(workload_path)
    kwargs = _common_kwargs(workload_path, workload, out_root, args)
    experiment_dir = os.path.join(out_root, EXPERIMENT_ID)

    t0 = time.perf_counter()
    if method == "rolled":
        from stream.api import optimize_rolled_schedules

        stats: dict = {}
        _best, pareto = optimize_rolled_schedules(
            **kwargs,
            max_schedule_evaluations=60,
            fold_tolerances=(0.0, 0.25, 1.0),
            fold_overlap="exact",
            stats_out=stats,
        )
        search_time = stats.get("search_time_s", 0.0) + stats.get("rolled", {}).get("time_s", 0.0)
        measured = _read_schedules_csv(os.path.join(experiment_dir, "rolled_search", "schedules.csv"))
    else:
        from stream.api import optimize_schedule_model

        level = int(method[-1])
        _best, pareto, stats = optimize_schedule_model(
            **kwargs,
            level=level,
            chunks_per_group=args.chunks,
            solve_time_limit_s=args.level3_solve_time if method == "level3" else args.solve_time,
            solver_workers=args.workers,
            area_levels=args.area_levels,
            energy_levels=args.energy_levels,
        )
        search_time = stats[f"optimization_l{level}"]["total_time_s"]
        measured = [
            row
            for row in _read_schedules_csv(
                os.path.join(experiment_dir, f"schedule_optimization_l{level}", "schedules.csv")
            )
            # The levels' own records (the space builder's single unrolled schedule is not theirs).
            if row["id"].startswith("opt")
        ]
    result = {
        "workload": workload,
        "method": method,
        "wall_time_s": time.perf_counter() - t0,
        "prepare_time_s": stats.get("prepare_time_s"),
        "search_time_s": search_time,
        "stats": stats,
        "pareto": [
            {
                key: record.get(key)
                for key in (
                    "id",
                    "kind",
                    "nb_cores",
                    "latency",
                    "energy",
                    "area",
                    "predicted_latency",
                    "predicted_energy",
                )
            }  # fmt: skip
            for record in pareto
        ],
        "measured": measured,
    }
    with open(os.path.join(out_root, f"result_{method}.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(f"{workload}/{method}: {len(pareto)} pareto point(s), search {search_time:.1f}s")


# --------------------------------------------------------------------------------------------- comparison -------


def _front(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    return [
        p
        for p in points
        if not any(all(o <= q for o, q in zip(other, p, strict=True)) and other != p for other in points)
    ]


def _method_points(result: dict) -> list[tuple[float, float, float]]:
    """The method's own measured points: the levels' front also holds the single unrolled schedule the space
    builder measured, which is not theirs."""
    keep = (lambda r: True) if result["method"] == "rolled" else (lambda r: str(r["id"]).startswith("opt"))
    return [(r["latency"], r["energy"], r["area"]) for r in result["pareto"] if keep(r)]


def _prediction_errors(result: dict, rows: list[dict]) -> dict:
    errors = {"latency": [], "energy": []}
    for row in rows:
        for name, bound in (("latency", "latency_bound"), ("energy", "energy_bound")):
            try:
                measured, predicted = float(row[name]), float(row[bound])
            except (KeyError, ValueError):
                continue
            if measured > 0 and math.isfinite(measured):
                errors[name].append((predicted - measured) / measured)
    out = {}
    for name, values in errors.items():
        if values:
            out[f"{name}_error_median_pct"] = round(100 * statistics.median(values), 1)
            out[f"{name}_error_max_abs_pct"] = round(100 * max(abs(v) for v in values), 1)
    return out


def _level_rows(output_dir: str, workload: str, method: str) -> list[dict]:
    """Every solution a level measured, with the model's prediction in `latency_bound` / `energy_bound`."""
    level = int(method[-1])
    path = os.path.join(output_dir, workload, EXPERIMENT_ID, f"schedule_optimization_l{level}", "schedules.csv")
    return [row for row in _read_schedules_csv(path) if row["id"].startswith("opt")]


def compare(output_dir: str, workloads: list[str]) -> None:
    from stream.opt.search_budget import hypervolume_3d

    rows = []
    for workload in workloads:
        results = {}
        for method in METHODS:
            path = os.path.join(output_dir, workload, f"result_{method}.json")
            if os.path.exists(path):
                with open(path) as f:
                    results[method] = json.load(f)
        if not results:
            continue
        fronts = {method: _front(_method_points(result)) for method, result in results.items()}
        union = [p for front in fronts.values() for p in front]
        ideal = [min(p[i] for p in union) for i in range(3)]
        nadir = [max(p[i] for p in union) * 1.1 for i in range(3)]
        span = [(n - i) or 1.0 for i, n in zip(ideal, nadir, strict=True)]
        best_edp_overall = min(p[0] * p[1] for p in union)
        for method, result in results.items():
            front = fronts[method]
            normalized = [tuple((p[i] - ideal[i]) / span[i] for i in range(3)) for p in front]
            best = min(front, key=lambda p: p[0] * p[1])
            stats = result["stats"]
            measured_count = (
                stats.get("measured", 0) + stats.get("rolled", {}).get("measured", 0)
                if method == "rolled"
                else stats[f"optimization_l{method[-1]}"].get("measured", 0)
            )
            rows.append(
                {
                    "workload": workload,
                    "method": method,
                    "prepare_time_s": result["prepare_time_s"],
                    "search_time_s": round(result["search_time_s"], 1),
                    "measured": measured_count,
                    "front_points": len(front),
                    "best_edp": best[0] * best[1],
                    "edp_gap_pct": round(100 * (best[0] * best[1] / best_edp_overall - 1), 1),
                    "min_latency": min(p[0] for p in front),
                    "min_energy": min(p[1] for p in front),
                    "min_area": min(p[2] for p in front),
                    "hypervolume": round(hypervolume_3d(normalized, (1.0, 1.0, 1.0)), 4),
                    **(
                        _prediction_errors(result, _level_rows(output_dir, workload, method))
                        if method != "rolled"
                        else {}
                    ),
                }
            )
        _plot(workload, fronts, os.path.join(output_dir, f"fronts_{workload}.png"))

    columns = sorted({key for row in rows for key in row}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(os.path.join(output_dir, "comparison.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(rows, f, indent=1)
    for row in rows:
        print(
            f"{row['workload']:>9} {row['method']:>7}: search {row['search_time_s']:>8}s, "
            f"{row['measured']:>4} measured, {row['front_points']:>3} on front, "
            f"best EDP {row['best_edp']:.4g} (+{row['edp_gap_pct']}%), "
            f"HV {row['hypervolume']}"
            + (
                f", latency error median {row.get('latency_error_median_pct')}% "
                f"(max {row.get('latency_error_max_abs_pct')}%)"
                if "latency_error_median_pct" in row
                else ""
            )
        )


def _plot(workload: str, fronts: dict[str, list], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (x, y, xl, yl) in zip(
        axes, ((0, 1, "latency [cycles]", "energy [pJ]"), (0, 2, "latency [cycles]", "area [a.u.]")), strict=True
    ):
        for method, front in fronts.items():
            if not front:
                continue
            ax.scatter([p[x] for p in front], [p[y] for p in front], label=f"{method} ({len(front)})",
                       color=COLORS[method], marker=MARKERS[method], s=36, alpha=0.85, edgecolors="none")  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(f"{workload}: measured Pareto fronts")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------- driver -------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workloads", nargs="+", default=["resnet"], choices=WORKLOADS)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--chunks", type=int, default=8, help="level 2: tasks per layer group.")
    parser.add_argument("--solve-time", type=float, default=20.0, help="CP-SAT time limit per solve, seconds.")
    parser.add_argument(
        "--level3-solve-time", type=float, default=60.0, help="CP-SAT time limit per level-3 solve, seconds."
    )
    parser.add_argument("--area-levels", type=int, default=5)
    parser.add_argument("--energy-levels", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/schedule_optimization_comparison")
    parser.add_argument("--log-dir", default="logs/schedule_optimization_comparison")
    parser.add_argument("--compare-only", action="store_true", help="Only rebuild the comparison from results.")
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
    if not args.compare_only:
        os.makedirs(args.log_dir, exist_ok=True)
        for workload in args.workloads:
            # rolled first: it builds the fronts and the cost matrix the levels reuse.
            for method in [m for m in METHODS if m in args.methods]:
                log_path = os.path.join(args.log_dir, f"{workload}_{method}.log")
                command = [sys.executable, __file__, "--single", workload, method, "--chunks", str(args.chunks),
                           "--solve-time", str(args.solve_time), "--level3-solve-time", str(args.level3_solve_time),
                           "--area-levels", str(args.area_levels),
                           "--energy-levels", str(args.energy_levels), "--workers", str(args.workers),
                           "--output-dir", args.output_dir]  # fmt: skip
                print(f"[{time.strftime('%H:%M:%S')}] {workload}/{method} -> {log_path}", flush=True)
                with open(log_path, "w") as log:
                    code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
                if code:
                    print(f"{workload}/{method} failed (exit {code}); see {log_path}")
    compare(args.output_dir, args.workloads)


if __name__ == "__main__":
    main()
