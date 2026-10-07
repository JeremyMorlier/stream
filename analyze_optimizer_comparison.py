"""Summarize `benchmark_optimizer_comparison.py` runs: convergence curves, Pareto fronts and a comparison table.

Reads `<results-dir>/<workload>/<method>/{trace.csv, summary.json}` and writes into `<results-dir>`:
- `convergence_<workload>.png`: best-so-far EDP against wall time (log-log), one line per method, with a marker
  where each run stopped;
- `pareto_<workload>.png`: every method's non-dominated (latency, energy) points, area as marker size;
- `comparison.csv`: per run -- stop reason, time to stop, time to reach within 1% of its own final best, best
  EDP (and its gap to the best method on that workload), the best point, evaluation count and the normalized
  (latency, energy, area) hypervolume.

The hypervolume of each workload is taken on objectives normalized to [ideal, nadir * 1.1] over the union of
all methods' fronts, so methods on one workload share a reference point and the value lies in [0, 1].

Run: python analyze_optimizer_comparison.py [--results-dir outputs/optimizer_comparison]
"""

import argparse
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stream.opt.search_budget import hypervolume_3d  # noqa: E402

COLORS = {"rolled": "#2a6fdb", "graph_ga": "#d1495b", "grid_2x2": "#2e933c", "grid_4x4": "#e09f3e"}


def read_trace(path: str) -> list[dict[str, float]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [
        {key: float(row[key]) for key in ("t", "n_eval", "latency", "energy", "area", "edp", "best_edp", "feasible")}
        for row in rows
    ]


def pareto_points(trace: list[dict[str, float]]) -> list[tuple[float, float, float]]:
    points = sorted({(r["latency"], r["energy"], r["area"]) for r in trace if r["feasible"]})
    front: list[tuple[float, float, float]] = []
    for point in points:
        dominated = any(all(o <= p for o, p in zip(other, point, strict=True)) and other != point for other in points)
        if not dominated:
            front.append(point)
    return front


def load_runs(results_dir: str) -> dict[str, dict[str, dict]]:
    runs: dict[str, dict[str, dict]] = {}
    for workload in sorted(os.listdir(results_dir)):
        workload_dir = os.path.join(results_dir, workload)
        if not os.path.isdir(workload_dir):
            continue
        for method in sorted(os.listdir(workload_dir)):
            run_dir = os.path.join(workload_dir, method)
            trace_path, summary_path = os.path.join(run_dir, "trace.csv"), os.path.join(run_dir, "summary.json")
            if not (os.path.exists(trace_path) and os.path.exists(summary_path)):
                continue
            with open(summary_path) as f:
                summary = json.load(f)
            trace = read_trace(trace_path)
            runs.setdefault(workload, {})[method] = {"summary": summary, "trace": trace, "front": pareto_points(trace)}
    return runs


def normalized_hypervolumes(methods: dict[str, dict]) -> dict[str, float]:
    union = [point for run in methods.values() for point in run["front"]]
    if not union:
        return dict.fromkeys(methods, 0.0)
    ideal = [min(p[i] for p in union) for i in range(3)]
    nadir = [max(p[i] for p in union) * 1.1 for i in range(3)]
    # A fixed-hardware method has one area for every point: give the axis a non-zero span either way.
    span = [(n - i) or 1.0 for i, n in zip(ideal, nadir, strict=True)]

    def normalize(point):
        return tuple((point[i] - ideal[i]) / span[i] for i in range(3))

    return {
        method: hypervolume_3d([normalize(p) for p in run["front"]], (1.0, 1.0, 1.0)) for method, run in methods.items()
    }


def plot_convergence(workload: str, methods: dict[str, dict], path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for method, run in methods.items():
        points = [(r["t"], r["best_edp"]) for r in run["trace"] if math.isfinite(r["best_edp"])]
        if not points:
            continue
        stop = run["summary"]["time_to_stop"]
        times, values = zip(*points, strict=True)
        times, values = [*times, max(stop, times[-1])], [*values, values[-1]]
        color = COLORS.get(method)
        ax.step(times, values, where="post", label=method, color=color, linewidth=1.8)
        ax.plot(times[-1], values[-1], marker="o", color=color)
        ax.annotate(
            run["summary"]["stop_reason"],
            (times[-1], values[-1]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time [s]")
    ax.set_ylabel("best EDP so far [cycles x pJ]")
    ax.set_title(f"{workload}: convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fronts(workload: str, methods: dict[str, dict], path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    all_areas = [p[2] for run in methods.values() for p in run["front"]]
    max_area = max(all_areas, default=1.0)
    for method, run in methods.items():
        if not run["front"]:
            continue
        latency, energy, area = zip(*run["front"], strict=True)
        sizes = [15 + 120 * a / max_area for a in area]
        ax.scatter(latency, energy, s=sizes, label=method, color=COLORS.get(method), alpha=0.7, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("latency [cycles]")
    ax.set_ylabel("energy [pJ]")
    ax.set_title(f"{workload}: Pareto fronts (marker size = area)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="outputs/optimizer_comparison")
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    rows = []
    for workload, methods in runs.items():
        plot_convergence(workload, methods, os.path.join(args.results_dir, f"convergence_{workload}.png"))
        plot_fronts(workload, methods, os.path.join(args.results_dir, f"pareto_{workload}.png"))
        hypervolumes = normalized_hypervolumes(methods)
        best_overall = min(run["summary"]["best_edp"] for run in methods.values())
        for method, run in methods.items():
            summary = run["summary"]
            best = summary.get("best_point") or {}
            rows.append(
                {
                    "workload": workload,
                    "method": method,
                    "stop_reason": summary["stop_reason"],
                    "time_to_stop_s": round(summary["time_to_stop"], 1),
                    "time_to_within_1pct_s": summary["time_to_within_1pct"]
                    and round(summary["time_to_within_1pct"], 1),
                    "n_eval": summary["n_eval"],
                    "best_edp": summary["best_edp"],
                    "edp_gap_to_best_pct": round(100 * (summary["best_edp"] / best_overall - 1), 2)
                    if math.isfinite(summary["best_edp"])
                    else math.inf,
                    "best_latency": best.get("latency"),
                    "best_energy": best.get("energy"),
                    "best_area": best.get("area"),
                    "pareto_points": len(run["front"]),
                    "hypervolume": round(hypervolumes[method], 4),
                    "error": summary.get("error"),
                }
            )
    out_path = os.path.join(args.results_dir, "comparison.csv")
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(
            f"{row['workload']:>9} {row['method']:>9}: {row['stop_reason']:>9} at {row['time_to_stop_s']:>8}s, "
            f"1% at {row['time_to_within_1pct_s']}s, {row['n_eval']:>5} evals, best EDP {row['best_edp']:.4g} "
            f"(+{row['edp_gap_to_best_pct']}%), HV {row['hypervolume']}"
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
