"""Summarize `benchmark_optimizer_comparison.py` runs: convergence curves, Pareto fronts and a comparison table.

Reads `<results-dir>/<workload>/<method>/{trace.csv, summary.json}` and writes into `<results-dir>`:
- `convergence_<workload>.png`: best-so-far EDP against wall time (log-log), one line per method, with a marker
  where each run stopped;
- `pareto_<workload>.png`: every method's non-dominated (latency, energy) points, area as marker size;
- `comparison.csv`: per run -- stop reason, time the stop was declared and real end time (the generation under
  way when the budget stops always finishes), time to reach within 1% of its own final best, best
  EDP (and its gap to the best method on that workload), the best point, evaluation count and the normalized
  (latency, energy, area) hypervolume;
- `comparison.html`: an interactive page in the style of the rolled exploration's `topologies.html` -- per
  workload, the convergence curves, every measured design on a selectable trade-off projection, and the hardware
  topology of each run's best design (plus every design of the rolled candidates' fronts), see
  `stream.visualization.comparison_report`.

The hypervolume of each workload is taken on objectives normalized to [ideal, nadir * 1.1] over the union of
all methods' fronts, so methods on one workload share a reference point and the value lies in [0, 1].

Run: python analyze_optimizer_comparison.py [--results-dir outputs/optimizer_comparison]
"""

import argparse
import csv
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from zigzag.utils import pickle_load  # noqa: E402

from stream.opt.search_budget import hypervolume_3d  # noqa: E402
from stream.visualization.comparison_report import write_comparison_report  # noqa: E402
from stream.visualization.hardware_graph import core_layers_from_workload, extract_topology  # noqa: E402
from stream.visualization.topology_report import design_payload  # noqa: E402

METHOD_ORDER = ("rolled", "graph_ga", "grid_2x2", "grid_4x4")

COLORS = {"rolled": "#2a6fdb", "graph_ga": "#d1495b", "grid_2x2": "#2e933c", "grid_4x4": "#e09f3e"}


def read_trace(path: str) -> list[dict[str, float]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [
        {
            **{
                key: float(row[key])
                for key in ("t", "n_eval", "latency", "energy", "area", "edp", "best_edp", "feasible")
            },
            "info": row.get("info", ""),
        }
        for row in rows
    ]


def step2_start(trace: list[dict]) -> float | None:
    """When a two-step graph_ga run moved from the hardware graph (step 1) to the mapping (step 2)."""
    return next((row["t"] for row in trace if str(row["info"]).startswith("step2")), None)


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
        # Runs finish the generation under way after the stop is declared, so they end at `total_wall_time`.
        stop = max(run["summary"]["time_to_stop"], run["summary"].get("total_wall_time") or 0)
        times, values = zip(*points, strict=True)
        times, values = [*times, max(stop, times[-1])], [*values, values[-1]]
        color = COLORS.get(method)
        ax.step(times, values, where="post", label=method, color=color, linewidth=1.8)
        ax.plot(times[-1], values[-1], marker="o", color=color)
        switch = step2_start(run["trace"])
        if switch is not None:
            ax.axvline(switch, color=color, linestyle=":", linewidth=1)
            ax.annotate(
                "step 2", (switch, values[0]), fontsize=7, color=color, xytext=(3, -10), textcoords="offset points"
            )
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


def _finite(value: float | None) -> float | None:
    """JSON has no infinity: an infeasible or missing value becomes null."""
    return value if value is not None and math.isfinite(value) else None


def _best_design(run_dir: str, method: str) -> dict | None:
    """The run's best design with its hardware topology, read off the `scme.pickle` it saved."""
    path = os.path.join(run_dir, "scme.pickle")
    if not os.path.exists(path):
        return None
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
    try:
        scme = pickle_load(path)
        workload = scme.workload
        layer_names = {node.id: node.short_name for node in workload.node_list if hasattr(node, "short_name")}
        topology = extract_topology(
            scme.accelerator, core_layers=core_layers_from_workload(workload), layer_names=layer_names
        )
        area = sum(
            core.get_area() for core in scme.accelerator.core_list if core.id != scme.accelerator.offchip_core_id
        )
    except Exception as exc:  # a missing figure must not cost the whole report
        print(f"warning: no topology for {run_dir}: {exc!r}")
        return None
    nb_cores = sum(1 for node in topology.nodes if node.kind == "compute")
    record = {
        "id": f"{method}-best",
        "kind": "best",
        "nb_cores": nb_cores,
        "latency": float(scme.latency),
        "energy": float(scme.energy),
        "area": float(area),
    }
    return {**design_payload(record, topology), "method": method, "label": f"{method} best"}


def _rolled_front_designs(run_dir: str) -> list[dict]:
    """Every design on each rolled tiling candidate's (latency, energy, area) front, as its stage saved them."""
    designs = []
    search_dir = os.path.join(run_dir, "tiling_search")
    if not os.path.isdir(search_dir):
        return designs
    for candidate in sorted(os.listdir(search_dir)):
        path = os.path.join(search_dir, candidate, "rolled_search", "topologies.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for design in json.load(f)["designs"]:
                key = candidate.removeprefix("candidate_")[:6]
                designs.append({**design, "method": "rolled", "label": f"rolled {key} {design['id']}"})
    return designs


def build_report_data(results_dir: str, runs: dict[str, dict[str, dict]], rows: list[dict]) -> dict:
    """Everything `comparison.html` shows, as plain JSON-safe data."""
    row_of = {(row["workload"], row["method"]): row for row in rows}
    by_workload = {}
    budget = {}
    for workload, methods in runs.items():
        order = [m for m in METHOD_ORDER if m in methods] + sorted(m for m in methods if m not in METHOD_ORDER)
        workload_runs, designs = {}, []
        for method in order:
            run, summary, row = methods[method], methods[method]["summary"], row_of[(workload, method)]
            budget = {key: summary[key] for key in ("max_time_s", "window_s", "rel_tol")}
            front = set(run["front"])
            curve, last = [], math.inf
            for entry in run["trace"]:
                if math.isfinite(entry["best_edp"]) and entry["best_edp"] < last:
                    last = entry["best_edp"]
                    curve.append([max(entry["t"], 1e-3), last])
            end_time = max(summary["time_to_stop"], summary.get("total_wall_time") or 0)
            workload_runs[method] = {
                "curve": curve,
                "end_time": end_time,
                "stop_reason": summary["stop_reason"],
                "stop_declared": summary["time_to_stop"] if summary["stop_reason"] in ("time", "converged") else None,
                "step2_start": step2_start(run["trace"]),
                "time_to_within_1pct": summary.get("time_to_within_1pct"),
                "n_eval": summary["n_eval"],
                "best_edp": _finite(summary["best_edp"]),
                "edp_gap_pct": _finite(row["edp_gap_to_best_pct"]),
                "pareto_points": row["pareto_points"],
                "hypervolume": row["hypervolume"],
                "points": [
                    {
                        "latency": entry["latency"],
                        "energy": entry["energy"],
                        "area": entry["area"],
                        "pareto": (entry["latency"], entry["energy"], entry["area"]) in front,
                    }
                    for entry in run["trace"]
                    if entry["feasible"]
                ],
            }
            run_dir = os.path.join(results_dir, workload, method)
            best = _best_design(run_dir, method)
            if best is not None:
                designs.append(best)
            if method == "rolled":
                designs += _rolled_front_designs(run_dir)
        best_design = min(range(len(designs)), key=lambda i: designs[i]["latency"] * designs[i]["energy"], default=0)
        by_workload[workload] = {
            "methods": order,
            "runs": workload_runs,
            "designs": designs,
            "best_design": best_design,
        }
    return {"workloads": sorted(by_workload), "by_workload": by_workload, "budget": budget}


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
                    "total_wall_time_s": round(summary.get("total_wall_time") or summary["time_to_stop"], 1),
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
                    "step2_start_s": step2_start(run["trace"]),
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
    html_path = os.path.join(args.results_dir, "comparison.html")
    write_comparison_report(build_report_data(args.results_dir, runs, rows), html_path)
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
