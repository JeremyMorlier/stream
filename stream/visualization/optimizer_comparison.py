"""Static figures comparing optimizer runs on a common wall-clock budget (see
`benchmarks/benchmark_optimizer_comparison.py` and `benchmarks/analyze_optimizer_comparison.py`): best-EDP
convergence over time and the methods' Pareto fronts."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHOD_COLORS = {
    "rolled": "#2a6fdb",
    "graph_ga": "#d1495b",
    "grid_2x2": "#2e933c",
    "grid_4x4": "#e09f3e",
    "tiling_placement_cpsat": "#1b4f9c",
    "tiling_placement_gurobi": "#6a9be8",
    "tiling_coreopt_cpsat": "#0f2f66",
    "tiling_coreopt_gurobi": "#3c7fd0",
    "edge_tpu_template": "#8a5cd6",
}


def step2_start(trace: list[dict]) -> float | None:
    """When a two-step graph_ga run moved from the hardware graph (step 1) to the mapping (step 2)."""
    return next((row["t"] for row in trace if str(row["info"]).startswith("step2")), None)


def plot_convergence(workload: str, methods: dict[str, dict], path: str) -> None:
    """Best EDP so far against wall time (log-log), one step line per method, with the stop reason at the end of
    each run and a dotted line where a two-step run entered step 2. `methods` maps a method to its run:
    `{"trace": [...], "summary": {...}}` as `analyze_optimizer_comparison.load_runs` reads them."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for method, run in methods.items():
        points = [(r["t"], r["best_edp"]) for r in run["trace"] if math.isfinite(r["best_edp"])]
        if not points:
            continue
        # Runs finish the generation under way after the stop is declared, so they end at `total_wall_time`.
        stop = max(run["summary"]["time_to_stop"], run["summary"].get("total_wall_time") or 0)
        times, values = zip(*points, strict=True)
        times, values = [*times, max(stop, times[-1])], [*values, values[-1]]
        color = METHOD_COLORS.get(method)
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
    """Every method's non-dominated `(latency, energy)` points (`run["front"]`), marker size proportional to
    area."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    all_areas = [p[2] for run in methods.values() for p in run["front"]]
    max_area = max(all_areas, default=1.0)
    for method, run in methods.items():
        if not run["front"]:
            continue
        latency, energy, area = zip(*run["front"], strict=True)
        sizes = [15 + 120 * a / max_area for a in area]
        ax.scatter(
            latency, energy, s=sizes, label=method, color=METHOD_COLORS.get(method), alpha=0.7, edgecolors="none"
        )
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
