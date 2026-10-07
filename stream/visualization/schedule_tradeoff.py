"""Visualize the `(latency, energy, area)` trade-off across the schedules `ScheduleExplorationStage` measured.

Three objectives don't fit in one readable plot, so the figure shows all three pairwise projections -- each
coloured by the *third* objective, which is what makes a trade-off visible rather than just a scatter -- plus a
3D view. Measured-but-dominated schedules stay in grey as the backdrop the front is drawn against.
"""

import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- the stage runs headless, inside the pipeline
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401 -- registers the '3d' projection

logger = logging.getLogger(__name__)

AXES = {
    "latency": "Latency [cycles]",
    "energy": "Energy [pJ]",
    # `Core.get_area()` adds the operational array's area (from the hardware description's unit area) to the
    # memories' CACTI area, so the sum carries no single physical unit -- don't label it mm^2.
    "area": "Area [a.u.]",
}
PROJECTIONS = (("latency", "energy", "area"), ("latency", "area", "energy"), ("energy", "area", "latency"))

# One marker per schedule family, so an unrolled schedule and the rolled (core-reusing) variant folded out of
# it stay tellable apart on a shared front. Anything unlabelled falls back to the unrolled marker.
KIND_MARKERS = {"unrolled": "o", "rolled": "^"}
KIND_ORDER = ("unrolled", "rolled")


def record_key(record: dict[str, Any]) -> Any:
    """Identity of a measured schedule. `id` where present -- a rolled variant shares its source's `ranks`, so
    keying on `ranks` alone would make the two collide and mark the wrong points as Pareto."""
    return record.get("id", record.get("ranks"))


def pareto_staircase(points: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    """Staircase through the 2D-non-dominated points of a projection. A schedule on the 3D front need not be on
    a 2D projection's front, so the line is drawn through the projected front only -- it marks the achievable
    boundary, while the off-line front points are the ones paying for their third objective."""
    non_dominated: list[tuple[float, float]] = []
    best_y = float("inf")
    for x, y in sorted(points):
        if y < best_y:
            non_dominated.append((x, y))
            best_y = y
    xs: list[float] = []
    ys: list[float] = []
    for index, (x, y) in enumerate(non_dominated):
        if index:
            xs.append(x)
            ys.append(non_dominated[index - 1][1])
        xs.append(x)
        ys.append(y)
    return xs, ys


def plot_schedule_tradeoffs(
    all_results: list[dict[str, Any]],
    pareto_results: list[dict[str, Any]],
    fig_path: str,
) -> None:
    """Save the trade-off figure for one schedule search.

    Args:
        all_results: every measured schedule (dicts with `latency`, `energy`, `area`, `ranks`).
        pareto_results: the subset on the 3D Pareto front.
        fig_path: where to write the PNG.
    """
    if not all_results:
        logger.warning("plot_schedule_tradeoffs: nothing measured, skipping the plot.")
        return

    pareto_keys = {record_key(record) for record in pareto_results}
    dominated = [record for record in all_results if record_key(record) not in pareto_keys]
    kinds = [kind for kind in KIND_ORDER if any(record.get("kind", "unrolled") == kind for record in pareto_results)]
    by_kind = {kind: [record for record in pareto_results if record.get("kind", "unrolled") == kind] for kind in kinds}

    figure = plt.figure(figsize=(14, 10))
    for index, (x_key, y_key, color_key) in enumerate(PROJECTIONS):
        axis = figure.add_subplot(2, 2, index + 1)
        if dominated:
            axis.scatter(
                [record[x_key] for record in dominated],
                [record[y_key] for record in dominated],
                s=18,
                c="0.75",
                label=f"dominated ({len(dominated)})",
                zorder=1,
            )
        xs, ys = pareto_staircase([(record[x_key], record[y_key]) for record in pareto_results])
        axis.plot(xs, ys, color="0.4", linewidth=1.2, linestyle="--", zorder=2)
        # One shared colour scale across the families, so the third-objective colour stays comparable.
        colors = [record[color_key] for record in pareto_results]
        vmin, vmax = min(colors), max(colors)
        scatter = None
        for kind in kinds:
            records = by_kind[kind]
            scatter = axis.scatter(
                [record[x_key] for record in records],
                [record[y_key] for record in records],
                s=70,
                c=[record[color_key] for record in records],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                marker=KIND_MARKERS[kind],
                edgecolors="black",
                linewidths=0.6,
                label=f"{kind} ({len(records)})",
                zorder=3,
            )
        if scatter is not None:
            figure.colorbar(scatter, ax=axis, label=AXES[color_key])
        axis.set_xlabel(AXES[x_key])
        axis.set_ylabel(AXES[y_key])
        axis.set_title(f"{y_key} vs {x_key}, coloured by {color_key}")
        axis.legend(loc="upper right", fontsize=8)
        axis.grid(True, alpha=0.3)

    axis3d = figure.add_subplot(2, 2, 4, projection="3d")
    if dominated:
        axis3d.scatter(
            [record["latency"] for record in dominated],
            [record["energy"] for record in dominated],
            [record["area"] for record in dominated],
            s=12,
            c="0.75",
            depthshade=False,
        )
    for kind in kinds:
        records = by_kind[kind]
        axis3d.scatter(
            [record["latency"] for record in records],
            [record["energy"] for record in records],
            [record["area"] for record in records],
            s=55,
            c="tab:orange" if kind == "unrolled" else "tab:blue",
            marker=KIND_MARKERS[kind],
            edgecolors="black",
            linewidths=0.5,
            label=f"{kind} ({len(records)})",
            depthshade=False,
        )
    if len(kinds) > 1:
        axis3d.legend(loc="upper left", fontsize=8)
    axis3d.set_xlabel(AXES["latency"])
    axis3d.set_ylabel(AXES["energy"])
    axis3d.set_zlabel(AXES["area"])
    axis3d.set_title(f"{len(pareto_results)} pareto of {len(all_results)} measured schedules")

    figure.suptitle("Schedule trade-offs over per-tile pareto core designs", fontsize=14)
    figure.tight_layout()
    figure.savefig(fig_path, dpi=150)
    plt.close(figure)
    logger.info(f"plot_schedule_tradeoffs: saved trade-off figure to {fig_path}.")
