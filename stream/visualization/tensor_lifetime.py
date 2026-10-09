"""Gantt charts of tensor lifetimes per memory in a steady-state schedule (see
`stream.cost_model.steady_state_tensor_lifetime.TensorLifetimeAnalyzer`)."""

import logging
from typing import TYPE_CHECKING

import seaborn
from matplotlib import patches
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from stream.cost_model.steady_state_tensor_lifetime import TensorLifetimeAnalyzer

logger = logging.getLogger(__name__)


def plot_tensor_lifetimes(
    analyzer: "TensorLifetimeAnalyzer", figsize: tuple[int, int] = (10, 6), filename_prefix: str = "tensor_lifetime"
) -> None:
    """
    For each memory/resource, plot a Gantt chart:
    - Top: computation/transfer nodes scheduled on that resource (not tensor nodes)
    - Bottom: tensor lifetimes (rectangles) for tensors stored in that memory
    """
    color_palette = seaborn.color_palette("pastel")
    tensor_color_map = {}
    node_color_map = {}
    all_tensors = [lt.tensor for lt in analyzer.lifetimes]
    all_nodes = [n for n in analyzer.workload.nodes() if n not in all_tensors]
    for i, tensor in enumerate(all_tensors):
        tensor_color_map[tensor] = color_palette[i % len(color_palette)]
    for i, node in enumerate(all_nodes):
        node_color_map[node] = color_palette[i % len(color_palette)]

    for mem, lifetimes in analyzer.get_lifetimes_by_memory().items():
        fig, ax = plt.subplots(figsize=figsize)
        # 1. Plot tensor lifetimes (bottom): rectangles for each tensor
        y_base = 0
        y_gap = 0
        for lt in sorted(lifetimes, key=lambda i: i.start):
            start = lt.start
            end = lt.end
            color = tensor_color_map.get(lt.tensor, "#bbbbbb")
            height = lt.tensor.size
            ax.add_patch(patches.Rectangle((start, y_base), end - start, height, facecolor=color, edgecolor="black"))
            text_x = (start + end) / 2
            text_y = y_base + height / 2
            text = lt.tensor.node_name
            ax.text(text_x, text_y, text, ha="center", va="center", fontsize=9)
            y_base += height + y_gap
        # 2. Plot schedule (top): all computation/transfer nodes on this resource
        nodes_on_mem = [
            n
            for n in analyzer.workload.nodes()
            if getattr(n, "chosen_resource_allocation", None) == mem and n not in all_tensors
        ]
        nodes_on_mem = sorted(
            nodes_on_mem, key=lambda n: (analyzer.node_start_times.get(n, 0), analyzer.node_finish_times.get(n, 0))
        )
        y_sched = y_base
        height_sched = y_base / 6  # Fixed height for schedule relative to the tensors
        for node in nodes_on_mem:
            start = analyzer.node_start_times.get(node, 0)
            end = analyzer.node_finish_times.get(node, 0)
            color = node_color_map.get(node, "#cccccc")
            ax.add_patch(
                patches.Rectangle((start, y_sched), end - start, height_sched, facecolor=color, edgecolor="black")
            )
            text = node.plot_name
            text_x = (start + end) / 2
            text_y = y_sched + height_sched / 2
            ax.text(text_x, text_y, text, ha="center", va="bottom", fontsize=9)
        # 3. Formatting
        max_time = max(analyzer.node_finish_times.values(), default=1)
        ax.set_xlim(0, max_time + 1)
        ax.set_ylim(0, y_base + height_sched)
        ax.set_yticks([])
        ax.set_xlabel("Time")
        ax.set_ylabel("Tensors (bottom) / Schedule (top)")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        # 4. Save
        fname = f"{filename_prefix}_mem{mem}.png"
        fig.tight_layout()
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved tensor lifetime visualization for memory {mem} to {fname}")
