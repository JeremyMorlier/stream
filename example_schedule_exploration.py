"""Minimal example of `ScheduleExplorationStage` (via `stream.api.optimize_schedules`).

Where `optimize_tiling` collapses each unique tile's core Pareto front to a single design before scheduling,
this searches the *product* of those fronts and measures whole schedules, returning every schedule that is
non-dominated on `(latency, energy, area)` at once -- plus a trade-off figure showing them.

The search is a branch and bound: the accelerator topology and each `(tile, design)` cost model result are
computed once, and lower bounds on all three objectives prune combinations whose every completion is already
dominated. See `stream/stages/generation/schedule_exploration.py` for why each bound holds.

Run: python example_schedule_exploration.py
"""

import logging as _logging
import os
from pathlib import Path

import onnx
import torch
from onnx.shape_inference import infer_shapes_path
from torch import nn

from stream.api import optimize_schedules

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


class TwoConvBlock(nn.Module):
    """Same tiny block as `example_tiling_exploration.py`, so the search finishes in well under a minute."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, input_tensor):
        return self.conv2(self.conv1(input_tensor))


############################################INPUTS############################################
output_path = "outputs"
hardware = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
mapping_path = "stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml"
mode = "fused"
experiment_id = "schedule_exploration_example"

Path(f"{output_path}/").mkdir(parents=True, exist_ok=True)
Path(f"{output_path}/{experiment_id}").mkdir(parents=True, exist_ok=True)
Path(f"{output_path}/{experiment_id}/workload/").mkdir(parents=True, exist_ok=True)
workload_path = f"{output_path}/{experiment_id}/workload/two_conv_example.onnx"
#################################################################################################

os.makedirs(os.path.dirname(workload_path), exist_ok=True)
dummy_input = torch.randn(1, 16, 32, 32)
torch.onnx.export(
    TwoConvBlock().eval(),
    dummy_input,
    workload_path,
    input_names=["input"],
    output_names=["output"],
    dynamo=False,  # the dynamo exporter omits Conv's kernel_shape attribute, which Stream's parser requires
)
infer_shapes_path(workload_path, workload_path)

nb_nodes = len(onnx.load(workload_path).graph.node)
layer_stacks = [tuple(range(nb_nodes))]

best_scme, pareto_schedules = optimize_schedules(
    hardware=hardware,
    workload=workload_path,
    mapping=mapping_path,
    mode=mode,
    layer_stacks=layer_stacks,
    experiment_id=experiment_id,
    output_path=output_path,
    skip_if_exists=False,
    sort_key="latency",
    # Per-tile core search (inherited from CoreArchitectureExplorationStage): its front size is the schedule
    # search's branching factor.
    nb_core_ga_generations=10,
    nb_core_ga_individuals=16,
    core_pareto_points=10,
    # Schedule search.
    max_schedule_evaluations=200,
    prune_schedules=True,
)

print(f"{len(pareto_schedules)} pareto schedule(s):")
for record in sorted(pareto_schedules, key=lambda r: r["latency"]):
    print(
        f"  ranks={record['ranks']} latency={record['latency']:.6g} "
        f"energy={record['energy']:.6g} area={record['area']:.6g}"
    )
print(f"Best by latency: latency={best_scme.latency}, energy={best_scme.energy}")
print("Trade-off figure: outputs/schedule_exploration_example/schedule_search/schedule_tradeoffs.png")
