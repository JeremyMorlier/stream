"""Minimal example of `RolledScheduleExplorationStage` (via `stream.api.optimize_rolled_schedules`).

Where `optimize_schedules` searches which core *design* each layer gets on a fully unrolled accelerator -- one
dedicated core per `(layer, inter-core group)` -- this adds the *rolling* axis: cores that have finished their
layer are reused for later ones, so the accelerator needs fewer of them. Each measured schedule is folded onto
fewer cores, the communication graph is re-derived (traffic between merged cores becomes free; new links appear
between merged groups), and the folded accelerator is re-measured with the real scheduler.

Both families land on one `(latency, energy, area)` front, so the printout shows the whole trade-off: unrolled
designs at the fast end, folded ones at the cheap end.

Run: python example_rolled_schedule_exploration.py
"""

import logging as _logging
import os
from pathlib import Path

import onnx
import torch
from onnx.shape_inference import infer_shapes_path
from torch import nn

from stream.api import optimize_rolled_schedules
from stream.visualization.tikz_export import write_figures

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 1x1: reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3: spatial convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1: expand channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        # Projection shortcut when dimensions differ
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        out = self.relu(out)

        return out


class TwoConvBlock(nn.Module):
    """Same tiny block as `example_schedule_exploration.py`, so the search finishes in well under a minute."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, input_tensor):
        return self.conv2(self.conv1(input_tensor))


############################################PATHS############################################
output_path = "outputs"
hardware = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
workload_path = "stream/inputs/examples/workload/two_conv_example.onnx"
mapping_path = "stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml"
mode = "fused"
experiment_id = "rolled_schedule_exploration_long"

Path(f"{output_path}/").mkdir(parents=True, exist_ok=True)
Path(f"{output_path}/{experiment_id}").mkdir(parents=True, exist_ok=True)
workload_path = f"{output_path}/{experiment_id}/workload.onnx"
#################################################################################################

os.makedirs(os.path.dirname(workload_path), exist_ok=True)
dummy_input = torch.randn(1, 256, 56, 56)
torch.onnx.export(
    Bottleneck(256, 64).eval(),
    dummy_input,
    workload_path,
    input_names=["input"],
    output_names=["output"],
    dynamo=False,  # the dynamo exporter omits Conv's kernel_shape attribute, which Stream's parser requires
)
infer_shapes_path(workload_path, workload_path)

nb_nodes = len(onnx.load(workload_path).graph.node)
layer_stacks = [tuple(range(nb_nodes))]

best_scme, pareto_schedules = optimize_rolled_schedules(
    hardware=hardware,
    workload=workload_path,
    mapping=mapping_path,
    mode=mode,
    layer_stacks=layer_stacks,
    experiment_id=experiment_id,
    output_path=output_path,
    skip_if_exists=False,
    sort_key="latency",
    # Per-tile core search (inherited): its front size is the schedule search's branching factor, and its
    # designs are the pool a folded core picks from.
    nb_core_ga_generations=16,
    nb_core_ga_individuals=16,
    core_pareto_points=10,
    # Unrolled schedule search.
    max_schedule_evaluations=60,
    # Rolling. Under layer fusion the two convs interleave, so the strict tolerance often admits no fold at
    # all -- the ladder is what buys area by paying serialization.
    fold_tolerances=(0.0, 0.25, 1.0),
    fold_overlap="exact",
)

print(f"{len(pareto_schedules)} pareto schedule(s) on the combined front:")
for record in sorted(pareto_schedules, key=lambda r: r["latency"]):
    fold = "" if record["kind"] == "unrolled" else f"  (from {record['source_id']}, t={record['fold_tolerance']:g})"
    print(
        f"  {record['kind']:<8} cores={record['nb_cores']:<3} latency={record['latency']:.6g} "
        f"energy={record['energy']:.6g} area={record['area']:.6g}{fold}"
    )
print(f"Best by latency: latency={best_scme.latency}, energy={best_scme.energy}")
print(f"Artifacts: {output_path}/{experiment_id}/rolled_search/")

# The same front the stage just drew as `topologies.html`, re-emitted as LaTeX. `pareto_topologies_*.tex` is
# the one figure of the pair: the area-vs-latency front with the accelerators behind four of its points drawn
# as insets above it, each tied to its point. `figures.tex` next to it is a standalone document -- compile it
# with pdflatex to check the fragments before they go into a paper.
figures = write_figures(
    f"{output_path}/{experiment_id}/rolled_search",
    x_key="latency",
    y_key="area",
    color_key="energy",
    nb_topologies=4,
    combined_float_env="figure*",  # the picture is 16 cm wide, so a two-column paper wants the starred float
)
print(f"Combined LaTeX figure: {figures['combined']}")
print(f"Preview document:      {figures['preview']} (pdflatex it in place)")
