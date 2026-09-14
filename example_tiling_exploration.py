"""Minimal example of `TilingExplorationStage` (via `stream.api.optimize_tiling`).

`TilingExplorationStage` searches per-layer, per-dimension intra-/inter-core tiling assignments with a
genetic algorithm: every dimension of every layer becomes a gene, assigned either to intra-core tiling or
inter-core tiling with a factor of 1 (no split) or one of that dimension's prime divisors. For every
candidate individual it re-runs the full downstream pipeline (tiling -> cost estimation -> allocation) once,
and returns the best candidate by `sort_key` together with a sample of other good candidates (the search's
hall of fame). See `stream/stages/generation/tiling_exploration.py` for the full algorithm.

Run: python example_tiling_exploration.py
"""

import logging as _logging
import os

import onnx
import torch
from onnx.shape_inference import infer_shapes_path
from torch import nn

from stream.api import optimize_tiling

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


class TwoConvBlock(nn.Module):
    """A tiny two-conv-layer block -- kept deliberately small so the tiling GA below runs in well under a
    minute; swap in any other ONNX model for a real search."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, input_tensor):
        return self.conv2(self.conv1(input_tensor))


############################################INPUTS############################################
hardware = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
workload_path = "stream/inputs/examples/workload/two_conv_example.onnx"
mapping_path = "stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml"
mode = "fused"
experiment_id = "tiling_exploration_example"
#################################################################################################

# Generate the example workload if it isn't already there.
if not os.path.exists(workload_path):
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
layer_stacks = [tuple(range(nb_nodes))]  # fuse the whole (tiny) block into a single layer stack

best_scme, all_candidates = optimize_tiling(
    hardware=hardware,
    workload=workload_path,
    mapping=mapping_path,
    mode=mode,
    layer_stacks=layer_stacks,
    experiment_id=experiment_id,
    output_path="outputs",
    skip_if_exists=False,
    # TilingExplorationStage's own outer GA -- sweeps candidate tiling configurations.
    nb_tiling_ga_generations=2,
    nb_tiling_ga_individuals=4,
    # Per-candidate GeneticAlgorithmAllocationStage's inner GA -- sizes each candidate's core-allocation search.
    nb_ga_generations=2,
    nb_ga_individuals=2,
    sort_key="latency",
    profile=True,  # log a per-stage timing breakdown when it's done, see TilingExplorationStage.run()
)

print(f"Evaluated {len(all_candidates)} tiling candidate(s).")
print(f"Best candidate: latency={best_scme.latency}, energy={best_scme.energy}")
