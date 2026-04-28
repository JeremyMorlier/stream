import logging as _logging
import re
from pathlib import Path

import onnx
from onnx.shape_inference import infer_shapes_path
import torch
import yaml

from stream.api import optimize_allocation_co
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

DEFAULT_MAPPING = [
    {"name": "default", "core_allocation": [0], "intra_core_tiling": ["D, all"]},
    {"name": "Conv", "core_allocation": [0], "intra_core_tiling": ["OY, all"]},
    {"name": "Relu", "core_allocation": [0], "intra_core_tiling": ["D, all"]},
]


class ConvReluModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


def export_conv_relu_onnx(workload_path: Path) -> None:
    model = ConvReluModel().eval()
    x = torch.randn(1, 8, 96, 96)
    workload_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            x,
            str(workload_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            x,
            str(workload_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
        )
    infer_shapes_path(str(workload_path), str(workload_path))

    node_count = len(onnx.load(str(workload_path)).graph.node)
    _logging.getLogger(__name__).info("Exported Conv+ReLU ONNX to %s with %d node(s)", workload_path, node_count)


def write_default_mapping(mapping_path: Path) -> None:
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(DEFAULT_MAPPING, f, sort_keys=False)


############################################INPUTS############################################
accelerator = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
artifacts_dir = Path("outputs/main_stream_co_tiling")
workload_path = artifacts_dir / "conv_relu.onnx"
mapping_path = artifacts_dir / "mapping.yaml"
mode = "fused"
layer_stacks = [(0, 1)]
##############################################################################################

if not workload_path.exists():
    export_conv_relu_onnx(workload_path)

if not mapping_path.exists():
    write_default_mapping(mapping_path)

################################PARSING###############################
hw_name = accelerator.split("/")[-1].split(".")[0]
wl_name = re.split(r"/|\\.", str(workload_path))[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\\.", str(workload_path))[-2]
experiment_id = f"{hw_name}-{wl_name}-{mode}-constraint_optimization-intra_tiling_explore"
######################################################################

scme = optimize_allocation_co(
    hardware=accelerator,
    workload=str(workload_path),
    mapping=str(mapping_path),
    mode=mode,
    layer_stacks=layer_stacks,
    experiment_id=experiment_id,
    output_path="outputs",
    skip_if_exists=False,
    explore_intra_core_tiling=True,
    explore_intra_core_tiling_dims=["K", "OY", "D", "OX"],
    max_explored_tiling_configurations=256,
)

############PLOTTING#############
section_start_percent = (0,)
percent_shown = (100,)
#################################

#########################PLOTTING PATHS##############################
memory_fig_path = f"outputs/{experiment_id}/memory.png"
json_path = f"outputs/{experiment_id}/scme.json"
#####################################################################

#####################CostModelEvaluationLUT LOAD#############################
cost_lut_path = f"outputs/{experiment_id}/cost_lut_post_co.pickle"
cost_lut = CostModelEvaluationLUT(cost_lut_path)
#############################################################################

# Plotting memory usage of best SCME
plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path)

# Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)
print(scme.latency, scme.energy)
