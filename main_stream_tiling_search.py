import logging as _logging
import re

import onnx
import torch
import torch.nn as nn
from onnx.shape_inference import infer_shapes_path

from stream.api import optimize_tiling
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


###########################################RESNET50 FIRST BOTTLENECK BLOCK#######################################
class ResNet50FirstBottleneck(nn.Module):
    """The first bottleneck block of ResNet50's layer1: in_channels=64 (stem output), mid_channels=64,
    out_channels=256, stride=1. Needs a downsample (1x1 conv + BN) on the identity path since in_channels !=
    out_channels."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.downsample_conv = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.downsample_bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input_tensor):
        identity = self.downsample_bn(self.downsample_conv(input_tensor))
        out = self.relu(self.bn1(self.conv1(input_tensor)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + identity
        out = self.relu(out)
        return out


bottleneck_onnx_path = "stream/inputs/examples/workload/resnet50_first_bottleneck.onnx"
# Stem output feature map for ResNet50 on a 224x224 input is 56x56x64
dummy_input = torch.randn(1, 64, 56, 56)
torch.onnx.export(
    ResNet50FirstBottleneck().eval(),
    dummy_input,
    bottleneck_onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamo=False,  # the dynamo exporter omits Conv's kernel_shape attribute, which Stream's parser requires
)
# Infer the shapes of intermediate tensors and save it back to the same file
infer_shapes_path(bottleneck_onnx_path, bottleneck_onnx_path)

# Count the number of nodes in the block and fuse them all into a single layer stack
bottleneck_model = onnx.load(bottleneck_onnx_path)
nb_bottleneck_nodes = len(bottleneck_model.graph.node)
_logging.getLogger(__name__).info(f"ResNet50 first bottleneck block has {nb_bottleneck_nodes} ONNX node(s).")
####################################################################################################################

############################################INPUTS############################################
accelerator = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
workload_path = bottleneck_onnx_path
mapping_path = "stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml"
mode = "fused"
layer_stacks = [tuple(range(nb_bottleneck_nodes))]
nb_ga_generations = 2
nb_ga_individuals = 2
##############################################################################################

################################PARSING###############################
hw_name = accelerator.split("/")[-1].split(".")[0]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
experiment_id = f"{hw_name}-{wl_name}-{mode}-tiling_bottleneck3"
######################################################################

##############PLOTTING###############
plot_full_schedule = True
draw_dependencies = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
#####################################

################################PATHS################################
timeline_fig_path_plotly = f"outputs/{experiment_id}/schedule.html"
memory_fig_path = f"outputs/{experiment_id}/memory.png"
json_path = f"outputs/{experiment_id}/scme.json"
#####################################################################

scme, all_results = optimize_tiling(
    hardware=accelerator,
    workload=workload_path,
    mapping=mapping_path,
    mode=mode,
    layer_stacks=layer_stacks,
    nb_ga_generations=nb_ga_generations,
    nb_ga_individuals=nb_ga_individuals,
    experiment_id=experiment_id,
    output_path="outputs",
    skip_if_exists=False,
    profile=True,
    max_workers=4,
    nb_tiling_ga_generations=1,
    nb_tiling_ga_individuals=2,
    nb_core_ga_generations=4,
    nb_core_ga_individuals=4,
    core_max_workers=4,
    core_pareto_points=4,
)
print(f"Evaluated {len(all_results)} tiling candidate(s).")

# Plotting memory usage of best SCME
plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path)

if all_results:
    # Load in the CostModelEvaluationLUT from the best candidate's run (skipped if `scme` was loaded from cache,
    # i.e. `skip_if_exists=True` and a previous run's result already existed)
    best_candidate_index = next(
        extra_info["candidate_index"] for result_scme, extra_info in all_results if result_scme is scme
    )
    cost_lut_path = f"outputs/{experiment_id}/tiling_search/candidate_{best_candidate_index}/cost_lut.pickle"
    cost_lut = CostModelEvaluationLUT(cost_lut_path)

    # Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
    convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)

print(scme.latency, scme.energy)
