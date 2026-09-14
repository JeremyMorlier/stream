import logging as _logging
import re

import torch
from onnx.shape_inference import infer_shapes_path
from torch import nn

from stream.api import optimize_allocation_co
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


###########################################TWO-CONV MODEL#######################################
class TwoConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Added conv3 for testing

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x + self.conv3(input_tensor)
        return x


two_conv_onnx_path = "stream/inputs/examples/workload/two_conv.onnx"
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(
    TwoConvNet().eval(),
    dummy_input,
    two_conv_onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamo=False,  # the dynamo exporter omits Conv's kernel_shape attribute, which Stream's parser requires
)
# Infer the shapes of intermediate tensors and save it back to the same file
infer_shapes_path(two_conv_onnx_path, two_conv_onnx_path)
##################################################################################################

############################################INPUTS############################################
accelerator = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
workload_path = two_conv_onnx_path
mapping_path = "stream/inputs/examples/mapping/tpu_like_multi_inter.yaml"
mode = "fused"
layer_stacks = [(0, 1)]
##############################################################################################

################################PARSING###############################
hw_name = accelerator.rsplit("/", maxsplit=1)[-1].split(".", maxsplit=1)[0]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
experiment_id = f"{hw_name}-{wl_name}-{mode}-constraint_optimization"
######################################################################

scme = optimize_allocation_co(
    hardware=accelerator,
    workload=workload_path,
    mapping=mapping_path,
    mode=mode,
    layer_stacks=layer_stacks,
    experiment_id=experiment_id,
    output_path="outputs",
    skip_if_exists=False,
)

############PLOTTING#############
plot_full_schedule = True
draw_dependencies = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
#################################

#########################PLOTTING PATHS##############################
timeline_fig_path_plotly = f"outputs/{experiment_id}/schedule.html"
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
