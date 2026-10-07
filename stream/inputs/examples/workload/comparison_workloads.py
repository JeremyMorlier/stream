"""Workloads of the optimizer comparison (`benchmark_optimizer_comparison.py`), exported to ONNX from torch.

- `export_resnet_bottleneck`: the ResNet-50 first bottleneck `example_rolled_schedule_exploration.py` searches.
- `export_attention_block`: the attention half of an LLM decoder layer -- Q/K/V projections, multi-head scaled
  dot-product attention, output projection and the residual connection.

Both use the legacy exporter (`dynamo=False`): the dynamo one drops Conv's `kernel_shape`, which Stream's parser
requires.
"""

import math
import os

import onnx
import torch
from onnx.shape_inference import infer_shapes_path
from torch import nn


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


class AttentionBlock(nn.Module):
    """`x + W_o · MHA(W_q x, W_k x, W_v x)`. Projections carry no bias (as in LLaMA-style layers) so the graph is
    MatMuls, the per-head Reshape/Transpose, one Softmax and the residual Add. The 1/sqrt(d_head) scale is folded
    into `W_q` rather than emitted as a separate elementwise node."""

    def __init__(self, d_model: int, heads: int):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.d_head = d_model // heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.q.weight.mul_(1 / math.sqrt(self.d_head))

    def forward(self, x):
        batch, seq, d_model = x.shape
        q = self.q(x).reshape(batch, seq, self.heads, self.d_head).transpose(1, 2)
        k = self.k(x).reshape(batch, seq, self.heads, self.d_head).permute(0, 2, 3, 1)
        v = self.v(x).reshape(batch, seq, self.heads, self.d_head).transpose(1, 2)
        attention = torch.softmax(torch.matmul(q, k), dim=-1)
        context = torch.matmul(attention, v).transpose(1, 2).reshape(batch, seq, d_model)
        return x + self.o(context)


def _export(module: nn.Module, dummy_input: torch.Tensor, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(
        module.eval(),
        dummy_input,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,  # the dynamo exporter omits Conv's kernel_shape attribute, which Stream's parser requires
    )
    infer_shapes_path(path, path)
    return path


def export_resnet_bottleneck(path: str) -> str:
    return _export(Bottleneck(256, 64), torch.randn(1, 256, 56, 56), path)


def export_attention_block(path: str, seq: int = 128, d_model: int = 512, heads: int = 8) -> str:
    return _export(AttentionBlock(d_model, heads), torch.randn(1, seq, d_model), path)


def single_stack(path: str) -> list[tuple[int, ...]]:
    """Every node of the model in one fused layer stack, as the rolled example uses."""
    return [tuple(range(len(onnx.load(path).graph.node)))]
