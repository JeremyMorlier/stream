"""Tests for the slices the inter-layer edges of the tiled workload carry, and for the scheduler moving only them.

A consumer tile reads only part of a producer tile's output when the two layers are tiled differently. The edges
carry that part (`TiledWorkloadGenerationStage.get_rtree_edge_slice`, `get_inter_edges_hybrid`), whatever the
layers call their dimensions: a Conv's OY and OX are a Relu's D and K, and attention's MatMul and Add both have a
D that means different things. The scheduler transfers that slice (`CoalaScheduler.edge_slice`) instead of the
producer's whole tile, which used to overflow the consumer cores sized for their own tiles.
"""

import os
from collections import Counter

from zigzag.datatypes import Constants, LayerDim

from stream.api import optimize_allocation_ga
from stream.cost_model.scheduler import CoalaScheduler
from stream.inputs.examples.workload.comparison_workloads import (
    export_attention_block,
    export_resnet_bottleneck,
    single_stack,
)
from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.tiled_workload_generation import TiledWorkloadGenerationStage
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import MainStage, Stage
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.tensor import Tensor

HARDWARE = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
MAPPING_DIR = "stream/inputs/examples/mapping"


class _CaptureWorkload(Stage):
    def is_leaf(self) -> bool:
        return True

    def run(self):
        yield self.kwargs["workload"], {}


def tiled_workload(workload_path: str, mapping: str, out_dir: str):
    return MainStage(
        [
            AcceleratorParserStage,
            ONNXModelParserStage,
            LayerStacksGenerationStage,
            TilingGenerationStage,
            TiledWorkloadGenerationStage,
            _CaptureWorkload,
        ],
        accelerator=HARDWARE,
        workload_path=workload_path,
        mapping_path=os.path.join(MAPPING_DIR, mapping),
        layer_stacks=single_stack(workload_path),
        mode="fused",
        tiled_workload_path=os.path.join(out_dir, "tiled_workload.pickle"),
        loma_lpf_limit=6,
    ).run()[0][0]


def inter_layer_edges(workload, producer_name: str, consumer_name: str):
    return [
        (producer, consumer, data)
        for producer, consumer, data in workload.edges(data=True)
        if isinstance(producer, ComputationNode)
        and producer.name == producer_name
        and consumer.name == consumer_name
    ]


def test_resnet_relu_reads_its_columns_of_the_conv_tile(tmp_path):
    """A Relu tile (one row D, a quarter of the columns K) reads a quarter of the Conv tile's row (OY, all OX)."""
    workload = tiled_workload(
        export_resnet_bottleneck(str(tmp_path / "resnet.onnx")), "tpu_like_quad_core_ga.yaml", str(tmp_path)
    )
    edges = inter_layer_edges(workload, "/conv1/Conv", "/relu/Relu")
    assert edges
    for producer, consumer, data in edges:
        output = producer.operand_tensors[Constants.OUTPUT_LAYER_OP]
        slice_ranges = dict(zip(output.loop_dimensions, data["slice"], strict=True))
        # Relu's D and K are the Conv's OY and OX
        assert slice_ranges[LayerDim("OY")] == consumer.loop_ranges[LayerDim("D")]
        assert slice_ranges[LayerDim("OX")] == consumer.loop_ranges[LayerDim("K")]
        assert data["bits"] * 4 == output.size
    # Every input element of the Relu is sent once
    relu_inputs = {consumer: consumer.operand_tensors[data["operand"]].size for _, consumer, data in edges}
    assert sum(data["bits"] for _, _, data in edges) == sum(relu_inputs.values())


def test_attention_add_reads_one_element_of_each_matmul_row(tmp_path):
    """The Add is tiled along d_model (its D), the MatMul along the sequence (its D): an Add tile reads one element
    of each MatMul row, not the row."""
    workload_path = export_attention_block(str(tmp_path / "attention.onnx"), seq=16, d_model=64, heads=2)
    workload = tiled_workload(workload_path, "tpu_like_quad_core_attention.yaml", str(tmp_path))
    edges = inter_layer_edges(workload, "/o/MatMul", "/Add")
    assert edges
    for producer, _, data in edges:
        assert data["bits"] == producer.operand_precision[Constants.FINAL_OUTPUT_LAYER_OP]
    # Through the heads' reshape and transpose the slice is the consumer's window
    edges = inter_layer_edges(workload, "/k/MatMul", "/MatMul")
    assert edges and all(data["slice"][0] == "window" for _, _, data in edges)


def test_edge_slice_identity():
    """A slice is a tensor of its own, by its ranges or (through a reshape) its window; none without a smaller one."""

    class _Links:
        def layer_to_mem_op(self, layer_op):
            return layer_op

    origin = type("Origin", (), {"id": 3, "sub_id": 0, "memory_operand_links": _Links()})()
    tensor = Tensor(64, origin, Constants.OUTPUT_LAYER_OP, ["K", "OX"], ((0, 4), (0, 16)))
    assert CoalaScheduler.edge_slice(tensor, {"bits": 64, "slice": ((0, 4), (0, 16))}) is None
    assert CoalaScheduler.edge_slice(tensor, {"bits": 16}) is None

    by_ranges = CoalaScheduler.edge_slice(tensor, {"bits": 16, "slice": ((0, 4), (0, 4))})
    assert by_ranges.size == 16 and by_ranges.loop_ranges == ((0, 4), (0, 4))
    window_a = CoalaScheduler.edge_slice(tensor, {"bits": 8, "slice": ("window", ((0, 1), (0, 8)))})
    window_b = CoalaScheduler.edge_slice(tensor, {"bits": 8, "slice": ("window", ((1, 2), (0, 8)))})
    hashes = {tensor.equality_hash, by_ranges.equality_hash, window_a.equality_hash, window_b.equality_hash}
    assert len(hashes) == 4
    again = CoalaScheduler.edge_slice(tensor, {"bits": 8, "slice": ("window", ((0, 1), (0, 8)))})
    assert again.equality_hash == window_a.equality_hash


def test_scheduler_sends_relu_tiles_their_slice(tmp_path):
    """Scheduled on the quad core, the Conv tiles' outputs reach the Relu cores as the Relu tiles' slices."""
    workload_path = export_resnet_bottleneck(str(tmp_path / "resnet.onnx"))
    scme = optimize_allocation_ga(
        HARDWARE,
        workload_path,
        os.path.join(MAPPING_DIR, "tpu_like_quad_core_ga.yaml"),
        "fused",
        single_stack(workload_path),
        nb_ga_generations=1,
        nb_ga_individuals=2,
        experiment_id="edge_slices",
        output_path=str(tmp_path),
    )
    sent = Counter(
        task.tensor.size
        for event in scme.accelerator.communication_manager.events
        for task in event.tasks
        if task.tensor.origin.name == "/conv1/Conv" and task.tensor.layer_operand == Constants.OUTPUT_LAYER_OP
    )
    conv_row_bits = 28672
    assert sent and conv_row_bits not in sent
    assert set(sent) == {conv_row_bits // 4}
