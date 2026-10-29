"""Tests for `memory_hierarchy_violations`, the check that rules a (tile, core) pair out before ZigZag runs.

A core passes only if each memory level can hold the operands' smallest slices and its top memories the whole
tile; an operand Stream spills off-chip (`layer_stacks`) no longer needs its whole tile on the core. A core that
passes must still be mappable by ZigZag.

Run directly with `python tests/test_memory_hierarchy_violations.py`, or collect with pytest.
"""

from zigzag.mapping.spatial_mapping import SpatialMapping
from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.parser.workload_factory import LayerNodeFactory

from stream.hardware.architecture.core_generator import (
    build_core_from_dict,
    build_rf_memory_dict,
    build_shared_memory_dict,
    core_dict_from_params,
)
from stream.stages.estimation.zigzag_core_mapping_estimation import (
    evaluate_node_on_core,
    memory_hierarchy_violations,
)
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.mapping import InterCoreMappingAttributes

OPERANDS = ("I1", "I2", "O")
LOOP_DIMS = ["B", "K", "C", "OY", "OX"]


def conv_tile(k: int, c: int, oy: int, ox: int) -> ComputationNode:
    """A 1x1 conv tile; with 8-bit activations its input holds `c * oy * ox * 8` bits."""
    node_data = {
        "id": 0,
        "name": "conv",
        "operator_type": "Conv",
        "equation": "O[b][k][oy][ox]+=W[k][c]*I[b][c][oy][ox]",
        "loop_dims": LOOP_DIMS,
        "loop_sizes": [1, k, c, oy, ox],
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"I": 0},
        "dimension_relations": [],
        "pr_loop_dims": [],
        "pr_loop_sizes": [],
        "padding": [],
    }
    mapping = InterCoreMappingAttributes(
        op_type="Conv",
        spatial_mapping=SpatialMapping({}),
        core_allocation=[0],
        intra_core_tiling=[],
        inter_core_tiling=[],
        layer_dimension_names=LOOP_DIMS,
    )
    node_attr = LayerNodeFactory(node_data, mapping_data=None).create_node_attr()
    return ComputationNode(node_id=0, node_name="conv", node_attr=node_attr, mapping_attr=mapping, op_type="Conv")


def core(sram_size: int, sram_bandwidth: int, upper_size: int | None):
    """An 8x8x2 core with a 256-bit RF per operand, a shared SRAM and, unless `upper_size` is None, an upper
    memory."""
    return build_core_from_dict(
        core_dict_from_params(
            "test_core",
            operands=OPERANDS,
            oa_dims=("D1", "D2", "D3"),
            oa_sizes=[8, 8, 2],
            unit_energy=0.05,
            rf_memories={operand: build_rf_memory_dict(operand, 256, 32, []) for operand in OPERANDS},
            sram_memory=build_shared_memory_dict(
                list(OPERANDS), sram_size, sram_bandwidth, sram_bandwidth, ["D1", "D2"]
            ),
            upper_memory=None
            if upper_size is None
            else build_shared_memory_dict(list(OPERANDS), upper_size, 512, 512, ["D1", "D2", "D3"]),
        )
    )


def test_roomy_core_passes_and_maps():
    tile = conv_tile(32, 32, 8, 8)
    roomy = core(2**18, 256, 2**21)
    assert memory_hierarchy_violations(tile, roomy) == []
    evaluate_node_on_core(tile, roomy, loma_lpf_limit=6, temporal_mapping_type=TemporalMappingType.UNEVEN)


def test_top_memory_must_hold_the_whole_tile():
    # No upper level: the 106144-bit SRAM is the top memory, and the input alone is 64 * 16 * 16 * 8 = 131072 bits.
    violations = memory_hierarchy_violations(conv_tile(64, 64, 16, 16), core(106144, 256, None))
    assert violations
    assert all("sram_shared" in violation for violation in violations)


def test_spilled_operands_need_only_their_slices():
    # Alone in its layer stack, the tile has every operand spilled to DRAM, so the SRAM only holds slices.
    assert memory_hierarchy_violations(conv_tile(64, 64, 16, 16), core(106144, 256, None), layer_stacks=[(0,)]) == []


if __name__ == "__main__":
    test_roomy_core_passes_and_maps()
    test_top_memory_must_hold_the_whole_tile()
    test_spilled_operands_need_only_their_slices()
    print("all passed")
