"""Tests for `stream.stages.generation.template_exploration`'s genome and `grid_generator.template_grid_dict`.

The genome is where the template search can go wrong silently: an inter-core factor larger than the grid, a
memory size CACTI was never precomputed for, or a decoded core the validator rejects would each only show up as
infinite fitness deep inside a run. These check the decoding against the real edge_tpu template, without running
ZigZag or the scheduler.

Run directly with `python tests/test_template_exploration.py`, or collect with `pytest`.
"""

import os

from zigzag.datatypes import LayerDim

from stream.hardware.architecture.grid_generator import template_grid_dict, tpu_grid_dict
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.stages.generation.tiled_workload_accelerator_generation import load_offchip_core_data, validate_core_yaml
from stream.stages.generation.template_exploration import (
    CORES_DIR,
    TemplateGeneLayout,
    TemplateParamRanges,
    _largest_divisor_at_most,
    load_template_core,
)
from stream.stages.generation.tiling_exploration import TilingGene

TEMPLATE = load_template_core(os.path.join(CORES_DIR, "edge_tpu_like.yaml"))
K, OY = LayerDim("K"), LayerDim("OY")
INTER_GENE = TilingGene(node_id=0, kind="inter", dim=None, candidates=[None, (K, 2), (K, 4), (K, 8), (OY, 2)])


def _layout() -> TemplateGeneLayout:
    return TemplateGeneLayout(TEMPLATE, [INTER_GENE], TemplateParamRanges(max_rows=2, max_cols=4))


def _values(layout: TemplateGeneLayout, rows: int, cols: int, inter_index: int) -> list[float]:
    choice_values = layout.template_values(rows, cols)
    choice_values[-1] = inter_index
    return layout.encode(choice_values)


def test_template_round_trip():
    layout = _layout()
    design = layout.decode(_values(layout, 2, 3, 0))
    assert (design.rows, design.cols) == (2, 3)
    assert design.oa_sizes == (8, 8, 4, 4)
    template_sizes = [TEMPLATE["memories"][name]["size"] for name in layout.memory_names]
    for size, template_size in zip(design.memory_sizes, template_sizes, strict=True):
        assert abs(size - template_size) / template_size < 0.25


def test_inter_factor_clamped_to_grid():
    layout = _layout()
    # (K, 8) on 2x2 = 4 cores becomes (K, 4); on 1x1 nothing of K fits, so no split.
    assert INTER_GENE.candidates[layout.decode(_values(layout, 2, 2, 3)).inter[0]] == (K, 4)
    assert layout.decode(_values(layout, 1, 1, 3)).inter[0] == 0
    # A factor that fits is kept.
    assert INTER_GENE.candidates[layout.decode(_values(layout, 2, 4, 3)).inter[0]] == (K, 8)


def test_every_gene_value_decodes_to_a_valid_core():
    layout = _layout()
    configs = layout.cacti_configs()
    for value in (0.0, 0.37, 0.999999):
        design = layout.decode([value] * len(layout.choices))
        core = layout.core_dict(design)
        for name, memory in core["memories"].items():
            assert memory["size"] % 8 == 0
            # Every memory size a design can pick has its CACTI cost precomputed.
            read_bandwidth = TemplateGeneLayout._read_bandwidth(TEMPLATE["memories"][name])
            assert (memory["mem_type"], memory["size"], read_bandwidth) in configs


FUSEMAX = load_template_core(os.path.join(CORES_DIR, "fusemax_like.yaml"))
FUSEMAX_VECTOR = load_template_core(os.path.join(CORES_DIR, "fusemax_vec_like.yaml"))


def _fusemax_layout() -> TemplateGeneLayout:
    ranges = TemplateParamRanges(max_rows=2, max_cols=2, oa_growth=1, intra_tiles=(2, 8, 32, 128))
    return TemplateGeneLayout(FUSEMAX, [INTER_GENE], ranges, vector_template=FUSEMAX_VECTOR)


def test_vector_template_and_intra_tiles_round_trip():
    """With a vector template and intra tile counts, the template itself decodes back, the most tiles included."""
    layout = _fusemax_layout()
    design = layout.decode(_values(layout, 2, 2, 0))
    assert design.oa_sizes == (256, 256) and design.vector_oa_sizes == (256,)
    assert design.intra_tiles == 128 and design.inter == (0,)
    for name, size in zip(layout.vector_memory_names, design.vector_memory_sizes, strict=True):
        assert abs(size - FUSEMAX_VECTOR["memories"][name]["size"]) / FUSEMAX_VECTOR["memories"][name]["size"] < 0.25
    assert layout.describe(design)["intra_tiles"] == 128
    # Designs differing only in the vector core or the tile count are told apart.
    smaller = layout.decode([0.0] * len(layout.choices))
    assert smaller.intra_tiles == 2 and smaller.key != design.key


def test_every_vector_gene_value_decodes_to_a_valid_core():
    layout = _fusemax_layout()
    configs = layout.cacti_configs()
    for value in (0.0, 0.37, 0.999999):
        design = layout.decode([value] * len(layout.choices))
        for core, template in ((layout.core_dict(design), FUSEMAX), (layout.vector_core_dict(design), FUSEMAX_VECTOR)):
            for name, memory in core["memories"].items():
                read_bandwidth = TemplateGeneLayout._read_bandwidth(template["memories"][name])
                assert (memory["mem_type"], memory["size"], read_bandwidth) in configs


def test_largest_divisor_at_most():
    assert _largest_divisor_at_most(128, 32) == 32
    assert _largest_divisor_at_most(56, 32) == 28
    assert _largest_divisor_at_most(7, 4) == 1
    assert _largest_divisor_at_most(16, 128) == 16


def test_template_grid_layout():
    grid = template_grid_dict({"name": "core"}, 2, 3, name="grid", simd_core={"name": "simd"}, offchip_core="off")
    assert list(grid["cores"]) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert grid["cores"][6] == {"name": "simd"} and grid["offchip_core_id"] == 7
    links = [tuple(c["cores"]) for c in grid["core_connectivity"] if c["type"] == "link"]
    # 2x3 mesh: 2 rows x 2 horizontal + 3 vertical links, plus one link from each compute core to the SIMD core.
    assert len(links) == 2 * 2 + 3 + 6
    assert grid["core_connectivity"][0] == {"type": "bus", "cores": list(range(8)), "bandwidth": 128}
    # The tpu_like grid is the same layout with core files.
    assert tpu_grid_dict(2, 3)["core_connectivity"] == grid["core_connectivity"]


def test_template_grid_keeps_its_links():
    """Built, a 2x2 edge_tpu grid joins its mesh neighbours and its SIMD core by links, the rest by the bus: the
    bus, which joins every pair, must not replace the links."""
    layout = TemplateGeneLayout(TEMPLATE, [], TemplateParamRanges())
    design = layout.decode(layout.encode(layout.template_values(2, 2)))
    data = template_grid_dict(
        layout.core_dict(design),
        2,
        2,
        name="grid",
        simd_core=validate_core_yaml(os.path.join(CORES_DIR, "simd.yaml")),
        offchip_core=load_offchip_core_data(),
    )
    data["core_memory_sharing"] = []
    accelerator = AcceleratorFactory(data).create()
    is_bus = {
        (sender.id, receiver.id): link.bus_id is not None and link.bus_id >= 0
        for sender, receiver, link in accelerator.cores.edges(data="cl")
    }
    mesh_and_simd = [(0, 1), (0, 2), (1, 3), (2, 3), *((core, 4) for core in range(4))]
    assert not any(is_bus[pair] or is_bus[pair[::-1]] for pair in mesh_and_simd)
    assert is_bus[(0, 3)] and is_bus[(0, 5)] and is_bus[(4, 5)]


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("All template exploration tests passed.")
