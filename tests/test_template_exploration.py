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
from stream.stages.generation.template_exploration import (
    CORES_DIR,
    TemplateGeneLayout,
    TemplateParamRanges,
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


def test_template_grid_layout():
    grid = template_grid_dict({"name": "core"}, 2, 3, name="grid", simd_core={"name": "simd"}, offchip_core="off")
    assert list(grid["cores"]) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert grid["cores"][6] == {"name": "simd"} and grid["offchip_core_id"] == 7
    links = [tuple(c["cores"]) for c in grid["core_connectivity"] if c["type"] == "link"]
    # 2x3 mesh: 2 rows x 2 horizontal + 3 vertical links, plus one link from each compute core to the SIMD core.
    assert len(links) == 2 * 2 + 3 + 6
    assert grid["core_connectivity"][-1] == {"type": "bus", "cores": list(range(8)), "bandwidth": 128}
    # The tpu_like grid is the same layout with core files.
    assert tpu_grid_dict(2, 3)["core_connectivity"] == grid["core_connectivity"]


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("All template exploration tests passed.")
