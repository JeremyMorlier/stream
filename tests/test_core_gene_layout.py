"""Tests for `CoreGeneLayout`, the core NSGA2's gene encoding (`stream.stages.generation.core_architecture_exploration`).

- size genes hold log2 of the size, so the low end of a range is reachable,
- an optional level's omit band leaves the level out, and a non-optional level is never left out,
- every core the layout decodes is in the CACTI precompute of its ranges.

Run with `python tests/test_core_gene_layout.py`, or collect with pytest.
"""

import math
import random

from stream.hardware.architecture.cacti_precompute import enumerate_cacti_configs
from stream.stages.generation.core_architecture_exploration import CoreGeneLayout, CoreParamRanges


def genes_with(layout: CoreGeneLayout, **sizes: float) -> list[float]:
    """The middle of every gene's bounds, with the given size genes (`upper=...` sets `upper_size`) replaced."""
    values = [(low + high) / 2 for low, high in zip(layout.low, layout.high, strict=True)]
    for level, value in sizes.items():
        values[layout.names.index(f"{level}_size")] = value
    return values


def test_size_genes_are_log2() -> None:
    ranges = CoreParamRanges()
    layout = CoreGeneLayout(ranges)
    upper = layout.names.index("upper_size")
    # Optional: one unit below the floor for the omit band.
    assert (layout.low[upper], layout.high[upper]) == (math.log2(ranges.upper_size_range[0]) - 1, 27)
    sram = layout.names.index("sram_size")
    assert (layout.low[sram], layout.high[sram]) == (16, 24)
    core = layout.decode_node(genes_with(layout, upper=20.0, sram=16.0), "c")
    assert core["memories"]["upper_shared"]["size"] == 2**20
    assert core["memories"]["sram_shared"]["size"] == 2**16


def test_optional_level_omit_band() -> None:
    layout = CoreGeneLayout(CoreParamRanges())
    assert "upper_shared" not in layout.decode_node(genes_with(layout, upper=19.0), "c")["memories"]
    assert "upper_shared" in layout.decode_node(genes_with(layout, upper=19.6), "c")["memories"]
    # The SRAM is not optional: its lowest gene is its smallest size.
    sram = layout.names.index("sram_size")
    assert layout.decode_node(genes_with(layout, sram=layout.low[sram]), "c")["memories"]["sram_shared"]["size"] > 0
    strict = CoreGeneLayout(CoreParamRanges(optional_levels=()))
    upper = strict.names.index("upper_size")
    assert "upper_shared" in strict.decode_node(genes_with(strict, upper=strict.low[upper]), "c")["memories"]


def test_decoded_cores_are_precomputed() -> None:
    ranges = CoreParamRanges()
    layout = CoreGeneLayout(ranges)
    configs = enumerate_cacti_configs(ranges)
    rng = random.Random(0)
    for _ in range(500):
        values = [rng.uniform(low, high) for low, high in zip(layout.low, layout.high, strict=True)]
        for memory in layout.decode_node(values, "c")["memories"].values():
            read_bw = next(port["bandwidth_max"] for port in memory["ports"] if port["type"] == "read")
            assert (memory["mem_type"], memory["size"], read_bw) in configs, memory


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"{name}: ok")
