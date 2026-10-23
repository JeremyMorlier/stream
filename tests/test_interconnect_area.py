"""Tests for the interconnect area model (`stream.hardware.architecture.noc.interconnect_area`).

- the router, link-wire and bus-wire areas of a small example match the GF22/AXI formulas worked out by hand,
- the separable form (`InterconnectTerms`) the branch and bound relies on gives the same total,
- link widths round up to a power of two within the bounds, and follow the traffic once a window is set,
- folding drops the links that became intra-core and keeps the widest of the pairs a link carries,
- `Accelerator.interconnect_area` reads a link's `width` before its bandwidth, and counts no link to the offchip
  core,
- a derived link is timed at its width with `link_timing="width"` (`with_link_widths`), and keeps the traffic
  bandwidth otherwise; the width levels a search chooses among are the powers of two within the bounds.

Run directly with `python tests/test_interconnect_area.py`, or collect with pytest.
"""

import math
from types import SimpleNamespace

from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.noc.interconnect_area import (
    InterconnectAreaModel,
    fold_link_widths,
    interconnect_area,
    interconnect_terms,
    unordered_link_widths,
)
from stream.stages.generation.tiled_workload_accelerator_generation import with_link_widths

MODEL = InterconnectAreaModel()
PITCH = (9e-5 + 4.44e-5) / 2


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)


def test_three_core_example_by_hand() -> None:
    # Cores of 1, 4 and 9 mm^2 (sides 1, 2, 3 mm) in a chain 0 -64b- 1 -128b- 2.
    widths = {(0, 1): 64, (1, 2): 128}
    area = interconnect_area(widths, {0: 1.0, 1: 4.0, 2: 9.0}, MODEL)

    def router_ge(ports: int, width: int) -> int:
        # s * demux(m + 1, bw) + m * mux(s, bw), summed port by port.
        return 29000 * ports + 22 * ports * width + 6 * width

    # Every router also has the bus and core ports, 128 bits each.
    expected_routers = 2e-7 * (router_ge(3, 256 + 64) + router_ge(4, 256 + 64 + 128) + router_ge(3, 256 + 128))
    assert close(area.routers, expected_routers)
    # AXI: 2w + 128 wires, over the mean of the two sides.
    expected_links = PITCH * ((2 * 64 + 128) * (1 + 2) / 2 + (2 * 128 + 128) * (2 + 3) / 2)
    assert close(area.links, expected_links)
    assert close(area.bus, PITCH * (2 * 128 + 128) * (1 + 2 + 3))
    assert close(area.total, expected_routers + expected_links + area.bus)


def test_separable_terms_match() -> None:
    widths = {(0, 1): 64, (1, 2): 128, (0, 2): 32}
    core_area = {0: 0.5, 1: 2.0, 2: 7.0}
    terms = interconnect_terms(widths, core_area, MODEL)
    by_hand = terms.routers + sum(
        (terms.link_coefficient[core] + terms.bus_coefficient) * math.sqrt(area) for core, area in core_area.items()
    )
    assert close(by_hand, interconnect_area(widths, core_area, MODEL).total)


def test_disabled_model_is_free() -> None:
    assert interconnect_area({(0, 1): 64}, {0: 1.0, 1: 1.0}, InterconnectAreaModel(enabled=False)).total == 0.0


def test_widths() -> None:
    widths = [MODEL.clamp_width(w) for w in (0, 1, 32, 33, 100, 512, 513, math.inf)]
    assert widths == [32, 32, 32, 64, 128, 512, 512, 512]
    # No window yet: traffic-derived links are as wide as allowed.
    assert MODEL.width_of_traffic(10) == 512
    windowed = InterconnectAreaModel(traffic_window_cycles=100)
    assert windowed.width_of_traffic(100 * 70) == 128
    # Both directions of a pair share one link, as wide as the busier one needs.
    assert unordered_link_widths({(0, 1): 100 * 70, (1, 0): 100 * 200, (2, 2): 10**9}, windowed) == {(0, 1): 256}


def test_fold_drops_intra_core_links_and_keeps_the_widest() -> None:
    widths = {(0, 1): 64, (0, 2): 256, (1, 3): 32, (2, 3): 128}
    # 0 and 1 merge into core 0; 2 and 3 into core 1.
    folded = fold_link_widths(widths, {0: 0, 1: 0, 2: 1, 3: 1})
    assert folded == {(0, 1): 256}


def test_accelerator_reads_link_width_and_skips_offchip() -> None:
    cores = [SimpleNamespace(id=i, get_area=lambda a=a: a) for i, a in enumerate((1.0, 4.0, 100.0))]

    def link(width, bus_id=-1, bandwidth=10**6):
        return {"cl": SimpleNamespace(width=width, bandwidth=bandwidth, bus_id=bus_id)}

    bus = link(None, bus_id=0, bandwidth=128)
    edges = [
        (cores[0], cores[1], link(64)),
        (cores[1], cores[0], link(64)),
        (cores[0], cores[2], link(None, bandwidth=32)),  # to the offchip core: a bus port, not a link
        *[(a, b, bus) for a in cores for b in cores if a is not b],
    ]
    accelerator = SimpleNamespace(
        core_list=cores, offchip_core_id=2, cores=SimpleNamespace(edges=lambda data: edges)
    )
    area = Accelerator.interconnect_area(accelerator)
    assert close(area.total, interconnect_area({(0, 1): 64}, {0: 1.0, 1: 4.0}, MODEL).total)
    # Without a width, the bandwidth is taken as one (clamped): the traffic-sized bandwidth here is far over it.
    edges[0][2]["cl"].width = edges[1][2]["cl"].width = None
    assert close(
        Accelerator.interconnect_area(accelerator).total,
        interconnect_area({(0, 1): 512}, {0: 1.0, 1: 4.0}, MODEL).total,
    )


def test_links_are_timed_at_their_width() -> None:
    links = [{"type": "link", "cores": [1, 0], "bandwidth": 4_000_000}]
    timed = with_link_widths(links, {(0, 1): 64}, timed=True)
    assert timed == [{"type": "link", "cores": [1, 0], "bandwidth": 64, "link_width": 64, "traffic_bits": 4_000_000}]
    untimed = with_link_widths(links, {(0, 1): 64})
    assert untimed == [{"type": "link", "cores": [1, 0], "bandwidth": 4_000_000, "link_width": 64}]
    assert with_link_widths(links, None, timed=True) is links
    assert MODEL.timed_widths and not InterconnectAreaModel(link_timing="traffic").timed_widths
    assert MODEL.width_levels == [32, 64, 128, 256, 512]
    assert InterconnectAreaModel(link_width_min=48, link_width_max=200).width_levels == [64, 128]


if __name__ == "__main__":
    test_three_core_example_by_hand()
    test_separable_terms_match()
    test_disabled_model_is_free()
    test_widths()
    test_fold_drops_intra_core_links_and_keeps_the_widest()
    test_accelerator_reads_link_width_and_skips_offchip()
    test_links_are_timed_at_their_width()
    print("All interconnect area tests passed.")
