"""Area of the on-chip interconnect: one router per compute core, the point-to-point links between cores and the
shared off-chip bus.

The formulas are the analytical part of the NoC cost model of Iff et al., "Sparse Hamming Graph: A Customizable
Network-on-Chip Topology" (arXiv 2211.13980, section IV-B), with the constants its toolchain ships for GF22 and
the AXI transport protocol (`inputs/technologies/gf22.xml`, `inputs/protocols/axi.xml`). GF22 matches the 22 nm
node CACTI runs at for the core memories (`cacti_precompute._TECHNOLOGY`). Its floorplanning and link routing
are not reused: they assume an R x C grid of identical tiles, while the topologies here are heterogeneous cores
with one link per pair of cores that exchange data.

- **Router.** A router with `s` input and `m` output ports of width `bw` is `s * demux(m + 1, bw) + m * mux(s,
  bw)` GE, with `demux(m, bw) = 28000 + 6 m bw` and `mux(s, bw) = 1000 + 16 s bw`. Summed over `k` ports whose
  widths add up to `W` (every port both an input and an output, widths possibly different) this is
  `29000 k + 22 k W + 6 W` GE. A compute core's router has one port per link, one to the bus and one to the core
  itself, the last two `bus_width` wide.
- **Wires.** An AXI link `w` bits wide is `2 w + 128` wires. Without a floorplan the two ends of a link are taken
  as adjacent, so it runs over the mean of their side lengths, `(sqrt(A_a) + sqrt(A_b)) / 2`; the bus chains every
  compute core, `sum sqrt(A)`. A wire takes the mean of the horizontal and vertical routing pitch. As in the
  paper nothing is routed over a core, so wire area adds to the core area.

Areas are in mm^2, and so must the core areas be (see `core_generator.MAC_UNIT_AREA_MM2`).

A link's width is a physical number of bits. Where the timing `bandwidth` is one too (hand-written accelerators,
the graph search) it is used as is, clamped to `[link_width_min, link_width_max]`. The derived topologies know
their links' traffic instead, so their width is derived from it -- what moves it within `traffic_window_cycles`
(`width_of_traffic`) -- or chosen by a search among `width_levels`. With `link_timing="width"` the scheduler then
times transfers at that same width; `"traffic"` keeps the older convention of a timing bandwidth equal to the core
pair's whole traffic, which makes every transfer take about one cycle while the area is priced at the width.
"""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

# AXI (protocols/axi.xml): wires of a link, and the per-port terms of the mux/demux router.
AXI_FIXED_WIRES = 128
ROUTER_GE_PER_PORT = 28000 + 1000
ROUTER_GE_PER_PORT_BIT = 6 + 16
ROUTER_GE_PER_BIT = 6
# A compute core's router ports besides its links: the bus and the core itself.
ROUTER_EXTRA_PORTS = 2


@dataclass(frozen=True)
class InterconnectAreaModel:
    """Technology and protocol constants of the interconnect area (see the module docstring), defaults GF22/AXI."""

    enabled: bool = True
    mm2_per_ge: float = 2e-7
    mm_per_horizontal_wire: float = 9e-5
    mm_per_vertical_wire: float = 4.44e-5
    link_width_min: int = 32
    link_width_max: int = 512
    bus_width: int = 128
    # Cycles a link has to move its core pair's traffic in, for traffic-derived widths. None until known (the
    # schedule space sets it after its cost-model warm-up); traffic-derived widths are then `link_width_max`.
    traffic_window_cycles: float | None = None
    # What the scheduler times a derived link's transfers with: its physical width ("width"), or the total traffic
    # of its core pair ("traffic", about one cycle per transfer -- the convention before widths were modelled).
    link_timing: Literal["width", "traffic"] = "width"

    def __post_init__(self) -> None:
        if self.link_timing not in ("width", "traffic"):
            raise ValueError(f"InterconnectAreaModel: unknown link_timing {self.link_timing!r}.")

    @property
    def timed_widths(self) -> bool:
        """Whether derived links are timed at their physical width."""
        return self.link_timing == "width"

    @property
    def width_levels(self) -> list[int]:
        """Every physical link width: the powers of two within `[link_width_min, link_width_max]`."""
        levels, width = [], self.clamp_width(self.link_width_min)
        while width <= self.link_width_max:
            levels.append(width)
            width *= 2
        return levels

    @property
    def mm_per_wire(self) -> float:
        """Routing pitch of one wire, with no floorplan to say which direction it runs."""
        return (self.mm_per_horizontal_wire + self.mm_per_vertical_wire) / 2

    @staticmethod
    def wires(width: float) -> float:
        """Wires of an AXI link `width` bits wide."""
        return 2 * width + AXI_FIXED_WIRES

    @staticmethod
    def router_ge(ports: int, total_width: float) -> float:
        """Gate equivalents of a router with `ports` ports whose widths add up to `total_width`."""
        return (
            ROUTER_GE_PER_PORT * ports
            + ROUTER_GE_PER_PORT_BIT * ports * total_width
            + ROUTER_GE_PER_BIT * total_width
        )

    def clamp_width(self, width: float) -> int:
        """A physical link width: the next power of two, within `[link_width_min, link_width_max]`."""
        if math.isnan(width) or width <= 0:
            return self.link_width_min
        if math.isinf(width):
            return self.link_width_max
        rounded = 1 << max(0, math.ceil(math.log2(width)))
        return int(min(self.link_width_max, max(self.link_width_min, rounded)))

    @classmethod
    def from_config(cls, config: "InterconnectAreaModel | Mapping | None") -> "InterconnectAreaModel":
        """A model from a stage kwarg: an instance, a dict of fields, or None for the defaults."""
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        return cls(**config)

    def width_of_traffic(self, bits: float) -> int:
        """Width of a link that moves `bits` within `traffic_window_cycles` (`link_width_max` while unknown)."""
        if not self.traffic_window_cycles:
            return self.link_width_max
        return self.clamp_width(bits / self.traffic_window_cycles)


@dataclass(frozen=True)
class InterconnectArea:
    """Interconnect area in mm^2, by part."""

    routers: float = 0.0
    links: float = 0.0
    bus: float = 0.0

    @property
    def total(self) -> float:
        return self.routers + self.links + self.bus

    def as_dict(self) -> dict[str, float]:
        return {"routers": self.routers, "links": self.links, "bus": self.bus, "total": self.total}


LinkWidths = Mapping[tuple[int, int], int]


def unordered_link_widths(pair_bits: Mapping[tuple[int, int], float], model: InterconnectAreaModel) -> dict:
    """Traffic-derived width per unordered core pair: a link serves both directions, so it is as wide as the busier
    one needs."""
    widths: dict[tuple[int, int], int] = {}
    for (a, b), bits in pair_bits.items():
        if a == b or bits <= 0:
            continue
        pair = (min(a, b), max(a, b))
        widths[pair] = max(widths.get(pair, 0), model.width_of_traffic(bits))
    return widths


def fold_link_widths(widths: LinkWidths, rolled_of: Mapping[int, int]) -> dict[tuple[int, int], int]:
    """The links of a fold: pairs landing on one core need none, and a link carrying several pairs is as wide as
    the widest of them -- their transfers share it in time, they do not run side by side."""
    folded: dict[tuple[int, int], int] = {}
    for (a, b), width in widths.items():
        rolled_a, rolled_b = rolled_of[a], rolled_of[b]
        if rolled_a == rolled_b:
            continue
        pair = (min(rolled_a, rolled_b), max(rolled_a, rolled_b))
        folded[pair] = max(folded.get(pair, 0), width)
    return folded


@dataclass(frozen=True)
class InterconnectTerms:
    """The interconnect area split by how it depends on the core areas, for searches that need it separable:
    `routers` does not depend on them, and the wires are linear in each core's side length -- the total is
    `routers + sum_c (link_coefficient[c] + bus_coefficient) * sqrt(A_c)` over the compute cores."""

    routers: float
    link_coefficient: dict[int, float]
    bus_coefficient: float

    def area(self, core_area: Mapping[int, float]) -> InterconnectArea:
        side = {core: math.sqrt(max(0.0, area)) for core, area in core_area.items()}
        links = sum(self.link_coefficient.get(core, 0.0) * side[core] for core in side)
        return InterconnectArea(routers=self.routers, links=links, bus=self.bus_coefficient * sum(side.values()))


def interconnect_terms(
    link_widths: LinkWidths, cores: Iterable[int], model: InterconnectAreaModel
) -> InterconnectTerms:
    """`InterconnectTerms` of the compute cores `cores` (every one of them on the bus), joined by `link_widths`
    (core pair -> width in bits). See the module docstring."""
    cores = list(cores)
    if not model.enabled or not cores:
        return InterconnectTerms(routers=0.0, link_coefficient={}, bus_coefficient=0.0)
    ports = dict.fromkeys(cores, ROUTER_EXTRA_PORTS)
    port_width = dict.fromkeys(cores, float(ROUTER_EXTRA_PORTS * model.bus_width))
    # A link's wires run half over each end's side length.
    link_coefficient = dict.fromkeys(cores, 0.0)
    for (a, b), width in link_widths.items():
        for core in (a, b):
            ports[core] += 1
            port_width[core] += width
            link_coefficient[core] += model.wires(width) * model.mm_per_wire / 2
    routers = model.mm2_per_ge * sum(model.router_ge(ports[core], port_width[core]) for core in cores)
    return InterconnectTerms(
        routers=routers,
        link_coefficient=link_coefficient,
        bus_coefficient=model.wires(model.bus_width) * model.mm_per_wire,
    )


def interconnect_area(
    link_widths: LinkWidths, core_area: Mapping[int, float], model: InterconnectAreaModel
) -> InterconnectArea:
    """Interconnect area of the compute cores `core_area` (id -> core area in mm^2, every one of them on the bus),
    joined by `link_widths` (core pair -> width in bits). See the module docstring."""
    return interconnect_terms(link_widths, core_area, model).area(core_area)


def physical_link_widths(links: Iterable[tuple[int, int, float]], model: InterconnectAreaModel) -> dict:
    """Widths of links given as `(core a, core b, physical width)`, one entry per unordered pair."""
    widths: dict[tuple[int, int], int] = {}
    for a, b, width in links:
        if a == b:
            continue
        pair = (min(a, b), max(a, b))
        widths[pair] = max(widths.get(pair, 0), model.clamp_width(width))
    return widths
