"""Draw the accelerator itself: which cores exist, what each one runs, and how they are wired.

Every other visualization in this package shows what happened *on* a piece of hardware -- a schedule, a memory
trace, a cost look-up. None of them show the hardware. That gap matters once the topology stops being a fixed
input: `derive_group_dedicated_topology` builds a core per `(layer, inter-core group)` and links it to whoever
it exchanges tensors with, and `derive_rolled_topology` then folds those cores together, dropping the links
that became intra-core traffic and creating new ones between the merged groups. The end result is a different
graph per design, and the point of this module is to make that graph legible.

Two things shape the drawing:

- **A bus is one shared wire, not every pair.** `AcceleratorFactory` implements a bus by creating a single
  `CommunicationLink` and wiring it into all `N*(N-1)` ordered core pairs, so drawing the raw graph gives a
  hairball that also misrepresents the hardware -- those edges are one contended resource. A bus is collapsed
  to a hub node with one spoke per member.
- **Extraction is separate from rendering.** `extract_topology` produces plain data, which the PNG renderer,
  the interactive report and the tests all consume. It also means a topology can be built straight from a
  `GroupDedicatedTopology`/`RolledTopology` without assembling a real `Accelerator` -- which would construct
  every `Core` and can shell out to CACTI.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- these run headless, inside the pipeline
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

logger = logging.getLogger(__name__)

NODE_KIND_T = Literal["compute", "offchip", "bus"]

COMPUTE_CMAP = "tab20"
OFFCHIP_COLOR = "0.55"
BUS_COLOR = "0.7"
MERGED_COLOR = "0.85"  # a folded core belongs to no single layer, so it gets no layer colour


@dataclass
class TopologyNode:
    id: int | str
    kind: NODE_KIND_T
    label: str
    layers: list[dict[str, Any]] = field(default_factory=list)
    design_key: str | None = None
    area: float | None = None
    memory_bits: int | None = None
    bandwidth: float | None = None
    column: int | None = None
    busy_cycles: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "layers": self.layers,
            "design_key": self.design_key,
            "area": self.area,
            "memory_bits": self.memory_bits,
            "bandwidth": self.bandwidth,
            "column": self.column,
            "busy_cycles": self.busy_cycles,
        }


@dataclass
class TopologyEdge:
    source: int | str
    target: int | str
    bandwidth: float
    directed: bool
    kind: Literal["link", "bus"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "bandwidth": self.bandwidth,
            "directed": self.directed,
            "kind": self.kind,
        }


@dataclass
class HardwareTopology:
    name: str
    offchip_core_id: int | None
    nodes: list[TopologyNode]
    edges: list[TopologyEdge]

    @property
    def compute_nodes(self) -> list[TopologyNode]:
        return [node for node in self.nodes if node.kind == "compute"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "offchip_core_id": self.offchip_core_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


# --------------------------------------------------------------------------------- label helpers -------------


def format_bandwidth(bandwidth: float) -> str:
    for limit, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(bandwidth) >= limit:
            return f"{bandwidth / limit:.3g}{suffix}"
    return f"{bandwidth:.3g}"


def _layer_label(layers: list[dict[str, Any]], max_layers: int) -> str:
    names = [layer["name"] for layer in layers]
    if not names:
        return ""
    if len(names) <= max_layers:
        return ", ".join(names)
    return ", ".join(names[:max_layers]) + f" (+{len(names) - max_layers} more)"


def _core_label(node: TopologyNode, max_layers: int) -> str:
    parts = [f"C{node.id}"]
    layer_text = _layer_label(node.layers, max_layers)
    if layer_text:
        parts.append(layer_text)
    if node.design_key:
        parts.append(node.design_key[:6])
    parts.append(f"A={node.area:.3g}" if node.area is not None else "A=?")
    return "\n".join(parts)


# ------------------------------------------------------------------------------------ extraction -------------


def core_layers_from_node_core_ids(node_core_ids: dict[int, list[int]]) -> dict[int, list[int]]:
    """Invert a node -> cores map into the core -> layers map the drawing needs. Unrolled topologies give each
    core one layer; rolled ones give some cores several."""
    core_layers: dict[int, list[int]] = {}
    for node_id, core_ids in node_core_ids.items():
        for core_id in core_ids:
            core_layers.setdefault(core_id, []).append(node_id)
    return core_layers


def core_layers_from_workload(workload: Any) -> dict[int, list[int]]:
    """Same map read off a pinned workload, for callers that have tiles but no topology object."""
    core_layers: dict[int, list[int]] = {}
    for tile in workload.node_list:
        core_id = getattr(tile, "chosen_core_allocation", None)
        if core_id is None:
            continue
        layers = core_layers.setdefault(core_id, [])
        if tile.id not in layers:
            layers.append(tile.id)
    return core_layers


def _layer_entries(
    core_id: int, core_layers: dict[int, list[int]] | None, layer_names: dict[int, str] | None
) -> list[dict[str, Any]]:
    layer_ids = (core_layers or {}).get(core_id, [])
    return [{"id": layer_id, "name": (layer_names or {}).get(layer_id, f"layer {layer_id}")} for layer_id in layer_ids]


def _assign_columns(nodes: list[TopologyNode], layer_order: list[int] | None) -> None:
    """Left-to-right position for each compute core: where its first layer sits in the workload's topological
    order, so the picture reads like the dataflow."""
    if layer_order is None:
        return
    rank_of_layer = {layer_id: index for index, layer_id in enumerate(layer_order)}
    for node in nodes:
        if node.kind != "compute":
            continue
        ranks = [rank_of_layer[layer["id"]] for layer in node.layers if layer["id"] in rank_of_layer]
        node.column = min(ranks) if ranks else None


def _collapse_edges(
    raw_edges: list[tuple[int, int, Any]], offchip_core_id: int | None
) -> tuple[list[TopologyEdge], dict[int, tuple[set[int], float]]]:
    """Split the core graph's edges into point-to-point edges and bus memberships.

    Point-to-point pairs arrive as two directed `CommunicationLink` objects; they collapse into one undirected
    edge when both directions are present *and* carry the same bandwidth. Anything asymmetric stays as two
    directed edges rather than being silently averaged."""
    buses: dict[int, tuple[set[int], float]] = {}
    directed: dict[tuple[int, int], float] = {}
    for source, target, link in raw_edges:
        bus_id = getattr(link, "bus_id", -1)
        if bus_id is not None and bus_id >= 0:
            # A bus link's own sender/receiver are the placeholder string "Any", so membership has to come from
            # the graph's endpoints.
            members, _ = buses.setdefault(bus_id, (set(), float(link.bandwidth)))
            members.update((source, target))
            continue
        directed[(source, target)] = float(link.bandwidth)

    edges: list[TopologyEdge] = []
    consumed: set[tuple[int, int]] = set()
    for (source, target), bandwidth in directed.items():
        if (source, target) in consumed:
            continue
        reverse = directed.get((target, source))
        if reverse is not None and math.isclose(reverse, bandwidth):
            consumed.add((target, source))
            edges.append(TopologyEdge(source, target, bandwidth, directed=False, kind="link"))
        else:
            edges.append(TopologyEdge(source, target, bandwidth, directed=True, kind="link"))
    return edges, buses


def _bus_nodes_and_spokes(
    buses: dict[int, tuple[set[int], float]],
) -> tuple[list[TopologyNode], list[TopologyEdge]]:
    nodes: list[TopologyNode] = []
    spokes: list[TopologyEdge] = []
    for bus_id, (members, bandwidth) in sorted(buses.items()):
        hub_id = f"bus{bus_id}"
        nodes.append(
            TopologyNode(
                id=hub_id,
                kind="bus",
                label=f"bus {bus_id}\n{format_bandwidth(bandwidth)}",
                bandwidth=bandwidth,
            )
        )
        for member in sorted(members):
            spokes.append(TopologyEdge(hub_id, member, bandwidth, directed=False, kind="bus"))
    return nodes, spokes


def extract_topology(  # noqa: PLR0913
    accelerator: Any,
    *,
    core_layers: dict[int, list[int]] | None = None,
    layer_names: dict[int, str] | None = None,
    design_keys: dict[int, str] | None = None,
    layer_order: list[int] | None = None,
    core_busy_cycles: dict[int, float] | None = None,
    max_layers_per_label: int = 3,
) -> HardwareTopology:
    """Plain-data topology of a built `Accelerator`.

    Structure, bandwidths, areas and the offchip core come from the accelerator itself. Which *layers* a core
    hosts and which design it carries cannot -- nothing on a `Core` records either -- so `core_layers` and
    `design_keys` are supplied by the caller, which already has them from the topology it assembled."""
    offchip_core_id = getattr(accelerator, "offchip_core_id", None)
    nodes: list[TopologyNode] = []
    for core in sorted(accelerator.core_list, key=lambda c: c.id):
        is_offchip = core.id == offchip_core_id or getattr(core, "type", "compute") == "memory"
        node = TopologyNode(
            id=core.id,
            kind="offchip" if is_offchip else "compute",
            label="",
            layers=[] if is_offchip else _layer_entries(core.id, core_layers, layer_names),
            design_key=None if is_offchip else (design_keys or {}).get(core.id),
            area=_safe(core.get_area),
            memory_bits=_safe(core.get_memory_capacity),
            busy_cycles=None if is_offchip else (core_busy_cycles or {}).get(core.id),
        )
        node.label = f"offchip C{core.id}" if is_offchip else _core_label(node, max_layers_per_label)
        nodes.append(node)

    raw_edges = [(source.id, target.id, data["cl"]) for source, target, data in accelerator.cores.edges(data=True)]
    edges, buses = _collapse_edges(raw_edges, offchip_core_id)
    bus_nodes, spokes = _bus_nodes_and_spokes(buses)
    nodes.extend(bus_nodes)
    edges.extend(spokes)

    _assign_columns(nodes, layer_order)
    return HardwareTopology(
        name=getattr(accelerator, "name", ""),
        offchip_core_id=offchip_core_id,
        nodes=nodes,
        edges=edges,
    )


def _safe(getter: Any) -> Any:
    """Core metrics are derived (CACTI areas, a single shared top memory) and can raise on unusual cores --
    `Accelerator.area` guards `get_area` the same way. A missing number must not cost us the whole figure."""
    try:
        return getter()
    except Exception:
        return None


def topology_from_connectivity(  # noqa: PLR0913
    *,
    name: str,
    node_core_ids: dict[int, list[int]],
    offchip_core_id: int,
    bus_connection: dict[str, Any],
    link_connections: list[dict[str, Any]],
    design_keys: dict[int, str] | None = None,
    areas: dict[int, float] | None = None,
    layer_names: dict[int, str] | None = None,
    layer_order: list[int] | None = None,
    core_busy_cycles: dict[int, float] | None = None,
    max_layers_per_label: int = 3,
) -> HardwareTopology:
    """Build the same topology straight from an `AcceleratorFactory` connectivity description.

    This is the path the exploration stages use: they have the topology dataclass and a design-key-per-core map
    already, and assembling a real `Accelerator` just to draw it would construct every `Core` (and can invoke
    CACTI) for nothing."""
    core_layers = core_layers_from_node_core_ids(node_core_ids)
    nodes: list[TopologyNode] = []
    for core_id in range(offchip_core_id):
        design_key = (design_keys or {}).get(core_id)
        node = TopologyNode(
            id=core_id,
            kind="compute",
            label="",
            layers=_layer_entries(core_id, core_layers, layer_names),
            design_key=design_key,
            area=(areas or {}).get(core_id),
            busy_cycles=(core_busy_cycles or {}).get(core_id),
        )
        node.label = _core_label(node, max_layers_per_label)
        nodes.append(node)
    nodes.append(TopologyNode(id=offchip_core_id, kind="offchip", label=f"offchip C{offchip_core_id}"))

    edges = [
        TopologyEdge(connection["cores"][0], connection["cores"][1], float(connection["bandwidth"]), False, "link")
        for connection in link_connections
    ]
    if bus_connection:
        hub_id = "bus0"
        bandwidth = float(bus_connection["bandwidth"])
        nodes.append(
            TopologyNode(id=hub_id, kind="bus", label=f"bus 0\n{format_bandwidth(bandwidth)}", bandwidth=bandwidth)
        )
        edges.extend(TopologyEdge(hub_id, core_id, bandwidth, False, "bus") for core_id in bus_connection["cores"])

    _assign_columns(nodes, layer_order)
    return HardwareTopology(name=name, offchip_core_id=offchip_core_id, nodes=nodes, edges=edges)


# ------------------------------------------------------------------------------------- rendering -------------


def layout_positions(
    topology: HardwareTopology, layout: Literal["layered", "spring"] = "layered"
) -> dict[Any, tuple[float, float]]:
    """Where each node goes. Shared with the interactive report so the two views agree.

    `layered` puts compute cores in columns by the layer they run, which reads like the dataflow; the offchip
    core and any bus hubs go in a band underneath rather than in a column of their own, so they don't stretch
    the layer spacing. `spring` is the fallback when no layer information was supplied -- with a fixed seed, so
    two designs of the same accelerator stay visually comparable."""
    graph = nx.Graph()
    graph.add_nodes_from(node.id for node in topology.nodes)
    graph.add_edges_from((edge.source, edge.target) for edge in topology.edges)

    compute = [node for node in topology.nodes if node.kind == "compute"]
    infra = [node for node in topology.nodes if node.kind != "compute"]
    has_columns = layout == "layered" and any(node.column is not None for node in compute)
    if not has_columns:
        positions = nx.spring_layout(graph, seed=0)
        return {key: (float(x), float(y)) for key, (x, y) in positions.items()}

    # Column index, then a stable vertical order within the column.
    by_column: dict[int, list[TopologyNode]] = {}
    for node in compute:
        by_column.setdefault(node.column if node.column is not None else 0, []).append(node)

    positions: dict[Any, tuple[float, float]] = {}
    columns = sorted(by_column)
    span = max(1, len(columns) - 1)
    for column_index, column in enumerate(columns):
        members = sorted(by_column[column], key=lambda n: n.id)
        height = max(1, len(members) - 1)
        for row, node in enumerate(members):
            x = column_index / span
            y = 0.5 if len(members) == 1 else 1.0 - row / height
            positions[node.id] = (x, y)

    for index, node in enumerate(sorted(infra, key=lambda n: str(n.id))):
        x = 0.5 if len(infra) == 1 else index / max(1, len(infra) - 1)
        positions[node.id] = (x, -0.6)
    return positions


def _edge_width(bandwidth: float, max_bandwidth: float) -> float:
    """Log-scaled, because an offchip bus (128) and a point-to-point link carrying a whole tensor (1e5+) differ
    by orders of magnitude -- linear widths would render the bus invisible."""
    if max_bandwidth <= 0:
        return 1.0
    return 1.0 + 4.0 * math.log1p(max(bandwidth, 0.0)) / math.log1p(max_bandwidth)


def _marker_size(label: str, fontsize: float, marker: str) -> float:
    """Marker area (points^2) big enough for the label to sit inside it.

    A core's label carries its id, the layers it runs, its design and its area, so a fixed marker size either
    wastes space or -- worse -- lets the text spill over the outline. Sized from the text instead: a diamond
    needs more area than a circle to contain the same box, hence the per-marker slack."""
    lines = label.split("\n") or [""]
    width = max((len(line) for line in lines), default=1) * fontsize * 0.62
    height = len(lines) * fontsize * 1.3
    radius = 0.5 * math.hypot(width, height) * (1.35 if marker == "D" else 1.12)
    return math.pi * radius**2


def _node_style(node: TopologyNode, layer_color: dict[int, Any], fontsize: float) -> tuple[Any, str, float]:
    """(face colour, marker, size). Offchip and bus differ from compute cores in *shape* as well as colour, so
    the figure survives greyscale printing. A folded core hosting several layers gets no layer colour -- it
    belongs to none of them."""
    if node.kind == "offchip":
        color, marker = OFFCHIP_COLOR, "s"
    elif node.kind == "bus":
        color, marker = BUS_COLOR, "D"
    elif len(node.layers) != 1:
        color, marker = MERGED_COLOR, "o"
    else:
        color, marker = layer_color.get(node.layers[0]["id"], MERGED_COLOR), "o"
    return color, marker, _marker_size(node.label, fontsize, marker)


def plot_hardware_graph(
    topology: HardwareTopology,
    fig_path: str,
    *,
    title: str | None = None,
    layout: Literal["layered", "spring"] = "layered",
    annotate_bandwidth: bool = True,
    max_edge_labels: int = 40,
) -> None:
    """Save the topology figure.

    Bandwidths are drawn as edge width and, while there are few enough edges to stay readable, as labels. Note
    a link's bandwidth here is derived from the *volume* the workload sends over it
    (`derive_group_dedicated_topology` sums the tile-edge bits), not from a physical wire width -- the same
    caveat as the "a.u." area axis in `schedule_tradeoff`."""
    if not topology.nodes:
        logger.warning("plot_hardware_graph: topology has no cores, skipping the plot.")
        return

    graph = nx.Graph()
    for node in topology.nodes:
        graph.add_node(node.id)
    for edge in topology.edges:
        graph.add_edge(edge.source, edge.target)
    positions = layout_positions(topology, layout)

    layer_ids = sorted({layer["id"] for node in topology.nodes for layer in node.layers})
    cmap = plt.get_cmap(COMPUTE_CMAP)
    layer_color = {layer_id: cmap(index % cmap.N) for index, layer_id in enumerate(layer_ids)}

    nb_nodes = len(topology.nodes)
    fontsize = 8.0
    figure, axis = plt.subplots(figsize=(max(10.0, 1.6 * nb_nodes), max(7.0, 1.0 * nb_nodes)))

    max_bandwidth = max((edge.bandwidth for edge in topology.edges), default=0.0)
    node_by_id = {node.id: node for node in topology.nodes}
    # Stop each edge at the node outline. The radius varies with the label, so it has to be per-endpoint.
    shrink = {
        node.id: math.sqrt(_node_style(node, layer_color, fontsize)[2] / math.pi) + 2.0 for node in topology.nodes
    }
    for edge in topology.edges:
        if edge.source not in node_by_id or edge.target not in node_by_id:
            continue
        axis.annotate(
            "",
            xy=positions[edge.target],
            xytext=positions[edge.source],
            arrowprops={
                "arrowstyle": "-|>" if edge.directed else "-",
                "color": BUS_COLOR if edge.kind == "bus" else "0.35",
                "linewidth": _edge_width(edge.bandwidth, max_bandwidth),
                "linestyle": (0, (4, 3)) if edge.kind == "bus" else "solid",
                "shrinkA": shrink[edge.source],
                "shrinkB": shrink[edge.target],
            },
            zorder=1,
        )

    for node in topology.nodes:
        color, marker, size = _node_style(node, layer_color, fontsize)
        axis.scatter(
            *positions[node.id],
            s=size,
            c=[color],
            marker=marker,
            edgecolors="black",
            linewidths=1.0,
            zorder=2,
        )
        axis.annotate(
            node.label,
            positions[node.id],
            ha="center",
            va="center",
            fontsize=fontsize,
            zorder=3,
        )

    if annotate_bandwidth and len(topology.edges) <= max_edge_labels:
        for edge in topology.edges:
            (x0, y0), (x1, y1) = positions[edge.source], positions[edge.target]
            axis.annotate(
                format_bandwidth(edge.bandwidth),
                ((x0 + x1) / 2, (y0 + y1) / 2),
                ha="center",
                va="center",
                fontsize=6,
                color="0.25",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
                zorder=4,
            )

    nb_compute = len(topology.compute_nodes)
    nb_links = sum(1 for edge in topology.edges if edge.kind == "link")
    axis.set_title(
        title or f"{topology.name}: {nb_compute} compute core(s), {nb_links} core-to-core link(s)",
        fontsize=12,
    )
    axis.set_axis_off()
    axis.margins(0.18)
    figure.tight_layout()
    figure.savefig(fig_path, dpi=150)
    plt.close(figure)
    logger.info(f"Saved hardware topology figure to {fig_path}.")
