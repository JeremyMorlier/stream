"""Tests for `stream.visualization.hardware_graph`'s topology extraction.

The part worth testing is the translation from Stream's core graph to something drawable, because that graph
does not mean what it looks like: `AcceleratorFactory` implements a bus by wiring *one* `CommunicationLink`
into every ordered pair of its cores, and a point-to-point link as *two* distinct directed objects. Get either
wrong and the picture is a hairball, or silently halves a bandwidth.

So these build the graph through the real `get_bidirectional_edges` + `CoreGraph` -- the same code the factory
calls -- with a stub standing in only for `Core`, which would otherwise drag in ZigZag and CACTI for nothing.

Run directly with `python test_hardware_graph.py`, or collect with `pytest test_hardware_graph.py`.
"""

import json
import os
import tempfile

from zigzag.utils import DiGraphWrapper

from stream.hardware.architecture.noc.communication_link import CommunicationLink, get_bidirectional_edges
from stream.visualization.hardware_graph import (
    core_layers_from_node_core_ids,
    extract_topology,
    plot_hardware_graph,
    topology_from_connectivity,
)


class _FakeCore:
    """Enough of a `Core` for the extraction: an id, a type, and the two derived metrics it asks for."""

    def __init__(self, core_id: int, core_type: str = "compute", area: float = 100.0):
        self.id = core_id
        self.type = core_type
        self._area = area

    def get_area(self) -> float:
        return self._area

    def get_memory_capacity(self) -> int:
        return 1024

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other) -> bool:
        return isinstance(other, _FakeCore) and other.id == self.id

    def __str__(self) -> str:
        return f"Core({self.id})"


class _FakeAccelerator:
    def __init__(self, cores, edges, offchip_core_id, name="test"):
        self.name = name
        self.core_list = cores
        self.offchip_core_id = offchip_core_id
        self.cores = DiGraphWrapper(edges)


def build(nb_cores: int, connections, offchip_core_id: int) -> _FakeAccelerator:
    """`connections` mirrors a yaml `core_connectivity`: ("bus", [ids], bw) or ("link", [a, b], bw)."""
    cores = [
        _FakeCore(i, "memory" if i == offchip_core_id else "compute", area=10.0 * (i + 1)) for i in range(nb_cores)
    ]
    edges = []
    bus_id = 0
    for kind, ids, bandwidth in connections:
        if kind == "bus":
            instance = CommunicationLink("Any", "Any", bandwidth, 0, bus_id=bus_id)
            bus_id += 1
            for index, a in enumerate(ids):
                for b in ids[index + 1 :]:
                    edges += get_bidirectional_edges(
                        cores[a],
                        cores[b],
                        bandwidth=bandwidth,
                        unit_energy_cost=0,
                        link_type="bus",
                        bus_instance=instance,
                    )
        else:
            a, b = ids
            edges += get_bidirectional_edges(
                cores[a], cores[b], bandwidth=bandwidth, unit_energy_cost=0, link_type="link"
            )
    return _FakeAccelerator(cores, edges, offchip_core_id)


def test_bus_collapses_to_one_hub():
    """A bus is one shared wire: N spokes to a hub, never the N*(N-1) pairwise edges the graph holds."""
    accelerator = build(4, [("bus", [0, 1, 2, 3], 128.0)], offchip_core_id=3)
    topology = extract_topology(accelerator)

    hubs = [node for node in topology.nodes if node.kind == "bus"]
    assert len(hubs) == 1, hubs
    assert len([e for e in topology.edges if e.kind == "bus"]) == 4
    assert [e for e in topology.edges if e.kind == "link"] == []
    assert hubs[0].bandwidth == 128.0


def test_point_to_point_pair_yields_one_edge():
    """The factory makes two directed links per pair; a symmetric pair must draw as one edge, not two."""
    accelerator = build(3, [("link", [0, 1], 4096.0)], offchip_core_id=2)
    topology = extract_topology(accelerator)

    links = [edge for edge in topology.edges if edge.kind == "link"]
    assert len(links) == 1, links
    assert links[0].directed is False
    assert links[0].bandwidth == 4096.0


def test_asymmetric_link_stays_two_directed_edges():
    """Differing bandwidths per direction must not be averaged away into one edge."""
    accelerator = build(3, [("link", [0, 1], 4096.0)], offchip_core_id=2)
    # Re-point one direction at a different bandwidth, as a hand-written yaml or a one-way rolled link would.
    for source, target, data in accelerator.cores.edges(data=True):
        if source.id == 1 and target.id == 0:
            data["cl"].bandwidth = 64.0
    topology = extract_topology(accelerator)

    links = [edge for edge in topology.edges if edge.kind == "link"]
    assert len(links) == 2, links
    assert all(edge.directed for edge in links)
    assert sorted(edge.bandwidth for edge in links) == [64.0, 4096.0]


def test_offchip_core_is_marked():
    accelerator = build(3, [("bus", [0, 1, 2], 128.0)], offchip_core_id=2)
    topology = extract_topology(accelerator)

    kinds = {node.id: node.kind for node in topology.nodes}
    assert kinds[2] == "offchip"
    assert [node.id for node in topology.compute_nodes] == [0, 1]


def test_two_buses_stay_separate():
    accelerator = build(5, [("bus", [0, 1, 4], 128.0), ("bus", [2, 3, 4], 64.0)], offchip_core_id=4)
    topology = extract_topology(accelerator)

    hubs = {node.id: node for node in topology.nodes if node.kind == "bus"}
    assert len(hubs) == 2, hubs
    members = {
        hub_id: {edge.target for edge in topology.edges if edge.kind == "bus" and edge.source == hub_id}
        for hub_id in hubs
    }
    assert sorted(members.values(), key=len) == [{0, 1, 4}, {2, 3, 4}]


def test_rolled_core_lists_all_hosted_layers():
    """A folded core runs several layers; the label carries all of them and the node is not layer-coloured."""
    accelerator = build(2, [("bus", [0, 1], 128.0)], offchip_core_id=1)
    topology = extract_topology(
        accelerator,
        core_layers={0: [0, 1]},
        layer_names={0: "conv1", 1: "conv2"},
        design_keys={0: "abcdef1234"},
    )

    core = next(node for node in topology.nodes if node.id == 0)
    assert [layer["name"] for layer in core.layers] == ["conv1", "conv2"]
    assert "conv1, conv2" in core.label
    assert "abcdef" in core.label


def test_core_layers_from_node_core_ids_inverts_the_map():
    assert core_layers_from_node_core_ids({0: [0, 1], 1: [0]}) == {0: [0, 1], 1: [0]}


def test_topology_from_connectivity_matches_the_accelerator_path():
    """The no-Accelerator path must produce the same cores, edges and bus hub as extraction does."""
    built = extract_topology(
        build(3, [("bus", [0, 1, 2], 128.0), ("link", [0, 1], 4096.0)], offchip_core_id=2),
        core_layers={0: [0], 1: [1]},
    )
    derived = topology_from_connectivity(
        name="test",
        node_core_ids={0: [0], 1: [1]},
        offchip_core_id=2,
        bus_connection={"type": "bus", "cores": [0, 1, 2], "bandwidth": 128.0},
        link_connections=[{"type": "link", "cores": [0, 1], "bandwidth": 4096.0}],
    )

    def summary(topology):
        return (
            sorted((str(node.id), node.kind) for node in topology.nodes),
            sorted((tuple(sorted(map(str, (e.source, e.target)))), e.bandwidth, e.kind) for e in topology.edges),
        )

    assert summary(built) == summary(derived)


def test_topology_json_roundtrips():
    """The interactive report serializes this; an unserializable field must fail here, not at write time."""
    topology = extract_topology(
        build(3, [("bus", [0, 1, 2], 128.0), ("link", [0, 1], 4096.0)], offchip_core_id=2),
        core_layers={0: [0], 1: [1]},
        layer_names={0: "conv1", 1: "conv2"},
        design_keys={0: "aaa", 1: "bbb"},
        layer_order=[0, 1],
    )
    restored = json.loads(json.dumps(topology.to_dict()))
    assert len(restored["nodes"]) == len(topology.nodes)
    assert restored["offchip_core_id"] == 2


def test_plot_writes_a_figure():
    topology = extract_topology(
        build(3, [("bus", [0, 1, 2], 128.0), ("link", [0, 1], 4096.0)], offchip_core_id=2),
        core_layers={0: [0], 1: [1]},
        layer_names={0: "conv1", 1: "conv2"},
        layer_order=[0, 1],
    )
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "topology.png")
        plot_hardware_graph(topology, path)
        assert os.path.getsize(path) > 0


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("All hardware graph tests passed.")
