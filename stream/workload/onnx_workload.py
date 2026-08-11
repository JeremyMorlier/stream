import logging
from typing import Any

import pydot
from networkx.drawing.nx_pydot import to_pydot  # type: ignore
from zigzag.utils import DiGraphWrapper

from stream.workload.computation.computation_node import ComputationNode
from stream.workload.node import Node

logger = logging.getLogger(__name__)

# Cycled through by layer id when coloring nodes in `ComputationNodeWorkload.visualize_to_file`
_LAYER_COLOR_PALETTE = (
    "#a2d5f2",
    "#ffcb9a",
    "#c2f0c2",
    "#eaff9a",
    "#f2a2d5",
    "#d5a2f2",
    "#a2f2d5",
    "#f2d5a2",
    "#c2c2f0",
    "#f0c2c2",
)


class ONNXWorkload(DiGraphWrapper[Node]):
    """Represents a Workload Graph"""

    def __init__(self, **attr: Any):
        """Collect all the algorithmic workload information here."""
        super().__init__(**attr)  # type: ignore
        self.node_id_to_obj: dict[int, Node] = {}
        # In case an ONNX node is converted into multiple Stream nodes, link id of first to last generated node
        self.first_to_last_expanded_id: dict[int, int] = {}

    def add(self, node_id: int, node_obj: Node):
        """
        Add a node object to the ONNX workload graph.
        This can be a different object based on if it's an "accelerateable" node or not.
        """
        self.node_id_to_obj[node_id] = node_obj

        self.add_node(node_obj)
        edges: list[tuple[Node, Node]] = []
        for parent_id in node_obj.input_operand_source.values():
            if parent_id in self.first_to_last_expanded_id:
                # If the parent node was replaced with multiple stream nodes, take the last generated one as parent
                parent_id_updated = self.first_to_last_expanded_id[parent_id]
            else:
                parent_id_updated = parent_id
            parent_node_obj = self.node_id_to_obj[parent_id_updated]
            edges.append((parent_node_obj, node_obj))
            self.add_edges_from(edges)

    def find_paths_with_intermediate_type(
        self, start: Node, end: Node, only_intermediate_type: type
    ) -> list[list[Node]]:
        """Returns all paths between `start` and `end` where the nodes in between must be of type
        `only_intermediate_type`

        # TODO move to `DiGraphWrapper` in ZigZag and make generic in `T`
        """
        valid_paths = [[start]]
        final_paths: list[list[Node]] = []
        while valid_paths:
            valid_path = valid_paths.pop(0)
            next_nodes = list(self.successors(valid_path[-1]))
            for next_node in next_nodes:
                if isinstance(next_node, only_intermediate_type):
                    valid_paths.append(valid_path + [next_node])
                elif next_node == end:
                    final_paths.append(valid_path + [next_node])
                else:
                    continue
        return final_paths


class ComputationNodeWorkload(DiGraphWrapper[ComputationNode]):
    """Workload graph with only ComputationNodes"""

    def get_sink_layer_ids(self):
        """Return the ids of layers where ALL sub-nodes have out-degree 0
        # TODO this might nog work yet! When there is intra-core tiling, edges between nodes in the same layer
        # TODO (with bits==0) are added, meaning some nodes have an out-degree > 0
        # TODO -> use get_real_nb_predecessors instead? or remove the empty intra-core edges?
        """
        out_degrees = self.out_degree()
        layer_ids = set(n.id for n, _ in out_degrees)
        sink_layer_ids = [
            curr_id
            for curr_id in layer_ids
            # x: (node, out_degree). Filter by id -> map to out_degree == 0 -> check if all are 0
            if all(map(lambda x: x[1] == 0, filter(lambda x: x[0].id == curr_id, out_degrees)))
        ]
        return sink_layer_ids

    def get_subgraph(self, nodes: list[ComputationNode]) -> "ComputationNodeWorkload":
        return self.subgraph(nodes)  # type: ignore

    def visualize_to_file(self, filepath: str = "tiled_workload_graph.dot"):
        """Visualize the tiled workload graph using Graphviz and save it to a DOT file.

        Nodes are laid out left to right in topological order, and vertically grouped by
        chosen_core_allocation (stacked top-to-bottom). Nodes are colored by the id of the
        layer they were tiled from.
        """
        dot = to_pydot(self)
        dot.set_rankdir("LR")
        dot.set_concentrate(True)

        # Group nodes by chosen core allocation (vertical stacking of clusters)
        core_to_nodes: dict[int | None, list[ComputationNode]] = {}
        for node in self.nodes():
            core_to_nodes.setdefault(node.chosen_core_allocation, []).append(node)
        sorted_cores = sorted(core_to_nodes.keys(), key=lambda c: (c is None, c))

        cluster_heads: list[str] = []
        for core in sorted_cores:
            nodes = core_to_nodes[core]
            cluster = pydot.Cluster(
                graph_name=f"core_{core}",
                label=f"Core {core}" if core is not None else "Unallocated",
                style="dashed",
            )
            for node in nodes:
                cluster.add_node(dot.get_node(str(node))[0])
            dot.add_subgraph(cluster)
            cluster_heads.append(str(nodes[0]))

        # Add invisible edges between cluster heads to stack clusters vertically
        for upper, lower in zip(cluster_heads, cluster_heads[1:], strict=False):
            dot.add_edge(pydot.Edge(upper, lower, style="invis"))

        # Color nodes by the id of the layer they were tiled from
        layer_ids = sorted(set(node.id for node in self.nodes()))
        layer_id_to_color = {
            layer_id: _LAYER_COLOR_PALETTE[i % len(_LAYER_COLOR_PALETTE)] for i, layer_id in enumerate(layer_ids)
        }

        for node in self.nodes():
            n = dot.get_node(str(node))[0]
            n.set_label(f"{node.short_name}\\n({node.id},{node.sub_id})")
            n.set_shape("box")
            n.set_style("filled")
            n.set_fillcolor(layer_id_to_color[node.id])

        # Highlight edges that carry no data (e.g. intra-core tiling edges with bits == 0)
        for u, v, data in self.edges(data=True):
            if data.get("bits") == 0:
                for e in dot.get_edge(str(u), str(v)):
                    e.set_color("red")

        dot.write_raw(filepath)
        logger.info(f"Tiled workload graph saved to {filepath}")
