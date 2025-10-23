import logging
from typing import Any, TypeAlias

import networkx as nx
from zigzag.datatypes import MemoryOperand

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.stage import Stage, StageCallable
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload
from itertools import combinations
import pulp
from gurobipy import Model, GRB, quicksum


logger = logging.getLogger(__name__)

STACK_T: TypeAlias = tuple[int, ...]


def single_node_subgraphs(G):
    """
    Generate a list of subgraphs each containing a single node from G.
    """
    subgraphs = []
    for node in G.nodes:
        sg = G.subgraph([node]).copy()  # create a copy of the subgraph
        subgraphs.append(sg)
    return subgraphs


def ilp_min_subgraphs_gurobi(G, subgraphs, cover_edges=True):
    """
    Selects a minimal set of subgraphs to reconstruct the original graph G
    without overlapping nodes (and optionally edges) using Gurobi.

    Parameters:
        G : networkx.DiGraph
            Original graph
        subgraphs : list of networkx.DiGraph
            Candidate subgraphs
        cover_edges : bool
            If True, ensure edges are covered exactly once as well

    Returns:
        selected_subgraphs : list of networkx.DiGraph
            Subgraphs chosen by the ILP
    """
    n = len(subgraphs)

    # Create Gurobi model
    model = Model("MinSubgraphs")
    model.Params.OutputFlag = 0  # Turn off solver output

    # Decision variables
    x = [model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in range(n)]

    # Objective: minimize number of subgraphs selected
    model.setObjective(quicksum(x), GRB.MINIMIZE)

    # Node coverage constraints: each node must appear in exactly one selected subgraph
    for v in G.nodes:
        model.addConstr(
            quicksum(x[i] for i, sg in enumerate(subgraphs) if v in sg.nodes) == 1, name=f"node_{v}_coverage"
        )

    # Solve the model
    model.optimize()

    # Extract selected subgraphs
    selected_subgraphs = [subgraphs[i] for i in range(n) if x[i].x > 0.5]

    return selected_subgraphs


def remove_necessary_subgraphs(subgraphs, necessary):
    """
    Removes all subgraphs that are included in the `necessary` list.
    Comparison is done structurally (by node and edge sets).
    """
    necessary_signatures = [(frozenset(sg.nodes), frozenset(sg.edges)) for sg in necessary]

    remaining = []
    for sg in subgraphs:
        sig = (frozenset(sg.nodes), frozenset(sg.edges))
        if sig not in necessary_signatures:
            remaining.append(sg)

    return remaining


def necessary_subgraphs(G, subgraphs):
    necessary_subgraphs = []
    for subgraph in subgraphs:
        is_necessary = True
        for subgraph2 in subgraphs:
            if subgraph2 != subgraph:
                if set([node.id for node in subgraph]) <= set([node.id for node in subgraph2]):
                    is_necessary = False
        if is_necessary:
            necessary_subgraphs.append(subgraph)
    return necessary_subgraphs


def find_constrained_subgraphs(G: nx.DiGraph):
    subgraphs = []
    visited = set()

    for component in nx.weakly_connected_components(G):
        subG = G.subgraph(component).copy()

        # Work from sinks upward
        for node in list(subG.nodes):
            if node in visited:
                continue

            # Build subgraph starting from this node
            sub_nodes = set()
            frontier = [node]
            while frontier:
                n = frontier.pop()
                if n in sub_nodes:
                    continue
                sub_nodes.add(n)

                # Count outgoing edges that leave this potential subgraph
                outgoing_outside = {u for u in sub_nodes for v in G.successors(u) if v not in sub_nodes}

                # Stop if more than 1 node sends outside
                if len(outgoing_outside) > 1:
                    sub_nodes.remove(n)
                    continue

                # Expand backwards
                for pred in G.predecessors(n):
                    if pred not in sub_nodes:
                        frontier.append(pred)

            visited |= sub_nodes
            subgraphs.append(G.subgraph(sub_nodes).copy())

    return subgraphs


def find_path(graph, source, computations_nodes):
    """
    run a BFS between the source and all other computations_nodes
    """
    paths = {}

    # BFS setup
    queue = [(source, [source])]
    visited = set()

    while queue:
        current_node, current_path = queue.pop(0)

        # Skip if already visited
        if current_node in visited:
            continue
        visited.add(current_node)

        # If current_node is a computation node and not the source, record the path
        if current_node in computations_nodes and current_node != source:
            paths[current_node] = 1
            continue  # No need to explore further from this node

        # Explore neighbors
        for neighbor in graph.neighbors(current_node):
            # Only proceed if neighbor is not a computation node (or is the source)
            # if not isinstance(neighbor, ComputationNode) or neighbor == source:
            #     print(neighbor)
            if neighbor not in visited:
                queue.append((neighbor, current_path + [neighbor]))
    return paths


def abstract_computation_graph(original_graph):
    """
    #TODO: add the output tensor size as the edge
    Creates a new digraph with only computation nodes, abstracting paths of non-computation nodes.

    Args:
        original_graph (nx.DiGraph): The original directed graph.

    Returns:
        nx.DiGraph: A new digraph with only computation nodes and abstracted edges.
    """
    # Create a new graph to store the abstracted computation graph
    abstracted_graph = nx.DiGraph()

    # Get all computation nodes
    computation_nodes = [node for node, attr in original_graph.nodes(data=True) if isinstance(node, ComputationNode)]

    # Add all computation nodes to the new graph
    for node in computation_nodes:
        # add some attributes to make the partitionning easier
        try:
            op = next(op for op in node.constant_operands)
            size = node.operand_size_bit[op]
        except StopIteration:
            size = 0

        intra_core_tiling = node.intra_core_tiling
        tiling = node.layer_dim_sizes[intra_core_tiling[0][0]]
        abstracted_graph.add_node(
            node,
            mem_size=size,
            mem_size_per_core={
                allocation: float(size / len(node.possible_core_allocation))
                for allocation in node.possible_core_allocation
            },
            tiling=tiling,
            **original_graph.nodes[node],
        )

    # For each pair of computation nodes, find all paths in the original graph
    # and add an edge in the abstracted graph if there is a path
    for u in computation_nodes:
        paths = find_path(original_graph, u, computation_nodes)
        for v in computation_nodes:
            if v != u and v in paths and paths[v] == 1:
                abstracted_graph.add_edge(u, v)

    return abstracted_graph


class LayerStacksGenerationStage(Stage):
    layer_stacks: list[STACK_T] | None

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        accelerator: Accelerator,
        workload: ComputationNodeWorkload,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.workload = workload

        self.layer_stacks = kwargs.get("layer_stacks", None)
        self.mode = kwargs.get("mode")
        self.stack_cutoff = kwargs.get("stack_cutoff", None)
        self.stack_cutoffs = kwargs.get("stack_cutoffs", None)

        # Get the weight capacity of all cores
        self.weight_capacities: dict[int, int] = {}
        for core in self.accelerator.cores.node_list:
            if core.id == self.accelerator.offchip_core_id:
                continue  # skip offchip core
            mem_op = MemoryOperand("I2")
            core_weight_capacity = core.memory_hierarchy.get_operand_top_level(mem_op).memory_instance.size
            self.weight_capacities[core.id] = core_weight_capacity

        # Total weight capacity in bits
        self.total_weight_capacity = sum(self.weight_capacities.values())

    def run(self):
        if self.mode == "fused":
            if self.layer_stacks is None:
                if self.stack_cutoff is not None:
                    self.layer_stacks = self.get_layer_stacks_fused_single_fixed()
                elif self.stack_cutoffs is not None:
                    self.layer_stacks = self.get_layer_stacks_fused_multiple_fixed()
                else:
                    self.layer_stacks = self.get_layer_stacks_fused_local_memory()
            else:
                self.layer_stacks = self.fill_layer_stacks_to_completion()

        elif self.mode == "lbl":
            self.layer_stacks = self.get_layer_stacks_lbl()
        else:
            raise ValueError("Unsupported mode for layer stack determination.")

        print(self.solve_constraint_programming())
        self.layer_stacks = self.solve_constraint_programming()
        logger.warning("%s", self.layer_stacks)
        self.only_keep_computation_node_ids()

        self.kwargs["accelerator"] = self.accelerator
        self.kwargs["workload"] = self.workload
        self.kwargs["layer_stacks"] = self.layer_stacks
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            **self.kwargs,
        )
        yield from sub_stage.run()

    def only_keep_computation_node_ids(self):
        """! Update the layer stacks to only keep ids of ComputationNodes"""
        assert self.layer_stacks is not None
        updated_layer_stacks: list[tuple[int, ...]] = []
        for stack in self.layer_stacks:
            update_stack: list[int] = []
            for layer_id in stack:
                try:
                    # Ignore node ids that do not exist
                    n = next(n for n in self.workload.node_list if n.id == layer_id)
                    if isinstance(n, ComputationNode):
                        update_stack.append(layer_id)
                except StopIteration:
                    pass
            updated_layer_stacks.append(tuple(update_stack))
        self.layer_stacks = updated_layer_stacks

    def get_layer_stacks_lbl(self):
        return [(id,) for id in sorted([n.id for n in self.workload.node_list if isinstance(n, ComputationNode)])]

    def fill_layer_stacks_to_completion(self):
        assert self.layer_stacks is not None
        stacks: list[tuple[int, ...]] = self.layer_stacks

        for node in self.workload.node_list:
            if not any(node.id in stack for stack in stacks):
                stacks += [(node.id,)]
        return stacks

    def get_layer_stacks_fused(self):
        cumsum = 0
        stacks: list[tuple[int, ...]] = []
        current_stack: list[int] = []
        for n in sorted(list(self.workload.node_list), key=lambda n: n.id):
            if isinstance(n, ComputationNode):
                id = n.id
                try:
                    op = next(op for op in n.constant_operands)
                except StopIteration:
                    current_stack.append(id)
                    continue
                size = n.operand_size_bit[op]
                cumsum += size
                ratio = cumsum / self.total_weight_capacity
                if ratio > 1:
                    stacks.append(tuple(current_stack))
                    current_stack = [id]
                    cumsum = size
                else:
                    current_stack.append(id)
        # Add last stack
        stacks.append(tuple(current_stack))

        return stacks

    def get_layer_stacks_fused_single(self):
        """
        Only the first set of layers will be fused, rest layer by layer"""
        cumsum = 0
        stacks: list[tuple[int, ...]] = []
        current_stack: list[int] = []
        first_complete = False
        for n in sorted(list(self.workload.node_list), key=lambda n: n.id):
            if isinstance(n, ComputationNode):
                id = n.id
                if first_complete:
                    stacks.append(tuple(current_stack))
                    current_stack = [id]
                    continue
                try:
                    op = next(op for op in n.constant_operands)
                except StopIteration:
                    current_stack.append(id)
                    continue
                size = n.operand_size_bit[op]
                cumsum += size
                ratio = cumsum / self.total_weight_capacity
                if ratio > 1:
                    stacks.append(tuple(current_stack))
                    current_stack = [id]
                    cumsum = size
                    first_complete = True
                else:
                    current_stack.append(id)
        # Add last stack
        stacks.append(tuple(current_stack))

        return stacks

    def check_intra_core_tiling(self, graph):
        tilings = nx.get_node_attributes(graph, "tiling").values()

        divisible = all((a % b == 0 or b % a == 0) for a in tilings for b in tilings if a != b)
        return divisible

    def check_memory_constraint(self, graph):
        subgraphs_mem_per_cores = nx.get_node_attributes(graph, "mem_size_per_core")

        for core_id in self.weight_capacities:
            core_allocated_mem = 0
            for _, node_mem_per_core in subgraphs_mem_per_cores.items():
                if core_id in node_mem_per_core:
                    core_allocated_mem += node_mem_per_core[core_id]

            if core_allocated_mem > self.weight_capacities[core_id]:
                return False

        return True

    def solve_constraint_programming(self):
        """
        Based on the workload and accelerator properties, return the possible layer stacks
        We solve a constraint optimization problem with partitionning the workload graph into subgraphthat are fuseable
        goal :
            maximize number of nodes in each subgraph
        constraint:
            - all nodes should be assigned
            - each subgraph needed memory should not exceed the corresponding cores memory
            - each subgraph should have one output node
                (so that the intermediate activations that could be removed are not needed)
            - all nodes in a subgraph share a common intra_core_tiling divisor
        We use a Backtracking approach to find all possible subgraphs, then we find all possible complete graph and we select the best(still using backtracking)
        """

        new_graph = abstract_computation_graph(self.workload)

        subgraphs = find_constrained_subgraphs(new_graph)

        # For each subgraph, we find if they match the constraints, otherwise we discard them
        valid_subgraphs = []
        for subgraph in subgraphs:
            if self.check_intra_core_tiling(subgraph):
                if self.check_memory_constraint(subgraph):
                    valid_subgraphs.append(subgraph)

        solution = ilp_min_subgraphs_gurobi(new_graph, valid_subgraphs + single_node_subgraphs(new_graph))

        return [[node.id for node in subgraph] for subgraph in solution]

    def get_layer_stacks_fused_local_memory(self):
        """
        Creates new stacks whenever the local memory capacity is exceeded.
        """

        cumsum = 0
        stacks: list[tuple[int, ...]] = []
        current_stack: list[int] = []
        current_stack_tiling: list[int] = []

        simd_cumsum = 0

        for n in sorted(list(self.workload.node_list), key=lambda n: n.id):
            if isinstance(n, ComputationNode):
                id = n.id
                intra_core_tiling = n.intra_core_tiling
                tiling = n.layer_dim_sizes[intra_core_tiling[0][0]]
                try:
                    op = next(op for op in n.constant_operands)
                except StopIteration:
                    if current_stack:  # Only append if not empty
                        stacks.append(tuple(current_stack))
                    current_stack = [id]
                    current_stack_tiling = [tiling]
                    cumsum = 0
                    simd_cumsum = 0
                    continue

                size = n.operand_size_bit[op]
                # Check divisibility with all existing tilings in the current stack
                divisible = all(
                    (a % b == 0 or b % a == 0)
                    for a in current_stack_tiling + [tiling]
                    for b in current_stack_tiling + [tiling]
                    if a != b
                )
                # logger.warning("%s %s %s", current_stack_tiling, tiling, n.id)
                # If adding this layer exceeds capacity or breaks tiling compatibility
                if (
                    (cumsum + size > self.total_weight_capacity and current_stack)
                    or not divisible
                    or len(current_stack) >= 20
                ):
                    if current_stack:  # Only append if not empty
                        stacks.append(tuple(current_stack))
                    current_stack = [id]
                    current_stack_tiling = [tiling]
                    cumsum = size
                    if 5 in n.core_allocation:
                        simd_cumsum = size
                    else:
                        simd_cumsum = 0
                else:
                    current_stack.append(id)
                    current_stack_tiling.append(tiling)
                    cumsum += size
                    if 5 in n.core_allocation:
                        simd_cumsum += size

        # Add the last stack if it's not empty
        if current_stack:
            stacks.append(tuple(current_stack))
        print(stacks)
        return stacks

    def get_layer_stacks_fused_single_fixed(self):
        """
        layers will be fused based on ids in stack cutoffs. if ratio of weights > 1, we switch to layer by layer
        """
        assert self.stack_cutoff is not None, "stack_cutoff should be defined."
        stacks = []
        current_stack = []
        for n in sorted(list(self.workload.node_list), key=lambda n: n.id):
            if isinstance(n, ComputationNode):
                id = n.id
                if id > self.stack_cutoff:
                    stacks.append(tuple(current_stack))
                    current_stack = [id]
                else:
                    current_stack.append(id)
        # Add last stack
        stacks.append(tuple(current_stack))

        return stacks

    def get_layer_stacks_fused_multiple_fixed(self):
        """
        Only the first set of layers will be fused until fixed id, rest layer by layer
        """
        assert self.stack_cutoffs is not None, "stack_cutoff should be defined."
        stacks = []
        current_stack = []
        assert len(self.stack_cutoffs) > 0
        stack_cutoff = self.stack_cutoffs[0]
        cutoff_idx = 1
        cumsum = 0
        lbl = False  # flag to switch to layer by layer
        for n in sorted(list(self.workload.node_list), key=lambda n: n.id):
            if isinstance(n, ComputationNode):
                id = n.id
                if lbl:
                    stacks.append(tuple(current_stack))
                    current_stack = [id]
                    continue
                if id > stack_cutoff:
                    stacks.append(tuple(current_stack))
                    current_stack = [id]
                    cumsum = 0
                    if cutoff_idx <= len(self.stack_cutoffs) - 1:
                        stack_cutoff = self.stack_cutoffs[cutoff_idx]
                        cutoff_idx += 1
                    else:
                        lbl = True
                try:
                    op = next(op for op in n.constant_operands)
                except StopIteration:
                    if id not in current_stack:
                        current_stack.append(id)
                    continue
                size = n.operand_size_bit[op]
                cumsum += size
                ratio = cumsum / self.total_weight_capacity
                if ratio > 1:
                    stacks.append(tuple(current_stack))
                    current_stack = [id]
                    cumsum = size
                    lbl = True
                elif not lbl:
                    if id not in current_stack:
                        current_stack.append(id)
        # Add last stack
        stacks.append(tuple(current_stack))

        return stacks
