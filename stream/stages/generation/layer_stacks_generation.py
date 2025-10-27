import logging
from typing import Any, TypeAlias

import networkx as nx
from zigzag.datatypes import MemoryOperand

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.stage import Stage, StageCallable
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload
from itertools import combinations
from gurobipy import Model, GRB, quicksum, Env
from collections import deque
import matplotlib.pyplot as plt

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

    with Env() as env:
        # Create Gurobi model
        model = Model("MinSubgraphs", env=env)
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

        # Precompute the dependency graph between all subgraphs
        dependency_graph = nx.DiGraph()
        dependency_graph.add_nodes_from(range(n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if any node in subgraph j has a predecessor in subgraph i
                for v in subgraphs[j].nodes:
                    predecessors = list(G.predecessors(v))
                    if any(p in subgraphs[i].nodes for p in predecessors):
                        dependency_graph.add_edge(i, j)
                        break

        # Callback to add lazy constraints for cycle detection in the dependency graph
        def no_cycles_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                # Get the current solution
                selected = [i for i in range(n) if model.cbGetSolution(x[i]) > 0.5]
                # Build the dependency graph for the selected subgraphs
                selected_dependency_graph = dependency_graph.subgraph(selected).copy()
                # Check for cycles in the selected dependency graph
                try:
                    nx.find_cycle(selected_dependency_graph, orientation="original")
                    # If a cycle is found, add a constraint to prevent this combination
                    model.cbLazy(quicksum(x[i] for i in selected) <= len(selected) - 1)
                except nx.NetworkXNoCycle:
                    pass  # No cycle, do nothing

        # Set the callback
        model.Params.LazyConstraints = 1
        model.optimize(no_cycles_callback)
        # Solve the model
        # model.optimize()

        print("ilp status", model.Status)
        # Extract selected subgraphs
        selected_subgraphs = [subgraphs[i] for i in range(n) if x[i].x > 0.5]

    return selected_subgraphs


def cycle_check(subgraphs, graph):
    # Create a mapping from subgraph index to its nodes
    subgraph_indices = list(range(len(subgraphs)))
    subgraph_nodes = [set(sg.nodes) for sg in subgraphs]

    # Build a dependency graph between subgraphs
    dependency_graph = nx.DiGraph()
    dependency_graph.add_nodes_from(subgraph_indices)

    # For each pair of subgraphs, check if any node in one has a predecessor in the other
    for i in subgraph_indices:
        for j in subgraph_indices:
            if i == j:
                continue
            # Check if any node in subgraph j has a predecessor in subgraph i
            for v in subgraphs[j].nodes:
                predecessors = list(graph.predecessors(v))
                if any(p in subgraph_nodes[i] for p in predecessors):
                    dependency_graph.add_edge(i, j)
                    break
    try:
        sorted_indices = list(nx.topological_sort(dependency_graph))
    except nx.NetworkXUnfeasible:
        # print("Cycle detected in dependency graph; returning original order.")
        return False

    # Return subgraphs in topological order
    return True


def topological_sort(selected_subgraphs, graph, draw_graph=True):
    # Create a mapping from subgraph index to its nodes
    subgraph_indices = list(range(len(selected_subgraphs)))
    subgraph_nodes = [set(sg.nodes) for sg in selected_subgraphs]

    # Build a dependency graph between subgraphs
    dependency_graph = nx.DiGraph()
    dependency_graph.add_nodes_from(subgraph_indices)

    # For each pair of subgraphs, check if any node in one has a predecessor in the other
    for i in subgraph_indices:
        for j in subgraph_indices:
            if i == j:
                continue
            # Check if any node in subgraph j has a predecessor in subgraph i
            for v in selected_subgraphs[j].nodes:
                predecessors = list(graph.predecessors(v))
                if any(p in subgraph_nodes[i] for p in predecessors):
                    dependency_graph.add_edge(i, j)
                    break

    # Optionally, draw the dependency graph
    if draw_graph:
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(dependency_graph)
        nx.draw(
            dependency_graph,
            pos,
            with_labels=True,
            node_size=1000,
            node_color="lightblue",
            font_size=10,
            font_weight="bold",
            arrowsize=20,
        )
        plt.title("Dependency Graph of Selected Subgraphs")
        plt.savefig("graph.png")
    import matplotlib.colors as mcolors

    draw_internal_graphs = True
    colors = list(mcolors.TABLEAU_COLORS.values())
    if draw_internal_graphs:
        plt.figure(figsize=(12, 8))

        # Create a combined graph for all subgraphs and their connections
        combined_graph = nx.DiGraph()
        for sg in selected_subgraphs:
            combined_graph.add_nodes_from(sg.nodes)
            combined_graph.add_edges_from(sg.edges)

        # Draw all subgraphs with unique colors
        pos = nx.spring_layout(combined_graph)
        for idx, sg in enumerate(selected_subgraphs):
            nx.draw_networkx_nodes(
                sg, pos, nodelist=sg.nodes, node_size=500, node_color=colors[idx % len(colors)], label=f"Subgraph {idx}"
            )
            nx.draw_networkx_edges(sg, pos, edge_color=colors[idx % len(colors)], width=2)
            nx.draw_networkx_labels(sg, pos, font_size=8, font_weight="bold")

        # Draw grey edges for connections between subgraphs
        for u, v in graph.edges:
            # Check if u and v are in different subgraphs
            u_in_subgraph = [idx for idx, sg_nodes in enumerate(subgraph_nodes) if u in sg_nodes]
            v_in_subgraph = [idx for idx, sg_nodes in enumerate(subgraph_nodes) if v in sg_nodes]
            if u_in_subgraph and v_in_subgraph and u_in_subgraph[0] != v_in_subgraph[0]:
                nx.draw_networkx_edges(
                    combined_graph, pos, edgelist=[(u, v)], edge_color="grey", width=1, style="dashed"
                )

        plt.title("Subgraphs with Input/Output Links (Grey)")
        plt.legend(loc="upper right")
        plt.savefig("subgraph.png")
    # Perform topological sort
    try:
        sorted_indices = list(nx.topological_sort(dependency_graph))
    except nx.NetworkXUnfeasible:
        print("Cycle detected in dependency graph; returning original order.")
        sorted_indices = subgraph_indices

    print(sorted_indices)
    print(
        "subgraph",
        [
            ([(node.id, node.type) for node in selected_subgraph], i)
            for i, selected_subgraph in enumerate(selected_subgraphs)
        ],
    )
    # Return subgraphs in topological order
    return [selected_subgraphs[i] for i in sorted_indices]


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


def find_subgraphs(G):
    """
    Find all connected subgraphs of a DiGraph where the subgraph has at most one outgoing source node.

    Parameters:
    G (nx.DiGraph): The input directed graph.

    Returns:
    list: A list of connected subgraphs (each as a set of nodes) that satisfy the condition.
    """
    valid_subgraphs = []

    # Iterate over all possible non-empty connected subgraphs
    for node in G.nodes():
        # Use BFS to explore connected subgraphs starting from 'node'
        queue = [(node, {node})]  # (current_node, subgraph_nodes)

        while queue:
            current_node, subgraph_nodes = queue.pop(0)

            # Check if the subgraph has at most one outgoing source node
            if len(subgraph_nodes) > 5:
                break
            outgoing_sources = set()
            for n in subgraph_nodes:
                for neighbor in G.neighbors(n):
                    if neighbor not in subgraph_nodes:
                        outgoing_sources.add(n)
                        break  # Only need to know if it has at least one outgoing edge

            if len(outgoing_sources) <= 1:
                if subgraph_nodes not in valid_subgraphs:
                    valid_subgraphs.append(subgraph_nodes)

            # Expand the subgraph by adding neighbors
            for neighbor in G.neighbors(current_node):
                if neighbor not in subgraph_nodes:
                    new_subgraph = subgraph_nodes.union({neighbor})
                    queue.append((neighbor, new_subgraph))

    return valid_subgraphs


def find_path(graph, source, computations_nodes):
    """
    Run a BFS between the source and all other computations_nodes.
    For each path, record the last edge's tensor size.
    """
    paths = {}  # Will store {computation_node: last_tensor_size}
    queue = [(source, [source], None)]  # (current_node, current_path, last_tensor_size)
    visited = set()

    while queue:
        current_node, current_path, last_tensor_size = queue.pop(0)
        if current_node in visited:
            continue
        visited.add(current_node)

        # If current_node is a computation node and not the source, record the last tensor size
        if current_node in computations_nodes and current_node != source:
            paths[current_node] = last_tensor_size
            continue  # No need to explore further from this node

        # Explore neighbors
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                # Get the tensor size of the edge current_node -> neighbor
                tensor_size = 0
                if hasattr(neighbor, "operand_size_bit"):
                    dict_input_size = neighbor.operand_size_bit
                    dict_input_operand_source = neighbor.input_operand_source
                    for operand_name, node_id in dict_input_operand_source.items():
                        if current_node.id == node_id:
                            tensor_size = dict_input_size[operand_name]

                queue.append((neighbor, current_path + [neighbor], tensor_size))

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
            # mem_size_per_core={
            #     allocation: float(size / len(node.possible_core_allocation))
            #     for allocation in node.possible_core_allocation
            # },
            mem_size_per_core={allocation: size for allocation in node.possible_core_allocation},
            tiling=tiling,
            **original_graph.nodes[node],
        )

    # For each pair of computation nodes, find all paths in the original graph
    # and add an edge in the abstracted graph if there is a path
    for u in computation_nodes:
        paths = find_path(original_graph, u, computation_nodes)
        for v in computation_nodes:
            if v != u and v in paths and paths[v] >= 1:
                tensor_size = float(paths[v] / len(node.possible_core_allocation))
                tensor_size = paths[v]
                # Add the edge to the abstracted graph with the tensor size
                abstracted_graph.add_edge(u, v, tensor_size=tensor_size)

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
                    # self.layer_stacks = self.get_layer_stacks_fused_local_memory()
                    # print(self.solve_constraint_programming())
                    self.layer_stacks = self.solve_constraint_programming()
            else:
                self.layer_stacks = self.fill_layer_stacks_to_completion()

        elif self.mode == "lbl":
            self.layer_stacks = self.get_layer_stacks_lbl()
        else:
            raise ValueError("Unsupported mode for layer stack determination.")

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

    def check_node_types(self, graph):
        """any subgraph with two gemm/matmul should not be considered as well as subgraph with more than 2 convolutions"""

        subgraph_types = [node.type for node in graph]
        if subgraph_types.count("conv") > 3:
            return False
        if subgraph_types.count("gemm") + subgraph_types.count("matmul") > 1:
            return False
        if (
            subgraph_types.count("gemm")
            + subgraph_types.count("matmul")
            + subgraph_types.count("conv")
            + subgraph_types.count("convtranspose")
            > 1
        ):
            return False
        # if subgraph_types.count("sub") > 0 and len(graph) > 1:
        #     return False
        return True

    def check_length_subgraph(self, graph, max_length: int = 5):
        if len(graph) >= max_length:
            return False
        else:
            return True

    def check_intra_core_tiling(self, graph):
        tilings = nx.get_node_attributes(graph, "tiling").values()

        divisible = all((a % b == 0 or b % a == 0) for a in tilings for b in tilings if a != b)
        return divisible

    def check_memory_constraint(self, subgraph, graph):
        subgraphs_mem_per_cores = nx.get_node_attributes(subgraph, "mem_size_per_core")
        subgraph_nodes = set(graph.nodes())
        for core_id in self.weight_capacities:
            core_allocated_mem = 0
            for node, node_mem_per_core in subgraphs_mem_per_cores.items():
                if core_id in node_mem_per_core:
                    core_allocated_mem += node_mem_per_core[core_id]

                for predecessor in graph.predecessors(node):
                    # print(predecessor in subgraph_nodes)
                    if predecessor not in subgraph_nodes:
                        # Get the edge data for the edge (predecessor -> node)
                        edge_data = graph.get_edge_data(predecessor, node)
                        if edge_data:
                            # Add the cost from the edge to the core's allocated memory
                            # Assuming the cost is stored as 'cost' or 'tensor_size'
                            cost = edge_data.get("tensor_size", 0)  # or "tensor_size"
                            print(cost, predecessor.id, node.id)
                            core_allocated_mem += cost
            if core_allocated_mem > self.weight_capacities[core_id]:
                return False

        return True

    def find_valid_subgraphs_bfs(self, G, max_size=None):
        """
        Find all connected subgraphs of G using BFS expansion that satisfy:
        - check_intra_core_tiling(subgraph)
        - check_memory_constraint(subgraph, G)

        Parameters
        ----------
        G : nx.DiGraph
            Original directed graph (assumed connected).
        checker : object
            Provides methods:
                - checker.check_intra_core_tiling(graph)
                - checker.check_memory_constraint(subgraph, graph)
        max_size : int, optional
            Optional limit on subgraph size (to avoid excessive exploration).

        Returns
        -------
        list of nx.DiGraph
            All connected subgraphs satisfying both checks.
        """

        valid_subgraphs = []
        visited_sets = set()

        for start_node in G.nodes:
            print(start_node.id)
            queue = deque()
            queue.append({start_node})

            while queue:
                current_nodes = queue.popleft()
                frozen = frozenset(current_nodes)

                if frozen in visited_sets:
                    continue
                visited_sets.add(frozen)

                subG = G.subgraph(current_nodes)

                # Run the checks
                if not self.check_node_types(subG):
                    continue
                if not self.check_intra_core_tiling(subG):
                    continue
                if not self.check_memory_constraint(subG, G):
                    continue

                outgoing_sources = set()
                for n in current_nodes:
                    for neighbor in G.successors(n):
                        if neighbor not in current_nodes:
                            outgoing_sources.add(n)
                            break

                if len(outgoing_sources) <= 1:
                    valid_subgraphs.append(subG.copy())

                # Store valid subgraph
                # valid_subgraphs.append(subG.copy())
                # print(len(valid_subgraphs), [[n.id for n in gra] for gra in valid_subgraphs])
                # Expand if below size limit
                if max_size and len(current_nodes) >= max_size:
                    continue

                # Expand by adding weak neighbors (successors or predecessors)
                neighbors = set()
                for node in current_nodes:
                    neighbors.update(G.successors(node))
                    neighbors.update(G.predecessors(node))

                # Only consider neighbors not already in the subgraph
                neighbors -= current_nodes

                for n in neighbors:
                    new_nodes = set(current_nodes)
                    new_nodes.add(n)
                    queue.append(new_nodes)

        return valid_subgraphs

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

        subgraphs = self.find_valid_subgraphs_bfs(new_graph, max_size=6)
        print(len(subgraphs))
        # For each subgraph, we find if they match the constraints, otherwise we discard them
        valid_subgraphs = []
        for subgraph in subgraphs:
            if self.check_node_types(subgraph):
                # if self.check_length_subgraph(subgraph, 30):
                if self.check_intra_core_tiling(subgraph):
                    if self.check_memory_constraint(subgraph, new_graph):
                        valid_subgraphs.append(subgraph)
        solution = ilp_min_subgraphs_gurobi(new_graph, valid_subgraphs + single_node_subgraphs(new_graph))

        sorted_solution = topological_sort(solution, new_graph)
        return [tuple(sorted([node.id for node in subgraph])) for subgraph in sorted_solution]

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
