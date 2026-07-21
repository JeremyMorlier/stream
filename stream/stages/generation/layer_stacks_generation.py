import logging
from collections import deque
from typing import Any, TypeAlias

import networkx as nx
from gurobipy import GRB, Env, Model, quicksum
from zigzag.datatypes import MemoryOperand

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.stage import Stage, StageCallable
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload

logger = logging.getLogger(__name__)

STACK_T: TypeAlias = tuple[int, ...]


def single_node_subgraphs(complete_graph):
    """
    Generate a list of subgraphs each containing a single node from complete_graph.
    """
    subgraphs = []
    for node in complete_graph.nodes:
        sg = complete_graph.subgraph([node]).copy()  # create a copy of the subgraph
        subgraphs.append(sg)
    return subgraphs


def create_dependency_graph(subgraphs, graph):
    """
    Creates a dependency graph between subgraphs, where an edge from i to j exists
    if any node in j has a predecessor in i. The edge weight is determined by the provided weight_function.

    Parameters:
        subgraphs : list of networkx.DiGraph
            List of candidate subgraphs
        graph : networkx.DiGraph
            Original graph
    Returns:
        dependency_graph : networkx.DiGraph
            Dependency graph between subgraphs, with edges weighted as specified
    """
    n = len(subgraphs)
    dependency_graph = nx.DiGraph()
    dependency_graph.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Check if any node in subgraph j has a predecessor in subgraph i
            has_dependency = False
            total_weight = 0.0
            for v in subgraphs[j].nodes:
                predecessors = list(graph.predecessors(v))
                for p in predecessors:
                    if p in subgraphs[i].nodes:
                        # Sum the weights of edges from p (in i) to v (in j)
                        if graph.has_edge(p, v):
                            total_weight += graph[p][v]["tensor_size"]
                            has_dependency = True
            if has_dependency:
                dependency_graph.add_edge(i, j, weight=total_weight)
    return dependency_graph


def ilp_min_subgraphs_gurobi(complete_graph, subgraphs, objective="minimize_subgraphs"):
    """
    Selects a minimal set of subgraphs to reconstruct the original graph G
    without overlapping nodes (and optionally edges) using Gurobi.

    Parameters:
        complete_graph : networkx.DiGraph
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

    dependency_graph = create_dependency_graph(subgraphs, complete_graph)
    with Env() as env:
        # Create Gurobi model
        model = Model("MinSubgraphs", env=env)
        model.Params.OutputFlag = 0  # Turn off solver output

        # Decision variables
        x = [model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in range(n)]

        # Objective: minimize number of subgraphs or sum of edges
        if objective == "minimize_subgraphs":
            model.setObjective(quicksum(x), GRB.MINIMIZE)
        elif objective == "minimize_tensor_movement":
            edge_weight_sum = 0
            for i, j, data in dependency_graph.edges(data=True):
                edge_weight_sum += data["weight"] * x[i] * x[j]
            model.setObjective(edge_weight_sum, GRB.MINIMIZE)
        else:
            raise ValueError("Invalid objective. Use 'minimize_subgraphs' or 'minimize_tensor_movement'.")

        # Node coverage constraints: each node must appear in exactly one selected subgraph
        for v in complete_graph.nodes:
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
                    predecessors = list(complete_graph.predecessors(v))
                    if any(p in subgraphs[i].nodes for p in predecessors):
                        dependency_graph.add_edge(i, j)
                        break

        # Callback to add lazy constraints for cycle detection in the dependency graph
        def no_cycles_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                # Get the current solution
                selected = [i for i in range(n) if model.cbGetSolution(x[i]) > 0.5]  # noqa: PLR2004
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
        logger.warning(f"ILP status: {model.Status}")
        # Extract selected subgraphs
        selected_subgraphs = [subgraphs[i] for i in range(n) if x[i].x > 0.5]  # noqa: PLR2004

    return selected_subgraphs


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

    # Perform topological sort
    try:
        sorted_indices = list(nx.topological_sort(dependency_graph))
    except nx.NetworkXUnfeasible:
        print("Cycle detected in dependency graph; returning original order.")
        sorted_indices = subgraph_indices

    # Return subgraphs in topological order
    return [selected_subgraphs[i] for i in sorted_indices]


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


def abstract_computation_graph(original_graph, weight_cap=None):
    """
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
        # Weights memory footprint
        size = 0
        for op in node.constant_operands:
            size += node.operand_size_bit[op]

        intra_core_tiling = node.intra_core_tiling

        tiling = 1
        for inv_intra_tiling in intra_core_tiling:
            if "all" in inv_intra_tiling[1]:
                tiling = tiling * node.layer_dim_sizes[inv_intra_tiling[0]]
            else:
                tiling = tiling * int(inv_intra_tiling[1])

        abstracted_graph.add_node(
            node,
            mem_size=int(size / len(node.possible_core_allocation)),
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

    def check_node_types(self, subgraph):
        """
        Check the graph for node types constraints that makes layer fusion difficult :
             - 3 conv or convtranspose max
             - 1 gemm or matmul max

        subgraph: Subgraph to check
        """

        subgraph_types = [node.type for node in subgraph]
        MAX_CONV = 3
        MAX_GEMM = 1
        if subgraph_types.count("conv") > MAX_CONV or subgraph_types.count("convtranspose") > MAX_CONV:
            return False
        if subgraph_types.count("gemm") + subgraph_types.count("matmul") > MAX_GEMM:
            return False

        if (subgraph_types.count("gemm") + subgraph_types.count("matmul")) > 0 and subgraph_types.count(
            "conv"
        ) + subgraph_types.count("convtranspose") > 1:
            return False
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
        for core_id in self.weight_capacities:
            core_allocated_mem = 0
            for _, node_mem_per_core in subgraphs_mem_per_cores.items():
                if core_id in node_mem_per_core:
                    core_allocated_mem += node_mem_per_core[core_id]

            if core_allocated_mem > self.weight_capacities[core_id]:
                return False

        return True

    def _is_valid_subgraph(self, subgraph, complete_graph):
        """Check if subgraph satisfies all validation constraints."""
        return (
            self.check_node_types(subgraph)
            and self.check_intra_core_tiling(subgraph)
            and self.check_memory_constraint(subgraph, complete_graph)
        )

    def _get_outgoing_sources(self, complete_graph, current_nodes):
        """Get nodes that have outgoing edges from the subgraph."""
        outgoing_sources = set()
        for n in current_nodes:
            has_external_successor = complete_graph.out_degree(n) == 0 or any(
                neighbor not in current_nodes for neighbor in complete_graph.successors(n)
            )
            if has_external_successor:
                outgoing_sources.add(n)
        return outgoing_sources

    def find_valid_subgraphs_bfs(self, complete_graph, max_size=None):
        """
        Find all connected subgraphs of complete_graph using BFS expansion that satisfy:
        - check_intra_core_tiling(subgraph)
        - check_memory_constraint(subgraph, complete_graph)

        Parameters
        ----------
        complete_graph : nx.DiGraph
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

        for start_node in complete_graph.nodes:
            queue = deque([{start_node}])

            while queue:
                current_nodes = queue.popleft()
                frozen = frozenset(current_nodes)

                if frozen in visited_sets:
                    continue
                visited_sets.add(frozen)

                subG = complete_graph.subgraph(current_nodes)

                # Run all checks together
                if not self._is_valid_subgraph(subG, complete_graph):
                    continue

                # Check outgoing sources constraint
                outgoing_sources = self._get_outgoing_sources(complete_graph, current_nodes)
                if len(outgoing_sources) <= 1:
                    valid_subgraphs.append(subG.copy())

                # Expand if below size limit
                if max_size and len(current_nodes) >= max_size:
                    continue

                # Expand by adding neighbors
                neighbors = set()
                for node in current_nodes:
                    neighbors.update(complete_graph.successors(node))
                    neighbors.update(complete_graph.predecessors(node))

                # Only consider neighbors not already in the subgraph
                neighbors -= current_nodes

                for n in neighbors:
                    queue.append(current_nodes | {n})

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
        We use a Backtracking approach to find all possible subgraphs, then we find all possible complete graph
        and we select the best (still using backtracking)
        """

        new_graph = abstract_computation_graph(self.workload, self.weight_capacities)

        valid_subgraphs = self.find_valid_subgraphs_bfs(new_graph, max_size=6)
        solution = ilp_min_subgraphs_gurobi(new_graph, valid_subgraphs + single_node_subgraphs(new_graph))

        sorted_solution = topological_sort(solution, new_graph)
        print(sorted_solution)
        return [tuple(sorted([node.id for node in subgraph])) for subgraph in sorted_solution]

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
