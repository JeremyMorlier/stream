from dataclasses import dataclass
from typing import Any

import numpy as np

from stream.workload.steady_state.node import SteadyStateNode
from stream.workload.steady_state.tensor import SteadyStateTensor
from stream.workload.steady_state.workload import SteadyStateWorkload


@dataclass(frozen=True)
class TensorLifetime:
    tensor: SteadyStateTensor
    memory: Any  # Typically the chosen_resource_allocation (e.g., core or memory id)
    start: int  # Time when tensor is produced (on its memory)
    end: int  # Time after last consumption (on its memory)


class TensorLifetimeAnalyzer:
    """
    Analyzes tensor lifetimes and memory usage in a SteadyStateWorkload, accounting for parallelism across resources.
    """

    def __init__(self, workload: SteadyStateWorkload):
        self.workload = workload
        self.tensor_nodes: list[SteadyStateTensor] = workload.tensor_nodes
        self.lifetimes: list[TensorLifetime] = []
        self.node_start_times: dict[SteadyStateNode, int] = {}
        self.node_finish_times: dict[SteadyStateNode, int] = {}
        self.resource_times: dict[Any, int] = {}
        self._schedule_nodes_per_resource()
        self._analyze_lifetimes()

    def _schedule_nodes_per_resource(self):
        """
        Assigns a finish time to each node, per resource, simulating parallel execution.
        Assumes each node has a .runtime property (default 1 if missing).
        """
        # Group nodes by their chosen_resource_allocation
        resource_to_nodes: dict[Any, list[SteadyStateNode]] = {}
        for node in self.workload.nodes():
            resource = getattr(node, "chosen_resource_allocation", None)
            resource_to_nodes.setdefault(resource, []).append(node)
        # For each resource, schedule nodes in topological order (local to that resource)
        for resource in resource_to_nodes.keys():
            self.resource_times[resource] = 0
        for node in self.workload.topological_sort():
            runtime = getattr(node, "runtime", None)
            if runtime is None:
                runtime = 1
            # Start time is max of finish times of all predecessors (on any resource)
            preds = list(self.workload.predecessors(node))
            if preds:
                start_time = max(self.node_finish_times.get(pred, 0) for pred in preds)
            else:
                start_time = 0
            start_time = max(start_time, self.resource_times[resource])
            finish_time = start_time + runtime
            self.node_start_times[node] = start_time
            self.node_finish_times[node] = finish_time
            self.resource_times[resource] = finish_time

    def _analyze_lifetimes(self):
        """
        For each tensor node, determine its production and last consumption time based on actual node finish times.
        """
        for tensor in self.tensor_nodes:
            # Produced by its predecessor (or itself if source)
            preds = list(self.workload.predecessors(tensor))
            if preds:
                prod_time = min(self.node_start_times.get(pred, 0) for pred in preds)
            else:
                prod_time = 0
            # Last consumed by its successors
            succs = list(self.workload.successors(tensor))
            if succs:
                cons_time = max(self.node_finish_times.get(succ, prod_time) for succ in succs)
            else:
                cons_time = prod_time
            mem = tensor.chosen_resource_allocation
            self.lifetimes.append(TensorLifetime(tensor, mem, prod_time, cons_time))

    def get_lifetimes_by_memory(self) -> dict[Any, list[TensorLifetime]]:
        """Group tensor lifetimes by their memory (resource)."""
        mem_map: dict[Any, list[TensorLifetime]] = {}
        for lt in self.lifetimes:
            mem_map.setdefault(lt.memory, []).append(lt)
        return mem_map

    def get_memory_usage_over_time(self, memory: Any) -> tuple[list[int], list[int]]:
        """
        For a given memory, return (times, memory_usage) lists.
        """
        events = []  # (time, size_change)
        for lt in self.get_lifetimes_by_memory().get(memory, []):
            size = getattr(lt.tensor, "size", 0)
            events.append((lt.start, size))
            events.append((lt.end + 1, -size))  # +1: dead after last use
        # Aggregate events
        events.sort()
        times = []
        mem_usage = []
        current = 0
        for t, delta in events:
            if times and t == times[-1]:
                mem_usage[-1] += delta
            else:
                times.append(t)
                mem_usage.append(current + delta)
            current = mem_usage[-1]
        return times, mem_usage

    def get_max_memory_usage(self, memory: Any) -> tuple[int, int]:
        times, usage = self.get_memory_usage_over_time(memory)
        if not usage:
            return 0, 0
        max_idx = int(np.argmax(usage))
        return times[max_idx], usage[max_idx]

    def get_alive_tensors_at(self, memory: Any, time: int) -> list[SteadyStateTensor]:
        return [lt.tensor for lt in self.get_lifetimes_by_memory().get(memory, []) if lt.start <= time <= lt.end]

    def summary(self) -> None:
        print("Tensor lifetimes by memory:")
        for mem, lifetimes in self.get_lifetimes_by_memory().items():
            print(f"\nMemory {mem}:")
            for lt in lifetimes:
                print(f"  Tensor {lt.tensor.node_name}: [{lt.start}, {lt.end}] size={getattr(lt.tensor, 'size', 0)}")
            t, m = self.get_max_memory_usage(mem)
            print(f"  Max usage: {m} at time {t}")

    def visualize(self, figsize: tuple[int, int] = (10, 6), filename_prefix: str = "tensor_lifetime") -> None:
        """Plot one Gantt chart per memory (see `stream.visualization.tensor_lifetime.plot_tensor_lifetimes`)."""
        from stream.visualization.tensor_lifetime import plot_tensor_lifetimes  # noqa: PLC0415

        plot_tensor_lifetimes(self, figsize=figsize, filename_prefix=filename_prefix)
