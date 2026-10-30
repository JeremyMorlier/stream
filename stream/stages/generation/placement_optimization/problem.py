"""The data of the placement optimization (`docs/source/schedule_optimization_formulation.md`, section 5),
independent of the solver.

Everything is integer-scaled the way both backends model it: latency in cycles, energy in `ENERGY_SCALE` pJ, area
in `1 / AREA_SCALE` mm^2. `PlacementProblem.from_space` builds it from a `ScheduleSpace`; the tests build it
directly.
"""

import logging
import math
import pickle
from dataclasses import MISSING, dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import networkx as nx

from stream.hardware.architecture.core_generator import build_core_from_dict
from stream.hardware.architecture.noc.interconnect_area import (
    ROUTER_EXTRA_PORTS,
    ROUTER_GE_PER_BIT,
    ROUTER_GE_PER_PORT,
    ROUTER_GE_PER_PORT_BIT,
    InterconnectAreaModel,
)
from stream.stages.estimation.zigzag_core_mapping_estimation import offchip_operands_for
from stream.stages.generation.schedule_evaluation import tile_key
from stream.stages.generation.schedule_optimization import AREA_SCALE, ENERGY_SCALE, SIDE_SCALE, LinkTransfer
from stream.stages.generation.schedule_space import ScheduleSpace
from stream.workload.computation.computation_node import ComputationNode

logger = logging.getLogger(__name__)

REFETCH_DENOMINATOR = 10**6


@dataclass(frozen=True)
class InterconnectCoefficients:
    """The interconnect area of one slot in the model's integer units (see `interconnect_area`): with `k` links of
    total width `W`, on a core of side `s` (in `SIDE_SCALE` units per mm),
    `ge * (per_used + per_port k + per_port_bit k W + per_bit W) + wire * (s (W + half_fixed_wires k) + s bus_wires)`.
    """

    ge: int  # area units per gate equivalent
    wire: int  # area units per wire and per side unit
    per_used: int  # GE of a used slot's router without links (its bus and core ports)
    per_port: int  # GE per link port
    per_port_bit: int  # GE per link port and per bit of total width
    per_bit: int  # GE per bit of total width
    half_fixed_wires: int  # wires a link runs along each end, beyond its width
    bus_wires: int  # wires of the bus along each core's side
    max_width: int  # widest link

    @classmethod
    def of(cls, model: InterconnectAreaModel) -> "InterconnectCoefficients":
        extra_width = ROUTER_EXTRA_PORTS * model.bus_width
        return cls(
            ge=round(model.mm2_per_ge * AREA_SCALE),
            wire=round(model.mm_per_wire * AREA_SCALE / SIDE_SCALE),
            per_used=(
                ROUTER_GE_PER_PORT * ROUTER_EXTRA_PORTS
                + ROUTER_GE_PER_PORT_BIT * ROUTER_EXTRA_PORTS * extra_width
                + ROUTER_GE_PER_BIT * extra_width
            ),
            per_port=ROUTER_GE_PER_PORT + ROUTER_GE_PER_PORT_BIT * extra_width,
            per_port_bit=ROUTER_GE_PER_PORT_BIT,
            per_bit=ROUTER_GE_PER_PORT_BIT * ROUTER_EXTRA_PORTS + ROUTER_GE_PER_BIT,
            half_fixed_wires=round(model.wires(0) / 2),
            bus_wires=round(model.wires(model.bus_width)),
            max_width=model.link_width_max,
        )


@dataclass
class OffchipLoad:
    """Tensors Stream loads from off-chip (constant operands, graph inputs) that the same groups use, each group
    needing them first in the same task. A slot loads them once, however many of their groups it hosts."""

    users: frozenset[int]
    first_task: dict[int, int]  # user group -> the task of that group that needs them first
    bits: int
    # Design of the slot receiving them -> energy units of the load: the DRAM read plus the write into the
    # receiving core's top memory, as `Accelerator.get_memory_energy_cost_of_transfer` counts it.
    energy: dict[str, int] = field(default_factory=dict)


@dataclass
class PlacementProblem:
    """Groups (the placement units: one per layer and inter-core slice), their tasks (the timing units, in chain
    order), and every cost the model needs, per design."""

    groups: list[int]
    layer_of_group: dict[int, int]
    design_keys: list[str]
    area_of: dict[str, float]  # mm^2
    tasks: list[tuple[int, int]]  # (group, index in the group's chain)
    task_tiles: list[list[Any]]  # tiles of each task (anything with `id` and `sub_id`), in sub id order
    duration: list[dict[str, int]]  # task -> design -> cycles
    group_energy: dict[int, dict[str, int]]  # group -> design -> energy units
    dependencies: list[tuple[int, int]]  # task precedences across groups (transitively reduced)
    chains: list[tuple[int, int]]  # consecutive tasks of one group
    max_cores: int | None = None
    allow_sibling_merge: bool = True
    # Unordered pair of communicating groups -> the width its link needs (traffic-derived), used as is when the
    # widths are not decisions, and as the hint when they are.
    group_link_widths: dict[tuple[int, int], int] = field(default_factory=dict)
    transfers: list[LinkTransfer] = field(default_factory=list)
    offchip_bits: dict[frozenset[int], int] = field(default_factory=dict)  # groups using a tensor -> its bits
    load_bits: list[int] = field(default_factory=list)  # bits each task loads first in its group (timed bus)
    bus_bandwidth: int = 1
    interconnect_model: InterconnectAreaModel | None = None
    width_levels: list[int] = field(default_factory=list)
    designs: dict[str, dict[str, Any]] = field(default_factory=dict)  # design key -> core dict (measurement only)
    dropped_dependencies: int = 0
    # The memory model (`from_space(memory_model=True)`), used by the CP-SAT refinements `shared_loads`,
    # `comm_energy` and `capacity` (see `cpsat_backend`).
    offchip_loads: list[OffchipLoad] = field(default_factory=list)
    # Per transfer: (sender's design -> energy units of reading it, receiver's design -> of writing it).
    transfer_energy: list[tuple[dict[str, int], dict[str, int]]] = field(default_factory=list)
    resident_capacity: dict[str, int] = field(default_factory=dict)  # design -> bits its top memories hold loads in
    refetch_pj_per_bit: float = 0.0  # a re-fetched bit: DRAM read and the costliest top-memory write
    stack_of_group: dict[int, int] = field(default_factory=dict)  # group -> index of its layer stack
    # (group, design) pairs where one of the group's tiles needs more than a top memory of the design holds, as
    # Stream's scheduler fills it (`_working_set_overflows`); the CP-SAT `capacity` refinement forbids them.
    forbidden: set[tuple[int, str]] = field(default_factory=set)

    # ------------------------------------------------------------------------------------- derived ----------

    @property
    def nb_slots(self) -> int:
        nb_groups = len(self.groups)
        return nb_groups if self.max_cores is None else max(1, min(self.max_cores, nb_groups))

    def slots_of(self, group: int) -> range:
        """Slots group `group` may use: the `i`-th group (in `groups` order) only slots `<= i`, which removes the
        relabelling symmetry, and never more than `nb_slots`."""
        index = self.groups.index(group)
        return range(min(index, self.nb_slots - 1) + 1)

    def can_share(self, g: int, h: int) -> bool:
        return g != h and (self.allow_sibling_merge or self.layer_of_group[g] != self.layer_of_group[h])

    @property
    def sibling_pairs(self) -> list[tuple[int, int]]:
        """Pairs of groups of one layer, which may not share a slot unless `allow_sibling_merge`."""
        if self.allow_sibling_merge:
            return []
        return [
            (a, b)
            for i, a in enumerate(self.groups)
            for b in self.groups[i + 1 :]
            if self.layer_of_group[a] == self.layer_of_group[b]
        ]

    @property
    def interconnect(self) -> InterconnectCoefficients | None:
        model = self.interconnect_model
        return InterconnectCoefficients.of(model) if model is not None and model.enabled else None

    @property
    def scaled_area(self) -> dict[str, int]:
        return {key: round(self.area_of[key] * AREA_SCALE) for key in self.design_keys}

    @property
    def side_of(self) -> dict[str, int]:
        return {key: round(math.sqrt(max(0.0, self.area_of[key])) * SIDE_SCALE) for key in self.design_keys}

    @property
    def previous_of(self) -> dict[int, int]:
        """Task -> the task before it in its group's chain."""
        return {b: a for a, b in self.chains}

    def load_durations(self) -> list[int]:
        return [math.ceil(bits / self.bus_bandwidth) for bits in self.load_bits]

    def horizon(self, offchip_bus: str, link_transfers: str) -> int:
        """A makespan every schedule fits in: every task, load and transfer one after the other, at its slowest."""
        horizon = sum(max(durations.values()) for durations in self.duration)
        if offchip_bus == "timed":
            horizon += sum(self.load_durations())
        if offchip_bus != "none":
            horizon = max(horizon, self.max_bus_time)
        if link_transfers == "timed" and self.width_levels:
            horizon += sum(transfer.duration(min(self.width_levels)) for transfer in self.transfers)
        return max(1, horizon)

    def unordered_task_pairs(self) -> list[tuple[int, int]]:
        """Task pairs `(k, k')`, `k < k'`, of two groups that may share a slot and that no chain of precedences
        orders: on a shared slot, the model has to order them itself."""
        nb_tasks = len(self.tasks)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(nb_tasks))
        graph.add_edges_from(self.dependencies + self.chains)
        # Descendants of every task as a bit set, in reverse topological order.
        reach = [0] * nb_tasks
        for task in reversed(list(nx.topological_sort(graph))):
            bits = 0
            for child in graph.successors(task):
                bits |= reach[child] | (1 << child)
            reach[task] = bits
        pairs = []
        for k in range(nb_tasks):
            g = self.tasks[k][0]
            for k2 in range(k + 1, nb_tasks):
                h = self.tasks[k2][0]
                if not self.can_share(g, h):
                    continue
                if (reach[k] >> k2) & 1 or (reach[k2] >> k) & 1:
                    continue
                pairs.append((k, k2))
        return pairs

    def variable_offchip_users(self) -> dict[frozenset[int], int]:
        """Off-chip tensors whose number of loads depends on the placement: used by several groups, two of which
        may share a slot (a core loads a tensor once). The others cost `bits * len(users)` whatever happens."""
        return {
            users: bits
            for users, bits in self.offchip_bits.items()
            if len(users) > 1 and any(self.can_share(a, b) for a in users for b in users if a < b)
        }

    @property
    def max_bus_time(self) -> int:
        """Bus cycles if every tensor were loaded once per group using it."""
        return math.ceil(sum(bits * len(users) for users, bits in self.offchip_bits.items()) / self.bus_bandwidth)

    @property
    def constant_offchip_bits(self) -> int:
        variable = self.variable_offchip_users()
        return sum(bits * len(users) for users, bits in self.offchip_bits.items() if users not in variable)

    def is_shared(self, load: OffchipLoad) -> bool:
        """Whether two of the load's groups may share a slot, so it may be loaded fewer times than it has users."""
        users = sorted(load.users)
        return any(self.can_share(a, b) for i, a in enumerate(users) for b in users[i + 1 :])

    def load_energy(self, slot_of_group: dict[int, int], design_of_slot: dict[int, str]) -> int:
        """Energy of the off-chip loads: each load once per slot hosting one of its groups."""
        total = 0
        for load in self.offchip_loads:
            slots = [slot_of_group[g] for g in load.users]
            total += sum(load.energy[design_of_slot[slot]] for slot in (set(slots) if self.is_shared(load) else slots))
        return total

    def transfer_energy_of(self, slot_of_group: dict[int, int], design_of_slot: dict[int, str]) -> int:
        """Energy of the transfers between groups on different slots (memory reads and writes; links are free)."""
        total = 0
        for transfer, (send, receive) in zip(self.transfers, self.transfer_energy, strict=True):
            p, q = slot_of_group[transfer.producer], slot_of_group[transfer.consumer]
            if p != q:
                total += send[design_of_slot[p]] + receive[design_of_slot[q]]
        return total

    def resident_bits(self, slot_of_group: dict[int, int]) -> dict[tuple[int, int], int]:
        """(slot, layer stack) -> bits of the loads kept on the slot while the stack runs."""
        resident: dict[tuple[int, int], int] = {}
        for load in self.offchip_loads:
            stack = self.stack_of_group.get(min(load.users), 0)
            slots = [slot_of_group[g] for g in load.users]
            for slot in set(slots) if self.is_shared(load) else slots:
                resident[slot, stack] = resident.get((slot, stack), 0) + load.bits
        return resident

    def overflow_bits(self, slot_of_group: dict[int, int], design_of_slot: dict[int, str]) -> int:
        """Bits of loads beyond what their slot's design can keep, summed over slots and stacks: each is re-fetched
        at least once."""
        return sum(
            max(0, bits - self.resident_capacity[design_of_slot[slot]])
            for (slot, _stack), bits in self.resident_bits(slot_of_group).items()
        )

    @property
    def refetch_ratio(self) -> tuple[int, int]:
        """Energy units per re-fetched bit as an integer ratio, so the model and `evaluate` round alike."""
        return round(self.refetch_pj_per_bit / ENERGY_SCALE * REFETCH_DENOMINATOR), REFETCH_DENOMINATOR

    def refetch_energy(self, overflow_bits: int) -> int:
        numerator, denominator = self.refetch_ratio
        return -(-overflow_bits * numerator // denominator)

    def size_report(self) -> dict[str, int]:
        return {
            "groups": len(self.groups),
            "slots": self.nb_slots,
            "tasks": len(self.tasks),
            "designs": len(self.design_keys),
            "dependencies": len(self.dependencies),
            "dropped_dependencies": self.dropped_dependencies,
            "transfers": len(self.transfers),
            "communicating_pairs": len(self.group_link_widths),
        }

    def evaluate(  # noqa: PLR0913
        self,
        slot_of_group: dict[int, int],
        design_of_slot: dict[int, str],
        starts: list[int],
        group_link_widths: dict[tuple[int, int], int] | None,
        offchip_bus: str,
        comm_energy: bool = False,
        capacity: str = "none",
    ) -> dict[str, int]:
        """The objectives of a solution, in the model's integer units, computed from its decisions alone. Model
        variables that are only bounded from below (the latency in the MILP, link literals, the area's products)
        can sit above these values in a solve that does not minimize them; these are what the model's objectives
        would be with every such variable at its least value, so a bound set to them still holds."""
        design_of_group = {g: design_of_slot[slot] for g, slot in slot_of_group.items()}
        ends = [
            start + self.duration[task][design_of_group[g]]
            for task, (start, (g, _index)) in enumerate(zip(starts, self.tasks, strict=True))
        ]
        latency = max(ends, default=0)
        overflow = self.overflow_bits(slot_of_group, design_of_slot) if capacity == "overflow" else 0
        if offchip_bus != "none":
            variable = self.variable_offchip_users()
            bits = self.constant_offchip_bits + sum(
                bits * len({slot_of_group[g] for g in users}) for users, bits in variable.items()
            )
            latency = max(latency, math.ceil((bits + overflow) / self.bus_bandwidth))
        energy = sum(self.group_energy[g][design_of_group[g]] for g in self.groups)
        if comm_energy:
            energy += self.load_energy(slot_of_group, design_of_slot)
            energy += self.transfer_energy_of(slot_of_group, design_of_slot)
        energy += self.refetch_energy(overflow)
        used = set(slot_of_group.values())
        scaled_area = self.scaled_area
        area = sum(scaled_area[design_of_slot[p]] for p in used)
        ic = self.interconnect
        if ic is not None:
            widths = group_link_widths if group_link_widths is not None else self.group_link_widths
            links: dict[tuple[int, int], int] = {}
            for (g, h), width in widths.items():
                p, q = slot_of_group[g], slot_of_group[h]
                if p != q:
                    pair = (min(p, q), max(p, q))
                    links[pair] = max(links.get(pair, 0), width)
            side_of = self.side_of
            for p in used:
                side = side_of[design_of_slot[p]]
                ports = sum(1 for pair in links if p in pair)
                total = sum(width for pair, width in links.items() if p in pair)
                router = ic.per_used + ic.per_port * ports + ic.per_port_bit * ports * total + ic.per_bit * total
                area += ic.ge * router
                area += ic.wire * side * (total + ic.half_fixed_wires * ports + ic.bus_wires)
        return {"latency": latency, "energy": energy, "area": area}

    def save(self, path: str) -> None:
        """Pickle the problem without its workload objects (tiles become `(id, sub_id)` namespaces, designs are
        dropped), so it can be solved again offline by either backend."""
        light = replace(
            self,
            task_tiles=[[SimpleNamespace(id=t.id, sub_id=t.sub_id) for t in members] for members in self.task_tiles],
            designs={},
        )
        with open(path, "wb") as f:
            pickle.dump(light, f)

    @staticmethod
    def load(path: str) -> "PlacementProblem":
        """A problem `save` wrote; fields added since it was written take their defaults."""
        with open(path, "rb") as f:
            problem = pickle.load(f)  # noqa: S301 -- written by `save`
        for name, spec in PlacementProblem.__dataclass_fields__.items():
            if name not in problem.__dict__:
                default = spec.default_factory() if spec.default_factory is not MISSING else spec.default
                setattr(problem, name, default)
        return problem

    # ------------------------------------------------------------------------------------- from a space -----

    @classmethod
    def from_space(  # noqa: PLR0912, PLR0915
        cls,
        space: ScheduleSpace,
        *,
        tasks_per_group: int | None,
        max_cores: int | None = None,
        allow_sibling_merge: bool = True,
        memory_model: bool = False,
        capacity_filter: bool = False,
    ) -> "PlacementProblem":
        """The problem of a schedule space with its full cost matrix. Groups are the unrolled cores (one per layer
        and inter-core slice). Each group's tiles are split, in tile order, into `tasks_per_group` contiguous tasks
        (`None`: one task per tile). As in `ScheduleOptimizationStage._build_problem`, task dependencies come from
        the tile edges, those inside mutually dependent task sets are dropped (fusion lets them interleave), and
        the rest are transitively reduced. With `memory_model`, the problem also gets what the CP-SAT refinements
        `comm_energy` and `capacity` need (`_add_memory_model`), and with `capacity_filter` the (group, design)
        pairs the scheduler cannot fit (`_forbidden_pairs`)."""
        groups: list[int] = list(space.unrolled_core_ids)
        layer_of_group = {core: node_id for node_id, cores in space.core_ids_of_node.items() for core in cores}
        needed_shapes = sorted(set(space.shape_class_of_tile.values()))
        designs = {
            key: core_dict
            for key, core_dict in space.distinct_designs().items()
            if all((shape, key) in space.costs.runtime for shape in needed_shapes)
        }
        if not designs:
            raise ValueError("PlacementProblem: no design is cost-modelled on every tile shape.")
        design_keys = sorted(designs)
        group_link_widths: dict[tuple[int, int], int] = space.link_widths() or {}

        tiles = [tile for tile in space.reference_workload.node_list if isinstance(tile, ComputationNode)]
        tiles_of_group: dict[int, list[Any]] = {group: [] for group in groups}
        for tile in tiles:
            tiles_of_group[space.unrolled_core_of_tile[tile_key(tile)]].append(tile)
        tasks: list[tuple[int, int]] = []
        task_tiles: list[list[Any]] = []
        task_of_tile: dict[tuple[int, int], int] = {}
        for group in groups:
            ordered = sorted(tiles_of_group[group], key=lambda tile: tile.sub_id)
            nb_tasks = len(ordered) if tasks_per_group is None else max(1, min(tasks_per_group, len(ordered)))
            size = math.ceil(len(ordered) / nb_tasks) if ordered else 1
            for index in range(nb_tasks):
                members = ordered[index * size : (index + 1) * size]
                if not members:
                    continue
                for tile in members:
                    task_of_tile[tile_key(tile)] = len(tasks)
                tasks.append((group, index))
                task_tiles.append(members)

        runtime, energy = space.costs.runtime, space.costs.energy
        shape_of = space.shape_class_of_tile
        duration = [
            {key: max(1, math.ceil(sum(runtime[(shape_of[tile_key(t)], key)] for t in members))) for key in design_keys}
            for members in task_tiles
        ]
        group_energy = {
            group: {
                key: round(sum(energy[(shape_of[tile_key(t)], key)] for t in tiles_of_group[group]) / ENERGY_SCALE)
                for key in design_keys
            }
            for group in groups
        }

        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(tasks)))
        bits_of_edge: dict[tuple[int, int], int] = {}
        # Bits of each edge per (sender, receiver) memory operand, for the transfer energy.
        operand_bits_of_edge: dict[tuple[int, int], dict[tuple[str, str], int]] = {}
        for producer, consumer, data in space.reference_workload.edges(data=True):
            if not (isinstance(producer, ComputationNode) and isinstance(consumer, ComputationNode)):
                continue
            edge = (task_of_tile[tile_key(producer)], task_of_tile[tile_key(consumer)])
            if edge[0] == edge[1]:
                continue
            graph.add_edge(*edge)
            if data.get("bits", 0) > 0:
                bits_of_edge[edge] = bits_of_edge.get(edge, 0) + data["bits"]
                operands = _edge_memory_operands(producer, consumer)
                per_operand = operand_bits_of_edge.setdefault(edge, {})
                per_operand[operands] = per_operand.get(operands, 0) + data["bits"]
        component_of = {
            task: index for index, component in enumerate(nx.strongly_connected_components(graph)) for task in component
        }
        kept = [(a, b) for a, b in graph.edges() if component_of[a] != component_of[b]]
        chains = [(i, i + 1) for i in range(len(tasks) - 1) if tasks[i][0] == tasks[i + 1][0]]
        dag = nx.DiGraph()
        dag.add_nodes_from(range(len(tasks)))
        dag.add_edges_from(kept + chains)
        chain_set = set(chains)
        dependencies = [edge for edge in nx.transitive_reduction(dag).edges() if edge not in chain_set]

        # Transfers: one per (producer group, consumer task), from the unreduced kept edges, leaving once the
        # producer group's last task feeding the consumer has ended. Traffic between groups without a link (inside
        # one layer) is not timed.
        merged: dict[tuple[int, int], list[int]] = {}
        operand_bits_of_transfer: dict[tuple[int, int], dict[tuple[str, str], int]] = {}
        unlinked_bits = 0
        kept_set = set(kept)
        for (a, b), bits in bits_of_edge.items():
            g, h = tasks[a][0], tasks[b][0]
            if g == h or (a, b) not in kept_set:
                continue
            if (min(g, h), max(g, h)) not in group_link_widths:
                unlinked_bits += bits
                continue
            entry = merged.setdefault((g, b), [a, 0])
            entry[0] = max(entry[0], a)
            entry[1] += bits
            per_operand = operand_bits_of_transfer.setdefault((g, b), {})
            for operands, operand_bits in operand_bits_of_edge[a, b].items():
                per_operand[operands] = per_operand.get(operands, 0) + operand_bits
        transfers = [
            LinkTransfer(
                producer=g,
                consumer=tasks[b][0],
                source=source,
                target=b,
                bits=bits,
                splits=max(1, task_tiles[b][0].get_total_inter_core_splits()),
            )
            for (g, b), (source, bits) in sorted(merged.items())
        ]
        if unlinked_bits:
            logger.warning(f"PlacementProblem: {unlinked_bits:.4g} bits between groups without a link are not timed.")

        offchip_bits, load_bits, load_tensors = _offchip_traffic(space, tiles, tasks, task_tiles)
        problem = cls(
            groups=groups,
            layer_of_group=layer_of_group,
            design_keys=design_keys,
            area_of={key: space.costs.area[key] for key in design_keys},
            tasks=tasks,
            task_tiles=task_tiles,
            duration=duration,
            group_energy=group_energy,
            dependencies=dependencies,
            chains=chains,
            max_cores=max_cores,
            allow_sibling_merge=allow_sibling_merge,
            group_link_widths=group_link_widths,
            transfers=transfers,
            offchip_bits=offchip_bits,
            load_bits=load_bits,
            bus_bandwidth=max(1, round(float(space.bus_connection["bandwidth"]))),
            interconnect_model=space.interconnect_model,
            width_levels=space.interconnect_model.width_levels,
            designs=designs,
            dropped_dependencies=graph.number_of_edges() - len(kept),
            offchip_loads=_bundle_loads(load_tensors),
        )
        if memory_model:
            transfer_operand_bits = [operand_bits_of_transfer[g, b] for (g, b) in sorted(merged)]
            _add_memory_model(problem, space, transfer_operand_bits, load_tensors)
        if capacity_filter:
            problem.forbidden = _forbidden_pairs(space, tiles, design_keys, designs)
        logger.info(f"PlacementProblem: {problem.size_report()}.")
        return problem


def _offchip_traffic(
    space: ScheduleSpace, tiles: list[ComputationNode], tasks: list[tuple[int, int]], task_tiles: list[list[Any]]
) -> tuple[dict[frozenset[int], int], list[int], dict[int, "LoadedTensor"]]:
    """The tensors Stream places off-chip at the start (constant operands, and the inputs of graph sources), keyed
    by the groups using them (a core loads one once, recognized by `equality_hash`), the bits each task loads
    first within its group, and each tensor on its own. See `ScheduleOptimizationStage._build_offchip_traffic`."""
    workload = space.reference_workload

    def loaded_tensors(tile: ComputationNode) -> list[tuple[int, int, str]]:
        loaded_ops = tile.constant_operands + tile.partially_constant_operands
        is_source = len(workload.in_edges(tile)) == 0
        return [
            (tensor.equality_hash, tensor.size, str(tile.memory_operand_links.layer_to_mem_op(op)))
            for op, tensor in tile.operand_tensors.items()
            if op != tile.output_operand and (op in loaded_ops or is_source)
        ]

    tensors: dict[int, LoadedTensor] = {}
    for tile in tiles:
        group = space.unrolled_core_of_tile[tile_key(tile)]
        for key, size, operand in loaded_tensors(tile):
            tensors.setdefault(key, LoadedTensor(bits=size, operand=operand)).users.add(group)
    offchip_bits: dict[frozenset[int], int] = {}
    for tensor in tensors.values():
        users = frozenset(tensor.users)
        offchip_bits[users] = offchip_bits.get(users, 0) + tensor.bits

    load_bits: list[int] = []
    seen_of_group: dict[int, set[int]] = {}
    for task, ((group, _index), members) in enumerate(zip(tasks, task_tiles, strict=True)):
        seen = seen_of_group.setdefault(group, set())
        bits = 0
        for tile in members:
            for key, size, _operand in loaded_tensors(tile):
                if key not in seen:
                    seen.add(key)
                    bits += size
                    tensors[key].first_task[group] = task
        load_bits.append(bits)
    return offchip_bits, load_bits, tensors


@dataclass
class LoadedTensor:
    """One tensor Stream loads from off-chip: its size, the memory operand it lands in, and its groups with the
    task of each that needs it first."""

    bits: int
    operand: str
    users: set[int] = field(default_factory=set)
    first_task: dict[int, int] = field(default_factory=dict)


def _bundle_loads(tensors: dict[int, LoadedTensor]) -> list[OffchipLoad]:
    """The tensors grouped into `OffchipLoad`s: the same groups, each needing them first in the same task."""
    bundles: dict[tuple, OffchipLoad] = {}
    for tensor in tensors.values():
        signature = tuple(sorted(tensor.first_task.items()))
        if signature not in bundles:
            bundles[signature] = OffchipLoad(users=frozenset(tensor.users), first_task=dict(tensor.first_task), bits=0)
        bundles[signature].bits += tensor.bits
    return list(bundles.values())


def _edge_memory_operands(producer: ComputationNode, consumer: ComputationNode) -> tuple[str, str]:
    """The memory operands a tile edge leaves the producer's core from and lands in on the consumer's."""
    sent = producer.memory_operand_links.layer_to_mem_op(producer.output_operand)
    received = next(
        (op for op, source in consumer.input_operand_source.items() if source == producer.id),
        consumer.input_operands[0],
    )
    return str(sent), str(consumer.memory_operand_links.layer_to_mem_op(received))


def _per_bit(instance: Any, cost: float) -> float:
    """pJ per bit of an access cost, at the instance's widest port (`get_memory_energy_cost_of_transfer`)."""
    return cost / instance.ports[0].bw_max


def _add_memory_model(
    problem: PlacementProblem,
    space: ScheduleSpace,
    transfer_operand_bits: list[dict[tuple[str, str], int]],
    tensors: dict[int, LoadedTensor],
) -> None:
    """Fill in the memory side of `problem` from the designs' memories (CACTI costs, through real cores) and the
    off-chip core's: the energy of each load and transfer per design, how many loaded bits each design keeps, and
    each group's layer stack."""
    offchip = space.accelerator.get_offchip_core()
    dram = offchip.get_top_memory_instance(next(iter(offchip.memory_hierarchy.get_operands())))
    dram_read = _per_bit(dram, dram.r_cost)
    loaded_operands = {tensor.operand for tensor in tensors.values()}
    read: dict[str, dict[str, float]] = {}
    write: dict[str, dict[str, float]] = {}
    for key in problem.design_keys:
        core = build_core_from_dict(problem.designs[key], core_id=0)
        instances = {str(op): core.get_top_memory_instance(op) for op in core.memory_hierarchy.get_operands()}
        read[key] = {op: _per_bit(instance, instance.r_cost) for op, instance in instances.items()}
        write[key] = {op: _per_bit(instance, instance.w_cost) for op, instance in instances.items()}
        holding = {id(instance): instance for op, instance in instances.items() if op in loaded_operands}
        problem.resident_capacity[key] = sum(instance.size for instance in holding.values())

    by_signature: dict[tuple, list[LoadedTensor]] = {}
    for tensor in tensors.values():
        by_signature.setdefault(tuple(sorted(tensor.first_task.items())), []).append(tensor)
    for load in problem.offchip_loads:
        members = by_signature[tuple(sorted(load.first_task.items()))]
        load.energy = {
            key: round(sum(t.bits * (dram_read + write[key][t.operand]) for t in members) / ENERGY_SCALE)
            for key in problem.design_keys
        }
    problem.transfer_energy = [
        (
            {key: round(sum(b * read[key][sent] for (sent, _r), b in bits.items()) / ENERGY_SCALE) for key in read},
            {key: round(sum(b * write[key][rcv] for (_s, rcv), b in bits.items()) / ENERGY_SCALE) for key in write},
        )
        for bits in transfer_operand_bits
    ]
    problem.refetch_pj_per_bit = dram_read + max(max(costs.values()) for costs in write.values())
    stacks = space.estimation_kwargs.get("layer_stacks") or []
    stack_of_layer = {layer: index for index, stack in enumerate(stacks) for layer in stack}
    problem.stack_of_group = {g: stack_of_layer.get(layer, 0) for g, layer in problem.layer_of_group.items()}


def _working_set_overflows(tile: ComputationNode, workload: Any, core: Any, layer_stacks: list) -> bool:
    """Whether Stream's scheduler cannot fit `tile` on `core`: what `CoalaScheduler.get_tensors_needed_for_node`
    brings into a top memory -- the constant operands, the *whole* output tensor of every predecessor tile (not
    just the slice the tile reads) -- plus the tile's own output, leaving out the operands `offchip_operands_for`
    streams from DRAM, exceeds that memory. `memory_hierarchy_violations` only checks the tile's own operand sizes,
    so a design can pass it and still fail here, when a producer's tiles are larger than its consumer's."""
    too_large = set(offchip_operands_for(tile, core, layer_stacks))
    needed: list[tuple[Any, Any]] = []  # (memory operand, tensor)
    for layer_op in tile.constant_operands:
        needed.append((tile.memory_operand_links.layer_to_mem_op(layer_op), tile.operand_tensors[layer_op]))
    for pred, _, data in workload.in_edges(tile, data=True):
        if pred.id != tile.id and isinstance(pred, ComputationNode):
            memory_op = tile.memory_operand_links.layer_to_mem_op(data["operand"])
            needed.append((memory_op, pred.operand_tensors[pred.output_operand]))
    output = tile.memory_operand_links.layer_to_mem_op(tile.output_operand)
    needed.append((output, tile.operand_tensors[tile.output_operand]))
    stored: dict[int, tuple[Any, dict[int, int]]] = {}  # top instance -> (instance, tensor id -> bits)
    for memory_op, tensor in needed:
        if memory_op in too_large:
            continue
        instance = core.get_top_memory_instance(memory_op)
        stored.setdefault(id(instance), (instance, {}))[1][id(tensor)] = tensor.size
    return any(sum(sizes.values()) > instance.size for instance, sizes in stored.values())


def _forbidden_pairs(
    space: ScheduleSpace, tiles: list[ComputationNode], design_keys: list[str], designs: dict[str, dict[str, Any]]
) -> set[tuple[int, str]]:
    """(group, design) pairs where one of the group's tiles overflows the design (`_working_set_overflows`)."""
    workload = space.reference_workload
    layer_stacks = space.estimation_kwargs["layer_stacks"]
    cores = {key: build_core_from_dict(designs[key], core_id=0) for key in design_keys}
    forbidden: set[tuple[int, str]] = set()
    for tile in tiles:
        group = space.unrolled_core_of_tile[tile_key(tile)]
        for key in design_keys:
            if (group, key) not in forbidden and _working_set_overflows(tile, workload, cores[key], layer_stacks):
                forbidden.add((group, key))
    return forbidden
