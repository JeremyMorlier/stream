"""The placement model with OR-Tools CP-SAT (`docs/source/schedule_optimization_formulation.md`, section 5).

Same semantics as `GurobiBackend`, written with CP-SAT's own constructs: a task is an optional interval on each
slot its group may use, present when the group sits there, and the slot's intervals do not overlap. Products of
variables (the router's ports times their width, a slot's side times its wires) are `AddMultiplicationEquality`.

By default, transfers do not contend with each other (left to the measurement): each one only delays its consumer
task by its duration at its group pair's chosen width, or not at all when both groups share a slot.

Refinements towards what Stream measures, off by default and not in `GurobiBackend` (`BackendOptions`; section 5.6
of the note):

- `shared_loads`: on the timed bus, a load that groups able to share a slot all use is one optional interval per
  slot, present when one of them sits there, instead of one per group.
- `comm_energy`: the energy counts the off-chip loads (DRAM read and top-memory write, once per slot for shared
  loads) and the transfers between groups on different slots (read at the producer's design, write at the
  consumer's). The cost-model energy only counts what a tile's own mapping accesses.
- `capacity`: `"filter"` forbids a group on a design one of its tiles overflows as the scheduler fills memories
  (`PlacementProblem.forbidden`). `"overflow"` also re-fetches once, on the bus and in energy, the loaded bits
  beyond what a slot's design keeps, per slot and layer stack.
- `link_contention`: transfers become intervals of their own, fetched on demand, sharing their link `splits` at a
  time per directed group pair (`"pair"`), or with every transfer on the same physical link (`"link"`).
"""

import logging
import math
import time
from typing import Any

from ortools.sat.python import cp_model

from stream.stages.generation.placement_optimization.backend import (
    BackendOptions,
    SolveResult,
    Status,
    make_solution,
)
from stream.stages.generation.placement_optimization.problem import PlacementProblem

logger = logging.getLogger(__name__)

# A link's bandwidth in the `link_contention="link"` cumulative: a transfer split `k` ways takes `ceil(SCALE / k)`.
LINK_SHARE_SCALE = 64

STATUS: dict[int, Status] = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
}


class CpSatBackend:
    name = "cpsat"

    def __init__(self, problem: PlacementProblem, options: BackendOptions):
        self.problem = problem
        self.options = options
        self.model = cp_model.CpModel()
        self.nb_solves = 0
        self._build()

    # --------------------------------------------------------------------------------------- model ---------

    def _check_options(self) -> None:
        pb, options = self.problem, self.options
        if options.link_contention != "none" and options.link_transfers != "timed":
            raise ValueError("CpSatBackend: `link_contention` needs `link_transfers='timed'`.")
        if options.comm_energy and (
            len(pb.transfer_energy) != len(pb.transfers) or any(not load.energy for load in pb.offchip_loads)
        ):
            raise ValueError("CpSatBackend: `comm_energy` needs the memory model (`from_space(memory_model=True)`).")
        if options.capacity == "overflow" and pb.offchip_loads and not pb.resident_capacity:
            raise ValueError("CpSatBackend: `capacity='overflow'` needs the memory model.")

    def _build(self) -> None:  # noqa: PLR0912, PLR0915
        self._check_options()
        pb, model, options = self.problem, self.model, self.options
        keys = pb.design_keys
        slots = range(pb.nb_slots)
        horizon = pb.horizon(options.offchip_bus, options.link_transfers)
        if options.capacity == "overflow":
            horizon += math.ceil(self._max_overflow_bits() / pb.bus_bandwidth)
        self.loaded_of: dict[tuple[frozenset[int], int], Any] = {}

        self.y = {(p, key): model.NewBoolVar(f"y_{p}_{key}") for p in slots for key in keys}
        self.used = [model.NewBoolVar(f"u_{p}") for p in slots]
        self.x: dict[tuple[int, int], Any] = {}
        for g in pb.groups:
            for p in pb.slots_of(g):
                self.x[g, p] = model.NewBoolVar(f"x_{g}_{p}")
            model.AddExactlyOne(self.x[g, p] for p in pb.slots_of(g))
        for p in slots:
            model.Add(sum(self.y[p, key] for key in keys) == self.used[p])
            hosted = [self.x[g, p] for g in pb.groups if (g, p) in self.x]
            model.Add(sum(hosted) >= self.used[p])
            for literal in hosted:
                model.AddImplication(literal, self.used[p])
            if p + 1 < pb.nb_slots:
                model.AddImplication(self.used[p + 1], self.used[p])
        if options.capacity != "none":
            self._forbid_overflowing_designs()
        for a, b in pb.sibling_pairs:
            for p in slots:
                if (a, p) in self.x and (b, p) in self.x:
                    model.AddBoolOr([self.x[a, p].Not(), self.x[b, p].Not()])

        # Energy and durations follow the design of the group's slot. With `comm_energy`, a group also pays for the
        # off-chip loads only it can make (those no other of their groups can share a slot with).
        group_energy = {}
        load_energy_of = self._unshared_load_energy() if options.comm_energy else {}
        for g in pb.groups:
            energies = {key: pb.group_energy[g][key] + load_energy_of.get(g, {}).get(key, 0) for key in keys}
            var = model.NewIntVar(min(energies.values()), max(energies.values()), f"e_{g}")
            for p in pb.slots_of(g):
                model.Add(var == sum(energies[key] * self.y[p, key] for key in keys)).OnlyEnforceIf(self.x[g, p])
            group_energy[g] = var
        # Tasks of one group with the same per-design durations share one duration variable.
        duration_var: dict[tuple, Any] = {}
        task_duration = []
        for (g, _index), durations in zip(pb.tasks, pb.duration, strict=True):
            signature = (g, tuple(durations[key] for key in keys))
            if signature not in duration_var:
                var = model.NewIntVar(min(durations.values()), max(durations.values()), f"d_{len(duration_var)}")
                for p in pb.slots_of(g):
                    model.Add(var == sum(durations[key] * self.y[p, key] for key in keys)).OnlyEnforceIf(self.x[g, p])
                duration_var[signature] = var
            task_duration.append(duration_var[signature])

        self.start, self.end = [], []
        intervals_of_slot: dict[int, list[Any]] = {p: [] for p in slots}
        for task, (g, _index) in enumerate(pb.tasks):
            start = model.NewIntVar(0, horizon, f"s_{task}")
            end = model.NewIntVar(0, horizon, f"c_{task}")
            model.Add(end == start + task_duration[task])
            self.start.append(start)
            self.end.append(end)
            for p in pb.slots_of(g):
                intervals_of_slot[p].append(
                    model.NewOptionalIntervalVar(start, task_duration[task], end, self.x[g, p], f"i_{task}_{p}")
                )
        for intervals in intervals_of_slot.values():
            model.AddNoOverlap(intervals)
        for a, b in pb.dependencies + pb.chains:
            model.Add(self.start[b] >= self.end[a])

        self.same_of: dict[tuple[int, int], Any] = {}
        self.width_choice: dict[tuple[int, int], dict[int, Any]] = {}
        self.group_width: dict[tuple[int, int], Any] = {}
        if options.link_transfers == "timed":
            self._add_links()

        if options.link_contention != "none":
            self._add_link_contention(horizon)

        overflow = self._add_overflow() if options.capacity == "overflow" else None
        finishes = list(self.end)
        if options.offchip_bus == "timed":
            self._add_timed_loads(horizon)
        if options.offchip_bus != "none":
            finishes.append(self._add_bus_time(overflow))
        self.latency = model.NewIntVar(0, horizon, "latency")
        model.AddMaxEquality(self.latency, finishes)
        self.energy = sum(group_energy.values())
        if options.comm_energy:
            self.energy = self.energy + self._add_shared_load_energy() + self._add_transfer_energy()
        if overflow is not None:
            self.energy = self.energy + self._add_refetch_energy(overflow)
        scaled_area = pb.scaled_area
        self.area = sum(scaled_area[key] * self.y[p, key] for p in slots for key in keys)
        if pb.interconnect is not None:
            self.area = self.area + self._add_interconnect_area()
        self.objective = {"latency": self.latency, "energy": self.energy, "area": self.area}

    def _same_of(self, transfer: Any) -> Any | None:
        """`_same_slot` of a transfer's group pair, built once per pair."""
        pair = (min(transfer.producer, transfer.consumer), max(transfer.producer, transfer.consumer))
        if pair not in self.same_of:
            self.same_of[pair] = self._same_slot(*pair)
        return self.same_of[pair]

    def _same_slot(self, g: int, h: int) -> Any | None:
        """A literal that may only be true when groups `g` and `h` sit on one slot (None if they never can). Only
        bounded from above: it frees their transfers, so the solver sets it whenever it can."""
        if not self.problem.can_share(g, h):
            return None
        together = []
        for p in range(self.problem.nb_slots):
            if (g, p) in self.x and (h, p) in self.x:
                literal = self.model.NewBoolVar(f"t_{g}_{h}_{p}")
                self.model.AddImplication(literal, self.x[g, p])
                self.model.AddImplication(literal, self.x[h, p])
                together.append(literal)
        if not together:
            return None
        same = self.model.NewBoolVar(f"same_{g}_{h}")
        self.model.Add(same <= sum(together))
        return same

    def _add_links(self) -> None:
        """One width per communicating group pair, among `width_levels` (hinted at its traffic-derived width), and
        each transfer a delay `ceil(bits / ceil(width / splits))` between its source and target tasks, zero when
        both groups share a slot."""
        pb, model = self.problem, self.model
        levels = pb.width_levels
        for (g, h), heuristic in pb.group_link_widths.items():
            choice = {width: model.NewBoolVar(f"z_{g}_{h}_{width}") for width in levels}
            model.AddExactlyOne(choice.values())
            width_var = model.NewIntVar(min(levels), max(levels), f"gw_{g}_{h}")
            model.Add(width_var == sum(width * literal for width, literal in choice.items()))
            hint = min((width for width in levels if width >= heuristic), default=max(levels))
            for width, literal in choice.items():
                model.AddHint(literal, width == hint)
            self.width_choice[g, h] = choice
            self.group_width[g, h] = width_var
        self.transfer_delay = []
        for index, transfer in enumerate(pb.transfers):
            pair = (min(transfer.producer, transfer.consumer), max(transfer.producer, transfer.consumer))
            same = self._same_of(transfer)
            durations = {width: transfer.duration(width) for width in levels}
            delay = model.NewIntVar(0, max(durations.values()), f"td_{index}")
            separated = sum(durations[width] * literal for width, literal in self.width_choice[pair].items())
            if same is None:
                model.Add(delay == separated)
            else:
                model.Add(delay == separated).OnlyEnforceIf(same.Not())
                model.Add(delay == 0).OnlyEnforceIf(same)
            self.transfer_delay.append(delay)
            if self.options.link_contention == "none":
                model.Add(self.start[transfer.target] >= self.end[transfer.source] + delay)

    def _add_timed_loads(self, horizon: int) -> None:
        """One load interval per task that loads anything, all on the bus's `NoOverlap`. A task starts once its
        load has ended, and a load cannot end before the previous task of its group has (fetched on demand).

        With `shared_loads`, the loads that groups sharing a slot can share (`PlacementProblem.is_shared`) leave the
        tasks' intervals: each gets one optional interval per slot, present when one of its groups sits there, that
        every task needing it on that slot waits for. It is fetched on demand of its first user task (in task
        order) when that one sits on the slot, and unconstrained otherwise."""
        pb, model = self.problem, self.model
        previous_of = pb.previous_of
        durations = pb.load_durations()
        intervals = []
        if self.options.shared_loads:
            durations = self._unshared_load_durations()
            intervals = self._add_shared_loads(horizon)
        for task, duration in enumerate(durations):
            if duration == 0:
                continue
            start = model.NewIntVar(0, horizon, f"ls_{task}")
            end = model.NewIntVar(0, horizon, f"le_{task}")
            intervals.append(model.NewIntervalVar(start, duration, end, f"load_{task}"))
            model.Add(self.start[task] >= end)
            if task in previous_of:
                model.Add(end >= self.end[previous_of[task]])
        model.AddNoOverlap(intervals)

    def _loaded(self, users: frozenset[int], p: int) -> Any | None:
        """A literal true iff one of `users` sits on slot `p` (None if none of them can), built once."""
        if (users, p) not in self.loaded_of:
            hosted = [self.x[g, p] for g in sorted(users) if (g, p) in self.x]
            literal = None
            if hosted:
                literal = self.model.NewBoolVar(f"load_{len(self.loaded_of)}_{p}")
                self.model.AddMaxEquality(literal, hosted)
            self.loaded_of[users, p] = literal
        return self.loaded_of[users, p]

    def _add_bus_time(self, overflow: Any | None = None) -> Any:
        """`ceil(bits / bandwidth)` of the off-chip loads, a tensor sent once to every slot one of its users sits
        on, plus the re-fetched `overflow` bits (`_add_overflow`)."""
        pb, model = self.problem, self.model
        variable_bits = []
        for users, bits in pb.variable_offchip_users().items():
            for p in range(pb.nb_slots):
                loaded = self._loaded(users, p)
                if loaded is not None:
                    variable_bits.append(bits * loaded)
        bandwidth = pb.bus_bandwidth
        max_bits = pb.max_bus_time * bandwidth + (self._max_overflow_bits() if overflow is not None else 0)
        bus_time = model.NewIntVar(0, math.ceil(max_bits / bandwidth), "bus_time")
        bits = pb.constant_offchip_bits + sum(variable_bits)
        if overflow is not None:
            bits = bits + overflow
        model.Add(bus_time * bandwidth >= bits)
        model.Add(bus_time * bandwidth <= bits + bandwidth - 1)
        return bus_time

    def _forbid_overflowing_designs(self) -> None:
        """A group never sits on a slot whose design one of its tiles overflows (`PlacementProblem.forbidden`),
        unless every design does: then the filter is dropped for that group, so the model stays feasible."""
        pb = self.problem
        for g in pb.groups:
            banned = [key for key in pb.design_keys if (g, key) in pb.forbidden]
            if len(banned) == len(pb.design_keys):
                logger.warning(f"CpSatBackend: every design overflows a tile of group {g}; not filtering it.")
                continue
            for p in pb.slots_of(g):
                for key in banned:
                    self.model.AddImplication(self.x[g, p], self.y[p, key].Not())

    def _shared_loads(self) -> list[Any]:
        return [load for load in self.problem.offchip_loads if self.problem.is_shared(load)]

    def _unshared_load_durations(self) -> list[int]:
        """`load_durations` without the shared loads, which `_add_shared_loads` times on their own."""
        pb = self.problem
        if not pb.offchip_loads:
            return pb.load_durations()
        bits = [0] * len(pb.tasks)
        for load in pb.offchip_loads:
            if not pb.is_shared(load):
                for task in load.first_task.values():
                    bits[task] += load.bits
        return [math.ceil(b / pb.bus_bandwidth) for b in bits]

    def _add_shared_loads(self, horizon: int) -> list[Any]:
        """The optional bus intervals of the shared loads (see `_add_timed_loads`)."""
        pb, model = self.problem, self.model
        previous_of = pb.previous_of
        intervals = []
        for index, load in enumerate(self._shared_loads()):
            duration = math.ceil(load.bits / pb.bus_bandwidth)
            if duration == 0:
                continue
            first_group = min(load.first_task, key=load.first_task.get)
            first_task = load.first_task[first_group]
            for p in range(pb.nb_slots):
                loaded = self._loaded(load.users, p)
                if loaded is None:
                    continue
                start = model.NewIntVar(0, horizon, f"sls_{index}_{p}")
                end = model.NewIntVar(0, horizon, f"sle_{index}_{p}")
                intervals.append(model.NewOptionalIntervalVar(start, duration, end, loaded, f"shared_{index}_{p}"))
                for g, task in load.first_task.items():
                    if (g, p) in self.x:
                        model.Add(self.start[task] >= end).OnlyEnforceIf(self.x[g, p])
                if first_task in previous_of and (first_group, p) in self.x:
                    model.Add(end >= self.end[previous_of[first_task]]).OnlyEnforceIf(self.x[first_group, p])
        return intervals

    def _unshared_load_energy(self) -> dict[int, dict[str, int]]:
        """Group -> design -> energy of the loads it makes on its own (each of their groups loads them)."""
        energy: dict[int, dict[str, int]] = {}
        for load in self.problem.offchip_loads:
            if self.problem.is_shared(load):
                continue
            for g in load.users:
                per_design = energy.setdefault(g, {})
                for key, value in load.energy.items():
                    per_design[key] = per_design.get(key, 0) + value
        return energy

    def _add_shared_load_energy(self) -> Any:
        """Each shared load costs its energy on every slot it is loaded on, at that slot's design."""
        terms = []
        for index, load in enumerate(self._shared_loads()):
            for p in range(self.problem.nb_slots):
                loaded = self._loaded(load.users, p)
                if loaded is None:
                    continue
                energy = self.model.NewIntVar(0, max(load.energy.values(), default=0), f"sle_e_{index}_{p}")
                design_energy = sum(load.energy[key] * self.y[p, key] for key in self.problem.design_keys)
                self.model.Add(energy == design_energy).OnlyEnforceIf(loaded)
                self.model.Add(energy == 0).OnlyEnforceIf(loaded.Not())
                terms.append(energy)
        return sum(terms)

    def _add_transfer_energy(self) -> Any:
        """A transfer between groups on different slots costs the read at the producer's design and the write at
        the consumer's; nothing when they share a slot."""
        pb, model = self.problem, self.model
        terms = []
        for index, (transfer, (send, receive)) in enumerate(zip(pb.transfers, pb.transfer_energy, strict=True)):
            parts = []
            for group, values, side in ((transfer.producer, send, "s"), (transfer.consumer, receive, "r")):
                part = model.NewIntVar(0, max(values.values(), default=0), f"te{side}_{index}")
                for p in pb.slots_of(group):
                    design_value = sum(values[key] * self.y[p, key] for key in pb.design_keys)
                    model.Add(part == design_value).OnlyEnforceIf(self.x[group, p])
                parts.append(part)
            energy = model.NewIntVar(0, max(send.values(), default=0) + max(receive.values(), default=0), f"te_{index}")
            same = self._same_of(transfer)
            if same is None:
                model.Add(energy == sum(parts))
            else:
                model.Add(energy == sum(parts)).OnlyEnforceIf(same.Not())
                model.Add(energy == 0).OnlyEnforceIf(same)
            terms.append(energy)
        return sum(terms)

    def _max_overflow_bits(self) -> int:
        return sum(load.bits * len(load.users) for load in self.problem.offchip_loads)

    def _add_overflow(self) -> Any:
        """Bits of loads a slot cannot keep: per slot and layer stack, the loads it holds while the stack runs
        beyond its design's `resident_capacity`, each re-fetched at least once (see
        `PlacementProblem.overflow_bits`). Only bounded from below."""
        pb, model = self.problem, self.model
        resident: dict[tuple[int, int], list[Any]] = {}
        for load in pb.offchip_loads:
            stack = pb.stack_of_group.get(min(load.users), 0)
            for p in range(pb.nb_slots):
                if pb.is_shared(load):
                    loaded = self._loaded(load.users, p)
                    hosted = [] if loaded is None else [loaded]
                else:
                    hosted = [self.x[g, p] for g in load.users if (g, p) in self.x]
                resident.setdefault((p, stack), []).extend(load.bits * literal for literal in hosted)
        max_bits = self._max_overflow_bits()
        terms = []
        for (p, stack), held in resident.items():
            if not held:
                continue
            capacity = sum(pb.resident_capacity[key] * self.y[p, key] for key in pb.design_keys)
            overflow = model.NewIntVar(0, max_bits, f"ov_{p}_{stack}")
            model.Add(overflow >= sum(held) - capacity)
            terms.append(overflow)
        total = model.NewIntVar(0, max_bits, "overflow")
        model.Add(total == sum(terms))
        self.overflow = total
        return total

    def _add_refetch_energy(self, overflow: Any) -> Any:
        """`ceil(overflow * pJ per bit)`, as `PlacementProblem.refetch_energy` rounds it."""
        numerator, denominator = self.problem.refetch_ratio
        energy = self.model.NewIntVar(0, self.problem.refetch_energy(self._max_overflow_bits()), "refetch_energy")
        self.model.Add(energy * denominator >= overflow * numerator)
        return energy

    def _add_link_contention(self, horizon: int) -> None:
        """Transfers as intervals of their own, between their source task's end and their target task's start,
        fetched on demand (they cannot end before the previous task of their consumer's group has). With `"pair"`,
        the transfers of one directed group pair share their link `splits` at a time (Stream's bandwidth
        fraction), as `ScheduleOptimizationStage._add_link_transfers` does. With `"link"`, every transfer on the
        physical link from slot `p` to slot `q` shares it: an optional copy per `(p, q)` its groups may sit on,
        demanding `1 / splits` of the link."""
        pb, model = self.problem, self.model
        previous_of = pb.previous_of
        intervals_of_pair: dict[tuple[int, int], list[Any]] = {}
        capacity_of_pair: dict[tuple[int, int], int] = {}
        copies_of_link: dict[tuple[int, int], list[tuple[Any, int]]] = {}
        for index, (transfer, delay) in enumerate(zip(pb.transfers, self.transfer_delay, strict=True)):
            start = model.NewIntVar(0, horizon, f"ts_{index}")
            end = model.NewIntVar(0, horizon, f"tend_{index}")
            interval = model.NewIntervalVar(start, delay, end, f"transfer_{index}")
            model.Add(start >= self.end[transfer.source])
            model.Add(self.start[transfer.target] >= end)
            if transfer.target in previous_of:
                model.Add(end >= self.end[previous_of[transfer.target]])
            g, h = transfer.producer, transfer.consumer
            if self.options.link_contention == "pair":
                intervals_of_pair.setdefault((g, h), []).append(interval)
                capacity_of_pair[g, h] = max(capacity_of_pair.get((g, h), 1), transfer.splits)
                continue
            demand = math.ceil(LINK_SHARE_SCALE / transfer.splits)
            for p in pb.slots_of(g):
                for q in pb.slots_of(h):
                    if p == q:
                        continue
                    on_link = model.NewBoolVar(f"tl_{index}_{p}_{q}")
                    model.AddBoolAnd([self.x[g, p], self.x[h, q]]).OnlyEnforceIf(on_link)
                    model.AddBoolOr([self.x[g, p].Not(), self.x[h, q].Not(), on_link])
                    copy = model.NewOptionalIntervalVar(start, delay, end, on_link, f"tc_{index}_{p}_{q}")
                    copies_of_link.setdefault((p, q), []).append((copy, demand))
        for pair, intervals in intervals_of_pair.items():
            if capacity_of_pair[pair] == 1:
                model.AddNoOverlap(intervals)
            else:
                model.AddCumulative(intervals, [1] * len(intervals), capacity_of_pair[pair])
        for copies in copies_of_link.values():
            model.AddCumulative([copy for copy, _ in copies], [demand for _, demand in copies], LINK_SHARE_SCALE)
        if copies_of_link:
            nb_copies = sum(len(copies) for copies in copies_of_link.values())
            logger.info(f"CpSatBackend: {nb_copies} transfer copies over {len(copies_of_link)} possible links.")

    def _add_interconnect_area(self) -> Any:  # noqa: PLR0914
        """See `ScheduleOptimizationStage._add_interconnect_area`: a link joins slots `p < q` as soon as two
        communicating groups sit on them, as wide as the widest such pair; per used slot, the router and the wires
        follow its link count `k` and total width `W`."""
        pb, model = self.problem, self.model
        ic = pb.interconnect
        max_width = ic.max_width
        pairs: dict[tuple[int, int], tuple[Any, Any]] = {}
        for (g, h), width in pb.group_link_widths.items():
            for p in pb.slots_of(g):
                for q in pb.slots_of(h):
                    if q == p:
                        continue
                    a, b = min(p, q), max(p, q)
                    if (a, b) not in pairs:
                        link = model.NewBoolVar(f"l_{a}_{b}")
                        link_width = model.NewIntVar(0, max_width, f"w_{a}_{b}")
                        model.Add(link_width <= max_width * link)
                        pairs[a, b] = (link, link_width)
                    link, link_width = pairs[a, b]
                    both = self.x[g, p] + self.x[h, q] - 1
                    model.Add(link >= both)
                    if (g, h) in self.group_width:
                        model.Add(link_width >= self.group_width[g, h] - max_width + max_width * both)
                    else:
                        model.Add(link_width >= width * both)
        side_of = pb.side_of
        max_side = max(side_of.values(), default=0)
        terms = []
        for p in range(pb.nb_slots):
            incident = [pair for pair in pairs if p in pair]
            side = sum(side_of[key] * self.y[p, key] for key in pb.design_keys)
            terms.append(ic.ge * ic.per_used * self.used[p] + ic.wire * ic.bus_wires * side)
            if not incident:
                continue
            nb = len(incident)
            ports = model.NewIntVar(0, nb, f"k_{p}")
            model.Add(ports == sum(pairs[pair][0] for pair in incident))
            width = model.NewIntVar(0, nb * max_width, f"W_{p}")
            model.Add(width == sum(pairs[pair][1] for pair in incident))
            port_width = model.NewIntVar(0, nb * nb * max_width, f"kW_{p}")
            model.AddMultiplicationEquality(port_width, [ports, width])
            half_wires = model.NewIntVar(0, nb * (max_width + ic.half_fixed_wires), f"hw_{p}")
            model.Add(half_wires == width + ic.half_fixed_wires * ports)
            wire_length = model.NewIntVar(0, max_side * nb * (max_width + ic.half_fixed_wires), f"wl_{p}")
            side_var = model.NewIntVar(0, max_side, f"side_{p}")
            model.Add(side_var == side)
            model.AddMultiplicationEquality(wire_length, [side_var, half_wires])
            terms.append(
                ic.ge * (ic.per_port * ports + ic.per_port_bit * port_width + ic.per_bit * width)
                + ic.wire * wire_length
            )
        self.link_vars = pairs
        return sum(terms)

    # --------------------------------------------------------------------------------------- solving -------

    def solve(self, objective: str, bounds: dict[str, int], time_limit: float) -> SolveResult:
        """Each bound is a constraint guarded by a fresh literal, and only this solve's literals are assumed:
        earlier bounds stay in the model but constrain nothing."""
        model = self.model
        literals = []
        for name, bound in bounds.items():
            literal = model.NewBoolVar(f"eps_{name}_{self.nb_solves}")
            model.Add(self.objective[name] <= bound).OnlyEnforceIf(literal)
            literals.append(literal)
        self.nb_solves += 1
        model.ClearAssumptions()
        model.AddAssumptions(literals)
        model.Minimize(self.objective[objective])
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = self.options.workers
        solver.parameters.cp_model_presolve = self.options.cpsat_presolve
        t0 = time.perf_counter()
        status = solver.Solve(model)
        elapsed = time.perf_counter() - t0
        name = STATUS.get(status, "UNKNOWN")
        if name not in ("OPTIMAL", "FEASIBLE"):
            return SolveResult(status=name, solution=None, time_s=elapsed)
        model_values = {key: int(solver.Value(expr)) for key, expr in self.objective.items()}
        starts = [solver.Value(start) for start in self.start]
        solution, values = make_solution(
            self.problem,
            slot_of_group={g: p for (g, p), var in self.x.items() if solver.Value(var)},
            design_of_slot={p: key for (p, key), var in self.y.items() if solver.Value(var)},
            starts=starts,
            offchip_bus=self.options.offchip_bus,
            model_values=model_values,
            comm_energy=self.options.comm_energy,
            capacity=self.options.capacity,
            group_link_widths=(
                {
                    pair: next(width for width, var in choice.items() if solver.Value(var))
                    for pair, choice in self.width_choice.items()
                }
                if self.width_choice
                else None
            ),
        )
        model.ClearHints()
        for var in [*self.x.values(), *self.y.values()]:
            model.AddHint(var, solver.Value(var))
        for choice in self.width_choice.values():
            for var in choice.values():
                model.AddHint(var, solver.Value(var))
        return SolveResult(
            status=name,
            solution=solution,
            time_s=elapsed,
            objective_value=values[objective],
            objective_bound=solver.BestObjectiveBound(),
            starts=starts,
        )

    def size(self) -> dict[str, int]:
        proto = self.model.Proto()
        return {"variables": len(proto.variables), "constraints": len(proto.constraints)}
