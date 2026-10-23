"""The placement model with OR-Tools CP-SAT (`docs/source/schedule_optimization_formulation.md`, section 5).

Same semantics as `GurobiBackend`, written with CP-SAT's own constructs: a task is an optional interval on each
slot its group may use, present when the group sits there, and the slot's intervals do not overlap. Products of
variables (the router's ports times their width, a slot's side times its wires) are `AddMultiplicationEquality`.

Transfers do not contend with each other (left to the measurement): each one only delays its consumer task by its
duration at its group pair's chosen width, or not at all when both groups share a slot.
"""

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

    def _build(self) -> None:  # noqa: PLR0912, PLR0915
        pb, model, options = self.problem, self.model, self.options
        keys = pb.design_keys
        slots = range(pb.nb_slots)
        horizon = pb.horizon(options.offchip_bus, options.link_transfers)

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
        for a, b in pb.sibling_pairs:
            for p in slots:
                if (a, p) in self.x and (b, p) in self.x:
                    model.AddBoolOr([self.x[a, p].Not(), self.x[b, p].Not()])

        # Energy and durations follow the design of the group's slot.
        group_energy = {}
        for g in pb.groups:
            energies = pb.group_energy[g]
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

        self.width_choice: dict[tuple[int, int], dict[int, Any]] = {}
        self.group_width: dict[tuple[int, int], Any] = {}
        if options.link_transfers == "timed":
            self._add_links()

        finishes = list(self.end)
        if options.offchip_bus == "timed":
            self._add_timed_loads(horizon)
        if options.offchip_bus != "none":
            finishes.append(self._add_bus_time())
        self.latency = model.NewIntVar(0, horizon, "latency")
        model.AddMaxEquality(self.latency, finishes)
        self.energy = sum(group_energy.values())
        scaled_area = pb.scaled_area
        self.area = sum(scaled_area[key] * self.y[p, key] for p in slots for key in keys)
        if pb.interconnect is not None:
            self.area = self.area + self._add_interconnect_area()
        self.objective = {"latency": self.latency, "energy": self.energy, "area": self.area}

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
        same_of: dict[tuple[int, int], Any] = {}
        for index, transfer in enumerate(pb.transfers):
            pair = (min(transfer.producer, transfer.consumer), max(transfer.producer, transfer.consumer))
            if pair not in same_of:
                same_of[pair] = self._same_slot(*pair)
            same = same_of[pair]
            durations = {width: transfer.duration(width) for width in levels}
            delay = model.NewIntVar(0, max(durations.values()), f"td_{index}")
            separated = sum(durations[width] * literal for width, literal in self.width_choice[pair].items())
            if same is None:
                model.Add(delay == separated)
            else:
                model.Add(delay == separated).OnlyEnforceIf(same.Not())
                model.Add(delay == 0).OnlyEnforceIf(same)
            model.Add(self.start[transfer.target] >= self.end[transfer.source] + delay)

    def _add_timed_loads(self, horizon: int) -> None:
        """One load interval per task that loads anything, all on the bus's `NoOverlap`. A task starts once its
        load has ended, and a load cannot end before the previous task of its group has (fetched on demand)."""
        pb, model = self.problem, self.model
        previous_of = pb.previous_of
        intervals = []
        for task, duration in enumerate(pb.load_durations()):
            if duration == 0:
                continue
            start = model.NewIntVar(0, horizon, f"ls_{task}")
            end = model.NewIntVar(0, horizon, f"le_{task}")
            intervals.append(model.NewIntervalVar(start, duration, end, f"load_{task}"))
            model.Add(self.start[task] >= end)
            if task in previous_of:
                model.Add(end >= self.end[previous_of[task]])
        model.AddNoOverlap(intervals)

    def _add_bus_time(self) -> Any:
        """`ceil(bits / bandwidth)` of the off-chip loads, a tensor sent once to every slot one of its users sits
        on."""
        pb, model = self.problem, self.model
        variable_bits = []
        for index, (users, bits) in enumerate(pb.variable_offchip_users().items()):
            for p in range(pb.nb_slots):
                hosted = [self.x[g, p] for g in users if (g, p) in self.x]
                if hosted:
                    loaded = model.NewBoolVar(f"load_{index}_{p}")
                    model.AddMaxEquality(loaded, hosted)
                    variable_bits.append(bits * loaded)
        bandwidth = pb.bus_bandwidth
        bus_time = model.NewIntVar(0, pb.max_bus_time, "bus_time")
        bits = pb.constant_offchip_bits + sum(variable_bits)
        model.Add(bus_time * bandwidth >= bits)
        model.Add(bus_time * bandwidth <= bits + bandwidth - 1)
        return bus_time

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
