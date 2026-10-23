"""The placement model as a Gurobi MILP (`docs/source/schedule_optimization_formulation.md`, section 5).

Same semantics as `CpSatBackend`, linearized:

- a group's cost on a slot follows the slot's design through `z[g, p, c] = x[g, p] y[p, c]` (`sum_c z = x`,
  `z <= y`), so durations and energies are linear in `z`;
- two groups share a slot when `q[g, h] = sum_p x[g, p] x[h, p]` (one linearized product per slot) is 1, and then
  every pair of their tasks that no precedence orders gets an ordering binary with big-M disjunctions;
- timed loads are disjunctions between every pair of loads on the one bus;
- the router's `k W` product uses a one-hot of the slot's port count `k`, and a slot's wire length (its side,
  set by its design, times its wires) a product per design. Both are only bounded from below, which is exact at an
  optimum as area only ever appears with positive coefficients.

Every variable is integer, as all data is: objective values are evaluated on the rounded solution, so they can be
used as the bounds of the next solve.
"""

import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from stream.stages.generation.placement_optimization.backend import (
    BackendOptions,
    SolveResult,
    Status,
    make_solution,
)
from stream.stages.generation.placement_optimization.problem import PlacementProblem

BINARY_THRESHOLD = 0.5


def _value(expr: Any) -> int:
    """The value of a linear expression on the rounded values of its (integer) variables."""
    if isinstance(expr, gp.Var):
        return round(expr.X)
    if isinstance(expr, int | float):
        return round(expr)
    total = expr.getConstant()
    for i in range(expr.size()):
        total += expr.getCoeff(i) * round(expr.getVar(i).X)
    return round(total)


class GurobiBackend:
    name = "gurobi"

    def __init__(self, problem: PlacementProblem, options: BackendOptions):
        self.problem = problem
        self.options = options
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        self.model = gp.Model("placement", env=self.env)
        params = self.model.Params
        params.Threads = options.workers
        params.MIPGap = options.mip_gap
        params.IntegralityFocus = 1
        if options.log_path is not None:
            params.LogFile = options.log_path
        self.unordered_pairs = 0
        self._build()
        self.model.update()

    def __del__(self) -> None:
        try:
            self.model.dispose()
            self.env.dispose()
        except Exception:  # noqa: S110 -- interpreter shutdown
            pass

    # --------------------------------------------------------------------------------------- model ---------

    def _build(self) -> None:  # noqa: PLR0912, PLR0915
        pb, m, options = self.problem, self.model, self.options
        keys = pb.design_keys
        slots = range(pb.nb_slots)
        horizon = pb.horizon(options.offchip_bus, options.link_transfers)
        self.horizon = horizon

        self.y = {(p, key): m.addVar(vtype=GRB.BINARY, name=f"y_{p}_{key}") for p in slots for key in keys}
        self.used = [m.addVar(vtype=GRB.BINARY, name=f"u_{p}") for p in slots]
        self.x: dict[tuple[int, int], gp.Var] = {}
        for g in pb.groups:
            for p in pb.slots_of(g):
                self.x[g, p] = m.addVar(vtype=GRB.BINARY, name=f"x_{g}_{p}")
            m.addConstr(gp.quicksum(self.x[g, p] for p in pb.slots_of(g)) == 1)
        for p in slots:
            m.addConstr(gp.quicksum(self.y[p, key] for key in keys) == self.used[p])
            hosted = [self.x[g, p] for g in pb.groups if (g, p) in self.x]
            m.addConstr(gp.quicksum(hosted) >= self.used[p])
            for var in hosted:
                m.addConstr(var <= self.used[p])
            if p + 1 < pb.nb_slots:
                m.addConstr(self.used[p + 1] <= self.used[p])
        for a, b in pb.sibling_pairs:
            for p in slots:
                if (a, p) in self.x and (b, p) in self.x:
                    m.addConstr(self.x[a, p] + self.x[b, p] <= 1)

        # z = x * y: the design a group runs on.
        self.z: dict[tuple[int, int, str], gp.Var] = {}
        for (g, p), x in self.x.items():
            for key in keys:
                self.z[g, p, key] = var = m.addVar(vtype=GRB.BINARY, name=f"z_{g}_{p}_{key}")
                m.addConstr(var <= self.y[p, key])
            m.addConstr(gp.quicksum(self.z[g, p, key] for key in keys) == x)
        design_terms = {g: [(p, key) for p in pb.slots_of(g) for key in keys] for g in pb.groups}
        self.energy = gp.quicksum(
            pb.group_energy[g][key] * self.z[g, p, key] for g in pb.groups for p, key in design_terms[g]
        )

        self.start, self.end = [], []
        for task, ((g, _index), durations) in enumerate(zip(pb.tasks, pb.duration, strict=True)):
            start = m.addVar(vtype=GRB.INTEGER, lb=0, ub=horizon, name=f"s_{task}")
            end = m.addVar(vtype=GRB.INTEGER, lb=0, ub=horizon, name=f"c_{task}")
            m.addConstr(end == start + gp.quicksum(durations[key] * self.z[g, p, key] for p, key in design_terms[g]))
            self.start.append(start)
            self.end.append(end)
        for a, b in pb.dependencies + pb.chains:
            m.addConstr(self.start[b] >= self.end[a])

        # Same-slot indicators of the group pairs that need one: those with unordered tasks or transfers.
        unordered = pb.unordered_task_pairs()
        self.unordered_pairs = len(unordered)
        needed = {(min(pb.tasks[k][0], pb.tasks[k2][0]), max(pb.tasks[k][0], pb.tasks[k2][0])) for k, k2 in unordered}
        if options.link_transfers == "timed":
            needed |= {
                (min(t.producer, t.consumer), max(t.producer, t.consumer))
                for t in pb.transfers
                if pb.can_share(t.producer, t.consumer)
            }
        self.q: dict[tuple[int, int], Any] = {pair: self._same_slot(*pair) for pair in sorted(needed)}
        for k, k2 in unordered:
            g, h = pb.tasks[k][0], pb.tasks[k2][0]
            q = self.q[min(g, h), max(g, h)]
            order = m.addVar(vtype=GRB.BINARY, name=f"o_{k}_{k2}")  # 1: k before k2
            m.addConstr(self.start[k2] >= self.end[k] - horizon * (1 - order) - horizon * (1 - q))
            m.addConstr(self.start[k] >= self.end[k2] - horizon * order - horizon * (1 - q))

        self.width_choice: dict[tuple[int, int], dict[int, gp.Var]] = {}
        self.group_width: dict[tuple[int, int], Any] = {}
        if options.link_transfers == "timed":
            self._add_links()

        self.latency = m.addVar(vtype=GRB.INTEGER, lb=0, ub=horizon, name="latency")
        for end in self.end:
            m.addConstr(self.latency >= end)
        if options.offchip_bus == "timed":
            self._add_timed_loads(horizon)
        if options.offchip_bus != "none":
            m.addConstr(self.latency >= self._add_bus_time())
        scaled_area = pb.scaled_area
        self.area = gp.quicksum(scaled_area[key] * self.y[p, key] for p in slots for key in keys)
        if pb.interconnect is not None:
            self.area = self.area + self._add_interconnect_area()
        self.objective = {"latency": self.latency, "energy": self.energy, "area": self.area}

    def _same_slot(self, g: int, h: int) -> Any:
        """`sum_p x[g, p] x[h, p]`, exact (both bounds), or 0 when the two groups never share a slot."""
        m = self.model
        terms = []
        for p in range(self.problem.nb_slots):
            if (g, p) in self.x and (h, p) in self.x:
                v = m.addVar(vtype=GRB.BINARY, name=f"v_{g}_{h}_{p}")
                m.addConstr(v <= self.x[g, p])
                m.addConstr(v <= self.x[h, p])
                m.addConstr(v >= self.x[g, p] + self.x[h, p] - 1)
                terms.append(v)
        return gp.quicksum(terms) if terms else 0

    def _add_links(self) -> None:
        pb, m = self.problem, self.model
        levels = pb.width_levels
        for (g, h), heuristic in pb.group_link_widths.items():
            choice = {width: m.addVar(vtype=GRB.BINARY, name=f"w_{g}_{h}_{width}") for width in levels}
            m.addConstr(gp.quicksum(choice.values()) == 1)
            hint = min((width for width in levels if width >= heuristic), default=max(levels))
            for width, var in choice.items():
                var.Start = 1 if width == hint else 0
            self.width_choice[g, h] = choice
            self.group_width[g, h] = gp.quicksum(width * var for width, var in choice.items())
        for index, transfer in enumerate(pb.transfers):
            pair = (min(transfer.producer, transfer.consumer), max(transfer.producer, transfer.consumer))
            durations = {width: transfer.duration(width) for width in levels}
            separated = gp.quicksum(durations[width] * var for width, var in self.width_choice[pair].items())
            longest = max(durations.values())
            delay = m.addVar(vtype=GRB.INTEGER, lb=0, ub=longest, name=f"td_{index}")
            m.addConstr(delay >= separated - longest * self.q.get(pair, 0))
            m.addConstr(self.start[transfer.target] >= self.end[transfer.source] + delay)

    def _add_timed_loads(self, horizon: int) -> None:
        pb, m = self.problem, self.model
        previous_of = pb.previous_of
        loads = []
        for task, duration in enumerate(pb.load_durations()):
            if duration == 0:
                continue
            start = m.addVar(vtype=GRB.INTEGER, lb=0, ub=horizon, name=f"ls_{task}")
            m.addConstr(self.start[task] >= start + duration)
            if task in previous_of:
                m.addConstr(start + duration >= self.end[previous_of[task]])
            loads.append((start, duration))
        for i, (start_a, duration_a) in enumerate(loads):
            for start_b, duration_b in loads[i + 1 :]:
                order = m.addVar(vtype=GRB.BINARY)
                m.addConstr(start_b >= start_a + duration_a - horizon * (1 - order))
                m.addConstr(start_a >= start_b + duration_b - horizon * order)

    def _add_bus_time(self) -> gp.Var:
        pb, m = self.problem, self.model
        variable_bits = []
        for index, (users, bits) in enumerate(pb.variable_offchip_users().items()):
            for p in range(pb.nb_slots):
                hosted = [self.x[g, p] for g in users if (g, p) in self.x]
                if hosted:
                    loaded = m.addVar(vtype=GRB.BINARY, name=f"load_{index}_{p}")
                    for var in hosted:
                        m.addConstr(loaded >= var)
                    variable_bits.append(bits * loaded)
        bus_time = m.addVar(vtype=GRB.INTEGER, lb=0, ub=pb.max_bus_time, name="bus_time")
        m.addConstr(bus_time * pb.bus_bandwidth >= pb.constant_offchip_bits + gp.quicksum(variable_bits))
        return bus_time

    def _add_interconnect_area(self) -> gp.LinExpr:  # noqa: PLR0914
        pb, m = self.problem, self.model
        ic = pb.interconnect
        max_width = ic.max_width
        pairs: dict[tuple[int, int], tuple[gp.Var, gp.Var]] = {}
        for (g, h), width in pb.group_link_widths.items():
            for p in pb.slots_of(g):
                for q in pb.slots_of(h):
                    if q == p:
                        continue
                    a, b = min(p, q), max(p, q)
                    if (a, b) not in pairs:
                        link = m.addVar(vtype=GRB.BINARY, name=f"l_{a}_{b}")
                        link_width = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max_width, name=f"lw_{a}_{b}")
                        m.addConstr(link_width <= max_width * link)
                        pairs[a, b] = (link, link_width)
                    link, link_width = pairs[a, b]
                    both = self.x[g, p] + self.x[h, q] - 1
                    m.addConstr(link >= both)
                    if (g, h) in self.group_width:
                        m.addConstr(link_width >= self.group_width[g, h] - max_width + max_width * both)
                    else:
                        m.addConstr(link_width >= width * both)
        side_of = pb.side_of
        terms = []
        for p in range(pb.nb_slots):
            incident = [pair for pair in pairs if p in pair]
            side = gp.quicksum(side_of[key] * self.y[p, key] for key in pb.design_keys)
            terms.append(ic.ge * ic.per_used * self.used[p] + ic.wire * ic.bus_wires * side)
            if not incident:
                continue
            nb = len(incident)
            ports = gp.quicksum(pairs[pair][0] for pair in incident)
            width = gp.quicksum(pairs[pair][1] for pair in incident)
            # k W >= j W when the slot has j ports (one-hot `count`).
            count = [m.addVar(vtype=GRB.BINARY, name=f"k_{p}_{j}") for j in range(nb + 1)]
            m.addConstr(gp.quicksum(count) == 1)
            m.addConstr(ports == gp.quicksum(j * var for j, var in enumerate(count)))
            max_total = nb * max_width
            port_width = m.addVar(vtype=GRB.INTEGER, lb=0, ub=nb * max_total, name=f"kW_{p}")
            for j, var in enumerate(count[1:], start=1):
                m.addConstr(port_width >= j * width - j * max_total * (1 - var))
            # side * half wires: one product per design of the slot.
            half_wires = width + ic.half_fixed_wires * ports
            max_half_wires = nb * (max_width + ic.half_fixed_wires)
            wire_length = []
            for key in pb.design_keys:
                h = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max_half_wires, name=f"hw_{p}_{key}")
                m.addConstr(h >= half_wires - max_half_wires * (1 - self.y[p, key]))
                wire_length.append(side_of[key] * h)
            terms.append(
                ic.ge * (ic.per_port * ports + ic.per_port_bit * port_width + ic.per_bit * width)
                + ic.wire * gp.quicksum(wire_length)
            )
        self.link_vars = pairs
        return gp.quicksum(terms)

    # --------------------------------------------------------------------------------------- solving -------

    def solve(self, objective: str, bounds: dict[str, int], time_limit: float) -> SolveResult:
        """The bounds are constraints removed after the solve; the previous solution is the next one's start."""
        m = self.model
        bound_constraints = [m.addConstr(self.objective[name] <= bound) for name, bound in bounds.items()]
        m.setObjective(self.objective[objective], GRB.MINIMIZE)
        m.Params.TimeLimit = time_limit
        t0 = time.perf_counter()
        m.optimize()
        elapsed = time.perf_counter() - t0
        try:
            if m.SolCount == 0:
                status: Status = "INFEASIBLE" if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) else "UNKNOWN"
                return SolveResult(status=status, solution=None, time_s=elapsed)
            status = "OPTIMAL" if m.Status == GRB.OPTIMAL else "FEASIBLE"
            model_values = {key: _value(expr) for key, expr in self.objective.items()}
            starts = [round(start.X) for start in self.start]
            solution, values = make_solution(
                self.problem,
                slot_of_group={g: p for (g, p), var in self.x.items() if var.X > BINARY_THRESHOLD},
                design_of_slot={p: key for (p, key), var in self.y.items() if var.X > BINARY_THRESHOLD},
                starts=starts,
                offchip_bus=self.options.offchip_bus,
                model_values=model_values,
                group_link_widths=(
                    {
                        pair: next(width for width, var in choice.items() if var.X > BINARY_THRESHOLD)
                        for pair, choice in self.width_choice.items()
                    }
                    if self.width_choice
                    else None
                ),
            )
            for var in m.getVars():
                var.Start = var.X
            return SolveResult(
                status=status,
                solution=solution,
                time_s=elapsed,
                objective_value=values[objective],
                objective_bound=m.ObjBound,
                starts=starts,
            )
        finally:
            m.remove(bound_constraints)
            m.update()

    def size(self) -> dict[str, int]:
        return {
            "variables": self.model.NumVars,
            "constraints": self.model.NumConstrs,
            "unordered_task_pairs": self.unordered_pairs,
        }
