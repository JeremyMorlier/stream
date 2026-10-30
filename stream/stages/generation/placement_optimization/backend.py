"""What the stage needs from a solver: build the model of a `PlacementProblem` once, then minimize one objective
under epsilon bounds on the others, as many times as the sweep asks."""

import logging
from dataclasses import dataclass
from typing import Literal, Protocol

from stream.stages.generation.placement_optimization.problem import PlacementProblem
from stream.stages.generation.schedule_optimization import AREA_SCALE, ENERGY_SCALE, ModelSolution

logger = logging.getLogger(__name__)

Status = Literal["OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN"]


@dataclass(frozen=True)
class BackendOptions:
    offchip_bus: Literal["none", "aggregate", "timed"] = "none"
    link_transfers: Literal["none", "timed"] = "timed"
    workers: int = 8
    # Gurobi's relative MIP gap; 0 to prove optimality as CP-SAT does, for a like-for-like comparison.
    mip_gap: float = 0.0
    log_path: str | None = None
    # CP-SAT 9.15.6755 presolve proves a wrong optimum on this model once transfers are freed for groups sharing a
    # core: on the ResNet bottleneck it reports an energy optimum of 220.7M pJ as OPTIMAL while 191.9M is feasible
    # (found by CP-SAT without presolve, by Gurobi, and by CP-SAT 9.12 / 9.14 on the exported model). Off until fixed.
    cpsat_presolve: bool = False
    # Refinements of the CP-SAT model towards what Stream measures; see the module docstring of `cpsat_backend`.
    # The defaults keep the model as it was, and `GurobiBackend` implements none of them.
    shared_loads: bool = False  # timed bus: a tensor shared by groups on one slot is loaded once
    comm_energy: bool = False  # energy of off-chip loads and core-to-core transfers (memory reads and writes)
    capacity: Literal["none", "filter", "overflow"] = "none"  # core memory capacity
    link_contention: Literal["none", "pair", "link"] = "none"  # transfers sharing a link

    @property
    def refinements(self) -> dict[str, object]:
        """The refinements set away from their defaults."""
        defaults = BackendOptions()
        return {
            name: getattr(self, name)
            for name in ("shared_loads", "comm_energy", "capacity", "link_contention")
            if getattr(self, name) != getattr(defaults, name)
        }


@dataclass
class SolveResult:
    status: Status
    solution: ModelSolution | None
    time_s: float
    objective_value: int | None = None  # the minimized objective (scaled units)
    objective_bound: float | None = None  # the solver's best lower bound on it
    starts: list[int] | None = None  # start of every task

    @property
    def gap(self) -> float | None:
        """Relative gap between the solution and the bound on the minimized objective."""
        if self.objective_value is None or self.objective_bound is None:
            return None
        return max(0.0, self.objective_value - self.objective_bound) / max(1.0, abs(self.objective_value))


class PlacementBackend(Protocol):
    name: str

    def __init__(self, problem: PlacementProblem, options: BackendOptions) -> None: ...

    def solve(self, objective: str, bounds: dict[str, int], time_limit: float) -> SolveResult:
        """Minimize `objective` subject to `objective_i <= bound_i` for each entry of `bounds` (scaled units).
        Bounds hold for this solve only; the previous solution warm-starts the next."""
        ...

    def size(self) -> dict[str, int]:
        """Variables and constraints of the model."""
        ...


def make_solution(  # noqa: PLR0913, PLR0917
    problem: PlacementProblem,
    slot_of_group: dict[int, int],
    design_of_slot: dict[int, str],
    starts: list[int],
    group_link_widths: dict[tuple[int, int], int] | None,
    offchip_bus: str,
    model_values: dict[str, int],
    comm_energy: bool = False,
    capacity: str = "none",
) -> tuple[ModelSolution, dict[str, int]]:
    """A `ModelSolution` from a backend's decisions, with its objectives recomputed by `PlacementProblem.evaluate`
    (scaled units, returned too): they never exceed the model's own values `model_values`, and the next solve can
    be bounded by them. A task's tiles all take its start time and keep their sub id order inside it."""
    used = set(slot_of_group.values())
    design_of_slot = {slot: key for slot, key in design_of_slot.items() if slot in used}
    values = problem.evaluate(
        slot_of_group,
        design_of_slot,
        starts,
        group_link_widths,
        offchip_bus,
        comm_energy=comm_energy,
        capacity=capacity,
    )
    for name, value in values.items():
        if value > model_values[name]:
            logger.warning(f"make_solution: {name} {value} recomputed above the model's {model_values[name]}.")
    timed = [
        (starts[task], tile.sub_id, tile.id) for task, members in enumerate(problem.task_tiles) for tile in members
    ]
    solution = ModelSolution(
        slot_of_group=slot_of_group,
        design_of_slot=design_of_slot,
        latency=values["latency"],
        energy=values["energy"] * ENERGY_SCALE,
        area=values["area"] / AREA_SCALE,
        scheduling_order=tuple((layer, sub_id) for _start, sub_id, layer in sorted(timed)),
        group_link_widths=group_link_widths,
    )
    return solution, values
